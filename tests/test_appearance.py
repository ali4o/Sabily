"""Appearance flags (logo/hashtag/bottom-text) + quality lock — no GPU, no network.

No real ffmpeg, no real DB: subprocess and the job store are stubbed.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.config import settings
from app.main import app
from app.pipeline.transcribe import Word


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _set(name, value):
    orig = getattr(settings, name)
    object.__setattr__(settings, name, value)
    return orig


def _restore(name, orig):
    object.__setattr__(settings, name, orig)


def _words():
    return [Word(start=i * 0.5, end=i * 0.5 + 0.4, text=t)
            for i, t in enumerate("السلام عليكم AP36".split())]


def _styles(path):
    out = {"Sabily": 0, "Brand": 0, "Source": 0}
    for ln in path.read_text(encoding="utf-8").splitlines():
        for style in out:
            if f",{style}," in ln:
                out[style] += 1
    return out


# --------------------------------------------------------------------------- #
# 1. write_ass honours the three flags
# --------------------------------------------------------------------------- #

def test_write_ass_all_off_returns_none(tmp_path):
    from app.pipeline import subtitle

    out = subtitle.write_ass(_words(), 0, 5, tmp_path / "a.ass",
                             source_tag="#t", show_brand=False,
                             show_source=False, burn=False)
    assert out is None


def test_write_ass_burn_off_keeps_source_only(tmp_path):
    from app.pipeline import subtitle

    o1, o2 = _set("brand_watermark", False), None
    try:
        # NOTE: flags explicit — never rely on the ambient .env.
        out = subtitle.write_ass(_words(), 0, 5, tmp_path / "a.ass",
                                 source_tag="#t", burn=False, show_source=True)
        assert out is not None
        assert _styles(out)["Sabily"] == 0
        assert _styles(out)["Source"] == 1
    finally:
        _restore("brand_watermark", o1)


def test_write_ass_show_source_off(tmp_path):
    from app.pipeline import subtitle

    out = subtitle.write_ass(_words(), 0, 5, tmp_path / "a.ass",
                             source_tag="#t", show_source=False, burn=True,
                             show_brand=False)
    assert _styles(out)["Source"] == 0
    assert _styles(out)["Sabily"] >= 1


def test_write_ass_brand_text_when_no_logo(tmp_path):
    from app.pipeline import subtitle

    o1, o2 = _set("brand_logo", tmp_path / "missing.png"), _set("brand_watermark", True)
    try:
        out = subtitle.write_ass(_words(), 0, 5, tmp_path / "a.ass",
                                 source_tag="", show_brand=True)
        assert _styles(out)["Brand"] == 1
    finally:
        _restore("brand_logo", o1)
        _restore("brand_watermark", o2)


def test_write_ass_brand_text_replaced_by_logo(tmp_path):
    from app.pipeline import subtitle

    logo = tmp_path / "logo.png"
    logo.write_bytes(b"fake-png")
    o1, o2 = _set("brand_logo", logo), _set("brand_watermark", True)
    try:
        out = subtitle.write_ass(_words(), 0, 5, tmp_path / "a.ass",
                                 source_tag="", show_brand=True, burn=True,
                                 show_source=False)
        assert _styles(out)["Brand"] == 0
        # explicit off stays off even without a logo
        o3 = _set("brand_logo", tmp_path / "missing.png")
        try:
            out2 = subtitle.write_ass(_words(), 0, 5, tmp_path / "b.ass",
                                      source_tag="", show_brand=False, burn=True,
                                      show_source=False)
            assert _styles(out2)["Brand"] == 0
        finally:
            _restore("brand_logo", o3)
    finally:
        _restore("brand_logo", o1)
        _restore("brand_watermark", o2)


# --------------------------------------------------------------------------- #
# 1b. Arabic/English mixing + dash rules (display layer in write_ass)
# --------------------------------------------------------------------------- #

def test_mix_ar_en_latin_isolated_dash_unified():
    from app.pipeline.subtitle import LRI, PDI, mix_ar_en

    out = mix_ar_en("نستخدم تقنية AP36 في الخط")
    assert f"{LRI}AP36{PDI}" in out
    assert "تقنية" in out and "في" in out
    # em/en dashes unify, then Latin compounds re-glue
    assert mix_ar_en("AP — 36") == f"{LRI}AP-36{PDI}"
    assert mix_ar_en("a–b") == f"{LRI}a-b{PDI}"


def test_mix_ar_en_arabic_separator_keeps_spaces():
    from app.pipeline.subtitle import mix_ar_en

    assert mix_ar_en("الإنتاج - الجزء") == "الإنتاج - الجزء"
    assert "  " not in mix_ar_en("كلام   كثير")


def test_mix_ar_en_idempotent_and_arabic_hashtag_untouched():
    from app.pipeline.subtitle import mix_ar_en

    once = mix_ar_en("تقنية AP36 #قناة_اختبار 3")
    assert mix_ar_en(once) == once
    assert "#قناة_اختبار" in once


# --------------------------------------------------------------------------- #
# 2. render_clip threads flags into the filtergraph + verifies dimensions
# --------------------------------------------------------------------------- #

def _fake_run_factory(calls, target_size=(1080, 1920)):
    def fake_run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr="")
    return fake_run


def test_render_flags_in_filtergraph(monkeypatch, tmp_path):
    from app.pipeline import render

    logo = tmp_path / "logo.png"
    logo.write_bytes(b"fake-png")
    o1, o2 = _set("brand_logo", logo), _set("brand_watermark", True)
    monkeypatch.setattr(render, "verify_output", lambda *a, **k: None)
    try:
        crop = {"w": 608, "h": 1080, "x_expr": "0", "y": 0, "mode": "center"}
        ass = tmp_path / "s.ass"
        ass.write_text("x", encoding="utf-8")

        calls = []
        monkeypatch.setattr(render.subprocess, "run", _fake_run_factory(calls))
        render.render_clip(tmp_path / "s.mp4", tmp_path / "o.mp4", 0, 1,
                           crop, ass, target_size=(1080, 1920),
                           show_logo=True, burn_subtitles=True)
        vf = calls[-1][calls[-1].index("-vf") + 1]
        assert "movie=" in vf and "overlay=" in vf and "ass=" in vf

        calls.clear()
        render.render_clip(tmp_path / "s.mp4", tmp_path / "o.mp4", 0, 1,
                           crop, ass, target_size=(1080, 1920),
                           show_logo=False, burn_subtitles=False)
        vf = calls[-1][calls[-1].index("-vf") + 1]
        assert "movie=" not in vf and "ass=" not in vf
    finally:
        _restore("brand_logo", o1)
        _restore("brand_watermark", o2)


def test_overlay_logo_filter_show_flag(tmp_path):
    from app.pipeline import render

    logo = tmp_path / "logo.png"
    logo.write_bytes(b"fake-png")
    o1, o2 = _set("brand_logo", logo), _set("brand_watermark", True)
    try:
        assert render.overlay_logo_filter(1080, 1920, "tr", True) is not None
        assert render.overlay_logo_filter(1080, 1920, "tr", False) is None
        assert "overlay=" in render.overlay_logo_filter(720, 1280, "bl", True)
    finally:
        _restore("brand_logo", o1)
        _restore("brand_watermark", o2)


def test_verify_output_ok_and_mismatch(monkeypatch):
    from app.pipeline import media, render

    monkeypatch.setattr(media, "probe",
                        lambda p: {"width": 1440, "height": 2560})
    render.verify_output(Path("x.mp4"), (1440, 2560))  # must not raise
    with pytest.raises(RuntimeError, match="لا تطابق"):
        render.verify_output(Path("x.mp4"), (1080, 1920))

    def _boom(p):
        raise RuntimeError("no ffprobe")
    monkeypatch.setattr(media, "probe", _boom)
    with pytest.raises(RuntimeError, match="تعذّر التحقق"):
        render.verify_output(Path("x.mp4"), (1440, 2560))


# --------------------------------------------------------------------------- #
# 3. API stores flags per job and per clip; rerender replays them
# --------------------------------------------------------------------------- #

def test_job_options_store_appearance_flags(client, monkeypatch):
    captured = {}

    def fake_create(url, options=None):
        captured.update(options or {})
        return "abcdef012345"

    monkeypatch.setattr(jobs, "create_job", fake_create)
    monkeypatch.setattr(jobs, "enqueue", lambda jid: None)

    r = client.post("/api/jobs", json={
        "url": "https://example.com/video",
        "brand_watermark": False, "show_source": False, "burn_subtitles": False,
    })
    assert r.status_code == 200
    assert captured["brand_watermark"] is False
    assert captured["show_source"] is False
    assert captured["burn_subtitles"] is False

    captured.clear()
    r = client.post("/api/jobs", json={"url": "https://example.com/video"})
    assert r.status_code == 200
    assert "brand_watermark" not in captured
    assert "show_source" not in captured
    assert "burn_subtitles" not in captured


def test_clip_edit_stores_appearance_flags(client, monkeypatch, tmp_path):
    orig = settings.db_path
    object.__setattr__(settings, "db_path", tmp_path / "t.db")
    try:
        jobs.init_db()
        jid = jobs.create_job("https://example.com/v", {})
        cid = jobs.add_clip(jid, {"idx": 1, "start_sec": 0.0, "end_sec": 5.0,
                                  "meta": {"lines": []}})
        r = client.patch(f"/api/clips/{cid}", json={
            "brand_watermark": False, "show_source": True, "burn_subtitles": False,
        })
        assert r.status_code == 200
        meta = jobs.get_clip(cid)["meta"]
        assert meta["brand_watermark"] is False
        assert meta["show_source"] is True
        assert meta["burn_subtitles"] is False
    finally:
        object.__setattr__(settings, "db_path", orig)


def test_rerender_replays_meta_flags_and_quality(client, monkeypatch, tmp_path):
    orig_out, orig_work = settings.outputs_dir, settings.work_dir
    outs, work = tmp_path / "outs", tmp_path / "work"
    outs.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    object.__setattr__(settings, "outputs_dir", outs)
    object.__setattr__(settings, "work_dir", work)
    orig_db = settings.db_path
    object.__setattr__(settings, "db_path", tmp_path / "t.db")
    try:
        jobs.init_db()
        jid, cid = "abcdef012345", "0123456789ab"
        src = work / jid / "source.mp4"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"fake")
        meta = {"source_video": str(src), "quality": "QHD", "aspect": "9:16",
                "brand_watermark": False, "show_source": False,
                "burn_subtitles": False}
        clip = {"id": cid, "job_id": jid, "video_path": f"{jid}/clip_01.mp4",
                "start_sec": 0.0, "end_sec": 1.0, "meta": dict(meta)}
        seen = {}

        # NOTE 2026-09-12: **kw absorbs en_lines/turns/zoom_times/
        # bumpers added since this mock was written.
        def fake_write_ass(words, cs, ce, out, source_tag="", lines=None,
                           brand_pos=None, source_pos=None, brand_text=None,
                           target_size=None, show_brand=None,
                           show_source=None, burn=None, **kw):
            seen.update(target_size=target_size, show_brand=show_brand,
                        show_source=show_source, burn=burn)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("ass", encoding="utf-8")
            return out

        def fake_render(source, out_path, start, end, crop, ass_path=None,
                        target_size=None, brand_pos=None, show_logo=None,
                        burn_subtitles=None, **kw):
            seen.update(render_target=target_size, show_logo=show_logo,
                        burn_subtitles=burn_subtitles)
            return out_path

        monkeypatch.setattr("app.pipeline.subtitle.write_ass", fake_write_ass)
        monkeypatch.setattr("app.pipeline.render.render_clip", fake_render)
        monkeypatch.setattr(jobs, "get_clip",
                            lambda _cid: {"meta": dict(meta), **clip})
        monkeypatch.setattr(jobs, "update_clip", lambda _cid, **kw: None)
        r = client.post(f"/api/clips/{cid}/rerender")
        assert r.status_code == 200
        # quality locked to the stored QHD choice — nothing else
        assert seen["target_size"] == (1440, 2560)
        assert seen["render_target"] == (1440, 2560)
        # appearance flags replayed from meta
        assert seen["show_brand"] is False
        assert seen["show_source"] is False
        assert seen["burn"] is False
        assert seen["show_logo"] is False
        assert seen["burn_subtitles"] is False
    finally:
        object.__setattr__(settings, "outputs_dir", orig_out)
        object.__setattr__(settings, "work_dir", orig_work)
        object.__setattr__(settings, "db_path", orig_db)
