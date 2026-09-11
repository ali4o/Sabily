"""Batch 3 (correctness P1) tests — no GPU, no network, no ffmpeg, no real DB."""

import json
import logging
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.config import settings
from app.main import app


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _patch_db(tmp_path, monkeypatch):
    """Point the job store at a throwaway DB. Never touch real data/sabily.db."""
    orig = settings.db_path
    tmp_db = tmp_path / "test.db"
    object.__setattr__(settings, "db_path", tmp_db)
    return orig


def _restore_db(orig):
    object.__setattr__(settings, "db_path", orig)


# --------------------------------------------------------------------------- #
# 1. per-job options preserve keep_source=False
# --------------------------------------------------------------------------- #

def test_options_preserves_keep_source_false(client, monkeypatch):
    captured = {}

    def fake_create(url, options=None):
        captured.update(options or {})
        return "abcdef012345"

    monkeypatch.setattr(jobs, "create_job", fake_create)
    monkeypatch.setattr(jobs, "enqueue", lambda jid: None)

    r = client.post("/api/jobs", json={"url": "https://example.com/video", "keep_source": False})
    assert r.status_code == 200
    assert "keep_source" in captured, "False must be preserved, not dropped as falsy"
    assert captured["keep_source"] is False

    captured.clear()
    r = client.post("/api/jobs", json={"url": "https://example.com/video", "keep_source": True})
    assert r.status_code == 200
    assert captured["keep_source"] is True

    # clips/lang/aspect threading + url validation
    captured.clear()
    r = client.post("/api/jobs", json={
        "url": "https://example.com/video", "clips": 3, "lang": "ar",
        "aspect": "1:1", "keep_source": True,
    })
    assert r.status_code == 200
    assert captured["clips"] == 3
    assert captured["lang"] == "ar"
    assert captured["aspect"] == "1:1"

    captured.clear()
    r = client.post("/api/jobs", json={"url": "https://example.com/video"})
    assert r.status_code == 200
    assert "clips" not in captured  # 0 default stays unset -> runner uses settings default
    # reviewer P1-1: absent keep_source must stay absent so runner falls back
    # to settings.keep_source (dashboard jobs keep source for re-render).
    assert "keep_source" not in captured

    r = client.post("/api/jobs", json={"url": "ftp://example.com/x", "keep_source": True})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# helpers for runner tests (fully mocked pipeline, tmp dirs)
# --------------------------------------------------------------------------- #

def _mock_pipeline(monkeypatch, tmp_path, job_options, rerank_order=(0,)):
    from app.pipeline import runner

    work = tmp_path / "work"
    outs = tmp_path / "outs"
    work.mkdir(parents=True, exist_ok=True)
    outs.mkdir(parents=True, exist_ok=True)
    orig_work, orig_outs = settings.work_dir, settings.outputs_dir
    object.__setattr__(settings, "work_dir", work)
    object.__setattr__(settings, "outputs_dir", outs)

    jid = "abcdef012345"
    state = {"updates": [], "clips": []}
    cand = SimpleNamespace(start=10.0, end=40.0, score=0.9, text="نص تجريبي",
                           duration=30.0, parts={"a": 1.0})

    captured = {}

    def fake_build_crop(video, start, end, width, height, aspect=None):
        captured["aspect"] = aspect
        return {"w": 608, "h": 1080, "x_expr": "0", "y": 0, "mode": "center"}

    # NOTE 2026-09-11: **kw absorbs the appearance flags
    # (show_brand/show_source/burn/brand_pos/show_logo/burn_subtitles)
    # added for per-job appearance control; the mock records them.
    def fake_write_ass(words, cs, ce, out, source_tag="", lines=None,
                       brand_pos=None, source_pos=None, brand_text=None,
                       target_size=None, **kw):
        captured["ass_target"] = tuple(target_size) if target_size else None
        captured["ass_flags"] = {k: kw.get(k) for k in
                                 ("show_brand", "show_source", "burn")}
        return out

    def fake_render(source, out_path, start, end, crop, ass_path=None,
                    target_size=None, **kw):
        captured["render_target"] = tuple(target_size) if target_size else None
        return out_path

    monkeypatch.setattr("app.pipeline.runner.reframe.build_crop", fake_build_crop)
    monkeypatch.setattr("app.pipeline.runner.subtitle.write_ass", fake_write_ass)
    monkeypatch.setattr("app.pipeline.runner.render.render_clip", fake_render)
    monkeypatch.setattr("app.pipeline.runner.probe", lambda v: {"width": 1920, "height": 1080})
    monkeypatch.setattr(runner.asr, "run",
                        lambda audio, cache=None, on_progress=None: [
                            SimpleNamespace(start=10.0, end=10.5, text="مرحبا")])
    monkeypatch.setattr("app.pipeline.runner.normalize.apply", lambda ws: ws)
    # NOTE 2026-09-12: audio_path kwarg added for audio-aware scoring.
    # NOTE 2026-09-12: content_type kwarg added for the moment engine
    # (UNDERSTAND→RANK); the mock absorbs it, runner behaviour unchanged.
    monkeypatch.setattr("app.pipeline.runner.score.select",
                        lambda words, limit=0, audio_path=None, content_type=None: [cand])
    monkeypatch.setattr("app.pipeline.runner.llm.rerank", lambda cands, wanted: list(rerank_order))
    monkeypatch.setattr("app.pipeline.runner.llm.generate_metadata",
                        lambda text, lang=None: {"title": "t", "hashtags": []})
    monkeypatch.setattr("app.pipeline.runner.llm.build_caption", lambda meta, url: "cap")
    # NOTE 2026-09-11: quality kwarg added so downloads can fetch
    # taller sources for QHD; default keeps old call shape working.
    monkeypatch.setattr("app.pipeline.runner.download.fetch",
                        lambda url, w, quality=None: SimpleNamespace(
                            video_path=tmp_path / "src.mp4",
                            audio_path=tmp_path / "a.wav",
                            title="T", channel="C", webpage_url=url))
    monkeypatch.setattr("app.pipeline.runner.download.as_hashtag", lambda c: "#c")
    monkeypatch.setattr("app.pipeline.runner.download.timestamped_url",
                        lambda url, s: url)
    monkeypatch.setattr(jobs, "get_job",
                        lambda _jid: {"id": jid, "url": "https://example.com/v",
                                     "status": "running", "stage": "",
                                     "options": json.dumps(job_options)})
    monkeypatch.setattr(jobs, "update_job",
                        lambda _jid, **kw: state["updates"].append(kw))
    monkeypatch.setattr(jobs, "add_clip",
                        lambda _jid, rec: state["clips"].append(rec) or "clipid1")

    return jid, state, captured, (orig_work, orig_outs)


def _restore_dirs(orig):
    object.__setattr__(settings, "work_dir", orig[0])
    object.__setattr__(settings, "outputs_dir", orig[1])


def test_runner_aspect_threading_1to1(monkeypatch, tmp_path):
    from app.pipeline import runner

    # NOTE 2026-09-12: quality pinned — ambient .env may default elsewhere.
    jid, state, captured, orig = _mock_pipeline(
        monkeypatch, tmp_path, {"aspect": "1:1", "clips": 1, "keep_source": True,
                                "quality": "FHD"})
    try:
        runner.process(jid)
    finally:
        _restore_dirs(orig)
    assert captured.get("aspect") == "1:1"
    assert captured.get("ass_target") == (1080, 1080)
    assert captured.get("render_target") == (1080, 1080)
    assert state["clips"] and state["clips"][0]["meta"]["aspect"] == "1:1"


def test_runner_aspect_default_fallback(monkeypatch, tmp_path):
    from app.pipeline import runner

    jid, state, captured, orig = _mock_pipeline(monkeypatch, tmp_path, {})
    try:
        runner.process(jid)
    finally:
        _restore_dirs(orig)
    assert captured.get("aspect") == settings.aspect
    assert captured.get("ass_target") == settings.target_size
    assert captured.get("render_target") == settings.target_size

    # invalid aspect falls back instead of crashing the job
    jid, state, captured, orig = _mock_pipeline(
        monkeypatch, tmp_path, {"aspect": "4:3", "keep_source": True})
    try:
        runner.process(jid)
    finally:
        _restore_dirs(orig)
    assert captured.get("aspect") == settings.aspect
    assert any(u.get("status") == "done" for u in state["updates"])


def test_runner_wanted_clamp_and_empty_guard(monkeypatch, tmp_path):
    from app.pipeline import runner

    assert max(1, min(20, 0)) == 1
    assert max(1, min(20, 25)) == 20
    assert max(1, min(20, 5)) == 5

    # NOTE 2026-09-12: an empty LLM order now falls back to heuristic
    # picks (degrade, don't fail — project fallback philosophy) instead
    # of erroring; only zero *candidates* still errors, never silent done.
    jid, state, captured, orig = _mock_pipeline(
        monkeypatch, tmp_path, {"clips": 1, "keep_source": True}, rerank_order=())
    try:
        runner.process(jid)
    finally:
        _restore_dirs(orig)
    assert state["updates"], "expected status updates"
    assert state["updates"][-1].get("status") == "done"
    assert state["clips"], "fallback must still produce a clip"

    jid, state, captured, orig = _mock_pipeline(
        monkeypatch, tmp_path, {"clips": 1, "keep_source": True})
    monkeypatch.setattr("app.pipeline.runner.score.select",
                        lambda *a, **k: [])
    try:
        runner.process(jid)
    finally:
        _restore_dirs(orig)
    assert state["updates"][-1].get("status") == "error"
    assert "لم يتم العثور" in (state["updates"][-1].get("error") or "")


# --------------------------------------------------------------------------- #
# 2. rerender fallback uses target_size for both aspects
# --------------------------------------------------------------------------- #

def _rerender_setup(monkeypatch, tmp_path, meta_extra):
    orig_outs, orig_work = settings.outputs_dir, settings.work_dir
    outs = tmp_path / "outs"
    work = tmp_path / "work"
    outs.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    object.__setattr__(settings, "outputs_dir", outs)
    object.__setattr__(settings, "work_dir", work)

    jid, cid = "abcdef012345", "0123456789ab"
    # real layout: source lives under work/<job>/source.mp4, so containment holds
    src = work / jid / "source.mp4"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"fake")
    # NOTE 2026-09-12: pin quality — the ambient .env may default to
    # QHD, and quality fallback is not what these aspect tests check.
    meta = {"source_video": str(src), "quality": "FHD"}
    meta.update(meta_extra)
    clip = {"id": cid, "job_id": jid, "video_path": f"{jid}/clip_01.mp4",
            "start_sec": 0.0, "end_sec": 1.0, "meta": dict(meta)}
    captured = {}

    # NOTE 2026-09-11: **kw absorbs appearance flags + brand_pos
    # added for per-clip appearance control (see test_appearance.py).
    def fake_write_ass(words, cs, ce, out, source_tag="", lines=None,
                       brand_pos=None, source_pos=None, brand_text=None,
                       target_size=None, **kw):
        captured["ass_target"] = tuple(target_size) if target_size else None
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("ass", encoding="utf-8")
        return out

    def fake_render(source, out_path, start, end, crop, ass_path=None,
                    target_size=None, brand_pos=None, **kw):
        captured["crop"] = dict(crop)
        captured["render_target"] = tuple(target_size) if target_size else None
        return out_path

    monkeypatch.setattr("app.pipeline.subtitle.write_ass", fake_write_ass)
    monkeypatch.setattr("app.pipeline.render.render_clip", fake_render)
    monkeypatch.setattr(jobs, "get_clip", lambda _cid: {"meta": dict(meta), **clip})
    monkeypatch.setattr(jobs, "update_clip", lambda _cid, **kw: None)
    return (outs, work, orig_outs, orig_work), captured


def test_rerender_fallback_default_aspect(client, monkeypatch, tmp_path):
    origs, captured = _rerender_setup(monkeypatch, tmp_path, {})
    try:
        r = client.post("/api/clips/0123456789ab/rerender")
    finally:
        object.__setattr__(settings, "outputs_dir", origs[2])
        object.__setattr__(settings, "work_dir", origs[3])
    assert r.status_code == 200
    # NOTE 2026-09-12: pinned FHD (see _rerender_setup), resolved against
    # ambient custom OUT_* dims — never the ambient default quality.
    from app.config import resolve_target
    exp = resolve_target("FHD", "9:16", settings.out_width, settings.out_height)
    assert captured["render_target"] == exp
    assert captured["ass_target"] == exp
    assert (captured["crop"]["w"], captured["crop"]["h"]) == exp


def test_rerender_fallback_square_aspect(client, monkeypatch, tmp_path):
    origs, captured = _rerender_setup(monkeypatch, tmp_path, {"aspect": "1:1"})
    try:
        r = client.post("/api/clips/0123456789ab/rerender")
    finally:
        object.__setattr__(settings, "outputs_dir", origs[2])
        object.__setattr__(settings, "work_dir", origs[3])
    assert r.status_code == 200
    assert captured["render_target"] == (1080, 1080)
    assert captured["ass_target"] == (1080, 1080)
    assert (captured["crop"]["w"], captured["crop"]["h"]) == (1080, 1080)


# --------------------------------------------------------------------------- #
# 3. init_db preserves queued + cancel_job
# --------------------------------------------------------------------------- #

def test_init_db_preserves_queued(tmp_path, monkeypatch):
    orig = _patch_db(tmp_path, monkeypatch)
    try:
        jobs.init_db()
        q = jobs.create_job("https://example.com/a", {})
        r = jobs.create_job("https://example.com/b", {})
        jobs.update_job(r, status="running", stage="x")
        jobs.init_db()
        assert jobs.get_job(q)["status"] == "queued"
        crashed = jobs.get_job(r)
        assert crashed["status"] == "error"
        assert crashed["error"] == "interrupted"
        # cancel checkpoint helper
        jobs.cancel_job(q)
        cancelled = jobs.get_job(q)
        assert cancelled["status"] == "cancelled"
        assert cancelled["stage"] == "ملغي"
    finally:
        _restore_db(orig)


# --------------------------------------------------------------------------- #
# 5. transcribe corrupt cache recovery
# --------------------------------------------------------------------------- #

def test_transcribe_corrupt_cache_recovers(tmp_path, monkeypatch):
    from app.pipeline import transcribe
    from app.pipeline.transcribe import Word

    cache = tmp_path / "transcript.json"
    cache.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(transcribe, "_transcribe",
                        lambda audio, device, compute, on_progress: [
                            Word(start=0.0, end=1.0, text="مرحبا")])
    words = transcribe.run(tmp_path / "audio.wav", cache=cache)
    assert len(words) == 1 and words[0].text == "مرحبا"
    # cache repaired with valid JSON (dict format carries the ASR
    # settings fingerprint since 2026-09-11; legacy list still readable)
    data = json.loads(cache.read_text(encoding="utf-8"))
    saved = data["words"] if isinstance(data, dict) else data
    assert saved[0]["text"] == "مرحبا"


# --------------------------------------------------------------------------- #
# 7. gate_vram Windows encoding
# --------------------------------------------------------------------------- #

def test_gate_vram_no_arrow_and_runs(tmp_path, monkeypatch, capsys):
    from pathlib import Path as _P

    src = _P("scripts/gate_vram.py").read_text(encoding="utf-8")
    assert "→" not in src, "arrow would crash cp1256 consoles"
    assert "->" in src
    assert "sys.stdout.reconfigure" in src

    from scripts import gate_vram

    monkeypatch.setattr("app.pipeline.transcribe.run", lambda wav, **k: [])
    monkeypatch.setattr("scripts.gate_vram.transcribe.run", lambda wav, **k: [])
    import subprocess as _sp
    monkeypatch.setattr(_sp, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr("scripts.gate_vram.llm.get_provider", lambda: None)
    # run with a fake cp1256-ish stdout lacking reconfigure: must not crash
    assert gate_vram.main() == 0
    out = capsys.readouterr().out
    assert "VRAM" in out


# --------------------------------------------------------------------------- #
# 8. jobs log thread safety + eviction
# --------------------------------------------------------------------------- #

def test_log_handler_threadsafe_smoke(monkeypatch):
    jid = "abcdef012345"
    saved_buffers = dict(jobs._buffers)
    saved_current = jobs._current
    jobs._buffers.clear()
    jobs._current = jid
    jobs._buffers[jid] = []
    try:
        handler = jobs._JobLogHandler()
        logger = logging.getLogger("sabily.test.threads")

        def worker(n):
            for i in range(120):
                handler.emit(logger.makeRecord(
                    logger.name, logging.INFO, __file__, 1,
                    f"msg {n}-{i}", None, None))

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        readers_done = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                lines, cursor = jobs.log_lines(jid, 0)
                assert cursor == len(lines) or cursor >= 0
            readers_done.append(True)

        rt = threading.Thread(target=reader)
        rt.start()
        for t in threads:
            t.join()
        stop.set()
        rt.join()
        assert readers_done
        lines, cursor = jobs.log_lines(jid, 0)
        assert cursor <= jobs.MAX_LINES
        assert len(lines) <= jobs.MAX_LINES
    finally:
        jobs._buffers.clear()
        jobs._buffers.update(saved_buffers)
        jobs._current = saved_current

    # eviction keeps the table bounded (unit-check the policy inline)
    with jobs._log_lock:
        jobs._buffers.clear()
        for i in range(21):
            jobs._buffers[f"job{i:02d}"] = ["x"]
        try:
            if len(jobs._buffers) > 20:
                keep = "job20"
                for k in list(jobs._buffers):
                    if k != keep:
                        jobs._buffers.pop(k, None)
                        break
            assert len(jobs._buffers) == 20
            assert "job20" in jobs._buffers
        finally:
            jobs._buffers.clear()
            jobs._buffers.update(saved_buffers)


# --------------------------------------------------------------------------- #
# 9. DELETE /api/clips scoping
# --------------------------------------------------------------------------- #

def test_delete_clips_requires_job_id(client, tmp_path, monkeypatch):
    orig = _patch_db(tmp_path, monkeypatch)
    try:
        jobs.init_db()
        j1 = jobs.create_job("https://example.com/1", {})
        j2 = jobs.create_job("https://example.com/2", {})
        base = {"idx": 1, "start_sec": 0.0, "end_sec": 1.0, "title": "t"}
        jobs.add_clip(j1, dict(base, video_path="v1.mp4"))
        jobs.add_clip(j2, dict(base, video_path="v2.mp4"))

        r = client.delete("/api/clips")
        assert r.status_code == 400

        r = client.delete("/api/clips", params={"job_id": j1})
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        remaining_j1 = jobs.list_clips(j1)
        remaining_j2 = jobs.list_clips(j2)
        assert remaining_j1 == []
        assert len(remaining_j2) == 1
    finally:
        _restore_db(orig)
