"""frame_scale (shrink-in-output with blur fill) — no black edges.

Unit: filtergraph shape via stubbed ffmpeg. API: jobs/preview/clips accept
and validate the flag. Live: a real shrunk mp4 probes at full target dims.
NOTE 2026-09-12: all appearance flags explicit — ambient .env runs
BURN_SUBTITLES=false/SOURCE_TAG=false/QUALITY=QHD and must not leak in.
"""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import jobs, preview
from app.config import settings
from app.main import app


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _fake_run_factory(calls):
    def fake_run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr="")
    return fake_run


def _crop():
    return {"w": 404, "h": 718, "x_expr": "438", "y": 1, "mode": "manual"}


def test_glass_chain_shape_no_black(monkeypatch, tmp_path):
    from app.pipeline import render

    monkeypatch.setattr(render.subprocess, "run", _fake_run_factory(calls := []))
    monkeypatch.setattr(render, "verify_output", lambda *a, **k: None)
    render.render_clip(tmp_path / "s.mp4", tmp_path / "o.mp4", 0, 1,
                       _crop(), None, target_size=(720, 1280),
                       show_logo=False, burn_subtitles=False, frame_scale=0.6)
    vf = calls[-1][calls[-1].index("-vf") + 1]
    # blurred glossy backdrop + centered sharp frame, single -vf input
    assert "gblur=sigma=" in vf and "overlay=(W-w)/2:(H-h)/2" in vf
    assert "[0:v]" not in vf and not vf.endswith("[v0]")
    # shrunk foreground is 60% of 720x1280, even-sized
    assert "scale=432:768" in vf


def test_full_bleed_unchanged_without_flag(monkeypatch, tmp_path):
    from app.pipeline import render

    for fs in (None, 1.0, 1, 0.1, 99.0, "bogus"):
        calls = []
        monkeypatch.setattr(render.subprocess, "run", _fake_run_factory(calls))
        monkeypatch.setattr(render, "verify_output", lambda *a, **k: None)
        render.render_clip(tmp_path / "s.mp4", tmp_path / "o.mp4", 0, 1,
                           _crop(), None, target_size=(720, 1280),
                           show_logo=False, burn_subtitles=False,
                           frame_scale=fs)
        vf = calls[-1][calls[-1].index("-vf") + 1]
        assert vf.startswith("crop=404:718:438:1"), fs  # classic chain kept


def test_jobs_api_accepts_and_validates_frame_scale(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(jobs, "create_job",
                        lambda url, options=None: (seen.update(options or {}), "abcdef012345")[1])
    monkeypatch.setattr(jobs, "enqueue", lambda jid: None)
    r = client.post("/api/jobs", json={"url": "https://example.com/v", "frame_scale": 0.7})
    assert r.status_code == 200 and seen.get("frame_scale") == 0.7
    seen.clear()
    r = client.post("/api/jobs", json={"url": "https://example.com/v"})
    assert r.status_code == 200 and "frame_scale" not in seen
    assert client.post("/api/jobs", json={"url": "https://example.com/v", "frame_scale": 0.1}).status_code == 422
    assert client.post("/api/jobs", json={"url": "https://example.com/v", "frame_scale": 2}).status_code == 422


def test_preview_render_validates_frame_scale(client):
    pid = "abcdef012345"
    assert client.post(f"/api/preview/{pid}/render", json={"frame_scale": 0.2}).status_code == 422


def test_clip_edit_stores_frame_scale(client, monkeypatch, tmp_path):
    orig = settings.db_path
    object.__setattr__(settings, "db_path", tmp_path / "t.db")
    try:
        jobs.init_db()
        jid = jobs.create_job("https://example.com/v", {})
        cid = jobs.add_clip(jid, {"idx": 1, "start_sec": 0.0, "end_sec": 5.0,
                                  "meta": {"lines": []}})
        r = client.patch(f"/api/clips/{cid}", json={"frame_scale": 0.8})
        assert r.status_code == 200
        assert jobs.get_clip(cid)["meta"]["frame_scale"] == 0.8
        assert client.patch(f"/api/clips/{cid}", json={"frame_scale": 0.1}).status_code == 422
    finally:
        object.__setattr__(settings, "db_path", orig)


def test_runner_threads_frame_scale(monkeypatch, tmp_path):
    import json

    from app.pipeline import runner

    work, outs = tmp_path / "work", tmp_path / "outs"
    work.mkdir(parents=True, exist_ok=True)
    outs.mkdir(parents=True, exist_ok=True)
    ow, oo = settings.work_dir, settings.outputs_dir
    object.__setattr__(settings, "work_dir", work)
    object.__setattr__(settings, "outputs_dir", outs)
    seen, state = {}, {"updates": [], "clips": []}
    cand = SimpleNamespace(start=1.0, end=4.0, score=0.9, text="نص",
                           duration=3.0, parts={})
    try:
        monkeypatch.setattr(runner.reframe, "build_crop",
                            lambda *a, **k: dict(_crop(), mode="static"))
        monkeypatch.setattr(runner.subtitle, "write_ass", lambda *a, **k: None)
        monkeypatch.setattr(runner.render, "render_clip",
                            lambda *a, **k: seen.update(k) or a[1])
        monkeypatch.setattr(runner, "probe", lambda v: {"width": 1280, "height": 720})
        monkeypatch.setattr(runner.asr, "run", lambda *a, **k: [
            SimpleNamespace(start=1.0, end=1.5, text="مرحبا")])
        monkeypatch.setattr(runner.normalize, "apply", lambda ws: ws)
        monkeypatch.setattr(runner.score, "select", lambda *a, **k: [cand])
        monkeypatch.setattr(runner.llm, "rerank", lambda c, w: [0])
        monkeypatch.setattr(runner.llm, "generate_metadata",
                            lambda t, lang=None: {"title": "t", "hashtags": []})
        monkeypatch.setattr(runner.llm, "build_caption", lambda m, u: "cap")
        monkeypatch.setattr(runner.download, "fetch", lambda url, w, quality=None:
                            SimpleNamespace(video_path=tmp_path / "s.mp4",
                                            audio_path=tmp_path / "a.wav", title="T",
                                            channel="C", webpage_url=url))
        monkeypatch.setattr(runner.download, "as_hashtag", lambda c: "#c")
        monkeypatch.setattr(runner.download, "timestamped_url", lambda u, s: u)
        jid = "abcdef012345"
        monkeypatch.setattr(jobs, "get_job", lambda _j: {
            "id": jid, "url": "https://example.com/v", "status": "running",
            "stage": "", "options": json.dumps({"clips": 1, "keep_source": True,
                                                "quality": "HD", "frame_scale": 0.6})})
        monkeypatch.setattr(jobs, "update_job",
                            lambda _j, **kw: state["updates"].append(kw))
        monkeypatch.setattr(jobs, "add_clip",
                            lambda _j, rec: state["clips"].append(rec) or "c1")
        runner.process(jid)
    finally:
        object.__setattr__(settings, "work_dir", ow)
        object.__setattr__(settings, "outputs_dir", oo)
    assert seen.get("frame_scale") == 0.6
    assert state["clips"] and state["clips"][0]["meta"]["frame_scale"] == 0.6


def _needs_ffmpeg():
    if not shutil.which(settings.ffmpeg) or not shutil.which(settings.ffprobe):
        pytest.skip("ffmpeg/ffprobe not on PATH")


def test_live_shrunk_render_full_dims(tmp_path):
    """Real ffmpeg: 0.6-shrunk HD sample still probes exactly 720x1280."""
    _needs_ffmpeg()
    from app.pipeline.media import probe as _probe
    from app.pipeline.transcribe import Word

    pid = preview._new_id()
    d = preview.preview_dir(pid)
    d.mkdir(parents=True, exist_ok=True)
    src = d / "source.mp4"
    cmd = [settings.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=6",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
           "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast",
           "-c:a", "aac", str(src)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    import time
    preview._sessions[pid] = {
        "status": "ready", "url": "https://example.com/v",
        "width": 1280, "height": 720, "duration": 6.0, "frames": [],
        "suggest": {"x": 0.5, "scale": 1.0}, "title": "T", "error": "",
        "created_at": time.time(), "video": str(src), "tag": "#Test",
    }
    try:
        out = preview.render_sample(
            pid, quality="HD", aspect="9:16", crop_x=0.5, crop_scale=1.0,
            show_brand=False, show_source=True, burn=True, frame_scale=0.6,
            sample_text="نستخدم تقنية AP36 في الاختبار")
        assert (out["width"], out["height"]) == (720, 1280)
        assert out["frame_scale"] == 0.6
        assert Path("data/previews", pid, Path(out["video"]).name).stat().st_size > 1000
    finally:
        preview._sessions.pop(pid, None)
        shutil.rmtree(d, ignore_errors=True)


def test_black_fill_chain_shape(monkeypatch, tmp_path):
    from app.pipeline import render

    monkeypatch.setattr(render.subprocess, "run", _fake_run_factory(calls := []))
    monkeypatch.setattr(render, "verify_output", lambda *a, **k: None)
    render.render_clip(tmp_path / "s.mp4", tmp_path / "o.mp4", 0, 1,
                       _crop(), None, target_size=(720, 1280),
                       show_logo=False, burn_subtitles=False,
                       frame_scale=0.6, fill_mode="black")
    vf = calls[-1][calls[-1].index("-vf") + 1]
    # matte bars: pad to full output, no blur anywhere, unlabeled chain
    assert "pad=720:1280:(ow-iw)/2:(oh-ih)/2:color=black" in vf
    assert "gblur" not in vf and "[0:v]" not in vf and not vf.endswith("[v0]")
    # unknown fill falls back to blur, never crashes
    calls.clear()
    render.render_clip(tmp_path / "s.mp4", tmp_path / "o.mp4", 0, 1,
                       _crop(), None, target_size=(720, 1280),
                       show_logo=False, burn_subtitles=False,
                       frame_scale=0.6, fill_mode="neon")
    vf = calls[-1][calls[-1].index("-vf") + 1]
    assert "gblur=sigma=" in vf


def test_fill_mode_api_validation(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(jobs, "create_job",
                        lambda url, options=None: (seen.update(options or {}), "abcdef012345")[1])
    monkeypatch.setattr(jobs, "enqueue", lambda jid: None)
    r = client.post("/api/jobs", json={"url": "https://example.com/v",
                                       "frame_scale": 0.7, "fill_mode": "black"})
    assert r.status_code == 200 and seen.get("fill_mode") == "black"
    assert client.post("/api/jobs",
                       json={"url": "https://example.com/v", "fill_mode": "neon"}).status_code == 422
    pid = "abcdef012345"
    assert client.post(f"/api/preview/{pid}/render",
                       json={"fill_mode": "neon"}).status_code == 422


def test_runner_threads_fill_mode(monkeypatch, tmp_path):
    import json

    from app.pipeline import runner

    work, outs = tmp_path / "work", tmp_path / "outs"
    work.mkdir(parents=True, exist_ok=True)
    outs.mkdir(parents=True, exist_ok=True)
    ow, oo = settings.work_dir, settings.outputs_dir
    object.__setattr__(settings, "work_dir", work)
    object.__setattr__(settings, "outputs_dir", outs)
    seen, state = {}, {"updates": [], "clips": []}
    cand = SimpleNamespace(start=1.0, end=4.0, score=0.9, text="نص",
                           duration=3.0, parts={})
    try:
        monkeypatch.setattr(runner.reframe, "build_crop",
                            lambda *a, **k: dict(_crop(), mode="static"))
        monkeypatch.setattr(runner.subtitle, "write_ass", lambda *a, **k: None)
        monkeypatch.setattr(runner.render, "render_clip",
                            lambda *a, **k: seen.update(k) or a[1])
        monkeypatch.setattr(runner, "probe", lambda v: {"width": 1280, "height": 720})
        monkeypatch.setattr(runner.asr, "run", lambda *a, **k: [
            SimpleNamespace(start=1.0, end=1.5, text="مرحبا")])
        monkeypatch.setattr(runner.normalize, "apply", lambda ws: ws)
        monkeypatch.setattr(runner.score, "select", lambda *a, **k: [cand])
        monkeypatch.setattr(runner.llm, "rerank", lambda c, w: [0])
        monkeypatch.setattr(runner.llm, "generate_metadata",
                            lambda t, lang=None: {"title": "t", "hashtags": []})
        monkeypatch.setattr(runner.llm, "build_caption", lambda m, u: "cap")
        monkeypatch.setattr(runner.download, "fetch", lambda url, w, quality=None:
                            SimpleNamespace(video_path=tmp_path / "s.mp4",
                                            audio_path=tmp_path / "a.wav", title="T",
                                            channel="C", webpage_url=url))
        monkeypatch.setattr(runner.download, "as_hashtag", lambda c: "#c")
        monkeypatch.setattr(runner.download, "timestamped_url", lambda u, s: u)
        jid = "abcdef012345"
        monkeypatch.setattr(jobs, "get_job", lambda _j: {
            "id": jid, "url": "https://example.com/v", "status": "running",
            "stage": "", "options": json.dumps({"clips": 1, "keep_source": True,
                                                "quality": "HD", "frame_scale": 0.6,
                                                "fill_mode": "black"})})
        monkeypatch.setattr(jobs, "update_job",
                            lambda _j, **kw: state["updates"].append(kw))
        monkeypatch.setattr(jobs, "add_clip",
                            lambda _j, rec: state["clips"].append(rec) or "c1")
        runner.process(jid)
    finally:
        object.__setattr__(settings, "work_dir", ow)
        object.__setattr__(settings, "outputs_dir", oo)
    assert seen.get("fill_mode") == "black"
    assert state["clips"] and state["clips"][0]["meta"]["fill_mode"] == "black"


def test_live_black_bars_render_full_dims(tmp_path):
    """Real ffmpeg: 0.6 + black fill still probes exactly 720x1280."""
    _needs_ffmpeg()

    pid = preview._new_id()
    d = preview.preview_dir(pid)
    d.mkdir(parents=True, exist_ok=True)
    src = d / "source.mp4"
    cmd = [settings.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=6",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
           "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast",
           "-c:a", "aac", str(src)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    import time
    preview._sessions[pid] = {
        "status": "ready", "url": "https://example.com/v",
        "width": 1280, "height": 720, "duration": 6.0, "frames": [],
        "suggest": {"x": 0.5, "scale": 1.0}, "title": "T", "error": "",
        "created_at": time.time(), "video": str(src), "tag": "#Test",
    }
    try:
        out = preview.render_sample(
            pid, quality="HD", aspect="9:16", crop_x=0.5, crop_scale=1.0,
            show_brand=False, show_source=False, burn=False,
            frame_scale=0.6, fill_mode="black")
        assert (out["width"], out["height"]) == (720, 1280)
        assert out["fill_mode"] == "black"
    finally:
        preview._sessions.pop(pid, None)
        shutil.rmtree(d, ignore_errors=True)
