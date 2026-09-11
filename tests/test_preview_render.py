"""True-preview renders — real ffmpeg, synthetic source, no network, no GPU.

Covers POST /api/preview/{id}/render: the same manual_crop/build_crop ->
write_ass -> render_clip chain the job runner uses, so the <video> on the
preview page shows exactly what a job with the same options produces.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import preview
from app.config import settings
from app.main import app
from app.pipeline.media import probe


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _needs_ffmpeg():
    if not shutil.which(settings.ffmpeg) or not shutil.which(settings.ffprobe):
        pytest.skip("ffmpeg/ffprobe not on PATH")


def _make_source(path: Path) -> Path:
    """6s 720p test pattern with audio — stands in for a download."""
    cmd = [settings.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=6",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
           "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast",
           "-c:a", "aac", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-300:]
    return path


@pytest.fixture()
def session_id(tmp_path):
    _needs_ffmpeg()
    pid = preview._new_id()
    preview.preview_dir(pid).mkdir(parents=True, exist_ok=True)
    # NOTE: the source lives inside the session dir, exactly like
    # download.fetch(url, preview_dir(pid)) places it in production.
    src = _make_source(preview.preview_dir(pid) / "source.mp4")
    import time
    preview._sessions[pid] = {
        "status": "ready", "url": "https://example.com/v",
        "width": 1280, "height": 720, "duration": 6.0,
        "frames": [], "suggest": {"x": 0.5, "scale": 1.0, "face": False},
        "title": "T", "error": "", "created_at": time.time(),
        "video": str(src), "tag": "#Test",
    }
    yield pid
    preview._sessions.pop(pid, None)
    shutil.rmtree(preview.preview_dir(pid), ignore_errors=True)


def test_render_sample_manual_crop_dims(session_id):
    # NOTE 2026-09-12: every appearance flag explicit — the dev .env runs
    # BURN_SUBTITLES=false/SOURCE_TAG=false and must not leak into results.
    out = preview.render_sample(
        session_id, quality="HD", aspect="9:16",
        crop_x=0.5, crop_scale=1.0,
        show_brand=False, show_source=True, burn=True,
        sample_text="نستخدم تقنية AP36 في الاختبار")
    assert out["crop"]["mode"] == "manual"
    assert (out["width"], out["height"]) == (720, 1280)
    assert out["target_size"] == [720, 1280]
    got = probe(Path("data") / "previews" / session_id / Path(out["video"]).name)
    assert (got["width"], got["height"]) == (720, 1280)
    assert got["duration"] == pytest.approx(3.0, abs=0.6)


def test_render_sample_auto_uses_fit_without_faces(session_id):
    # NOTE 2026-09-12: synthetic pattern has no faces, so the auto path
    # must take mode "fit" (full frame over blurred fill) — and render it
    # through real ffmpeg. Guards the fit-without-logo -vf crash.
    out = preview.render_sample(
        session_id, quality="HD", aspect="9:16",
        show_brand=False, show_source=False, burn=False)
    assert out["crop"]["mode"] == "fit"
    assert (out["width"], out["height"]) == (720, 1280)


def test_render_api_validation(client, session_id):
    assert client.post("/api/preview/nope!!/render", json={}).status_code == 404
    assert client.post("/api/preview/abcdef012345/render", json={}).status_code == 404
    r = client.post(f"/api/preview/{session_id}/render", json={"crop_x": 9})
    assert r.status_code == 422
    r = client.post(f"/api/preview/{session_id}/render", json={"crop_scale": 0.1})
    assert r.status_code == 422
    # internal absolute video path must never leave the server
    r = client.get(f"/api/preview/{session_id}")
    assert r.status_code == 200
    assert "video" not in r.json()


def test_extract_defaults_to_five_frames(tmp_path):
    """Preview shows 5 real frames spread over the timeline."""
    _needs_ffmpeg()
    src = _make_source(tmp_path / "src.mp4")
    names = preview._extract_frames(src, 6.0, tmp_path)
    assert len(names) == 5
    assert all((tmp_path / n).is_file() for n in names)


def test_preview_cleanup_removes_old_sessions(tmp_path, monkeypatch):
    # NOTE 2026-09-12: every preview holds a full source video (~100s of
    # MB) — stale sessions must actually free disk, not just memory.
    import os
    import time as _time

    saved = dict(preview._sessions)
    orig_work = settings.work_dir
    object.__setattr__(settings, "work_dir", tmp_path)
    try:
        preview._sessions.clear()
        old, fresh = preview._new_id(), preview._new_id()
        for pid in (old, fresh):
            preview.preview_dir(pid).mkdir(parents=True, exist_ok=True)
            (preview.preview_dir(pid) / "frame_00.jpg").write_bytes(b"x")
            preview._sessions[pid] = {"status": "ready", "created_at": _time.time()}
        ancient = _time.time() - 7200
        os.utime(preview.preview_dir(old), (ancient, ancient))
        preview._sessions[old]["created_at"] = ancient
        removed = preview.cleanup(max_age_sec=3600)
        assert removed >= 1
        assert not preview.preview_dir(old).exists()
        assert old not in preview._sessions
        assert preview.preview_dir(fresh).exists()
        assert fresh in preview._sessions
    finally:
        preview._sessions.clear()
        preview._sessions.update(saved)
        object.__setattr__(settings, "work_dir", orig_work)


def test_sweep_temp_removes_only_stale_regenerables(tmp_path):
    import os
    import time as _time

    from app import jobs as _jobs

    orig_work = settings.work_dir
    object.__setattr__(settings, "work_dir", tmp_path)
    try:
        job = tmp_path / "abcdef012345"
        job.mkdir(parents=True, exist_ok=True)
        stale_audio = job / "audio_0123456789ab.m4a"
        fresh_audio = job / "audio_abcdef012345.m4a"
        stale_bumper = job / "bumper_intro_720x1280.mp4"
        keep_mp4 = job / "clip_01.mp4"
        for p in (stale_audio, fresh_audio, stale_bumper, keep_mp4):
            p.write_bytes(b"x" * 64)
        ancient = _time.time() - 10 * 86400
        os.utime(stale_audio, (ancient, ancient))
        os.utime(stale_bumper, (ancient - 30 * 86400, ancient - 30 * 86400))
        done = _jobs.sweep_temp()
        assert done == {"audio": 1, "bumper": 1}
        assert not stale_audio.exists() and not stale_bumper.exists()
        assert fresh_audio.exists() and keep_mp4.exists()
    finally:
        object.__setattr__(settings, "work_dir", orig_work)


def _write_disk_session(pid, directory, *, video, frames, width=1280, height=720):
    import json

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps({
        "width": width, "height": height, "duration": 6.0,
        "frames": frames, "suggest": {"x": 0.5, "scale": 1.0},
        "title": "T", "video": video, "tag": "#Test",
    }), encoding="utf-8")


def test_get_preview_rebuilds_from_disk(tmp_path):
    # NOTE 2026-09-12: a restart wipes in-memory sessions while frames stay
    # on disk — get_preview must rebuild instead of 404ing forever (this is
    # what left the modal spinning with a broken image).
    _needs_ffmpeg()
    src = _make_source(tmp_path / "src.mp4")
    pid = preview._new_id()
    d = preview.preview_dir(pid)
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, d / "source.mp4")
    src = d / "source.mp4"
    cmd = [settings.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-ss", "1", "-i", str(src), "-frames:v", "1", str(d / "frame_00.jpg")]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    _write_disk_session(pid, d, video=str(src), frames=["frame_00.jpg"])
    try:
        assert pid not in preview._sessions
        st = preview.get_preview(pid)
        assert st and st["status"] == "ready"
        assert st["frames"] == [f"/preview/{pid}/frame_00.jpg"]
        # NOTE: flags explicit — immune to ambient BURN=false/SOURCE=false.
        out = preview.render_sample(
            pid, quality="HD", aspect="9:16",
            crop_x=0.5, crop_scale=1.0,
            show_brand=False, show_source=False, burn=False)
        assert (out["width"], out["height"]) == (720, 1280)
    finally:
        preview._sessions.pop(pid, None)
        shutil.rmtree(d, ignore_errors=True)


def test_render_sample_rejects_foreign_video_path(tmp_path):
    # NOTE 2026-09-12: meta.json is disk-controlled, so a video path
    # outside previews/ must be refused before ffmpeg ever sees it.
    pid = preview._new_id()
    d = preview.preview_dir(pid)
    outside = tmp_path / "evil.mp4"
    outside.write_bytes(b"fake")
    (tmp_path / "frame_00.jpg").write_bytes(b"fake")
    import shutil as _sh
    d.mkdir(parents=True, exist_ok=True)
    _sh.copy(tmp_path / "frame_00.jpg", d / "frame_00.jpg")
    _write_disk_session(pid, d, video=str(outside), frames=["frame_00.jpg"])
    try:
        with pytest.raises(RuntimeError, match="غير صالح"):
            preview.render_sample(pid, quality="HD", aspect="9:16",
                                  crop_x=0.5, show_brand=False,
                                  show_source=False, burn=False)
    finally:
        preview._sessions.pop(pid, None)
        shutil.rmtree(d, ignore_errors=True)
