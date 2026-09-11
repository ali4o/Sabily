"""Crop calibration + center-biased tracking. No GPU, no network."""

from app.config import settings
from app.pipeline import reframe


def _set(name, value):
    orig = getattr(settings, name)
    object.__setattr__(settings, name, value)
    return orig


def test_manual_crop_math_and_clamps():
    # NOTE: 404x718 matches crop_size(1280, 720) exactly (even rounding)
    c = reframe.manual_crop(1280, 720, "9:16", 0.5, 1.0)
    assert c["mode"] == "manual"
    assert (c["w"], c["h"]) == (404, 718)
    assert c["x_expr"] == str((1280 - 404) // 2)
    # edges clamp inside the frame
    left = reframe.manual_crop(1280, 720, "9:16", 0.0, 1.0)
    assert left["x_expr"] == "0"
    right = reframe.manual_crop(1280, 720, "9:16", 1.0, 1.0)
    assert right["x_expr"] == str(1280 - 404)
    # narrower cut stays centered on demand + even + inside
    small = reframe.manual_crop(1280, 720, "9:16", 0.5, 0.5)
    assert small["w"] % 2 == 0 and small["h"] % 2 == 0
    assert small["w"] < c["w"] and int(small["x_expr"]) > int(c["x_expr"])
    # square aspect + invalid inputs fall back safely
    sq = reframe.manual_crop(1280, 720, "1:1", 0.5, 1.0)
    assert (sq["w"], sq["h"]) == (720, 720) and sq["mode"] == "manual"
    bad = reframe.manual_crop(1280, 720, "bogus", 9.0, 99.0)
    assert bad["mode"] == "manual" and int(bad["x_expr"]) == 1280 - bad["w"]


def test_pick_best_prefers_center(monkeypatch):
    import numpy as np

    assert reframe._pick_best([]) is None
    # a huge edge poster loses to a smaller centered host (default bias)
    cands = [(0.05, 40000.0), (0.5, 30000.0)]
    assert reframe._pick_best(cands) == 0.5
    # bias off restores pure largest-area
    orig = _set("face_center_bias", 0.0)
    try:
        assert reframe._pick_best(cands) == 0.05
    finally:
        object.__setattr__(settings, "face_center_bias", orig)


def test_detect_center_uses_bias(monkeypatch):
    import numpy as np

    class FakeFrontal:
        def detectMultiScale(self, *a, **k):
            # edge poster (200x200 @cx 0.1375) vs centered host
            # (190x190 @cx 0.4438): bias 0.3 flips the pick to the host
            return [(10, 100, 200, 200), (300, 100, 190, 190)]

    monkeypatch.setattr(reframe, "_detector",
                        lambda: ("haar", (FakeFrontal(), None)))
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    assert abs(reframe._detect_center(frame) - (300 + 95) / 800) < 1e-6


def test_runner_honors_crop_lock(monkeypatch, tmp_path):
    from app import jobs
    from app.pipeline import runner

    orig_db, orig_work, orig_out = (settings.db_path, settings.work_dir,
                                    settings.outputs_dir)
    object.__setattr__(settings, "db_path", tmp_path / "t.db")
    object.__setattr__(settings, "work_dir", tmp_path / "work")
    object.__setattr__(settings, "outputs_dir", tmp_path / "outs")
    try:
        jobs.init_db()
        jid = jobs.create_job("https://example.com/v",
                              {"crop_lock": {"x": 0.25, "scale": 0.8},
                               "burn_subtitles": False, "brand_watermark": False,
                               "show_source": False, "keep_source": False})
        from types import SimpleNamespace

        monkeypatch.setattr(
            runner.download, "fetch",
            lambda url, work, quality=None: SimpleNamespace(
                video_path=tmp_path / "src.mp4", audio_path=tmp_path / "a.wav",
                title="T", channel="C", webpage_url=url))
        monkeypatch.setattr(runner, "probe",
                            lambda v: {"width": 1280, "height": 720})
        monkeypatch.setattr(runner.asr, "run", lambda *a, **k: [])
        runner.process(jid)
        job = jobs.get_job(jid)
        assert job["status"] == "error"  # empty transcript guarded
        # now with words: build_crop must NOT run (locked, no sampling)
        from app.pipeline.transcribe import Word
        words = [Word(start=i * 0.5, end=i * 0.5 + 0.4, text=t)
                 for i, t in enumerate(("مرحبا " * 60).split())]

        def _boom(*a, **k):
            raise AssertionError("sampling must be skipped for locked crop")

        monkeypatch.setattr(runner.asr, "run", lambda *a, **k: words)
        monkeypatch.setattr(runner.reframe, "build_crop", _boom)
        monkeypatch.setattr(runner.score, "select", lambda *a, **k: [])
        runner.process(jid)
        job = jobs.get_job(jid)
        assert "لم يتم العثور" in (job.get("error") or "")
    finally:
        object.__setattr__(settings, "db_path", orig_db)
        object.__setattr__(settings, "work_dir", orig_work)
        object.__setattr__(settings, "outputs_dir", orig_out)


def test_preview_endpoints_validation():
    from fastapi.testclient import TestClient

    from app.main import app

    c = TestClient(app, raise_server_exceptions=False)
    assert c.post("/api/preview", json={"url": "ftp://x/y"}).status_code == 400
    assert c.post("/api/preview", json={"url": "x"}).status_code in (400, 422)
    assert c.get("/api/preview/nope!!").status_code == 404
    assert c.get("/api/preview/abcdef012345").status_code == 404
    r = c.post("/api/jobs", json={"url": "https://example.com/v", "crop_x": 9})
    assert r.status_code == 422
    r = c.post("/api/jobs", json={"url": "https://example.com/v", "crop_scale": 0.1})
    assert r.status_code == 422
