"""Quality presets — no GPU, no network, no ffmpeg."""

from app.config import (
    QUALITY_PRESETS,
    normalize_quality,
    resolve_target,
    settings,
)


def test_presets_dimensions():
    assert QUALITY_PRESETS["HD"]["9:16"] == (720, 1280)
    assert QUALITY_PRESETS["FHD"]["9:16"] == (1080, 1920)
    assert QUALITY_PRESETS["QHD"]["9:16"] == (1440, 2560)
    for q in ("HD", "FHD", "QHD"):
        w, h = QUALITY_PRESETS[q]["9:16"]
        assert w % 2 == 0 and h % 2 == 0
        w1, h1 = QUALITY_PRESETS[q]["1:1"]
        assert w1 == h1 and w1 % 2 == 0


def test_normalize_quality_aliases():
    assert normalize_quality("2K") == "QHD"
    assert normalize_quality("1440p") == "QHD"
    assert normalize_quality("1080p") == "FHD"
    assert normalize_quality("720") == "HD"
    assert normalize_quality("") == "FHD"
    assert normalize_quality(None) == "FHD"
    assert normalize_quality("8K") == "FHD"  # unknown -> fallback
    assert normalize_quality("qhd") == "QHD"


def test_resolve_target():
    assert resolve_target("HD", "9:16") == (720, 1280)
    assert resolve_target("QHD", "9:16") == (1440, 2560)
    assert resolve_target("QHD", "1:1") == (1440, 1440)
    assert resolve_target("FHD", "1:1", 1080, 1920) == (1080, 1080)
    assert resolve_target("bogus", "bogus") == (1080, 1920)  # double fallback
    # custom OUT_WIDTH/OUT_HEIGHT honoured for FHD
    assert resolve_target("FHD", "9:16", 720, 1280) == (720, 1280)


def test_default_target_is_fhd():
    # NOTE 2026-09-12: never assert ambient .env values (user may run
    # QUALITY=QHD) — assert the resolver contract instead.
    assert resolve_target("FHD", "9:16", 1080, 1920) == (1080, 1920)
    assert resolve_target("FHD", "1:1", 1080, 1920) == (1080, 1080)


def test_download_fetch_height():
    from app.pipeline.download import _fmt, fetch_height
    assert fetch_height(None) == settings.max_height
    assert fetch_height("HD") >= 720
    assert fetch_height("QHD") >= 1440
    assert "1440" in _fmt(1440)
    assert str(settings.max_height) in _fmt()


def test_api_accepts_quality_and_alias(client=None):
    # via TestClient when available (same pattern as batch3 tests)
    try:
        from fastapi.testclient import TestClient
        from app.main import app
        from app import jobs
    except Exception:
        return
    c = TestClient(app, raise_server_exceptions=False)
    seen: dict = {}
    orig_create, orig_enqueue = jobs.create_job, jobs.enqueue
    jobs.create_job = lambda url, options=None: (seen.update(options or {}), "abcdef012345")[1]
    jobs.enqueue = lambda jid: None
    try:
        r = c.post("/api/jobs", json={"url": "https://example.com/video", "quality": "2K"})
        assert r.status_code == 200
        assert seen.get("quality") == "QHD"
        seen.clear()
        r = c.post("/api/jobs", json={"url": "https://example.com/video"})
        assert r.status_code == 200
        assert "quality" not in seen
        seen.clear()
        r = c.post("/api/jobs", json={"url": "https://example.com/video", "quality": "8K"})
        assert r.status_code == 200
        assert "quality" not in seen  # invalid ignored, runner falls back
    finally:
        jobs.create_job, jobs.enqueue = orig_create, orig_enqueue
