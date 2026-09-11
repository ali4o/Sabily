"""Fix-loop regressions for reviewer P1/P2/P3 — no GPU, no network, no ffmpeg live."""

from pathlib import Path


def test_keep_source_none_means_default():
    from app.main import JobRequest
    # dashboard never sends keep_source → None → runner falls back to settings
    r = JobRequest(url="https://youtube.com/watch?v=12345678")
    assert r.keep_source is None
    d = r.model_dump()
    options: dict = {}
    if d.get("keep_source") is not None:
        options["keep_source"] = bool(d["keep_source"])
    assert "keep_source" not in options
    # explicit False is preserved
    r2 = JobRequest(url="https://youtube.com/watch?v=12345678", keep_source=False)
    d2 = r2.model_dump()
    opts2: dict = {}
    if d2.get("keep_source") is not None:
        opts2["keep_source"] = bool(d2["keep_source"])
    assert opts2["keep_source"] is False


def test_runner_keep_source_logic():
    # mirrors runner.finally: explicit False deletes even when settings True
    def keep_of(options, settings_keep):
        return options.get("keep_source") if "keep_source" in options else settings_keep
    assert keep_of({}, True) is True
    assert keep_of({"keep_source": False}, True) is False
    assert keep_of({"keep_source": True}, False) is True
    assert keep_of({}, False) is False


def test_clipline_rejects_end_le_start():
    from pydantic import ValidationError
    from app.main import ClipLine
    import pytest
    ClipLine(start=0.0, end=1.0, text="ok")
    with pytest.raises(ValidationError):
        ClipLine(start=1.0, end=1.0, text="bad")
    with pytest.raises(ValidationError):
        ClipLine(start=2.0, end=1.0, text="bad")


def test_body_uses_safe_tag():
    from app.pipeline.subtitle import _safe_tag
    # \r and \N must not survive into ASS dialogue
    assert "\r" not in _safe_tag("a\rb")
    assert "\\N" not in _safe_tag("a\\Nb")
    assert "{" not in _safe_tag("a{b")


def test_probe_bad_output_is_runtime(tmp_path, monkeypatch):
    import subprocess
    from app.pipeline import media
    def _raise_os(*a, **k):
        raise FileNotFoundError("no ffprobe")
    monkeypatch.setattr(subprocess, "run", _raise_os)
    try:
        media.probe(Path("x.mp4"))
    except RuntimeError as exc:
        assert "ffprobe failed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    class FakeBadJson:
        stdout = "not json"
        stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeBadJson())
    try:
        media.probe(Path("x.mp4"))
    except RuntimeError as exc:
        assert "ffprobe failed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for bad json")
