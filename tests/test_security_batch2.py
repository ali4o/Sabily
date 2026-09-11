"""Batch 2 (security hardening) tests — no GPU, no network, no ffmpeg, no real DB."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import _ID_RE, _require_id, app


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_require_id_rejects_traversal():
    from fastapi import HTTPException

    for bad in ["..", "../outputs", "a/b", "", "ABC123DEF456", "abcdef01234",
                "abcdef0123456", "gggggggggggg", "a/b", "..\\..", "/etc/passwd"]:
        with pytest.raises(HTTPException) as exc_info:
            _require_id(bad)
        assert exc_info.value.status_code == 404


def test_require_id_accepts_valid():
    assert _require_id("abcdef012345") == "abcdef012345"
    assert _require_id("0123456789ab") == "0123456789ab"
    assert _ID_RE.fullmatch("abcdef012345")


def test_job_endpoints_reject_invalid_id(client):
    # path params must 404 without touching the DB (validation first)
    assert client.get("/api/jobs/............").status_code == 404
    assert client.get("/api/jobs/abcdefghijkl").status_code == 404
    assert client.get("/api/jobs/............/log").status_code == 404
    assert client.delete("/api/jobs/............/source").status_code == 404
    assert client.delete("/api/clips/............").status_code == 404
    assert client.post("/api/clips/............/rerender").status_code == 404
    assert client.patch("/api/clips/............", json={"title": "x"}).status_code == 404


def test_get_clips_invalid_job_returns_empty(client):
    r = client.get("/api/clips", params={"job_id": ".."})
    assert r.status_code == 200
    assert r.json() == []
    r = client.get("/api/clips", params={"job_id": "../outputs"})
    assert r.status_code == 200
    assert r.json() == []
    r = client.get("/api/clips", params={"job_id": "not-hex"})
    assert r.status_code == 200
    assert r.json() == []


def test_cors_headers_absent(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    lowered = {k.lower(): v for k, v in r.headers.items()}
    assert "access-control-allow-origin" not in lowered
    # no wildcard
    assert lowered.get("access-control-allow-origin") != "*"
    # preflight must not grant foreign origins
    r2 = client.options(
        "/api/jobs",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"},
    )
    lowered2 = {k.lower(): v for k, v in r2.headers.items()}
    assert lowered2.get("access-control-allow-origin") != "*"


def test_update_job_unknown_column_raises():
    from app import jobs

    with pytest.raises(ValueError, match="unknown field"):
        jobs.update_job("abcdef012345", badcol="x")
    with pytest.raises(ValueError, match="unknown field"):
        jobs.update_job("abcdef012345", status="done", injection="1")


def test_update_clip_unknown_column_raises():
    from app import jobs

    with pytest.raises(ValueError, match="unknown field"):
        jobs.update_clip("abcdef012345", badcol="x")
    # allowed columns must not raise the allowlist error (they fail later on DB
    # only if the DB is touched — so check the allowlist directly)
    assert "title" in jobs._CLIP_COLS
    assert "meta" in jobs._CLIP_COLS
    assert "status" in jobs._JOB_COLS


def test_redact_hides_key():
    from app.pipeline import llm

    orig = settings.llm_api_key
    try:
        object.__setattr__(settings, "llm_api_key", "SECRETKEY123")
        out = llm._redact("failed https://example.com?key=SECRETKEY123 boom SECRETKEY123")
        assert "SECRETKEY123" not in out
        assert "[redacted]" in out
        # empty key → passthrough
        object.__setattr__(settings, "llm_api_key", "")
        assert llm._redact("plain error") == "plain error"
    finally:
        object.__setattr__(settings, "llm_api_key", orig)


def test_safe_tag_strips_braces_and_newlines():
    from app.pipeline.subtitle import _safe_tag

    assert "{" not in _safe_tag("a{b}c")
    assert "}" not in _safe_tag("a{b}c")
    assert _safe_tag("a{b}c") == "a(b)c"
    assert "\n" not in _safe_tag("a\nb")
    assert "\r" not in _safe_tag("a\rb")
    # literal ASS newline \N must not survive (would inject a line break)
    assert "\\N" not in _safe_tag("a\\Nb")
    assert "\\n" not in _safe_tag("a\\nb")
    # injection attempt neutralised
    evil = "{\\fs100}hello\nworld"
    safe = _safe_tag(evil)
    assert "{" not in safe and "}" not in safe and "\n" not in safe
    assert safe == "(\\fs100)hello world"


def test_write_ass_sanitizes_brand_and_source(tmp_path):
    from app.config import settings
    from app.pipeline import subtitle

    # NOTE 2026-09-11: repo ships a real logo PNG, which replaces the Brand
    # text line. Point at a missing logo so the Brand *text* sanitization
    # path (the injection surface) is exercised.
    orig_logo = settings.brand_logo
    object.__setattr__(settings, "brand_logo", tmp_path / "missing.png")
    try:
        out = tmp_path / "t.ass"
        subtitle.write_ass(
            [],
            0.0,
            10.0,
            out,
            source_tag="src {\\b1} tag",
            lines=[{"start": 0.0, "end": 1.0, "text": "hello"}],
            brand_text="brand {\\fs50}\nbreak",
            burn=True,
            show_source=True,
        )
        text = out.read_text(encoding="utf-8")
        brand_lines = [ln for ln in text.splitlines() if ",Brand," in ln]
        source_lines = [ln for ln in text.splitlines() if ",Source," in ln]
        assert brand_lines and source_lines
        for ln in brand_lines + source_lines:
            payload = ln.split(",,", 1)[1] if ",," in ln else ln
            assert "{" not in payload and "}" not in payload
            assert "\\N" not in payload
    finally:
        object.__setattr__(settings, "brand_logo", orig_logo)


def test_get_jobs_limit_bounds(client, monkeypatch):
    from app import jobs

    # invalid limits rejected by FastAPI validation (never reach DB)
    assert client.get("/api/jobs", params={"limit": -1}).status_code == 422
    assert client.get("/api/jobs", params={"limit": 0}).status_code == 422
    assert client.get("/api/jobs", params={"limit": 200}).status_code == 422
    assert client.get("/api/jobs", params={"limit": 101}).status_code == 422
    # valid limit reaches handler — stub the DB, never touch sabily.db
    monkeypatch.setattr(jobs, "list_jobs", lambda limit: [])
    r = client.get("/api/jobs", params={"limit": 1})
    assert r.status_code == 200
    assert r.json() == []


def test_clip_line_validation(client):
    cid = "abcdef012345"
    # negative start
    r = client.patch(f"/api/clips/{cid}",
                     json={"lines": [{"start": -1, "end": 1.0, "text": "x"}]})
    assert r.status_code == 422
    # missing text
    r = client.patch(f"/api/clips/{cid}", json={"lines": [{"start": 0, "end": 1.0}]})
    assert r.status_code == 422
    # text too long
    r = client.patch(f"/api/clips/{cid}",
                     json={"lines": [{"start": 0, "end": 1.0, "text": "x" * 201}]})
    assert r.status_code == 422
    # negative end
    r = client.patch(f"/api/clips/{cid}",
                     json={"lines": [{"start": 0, "end": -0.5, "text": "x"}]})
    assert r.status_code == 422


def test_edit_clip_converts_lines_to_dicts(client, monkeypatch, tmp_path):
    from app import jobs
    from app.pipeline import normalize

    monkeypatch.setattr(settings.__class__, "terms_file",
                        property(lambda self: tmp_path / "terms.json"), raising=False)
    cid = "abcdef012345"
    stored = {}

    def fake_get(_cid):
        return {"id": cid, "job_id": "abcdef012345", "meta": {"lines": [{"start": 0, "end": 1, "text": "old text here"}]}}

    def fake_update(_cid, **fields):
        stored.update(fields)

    monkeypatch.setattr(jobs, "get_clip", fake_get)
    # second get_clip after update returns stored meta
    orig_get = fake_get

    def fake_get2(_cid):
        if stored:
            return {"id": cid, "job_id": "abcdef012345", "meta": stored.get("meta", {})}
        return orig_get(_cid)

    monkeypatch.setattr(jobs, "get_clip", fake_get2)
    monkeypatch.setattr(jobs, "update_clip", fake_update)
    # avoid writing real glossary twice — use tmp terms file already patched
    r = client.patch(f"/api/clips/{cid}",
                     json={"lines": [{"start": 0, "end": 1.0, "text": "new text here"}]})
    assert r.status_code == 200
    assert isinstance(stored.get("meta"), dict)
    assert isinstance(stored["meta"]["lines"], list)
    assert all(isinstance(x, dict) for x in stored["meta"]["lines"])


def test_rerender_path_containment(client, monkeypatch, tmp_path):
    from app import jobs

    src = tmp_path / "src.mp4"
    src.write_bytes(b"fake")
    fake = {
        "id": "abcdef012345",
        "job_id": "abcdef012345",
        "video_path": "../../evil.mp4",
        "start_sec": 0.0,
        "end_sec": 1.0,
        "meta": {"source_video": str(src)},
    }
    monkeypatch.setattr(jobs, "get_clip", lambda _cid: dict(fake))
    r = client.post("/api/clips/abcdef012345/rerender")
    assert r.status_code == 400


def test_rerender_absolute_escape_rejected(client, monkeypatch, tmp_path):
    from app import jobs

    src = tmp_path / "src.mp4"
    src.write_bytes(b"fake")
    fake = {
        "id": "abcdef012345",
        "job_id": "abcdef012345",
        "video_path": "/etc/passwd",
        "start_sec": 0.0,
        "end_sec": 1.0,
        "meta": {"source_video": str(src)},
    }
    monkeypatch.setattr(jobs, "get_clip", lambda _cid: dict(fake))
    r = client.post("/api/clips/abcdef012345/rerender")
    # absolute path outside outputs_dir → 400 (or 409 only if source missing,
    # but source exists here so must be 400)
    assert r.status_code == 400


def test_non_loopback_guard():
    from app.main import _startup

    orig = settings.host
    try:
        object.__setattr__(settings, "host", "0.0.0.0")
        with pytest.raises(SystemExit):
            _startup()
    finally:
        object.__setattr__(settings, "host", orig)
