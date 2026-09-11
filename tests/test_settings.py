"""Settings store (.env) + stats API — never touches the real .env or DB."""

from fastapi.testclient import TestClient

from app import jobs, settings_store
from app.config import settings
from app.main import app


def _env(tmp_path, text="# test env\nPORT=7000\n"):
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    return p


def test_describe_sources_and_masking(tmp_path, monkeypatch):
    from app import settings_store as ss

    p = _env(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "sekret")
    d = ss.describe(tmp_path)["keys"]
    assert d["PORT"]["value"] == "7000" and d["PORT"]["source"] == "env_file"
    assert d["LLM_API_KEY"]["value"] == "" and d["LLM_API_KEY"]["has_value"] is True
    assert d["QUALITY"]["source"] in ("default", "environment")
    monkeypatch.delenv("LLM_API_KEY", raising=False)


def test_save_validates_and_preserves_comments(tmp_path):
    from app import settings_store as ss

    p = _env(tmp_path, "# keep me\nPORT=7000\nCUSTOM=1\n")
    out = ss.save({"PORT": "6768", "QUALITY": "qhd", "BURN_SUBTITLES": False}, tmp_path)
    assert out["saved"] == ["BURN_SUBTITLES", "PORT", "QUALITY"]
    text = p.read_text(encoding="utf-8")
    assert "# keep me" in text and "CUSTOM=1" in text
    assert "PORT=6768" in text and "QUALITY=QHD" in text
    assert "BURN_SUBTITLES=false" in text


def test_save_rejects_unknown_and_bad_values(tmp_path):
    import pytest

    from app import settings_store as ss

    _env(tmp_path)
    with pytest.raises(ValueError, match="غير معروفة"):
        ss.save({"NOPE": "1"}, tmp_path)
    with pytest.raises(ValueError):
        ss.save({"PORT": "abc"}, tmp_path)
    with pytest.raises(ValueError):
        ss.save({"PORT": "80"}, tmp_path)
    with pytest.raises(ValueError):
        ss.save({"QUALITY": "8K"}, tmp_path)
    with pytest.raises(ValueError):
        ss.save({"BURN_SUBTITLES": "maybe"}, tmp_path)


def test_save_secret_masked_keeps_old(tmp_path):
    from app import settings_store as ss

    p = _env(tmp_path, "LLM_API_KEY=realkey\n")
    ss.save({"LLM_API_KEY": "••••••", "PORT": "6767"}, tmp_path)
    assert "LLM_API_KEY=realkey" in p.read_text(encoding="utf-8")
    ss.save({"LLM_API_KEY": "newkey"}, tmp_path)
    assert "LLM_API_KEY=newkey" in p.read_text(encoding="utf-8")


def test_settings_endpoints_do_not_touch_real_env(client=None):
    # TestClient against the real app would read the real .env on GET
    # (read-only, safe); PUT is tested via tmp base through save().
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/api/settings")
    assert r.status_code == 200
    assert "QUALITY" in r.json()["keys"]
    r = c.put("/api/settings", json={"NOPE_KEY": "1"})
    assert r.status_code == 400
    r = c.put("/api/settings", json={})
    assert r.status_code == 400


def test_stats_endpoint_on_tmp_db(tmp_path):
    orig = settings.db_path
    object.__setattr__(settings, "db_path", tmp_path / "t.db")
    try:
        jobs.init_db()
        jid = jobs.create_job("https://example.com/v", {})
        jobs.add_clip(jid, {"idx": 1, "start_sec": 0.0, "end_sec": 60.0,
                            "meta": {}})
        c = TestClient(app, raise_server_exceptions=False)
        r = c.get("/api/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["jobs_total"] >= 1 and body["clips_total"] >= 1
        assert body["rendered_min"] >= 1.0
        assert set(body["disk_mb"]) == {"outputs", "work", "db"}
        assert body["recent"][0]["id"] == jid
    finally:
        object.__setattr__(settings, "db_path", orig)
