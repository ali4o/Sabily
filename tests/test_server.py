"""Server power controls — no real exit, no real subprocess."""

import os
import sys
import threading

from fastapi.testclient import TestClient

from app.main import app


class FakeTimer:
    seen: list = []

    def __init__(self, delay, fn):
        self.delay = delay
        self.fn = fn

    def start(self):
        FakeTimer.seen.append((self.delay, self.fn))
        self.fn()


def _patch_exit(monkeypatch):
    calls = []
    monkeypatch.setattr(threading, "Timer", FakeTimer)
    monkeypatch.setattr(os, "_exit", lambda code: calls.append(code))
    FakeTimer.seen.clear()
    return calls


def test_shutdown_schedules_exit(monkeypatch):
    calls = _patch_exit(monkeypatch)
    r = TestClient(app, raise_server_exceptions=False).post("/api/server/shutdown")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "action": "shutdown"}
    assert calls == [0]


def test_restart_spawns_fresh_process(monkeypatch):
    import subprocess

    from app.config import settings

    calls = _patch_exit(monkeypatch)
    spawned = []
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: spawned.append((a, k)))
    r = TestClient(app, raise_server_exceptions=False).post("/api/server/restart")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "action": "restart"}
    (args, kwargs), = spawned
    assert list(args[0][:3]) == [sys.executable, "-m", "app.main"]
    assert kwargs.get("cwd") == str(settings.base_dir)
    assert calls == [0]


def test_restart_spawn_failure_keeps_server_up(monkeypatch):
    import subprocess

    calls = _patch_exit(monkeypatch)

    def _boom(*a, **k):
        raise OSError("no spawn")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    r = TestClient(app, raise_server_exceptions=False).post("/api/server/restart")
    assert r.status_code == 500
    assert calls == []
