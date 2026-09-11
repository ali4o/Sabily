"""Batch 1 (P0 crash fixes) regression tests — no GPU, no network, no ffmpeg."""

import json
import string
import subprocess
import sys
from pathlib import Path


def test_dynamic_crop_escaping():
    from app.pipeline.render import _escape_x

    # static values unchanged (preserve existing behavior)
    assert _escape_x("420") == "420"
    assert _escape_x("0") == "0"

    # dynamic expression from reframe.py gets single-quoted
    dynamic = "if(lt(t,1.50),100,if(lt(t,3.00),200,300))"
    escaped = _escape_x(dynamic)
    assert escaped.startswith("'") and escaped.endswith("'")
    assert "if(lt(t,1.50)" in escaped

    # single quotes inside are escaped
    assert _escape_x("a'b,c") == "'a\\'b,c'"

    # full crop filter appears exactly once in the -vf chain
    crop = {"w": 608, "h": 1080, "x_expr": dynamic, "y": 0}
    vf = ",".join([
        f"crop={crop['w']}:{crop['h']}:{_escape_x(str(crop['x_expr']))}:{crop['y']}",
        "scale=1080:1920:flags=bicubic",
        "setsar=1",
    ])
    assert vf.count("crop=") == 1
    assert f"'{dynamic}'" in vf

    # static chain has no quotes at all
    static_crop = {"w": 608, "h": 1080, "x_expr": "420", "y": 0}
    static_vf = f"crop={static_crop['w']}:{static_crop['h']}:" \
                f"{_escape_x(str(static_crop['x_expr']))}:{static_crop['y']}"
    assert "'" not in static_vf


def test_header_helper_covers_all_placeholders():
    from app.pipeline.subtitle import HEADER, build_header, header_params

    fields = [fn for _, fn, _, _ in string.Formatter().parse(HEADER) if fn]
    assert fields, "expected HEADER to contain placeholders"
    params = header_params(1080, 1920)
    assert set(fields) == set(params.keys()), f"missing: {set(fields) - set(params.keys())}"

    header = build_header(1080, 1920)
    assert "{" not in header, "unformatted placeholder left in header"
    assert "1920" in header and "1080" in header


def test_doctor_section_no_crash_on_cp1256(monkeypatch, capsys):
    from scripts import doctor

    class FakeOut:
        def __init__(self, encoding):
            self.encoding = encoding
            self.buf = ""

        def write(self, s):
            # emulate a real cp1256 console: ─ cannot be encoded there
            if self.encoding and "cp1256" in self.encoding.lower():
                s.encode("cp1256")
            self.buf += s

        def flush(self):
            pass

    fake = FakeOut("cp1256")
    monkeypatch.setattr(sys, "stdout", fake)
    doctor.section("اختبار")
    assert "-" * 62 in fake.buf
    assert "─" not in fake.buf

    fake_utf8 = FakeOut("utf-8")
    monkeypatch.setattr(sys, "stdout", fake_utf8)
    assert doctor._sep() == "─" * 62

    # main() must attempt utf-8 reconfigure without crashing on streams
    # that lack reconfigure (covered by try/except) — just call helpers.
    monkeypatch.setattr(sys, "stdout", FakeOut("cp1256"))
    assert doctor._sep() == "-" * 62


def test_probe_zero_fps_fallback(monkeypatch):
    from app.pipeline import media

    payload = json.dumps({
        "streams": [{"width": 1920, "height": 1080, "r_frame_rate": "0/0"}],
        "format": {"duration": "10.0"},
    })

    class FakeCompleted:
        stdout = payload
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeCompleted())
    info = media.probe(Path("dummy.mp4"))
    assert info["fps"] == 25.0
    assert info["width"] == 1920 and info["height"] == 1080

    # missing dimensions raise a clear error instead of downstream crop failure
    bad_payload = json.dumps({
        "streams": [{"width": 0, "height": 0, "r_frame_rate": "30/1"}],
        "format": {"duration": "5.0"},
    })

    class FakeBad:
        stdout = bad_payload
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeBad())
    try:
        media.probe(Path("dummy.mp4"))
    except RuntimeError as exc:
        assert "أبعاد" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing dimensions")

    # ffprobe hard failure surfaces as RuntimeError
    def _raise(*a, **k):
        raise subprocess.CalledProcessError(1, "ffprobe", stderr="boom")

    monkeypatch.setattr(subprocess, "run", _raise)
    try:
        media.probe(Path("dummy.mp4"))
    except RuntimeError as exc:
        assert "ffprobe failed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for ffprobe failure")


def test_config_bad_int_falls_back(tmp_path, monkeypatch):
    from app import config

    monkeypatch.setenv("PORT", "not-a-number")
    assert config._int("PORT", 6767) == 6767

    monkeypatch.setenv("MIN_CLIP_SEC", "bad")
    assert config._float("MIN_CLIP_SEC", 20.0) == 20.0

    # absolute paths must not be joined onto BASE_DIR
    abs_dir = tmp_path / "abs_work"
    monkeypatch.setenv("WORK_DIR", str(abs_dir))
    assert config._path("WORK_DIR", "data") == Path(str(abs_dir))

    # relative paths still resolve under BASE_DIR (existing behavior)
    monkeypatch.setenv("WORK_DIR", "data")
    assert config._path("WORK_DIR", "data") == config.BASE_DIR / "data"
