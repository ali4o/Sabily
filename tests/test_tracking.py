"""Professional face tracking — pure logic + mocked video, no GPU."""

from app.config import settings
from app.pipeline import reframe


def _set(name, value):
    orig = getattr(settings, name)
    object.__setattr__(settings, name, value)
    return orig


def _restore(name, orig):
    object.__setattr__(settings, name, orig)


def test_fill_gaps_bridges_short_misses():
    filled, cov = reframe._fill_gaps(
        [(0.0, 0.5), (0.5, None), (1.0, 0.6), (2.0, None), (5.0, 0.7)], 1.5)
    assert cov == 3 / 5
    # 0.5s and 1.0s gaps bridged with last position; 3s gap breaks it
    assert filled == [(0.0, 0.5), (0.5, 0.5), (1.0, 0.6), (2.0, 0.6), (5.0, 0.7)]
    filled2, _ = reframe._fill_gaps([(0.0, 0.4), (3.0, None), (4.0, 0.5)], 1.5)
    assert filled2 == [(0.0, 0.4), (4.0, 0.5)]


def test_fill_gaps_empty_and_all_none():
    assert reframe._fill_gaps([], 1.5) == ([], 0.0)
    filled, cov = reframe._fill_gaps([(0.0, None), (0.5, None)], 5.0)
    assert filled == [] and cov == 0.0


def test_build_crop_fit_on_low_coverage(monkeypatch):
    monkeypatch.setattr(reframe, "_sample_face_centers",
                        lambda *a, **k: [(t, None) for t in (0.0, 0.5, 1.0, 1.5)])
    o1, o2 = _set("face_min_coverage", 0.25), _set("aspect", "9:16")
    try:
        crop = reframe.build_crop("v.mp4", 0, 2, 1280, 720, "9:16")
        assert crop["mode"] == "fit"
    finally:
        _restore("face_min_coverage", o1)
        _restore("aspect", o2)


def test_build_crop_tracks_when_covered(monkeypatch):
    pts = [(t, 0.3) for t in [round(i * 0.5, 2) for i in range(12)]]
    monkeypatch.setattr(reframe, "_sample_face_centers", lambda *a, **k: pts)
    crop = reframe.build_crop("v.mp4", 0, 6, 1280, 720, "9:16")
    assert crop["mode"] in ("static", "dynamic")


def test_detect_center_profile_second_opinion(monkeypatch):
    import numpy as np

    frontal_calls: list = []

    class FakeFrontal:
        def detectMultiScale(self, *a, **k):
            frontal_calls.append(1)
            return ()

    class FakeProfile:
        def detectMultiScale(self, *a, **k):
            return [(100, 50, 60, 60)]

    monkeypatch.setattr(reframe, "_detector", lambda: ("haar", (FakeFrontal(), FakeProfile())))
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    center = reframe._detect_center(frame)
    assert frontal_calls and center is not None
    assert abs(center - (100 + 30) / 400) < 1e-6


def test_full_frame_filter_shape():
    from app.pipeline import render

    vf = render.full_frame_filter(1080, 1920)
    assert "split=2" in vf and "gblur=sigma=" in vf
    assert "overlay=(W-w)/2:(H-h)/2" in vf and vf.endswith("[v0]")
    assert "force_original_aspect_ratio=increase" in vf


def test_render_clip_fit_branch(monkeypatch, tmp_path):
    from app.pipeline import render

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(render.subprocess, "run", fake_run)
    monkeypatch.setattr(render, "verify_output", lambda *a, **k: None)
    crop = {"w": 404, "h": 720, "x_expr": "0", "y": 0, "mode": "fit"}
    render.render_clip(tmp_path / "s.mp4", tmp_path / "o.mp4", 0, 1,
                       crop, None, target_size=(720, 1280),
                       show_logo=False, burn_subtitles=False)
    vf = calls[-1][calls[-1].index("-vf") + 1]
    # NOTE 2026-09-12: fit chain runs via -vf (single implicit input), so
    # it must not carry a "[0:v]" prefix nor a trailing "[v0]" when no logo
    # is chained — both made real ffmpeg fail (2 inputs/2 outputs). The bg
    # crop (crop=720:1280) is the blurred fill, not a blind center slice.
    assert "gblur=" in vf and "[0:v]" not in vf and not vf.endswith("[v0]")
    assert vf.startswith("split=2")
