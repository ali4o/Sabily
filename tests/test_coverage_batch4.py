"""Batch 4 (coverage + docs/UX) — no GPU, no network, no live ffmpeg, no real DB."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import settings
from app.pipeline.transcribe import Word


def _set_setting(name, value):
    orig = getattr(settings, name)
    object.__setattr__(settings, name, value)
    return orig


def _restore_setting(name, orig):
    object.__setattr__(settings, name, orig)


# --------------------------------------------------------------------------- #
# 1. download
# --------------------------------------------------------------------------- #

def test_download_as_hashtag_arabic_spaces_emoji_empty():
    from app.pipeline.download import as_hashtag

    assert as_hashtag("قناة الجزيرة") == "#قناة_الجزيرة"
    assert as_hashtag("My Channel Name") == "#My_Channel_Name"
    # emoji stripped
    assert as_hashtag("Hello \U0001f600 World") == "#Hello_World"
    # spaces collapse to single underscore
    assert as_hashtag("a   b") == "#a_b"
    assert as_hashtag("") == ""
    assert as_hashtag("!!!") == ""
    assert as_hashtag("   ") == ""


def test_download_timestamped_url_variants():
    from app.pipeline.download import timestamped_url

    yt_q = "https://www.youtube.com/watch?v=abc123"
    assert timestamped_url(yt_q, 65) == f"{yt_q}&t=65s"
    yt_no_q = "https://www.youtube.com/watch"
    assert timestamped_url(yt_no_q, 65) == f"{yt_no_q}?t=65s"
    yt_has_q = "https://www.youtube.com/watch?v=abc123&list=PL1"
    assert timestamped_url(yt_has_q, 10) == f"{yt_has_q}&t=10s"
    short = "https://youtu.be/abc123"
    assert timestamped_url(short, 5) == f"{short}?t=5s"
    vimeo = "https://vimeo.com/12345"
    assert timestamped_url(vimeo, 30) == f"{vimeo}#t=30s"
    other = "https://example.com/video.mp4"
    assert timestamped_url(other, 30) == other
    # negative clamps to 0
    assert timestamped_url(yt_q, -5).endswith("&t=0s")
    assert timestamped_url(yt_no_q, -5).endswith("?t=0s")
    assert timestamped_url(vimeo, -1).endswith("#t=0s")


def test_download_explain_branches():
    from app.pipeline.download import _explain

    assert "403" in _explain("ERROR: 403 Forbidden boom")
    assert "403" in _explain("forbidden access denied")
    assert "تسجيل دخول" in _explain("Sign in to confirm you are not a bot")
    assert "تسجيل دخول" in _explain("bot detected, need cookies")
    assert "تسجيل دخول" in _explain("cookies missing")
    assert "خاص أو محذوف" in _explain("This video is private")
    assert "خاص أو محذوف" in _explain("Video unavailable in your region")
    # age branch needs age+restrict without sign-in/bot/cookies keywords
    assert "مقيّد بالعمر" in _explain("This content is age restricted")
    # fallback echoes last line truncated
    fb = _explain("line1\nline2 boom happened")
    assert fb.startswith("فشل التحميل:")
    assert "boom happened" in fb


def test_download_fmt_contains_max_height():
    from app.pipeline.download import _fmt

    fmt = _fmt()
    assert str(settings.max_height) in fmt
    assert "height<=" in fmt


# --------------------------------------------------------------------------- #
# 2. llm
# --------------------------------------------------------------------------- #

def test_llm_parse_json_fenced():
    from app.pipeline.llm import _parse_json

    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json('```\n{"x": "y"}\n```') == {"x": "y"}


def test_llm_parse_json_prose():
    from app.pipeline.llm import _parse_json

    assert _parse_json('here is {"title": "x"} done') == {"title": "x"}
    assert _parse_json('{"a": [1, 2]}') == {"a": [1, 2]}


def test_llm_parse_json_invalid_raises():
    from app.pipeline.llm import _parse_json

    with pytest.raises(Exception):
        _parse_json("not json at all")


def test_llm_fallback_meta_limits():
    from app.pipeline.llm import _fallback_meta

    long_text = " ".join(["كلمة"] * 100)
    meta = _fallback_meta(long_text)
    assert len(meta["title"]) <= 60
    assert meta["hashtags"] == ["#Shorts", "#Reels", "#مقاطع"]
    assert _fallback_meta("")["title"] == "مقطع مختار"


def test_llm_generate_metadata_arabic_suffix(monkeypatch):
    from app.pipeline import llm

    captured = {}

    class Fake:
        def complete(self, system, user):
            captured["system"] = system
            captured["user"] = user
            return json.dumps({"title": "عنوان جيد", "summary": "ملخص", "hashtags": ["#ا"]})

    monkeypatch.setattr("app.pipeline.llm.get_provider", lambda: Fake())
    llm.generate_metadata("نص تجريبي للفيديو", lang="ar")
    assert "أعد JSON بالعربية فقط." in captured["user"]

    captured.clear()
    llm.generate_metadata("some english text here", lang="en")
    assert "أعد JSON بالعربية فقط." not in captured["user"]


def test_llm_generate_metadata_overlong_title_falls_back(monkeypatch):
    from app.pipeline import llm

    class Fake:
        def complete(self, system, user):
            return json.dumps({"title": "x" * 100, "summary": "s", "hashtags": ["#a"]})

    monkeypatch.setattr("app.pipeline.llm.get_provider", lambda: Fake())
    out = llm.generate_metadata("نص قصير للتجربة", lang="en")
    assert out["title"] != "x" * 80
    assert len(out["title"]) <= 80


def test_llm_generate_metadata_hashtags_string_split(monkeypatch):
    from app.pipeline import llm

    class Fake:
        def complete(self, system, user):
            return json.dumps({"title": "عنوان", "summary": "ملخص", "hashtags": "#واحد #اثنان"})

    monkeypatch.setattr("app.pipeline.llm.get_provider", lambda: Fake())
    out = llm.generate_metadata("نص", lang="en")
    assert out["hashtags"] == ["#واحد", "#اثنان"]


def test_llm_generate_metadata_exception_falls_back(monkeypatch):
    from app.pipeline import llm

    class Boom:
        def complete(self, system, user):
            raise RuntimeError("llm down")

    monkeypatch.setattr("app.pipeline.llm.get_provider", lambda: Boom())
    out = llm.generate_metadata("نص تجريبي للفيديو هنا", lang="en")
    assert out["hashtags"] == ["#Shorts", "#Reels", "#مقاطع"]


def test_llm_ollama_messages_gemma_vs_other():
    from app.pipeline.llm import OllamaProvider

    orig = _set_setting("llm_model", "gemma2:2b")
    try:
        msgs = OllamaProvider()._messages("SYS", "USER")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert "SYS" in msgs[0]["content"] and "USER" in msgs[0]["content"]
    finally:
        _restore_setting("llm_model", orig)

    orig = _set_setting("llm_model", "llama3:8b")
    try:
        msgs = OllamaProvider()._messages("SYS", "USER")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system" and msgs[0]["content"] == "SYS"
        assert msgs[1]["role"] == "user" and msgs[1]["content"] == "USER"
    finally:
        _restore_setting("llm_model", orig)


def test_llm_rerank_provider_none_and_flag_and_short(monkeypatch):
    from app.pipeline import llm

    cands = [SimpleNamespace(duration=30.0, text="نص %d" % i) for i in range(5)]
    monkeypatch.setattr("app.pipeline.llm.get_provider", lambda: None)
    assert llm.rerank(cands, 3) == [0, 1, 2]

    class Fake:
        def complete(self, s, u):
            return json.dumps({"order": [4, 3, 2, 1, 0]})

    monkeypatch.setattr("app.pipeline.llm.get_provider", lambda: Fake())
    orig = _set_setting("llm_rerank", False)
    try:
        assert llm.rerank(cands, 3) == [0, 1, 2]
    finally:
        _restore_setting("llm_rerank", orig)

    # len <= keep returns default without calling provider
    orig = _set_setting("llm_rerank", True)
    try:
        assert llm.rerank(cands[:2], 5) == [0, 1]
    finally:
        _restore_setting("llm_rerank", orig)


def test_llm_rerank_dedup_out_of_range_and_empty(monkeypatch):
    from app.pipeline import llm

    cands = [SimpleNamespace(duration=20.0, text="نص %d" % i) for i in range(3)]
    orig = _set_setting("llm_rerank", True)
    try:
        class Fake:
            def complete(self, s, u):
                return json.dumps({"order": [2, 2, 99, -1, 0]})

        monkeypatch.setattr("app.pipeline.llm.get_provider", lambda: Fake())
        assert llm.rerank(cands, 2) == [2, 0]

        class Empty:
            def complete(self, s, u):
                return json.dumps({"order": []})

        monkeypatch.setattr("app.pipeline.llm.get_provider", lambda: Empty())
        assert llm.rerank(cands, 2) == [0, 1]
    finally:
        _restore_setting("llm_rerank", orig)


def test_llm_build_caption_with_without_tags():
    from app.pipeline.llm import build_caption

    with_tags = build_caption({"summary": "شرح", "hashtags": ["#ا", "#ب"]}, "https://example.com")
    assert "شرح" in with_tags
    assert "https://example.com" in with_tags
    assert "#ا #ب" in with_tags

    without = build_caption({"summary": "شرح", "hashtags": []}, "https://example.com")
    assert "شرح" in without
    assert "https://example.com" in without
    assert "#" not in without


# --------------------------------------------------------------------------- #
# 3. reframe
# --------------------------------------------------------------------------- #

def test_reframe_smooth_converges():
    from app.pipeline.reframe import _smooth

    assert _smooth([(0, 0.5), (1, 0.5)]) == [(0, 0.5), (1, 0.5)]
    out = _smooth([(0, 0.0), (0.5, 1.0)])
    assert out[1][1] == pytest.approx(0.25)
    # long constant target converges
    pts = [(i * 0.5, 1.0) for i in range(30)]
    pts[0] = (0, 0.0)
    assert _smooth(pts)[-1][1] == pytest.approx(1.0, abs=0.01)


def test_reframe_thin_drops_jitter_and_caps():
    from app.pipeline.reframe import _thin

    pts = [(0, 0.5), (0.5, 0.51), (1.0, 0.505), (1.5, 0.52)]
    assert _thin(pts) == [(0, 0.5)]
    many = [(i * 0.5, i * 0.05) for i in range(100)]
    thinned = _thin(many)
    assert len(thinned) == 20


def test_reframe_x_for_clamps():
    from app.pipeline.reframe import _x_for

    assert _x_for(0.0, 1920, 600) == 0
    assert _x_for(1.0, 1920, 600) == 1920 - 600
    assert _x_for(-1.0, 1920, 600) == 0
    assert _x_for(2.0, 1920, 600) == 1920 - 600
    assert _x_for(0.5, 1920, 600) == int(0.5 * 1920 - 300)


def test_reframe_build_crop_center_on_empty_and_exception(monkeypatch):
    from app.pipeline import reframe
    from app.pipeline.media import crop_size

    cw, _ = crop_size(1920, 1080, "9:16")
    expect = str((1920 - cw) // 2)
    # NOTE 2026-09-12: zero detections now means full-frame fit (mode
    # "fit"), not a blind center slice — only a hard sampling *error*
    # still falls back to center.
    monkeypatch.setattr(reframe, "_sample_face_centers", lambda v, s, e: [])
    out = reframe.build_crop(Path("v.mp4"), 0, 5, 1920, 1080, "9:16")
    assert out["mode"] == "fit"

    monkeypatch.setattr(reframe, "_sample_face_centers",
                        lambda v, s, e: [(0.0, None), (0.5, None)])
    out0 = reframe.build_crop(Path("v.mp4"), 0, 5, 1920, 1080, "9:16")
    assert out0["mode"] == "fit"

    def _boom(v, s, e):
        raise RuntimeError("no cv2")

    monkeypatch.setattr(reframe, "_sample_face_centers", _boom)
    out2 = reframe.build_crop(Path("v.mp4"), 0, 5, 1920, 1080, "9:16")
    assert out2["mode"] == "center"
    assert out2["x_expr"] == expect


def test_reframe_build_crop_full_when_narrow():
    from app.pipeline import reframe

    # 720x1280 at 9:16 fills width exactly -> full
    out = reframe.build_crop(Path("v.mp4"), 0, 5, 720, 1280, "9:16")
    assert out["mode"] == "full"
    assert out["x_expr"] == "0"


def test_reframe_build_crop_static_when_dynamic_off(monkeypatch):
    from app.pipeline import reframe

    orig = _set_setting("dynamic_crop", False)
    try:
        monkeypatch.setattr(reframe, "_sample_face_centers",
                            lambda v, s, e: [(0, 0.2), (0.5, 0.8)])
        out = reframe.build_crop(Path("v.mp4"), 0, 5, 1920, 1080, "9:16")
        assert out["mode"] == "static"
        assert "if(lt(t," not in out["x_expr"]
    finally:
        _restore_setting("dynamic_crop", orig)


def test_reframe_build_crop_dynamic_expression(monkeypatch):
    from app.pipeline import reframe

    orig = _set_setting("dynamic_crop", True)
    try:
        monkeypatch.setattr(reframe, "_sample_face_centers",
                            lambda v, s, e: [(0, 0.2), (0.5, 0.5), (1.0, 0.8)])
        out = reframe.build_crop(Path("v.mp4"), 0, 5, 1920, 1080, "9:16")
        assert out["mode"] == "dynamic"
        assert "if(lt(t," in out["x_expr"]
    finally:
        _restore_setting("dynamic_crop", orig)


# --------------------------------------------------------------------------- #
# 4. subtitle
# --------------------------------------------------------------------------- #

def test_subtitle_group_lines_empty():
    from app.pipeline.subtitle import group_lines

    assert group_lines([]) == []


def test_subtitle_single_word_stretched_no_bleed():
    from app.pipeline.subtitle import MIN_LINE_SEC, group_lines

    words = [Word(start=0.0, end=0.2, text="مرحبا."), Word(start=5.0, end=5.2, text="ثانية.")]
    lines = group_lines(words)
    assert len(lines) == 2
    s0, e0, _ = lines[0]
    assert e0 - s0 >= MIN_LINE_SEC - 0.01
    assert e0 <= lines[1][0]  # never into next start


def test_subtitle_max_words_split():
    from app.pipeline.subtitle import group_lines

    words = [Word(start=i * 0.5, end=i * 0.5 + 0.4, text="w%d" % i) for i in range(7)]
    lines = group_lines(words)
    assert len(lines) >= 2
    assert lines[0][2].split() == ["w%d" % i for i in range(6)]


def test_subtitle_write_ass_skips_end_le_start(tmp_path):
    from app.config import settings
    from app.pipeline import subtitle

    # NOTE 2026-09-11: the repo now ships a real logo PNG, so the Brand
    # text line is replaced by the image overlay and an all-dropped body
    # yields None. Point at a missing logo to exercise the text path.
    orig = settings.brand_logo
    object.__setattr__(settings, "brand_logo", tmp_path / "missing.png")
    try:
        out = tmp_path / "t.ass"
        # NOTE: flags are explicit — the ambient .env may disable
        # burning/hashtags (e.g. BURN_SUBTITLES=false).
        subtitle.write_ass([], 0.0, 10.0, out,
                           lines=[{"start": 2.0, "end": 1.0, "text": "bad"}],
                           burn=True)
        text = out.read_text(encoding="utf-8")
        assert "bad" not in text
    finally:
        object.__setattr__(settings, "brand_logo", orig)


def test_subtitle_ts_negative():
    from app.pipeline.subtitle import _ts

    assert _ts(-1) == "0:00:00.00"
    assert _ts(0) == "0:00:00.00"


def test_subtitle_sanitize_braces_newlines(tmp_path):
    from app.pipeline import subtitle
    from app.pipeline.subtitle import LRI, PDI

    out = tmp_path / "t.ass"
    subtitle.write_ass([], 0.0, 10.0, out,
                       lines=[{"start": 0.0, "end": 1.0, "text": "a{b}\nc"}],
                       burn=True)
    text = out.read_text(encoding="utf-8")
    # NOTE 2026-09-11: Latin runs are wrapped in bidi isolates by mix_ar_en
    # (approved Arabic/English mixing); strip them before asserting the
    # original sanitization intent: braces/newlines never reach the file.
    plain = text.replace(LRI, "").replace(PDI, "")
    assert "a(b) c" in plain
    assert "{b}" not in text


def test_subtitle_empty_inside_no_tag_returns_none(tmp_path):
    from app.pipeline import subtitle

    out = tmp_path / "t.ass"
    assert subtitle.write_ass([], 0.0, 10.0, out, source_tag="") is None
    assert not out.exists()


def test_subtitle_editor_path_offset_and_clamp(tmp_path):
    from app.pipeline import subtitle

    out = tmp_path / "t.ass"
    # NOTE 2026-09-12: burn explicit — ambient .env may disable subtitles.
    subtitle.write_ass([], 10.0, 20.0, out,
                       lines=[{"start": 0.0, "end": 5.0, "text": "hi"},
                              {"start": 0.0, "end": 100.0, "text": "long"}],
                       burn=True, show_brand=False, show_source=False)
    text = out.read_text(encoding="utf-8")
    assert "0:00:00.00" in text
    assert "0:00:05.00" in text
    assert "0:00:10.00" in text  # clamped to clip duration


# --------------------------------------------------------------------------- #
# 5. normalize
# --------------------------------------------------------------------------- #

def test_normalize_multiword_span_and_punct():
    from app.pipeline.normalize import apply

    terms = {"replace": {"كي فريم": "Keyframe"}, "split": []}
    words = [Word(start=0.0, end=0.5, text="كي"), Word(start=0.5, end=1.0, text="فريم.")]
    out = apply(words, terms)
    assert len(out) == 1
    assert out[0].text == "Keyframe."
    assert out[0].start == 0.0 and out[0].end == 1.0


def test_normalize_single_replace():
    from app.pipeline.normalize import apply

    terms = {"replace": {"المحشوى": "المحتوى"}, "split": []}
    out = apply([Word(start=0.0, end=1.0, text="المحشوى")], terms)
    assert out[0].text == "المحتوى"


def test_normalize_empty_terms_identity():
    from app.pipeline.normalize import apply

    words = [Word(start=0.0, end=1.0, text="مرحبا")]
    out = apply(words, {"replace": {}, "split": []})
    assert [w.text for w in out] == ["مرحبا"]


def test_normalize_corrupted_terms_returns_defaults(tmp_path):
    from app.pipeline.normalize import DEFAULT_TERMS, load_terms

    bad = tmp_path / "terms.json"
    bad.write_text("{not valid", encoding="utf-8")
    orig = _set_setting("terms_file", bad)
    try:
        assert load_terms() == DEFAULT_TERMS
    finally:
        _restore_setting("terms_file", orig)


def test_normalize_load_terms_deepcopy_isolation(tmp_path):
    from app.pipeline.normalize import load_terms

    f = tmp_path / "terms.json"
    f.write_text(json.dumps({"replace": {"a": "b"}, "split": []}), encoding="utf-8")
    orig = _set_setting("terms_file", f)
    try:
        first = load_terms()
        first["replace"]["new"] = "x"
        second = load_terms()
        assert "new" not in second["replace"]
    finally:
        _restore_setting("terms_file", orig)


# --------------------------------------------------------------------------- #
# 6. render
# --------------------------------------------------------------------------- #

def test_render_escape_and_shaping(tmp_path):
    from app.pipeline.render import _escape_for_filter, subtitles_filter

    orig = _set_setting("fonts_dir", tmp_path / "empty_fonts")
    try:
        (tmp_path / "empty_fonts").mkdir(exist_ok=True)
        esc = _escape_for_filter(Path("C:/a:b'c"))
        assert "\\:" in esc
        assert "\\'" in esc
        f = subtitles_filter(Path("C:/a:b'c.ass"))
        assert "shaping=complex" in f
        assert "\\:" in f
    finally:
        _restore_setting("fonts_dir", orig)


def test_render_fontsdir_conditional(tmp_path):
    from app.pipeline.render import subtitles_filter

    d = tmp_path / "fonts"
    d.mkdir(exist_ok=True)
    orig = _set_setting("fonts_dir", d)
    try:
        ass = tmp_path / "s.ass"
        ass.write_text("x", encoding="utf-8")
        assert ":fontsdir=" not in subtitles_filter(ass)
        (d / "font.ttf").write_bytes(b"fake")
        assert ":fontsdir=" in subtitles_filter(ass)
    finally:
        _restore_setting("fonts_dir", orig)


def test_render_video_args_flags():
    from app.pipeline.render import _video_args

    gpu = _video_args(True)
    assert "h264_nvenc" in gpu
    cpu = _video_args(False)
    assert "libx264" in cpu


def test_render_clip_retries_x264(monkeypatch, tmp_path):
    from app.pipeline import render

    orig = _set_setting("encoder", "nvenc")
    try:
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if len(calls) == 1:
                return SimpleNamespace(returncode=1, stderr="nvenc fail")
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(render.subprocess, "run", fake_run)
        # output-dimension check is stubbed: it needs a real ffprobe,
        # covered separately by test_verify_output_* below.
        monkeypatch.setattr(render, "verify_output", lambda *a, **k: None)
        crop = {"w": 608, "h": 1080, "x_expr": "0", "y": 0, "mode": "center"}
        out = render.render_clip(tmp_path / "s.mp4", tmp_path / "o.mp4", 0, 1, crop, None)
        assert out == tmp_path / "o.mp4"
        assert len(calls) == 2
        assert "libx264" in calls[1]
    finally:
        _restore_setting("encoder", orig)


def test_render_clip_double_failure_raises(monkeypatch, tmp_path):
    from app.pipeline import render

    orig = _set_setting("encoder", "nvenc")
    try:
        monkeypatch.setattr(render.subprocess, "run",
                            lambda *a, **k: SimpleNamespace(returncode=1, stderr="boom"))
        crop = {"w": 608, "h": 1080, "x_expr": "0", "y": 0, "mode": "center"}
        with pytest.raises(RuntimeError):
            render.render_clip(tmp_path / "s.mp4", tmp_path / "o.mp4", 0, 1, crop, None)
    finally:
        _restore_setting("encoder", orig)


# --------------------------------------------------------------------------- #
# 7. media
# --------------------------------------------------------------------------- #

def test_media_crop_size_ratios_and_even():
    from app.pipeline.media import crop_size

    w, h = crop_size(1920, 1080, "9:16")
    assert w % 2 == 0 and h % 2 == 0
    assert abs(w / h - 9 / 16) < 0.02

    w1, h1 = crop_size(1920, 1080, "1:1")
    assert (w1, h1) == (1080, 1080)

    # portrait keeps full frame when already 9:16
    assert crop_size(1080, 1920, "9:16") == (1080, 1920)

    # odd dims still even
    w2, h2 = crop_size(1919, 1079, "9:16")
    assert w2 % 2 == 0 and h2 % 2 == 0

    w3, h3 = crop_size(1919, 1079, "1:1")
    assert w3 % 2 == 0 and h3 % 2 == 0


# --------------------------------------------------------------------------- #
# 8. score gaps
# --------------------------------------------------------------------------- #

def test_score_keyword_weights_stopwords_and_normalized():
    from app.pipeline.score import keyword_weights
    from app.pipeline.score import Sentence

    sents = [Sentence(0, 1, [Word(0, 0.5, "the"), Word(0.5, 1, "تفاح")])]
    weights = keyword_weights(sents)
    assert "the" not in weights
    assert weights  # normalized تفاح kept


def test_score_density_zero_and_peak():
    from app.pipeline.score import _density

    assert _density("a b c", 0) == 0.0
    assert _density("a b", -1) == 0.0
    assert _density(" ".join(["w"] * 26), 10.0) == pytest.approx(1.0)
    assert _density("w", 10.0) < 1.0


def test_score_keyword_empty_zero():
    from app.pipeline.score import _keyword_score

    assert _keyword_score([], {}) == 0.0


def test_score_suppress_boundary_kept():
    from app.pipeline.score import Candidate, suppress_overlaps

    c1 = Candidate(start=0, end=10, text="a", score=1.0)
    c2 = Candidate(start=8, end=18, text="b", score=0.9)
    kept = suppress_overlaps([c1, c2], limit=2)
    assert len(kept) == 2  # exactly 0.2 overlap is kept


def test_score_short_transcript_no_candidates():
    from app.pipeline.score import Sentence, generate_candidates

    sents = [Sentence(0, 5, [Word(0, 5, "مرحبا")])]
    assert generate_candidates(sents, min_sec=20, max_sec=75) == []


# --------------------------------------------------------------------------- #
# 9. jobs
# --------------------------------------------------------------------------- #

def test_jobs_crud_meta_roundtrip(tmp_path):
    from app import jobs

    orig = _set_setting("db_path", tmp_path / "test.db")
    try:
        jobs.init_db()
        jid = jobs.create_job("https://example.com/v", {"clips": 3})
        job = jobs.get_job(jid)
        assert job["url"] == "https://example.com/v"
        jobs.update_job(jid, status="running", stage="x")
        assert jobs.get_job(jid)["status"] == "running"

        meta = {"a": 1, "nested": {"b": [1, 2]}}
        cid = jobs.add_clip(jid, {"idx": 1, "start_sec": 0.0, "end_sec": 5.0,
                                  "title": "t", "caption": "c",
                                  "source_url": "u", "video_path": "v.mp4",
                                  "meta": meta})
        got = jobs.get_clip(cid)
        assert got["meta"] == meta
        jobs.update_clip(cid, title="t2")
        assert jobs.get_clip(cid)["title"] == "t2"
        assert len(jobs.list_clips(jid)) == 1
        assert jobs.list_jobs()
    finally:
        _restore_setting("db_path", orig)


def test_jobs_log_buffer_cap_and_slicing():
    import logging

    from app import jobs

    jid = "abcdef012345"
    saved_buffers = dict(jobs._buffers)
    saved_current = jobs._current
    jobs._buffers.clear()
    jobs._current = jid
    jobs._buffers[jid] = []
    try:
        handler = jobs._JobLogHandler()
        logger = logging.getLogger("sabily.test.batch4")
        for i in range(jobs.MAX_LINES + 50):
            handler.emit(logger.makeRecord(
                logger.name, logging.INFO, __file__, 1, f"msg {i}", None, None))
        lines, cursor = jobs.log_lines(jid, 0)
        assert cursor == jobs.MAX_LINES
        assert len(lines) == jobs.MAX_LINES
        # slicing from cursor offset
        tail, cursor2 = jobs.log_lines(jid, 10)
        assert len(tail) == jobs.MAX_LINES - 10
        assert cursor2 == jobs.MAX_LINES
        # after beyond end clamps
        tail2, _ = jobs.log_lines(jid, 10_000)
        assert tail2 == []
    finally:
        jobs._buffers.clear()
        jobs._buffers.update(saved_buffers)
        jobs._current = saved_current


# --------------------------------------------------------------------------- #
# 10. transcribe
# --------------------------------------------------------------------------- #

def test_transcribe_cache_hit_bypasses_model(tmp_path, monkeypatch):
    from app.pipeline import transcribe

    cache = tmp_path / "transcript.json"
    cache.write_text(json.dumps([{"start": 0.0, "end": 1.0, "text": "hi", "conf": 1.0}]),
                     encoding="utf-8")

    def _boom(audio, device, compute, on_progress):
        raise AssertionError("model must not be called on cache hit")

    monkeypatch.setattr(transcribe, "_transcribe", _boom)
    words = transcribe.run(tmp_path / "audio.wav", cache=cache)
    assert len(words) == 1 and words[0].text == "hi"


def test_transcribe_cuda_error_retries_cpu(tmp_path, monkeypatch):
    from app.pipeline import transcribe

    orig = _set_setting("whisper_device", "cuda")
    try:
        calls = []

        def fake(audio, device, compute, on_progress):
            calls.append(device)
            if len(calls) == 1:
                raise RuntimeError("cublas failed to load")
            return [Word(start=0.0, end=1.0, text="مرحبا")]

        monkeypatch.setattr(transcribe, "_transcribe", fake)
        words = transcribe.run(tmp_path / "a.wav", cache=None)
        assert [w.text for w in words] == ["مرحبا"]
        assert calls == ["cuda", "cpu"]
    finally:
        _restore_setting("whisper_device", orig)


def test_transcribe_non_cuda_raises_immediately(tmp_path, monkeypatch):
    from app.pipeline import transcribe

    orig = _set_setting("whisper_device", "cuda")
    try:
        calls = []

        def fake(audio, device, compute, on_progress):
            calls.append(device)
            raise RuntimeError("random io failure")

        monkeypatch.setattr(transcribe, "_transcribe", fake)
        with pytest.raises(RuntimeError):
            transcribe.run(tmp_path / "a.wav", cache=None)
        assert calls == ["cuda"]
    finally:
        _restore_setting("whisper_device", orig)


def test_transcribe_cpu_failure_raises(tmp_path, monkeypatch):
    from app.pipeline import transcribe

    orig = _set_setting("whisper_device", "cpu")
    try:
        monkeypatch.setattr(transcribe, "_transcribe",
                            lambda a, d, c, o: (_ for _ in ()).throw(RuntimeError("cpu boom")))
        with pytest.raises(RuntimeError):
            transcribe.run(tmp_path / "a.wav", cache=None)
    finally:
        _restore_setting("whisper_device", orig)


# --------------------------------------------------------------------------- #
# 11. runner keep_source
# --------------------------------------------------------------------------- #

def _mock_runner(monkeypatch, tmp_path, options):
    from app.pipeline import runner
    from app import jobs as _jobs

    work = tmp_path / "work"
    outs = tmp_path / "outs"
    work.mkdir(parents=True, exist_ok=True)
    outs.mkdir(parents=True, exist_ok=True)
    orig_work = _set_setting("work_dir", work)
    orig_outs = _set_setting("outputs_dir", outs)

    jid = "abcdef012345"
    cand = SimpleNamespace(start=10.0, end=40.0, score=0.9, text="نص تجريبي",
                           duration=30.0, parts={"a": 1.0})

    monkeypatch.setattr("app.pipeline.runner.reframe.build_crop",
                        lambda v, s, e, w, h, aspect=None:
                        {"w": 608, "h": 1080, "x_expr": "0", "y": 0, "mode": "center"})
    monkeypatch.setattr("app.pipeline.runner.subtitle.write_ass",
                        lambda *a, **k: (a[3] if len(a) > 3 else k.get("out")))
    monkeypatch.setattr("app.pipeline.runner.render.render_clip",
                        lambda *a, **k: a[1])
    monkeypatch.setattr("app.pipeline.runner.probe", lambda v: {"width": 1920, "height": 1080})
    monkeypatch.setattr(runner.asr, "run", lambda audio, cache=None, on_progress=None:
                        [Word(start=10.0, end=10.5, text="مرحبا")])
    monkeypatch.setattr("app.pipeline.runner.normalize.apply", lambda ws: ws)
    # NOTE 2026-09-12: the mock absorbs the runner's scoring kwargs
    # (audio_path, content_type) so process() runs instead of erroring.
    monkeypatch.setattr("app.pipeline.runner.score.select", lambda words, limit=0, audio_path=None, content_type=None: [cand])
    monkeypatch.setattr("app.pipeline.runner.llm.rerank", lambda cands, wanted: [0])
    monkeypatch.setattr("app.pipeline.runner.llm.generate_metadata",
                        lambda text, lang=None: {"title": "t", "hashtags": []})
    monkeypatch.setattr("app.pipeline.runner.llm.build_caption", lambda meta, url: "cap")
    monkeypatch.setattr("app.pipeline.runner.download.fetch",
                        lambda url, w: SimpleNamespace(
                            video_path=tmp_path / "src.mp4",
                            audio_path=tmp_path / "a.wav",
                            title="T", channel="C", webpage_url=url))
    monkeypatch.setattr("app.pipeline.runner.download.as_hashtag", lambda c: "#c")
    monkeypatch.setattr("app.pipeline.runner.download.timestamped_url", lambda url, s: url)
    monkeypatch.setattr(_jobs, "get_job",
                        lambda _jid: {"id": jid, "url": "https://example.com/v",
                                      "status": "running", "stage": "",
                                      "options": json.dumps(options)})
    monkeypatch.setattr(_jobs, "update_job", lambda _jid, **kw: None)
    monkeypatch.setattr(_jobs, "add_clip", lambda _jid, rec: "clipid1")
    return jid, (orig_work, orig_outs)


def test_runner_keep_source_false_deletes(monkeypatch, tmp_path):
    from app.pipeline import runner

    jid, (ow, oo) = _mock_runner(monkeypatch, tmp_path, {"keep_source": False})
    try:
        runner.process(jid)
        assert not (settings.work_dir / jid).exists()
    finally:
        _restore_setting("work_dir", ow)
        _restore_setting("outputs_dir", oo)


def test_runner_keep_source_true_keeps(monkeypatch, tmp_path):
    from app.pipeline import runner

    jid, (ow, oo) = _mock_runner(monkeypatch, tmp_path, {"keep_source": True})
    try:
        runner.process(jid)
        assert (settings.work_dir / jid).exists()
    finally:
        _restore_setting("work_dir", ow)
        _restore_setting("outputs_dir", oo)


def test_runner_keep_source_missing_falls_back_to_settings(monkeypatch, tmp_path):
    from app.pipeline import runner

    # settings False -> deletes even when key missing
    orig_keep = _set_setting("keep_source", False)
    jid, (ow, oo) = _mock_runner(monkeypatch, tmp_path, {})
    try:
        runner.process(jid)
        assert not (settings.work_dir / jid).exists()
    finally:
        _restore_setting("work_dir", ow)
        _restore_setting("outputs_dir", oo)
        _restore_setting("keep_source", orig_keep)

    # settings True -> keeps when key missing
    orig_keep = _set_setting("keep_source", True)
    jid, (ow, oo) = _mock_runner(monkeypatch, tmp_path, {})
    try:
        runner.process(jid)
        assert (settings.work_dir / jid).exists()
    finally:
        _restore_setting("work_dir", ow)
        _restore_setting("outputs_dir", oo)
        _restore_setting("keep_source", orig_keep)
