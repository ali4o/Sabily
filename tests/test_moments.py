"""Moment engine (UNDERSTAND→SEGMENT→DISCOVER→RANK→REFINE).

Pure heuristics: no GPU, no network, no ffmpeg. Every test builds its own
words — never the user's DB or .env.
"""

import pytest

from app.pipeline import score
from app.pipeline.score import (
    CONTENT_PROFILES,
    build_sentences,
    classify_moment,
    detect_content_type,
    detect_turns,
    diversify,
    explain,
    normalize_content_type,
    segment_topics,
    select,
    trim_edges,
)
from app.pipeline.transcribe import Word


def w(spec, start=0.0, step=0.5):
    return [Word(start=start + i * step, end=start + i * step + 0.45, text=t)
            for i, t in enumerate(spec.split())]


def test_classify_moment_types():
    cases = {
        "لماذا يفشل أغلب الناس في الالتزام؟": "question",
        "قبل سنوات حدث معي موقف غريب في العمل.": "story",
        "السبب الرئيسي هو أنك تعتمد على الإرادة وحدها.": "insight",
        "احذر أن تضيع وقتك في إعادة القراءة.": "statement",
        "أثبتت دراسة أن 70% من الطلاب ينسون بسرعة.": "fact",
        "هناك ثلاث خطوات: قلل، ثم كرر، ثم كافئ نفسك.": "framework",
        "برأيي أكبر مشكلة في التعليم اليوم هي الحفظ.": "opinion",
        "خليني أوضح لك لماذا يحدث هذا بمثال بسيط.": "explanation",
        "الخلاصة إذن ابدأ بخطوة صغيرة اليوم.": "conclusion",
    }
    for text, kind in cases.items():
        got, conf = classify_moment(text)
        assert got == kind, f"{text!r} -> {got}"
        assert conf > 0
    assert classify_moment("اليوم نتحدث عن موضوع عام جدا.")[0] == "general"


def test_profiles_sum_to_one_and_aliases():
    for name, prof in CONTENT_PROFILES.items():
        assert abs(sum(prof.values()) - 1.0) < 1e-9, name
    assert normalize_content_type("Podcast") == "podcast"
    assert normalize_content_type("درس") == "general"  # latin aliases only
    assert normalize_content_type("bogus") == "general"
    assert normalize_content_type("") == "auto"
    assert normalize_content_type(None) == "auto"


def test_detect_content_type_shapes():
    # multi-speaker + stories -> podcast (long-form conversational)
    turns = [{"start": i * 100, "end": i * 100 + 90, "speaker": i % 2 + 1}
             for i in range(8)]
    sents = build_sentences(w("قبل سنوات حدث معي موقف مضحك جدا في السفر. " * 60))
    assert detect_content_type(sents, turns, duration=800.0) == "podcast"
    # multi-speaker + questions -> interview
    sents = build_sentences(w("لماذا حدث هذا؟ وكيف تعاملت معه؟ " * 60))
    assert detect_content_type(sents, turns, duration=800.0) == "interview"
    # long monologue -> lecture, short monologue -> lesson
    long_words = w("الفكرة الأساسية هي التركيز والانتباه الكامل. " * 220)
    assert detect_content_type(build_sentences(long_words), []) == "lecture"
    short_words = w("الفكرة الأساسية هي التركيز والانتباه الكامل. " * 10)
    assert detect_content_type(build_sentences(short_words), []) == "lesson"
    assert detect_content_type([]) == "general"


def test_segment_topics_splits_distinct_blocks():
    a = "النوم مهم جدا لصحة الدماغ والذاكرة والتركيز. "
    b = "الاستثمار في الأسهم يحتاج صبرا وتنويعا للمحفظة. "
    sents = build_sentences(w((a * 8 + b * 8).strip()))
    topics = segment_topics(sents)
    assert len(topics) >= 2
    assert all(t.keywords for t in topics)
    assert segment_topics(build_sentences(w("جملة واحدة فقط هنا."))) != []


def test_standalone_prefers_self_contained_openers():
    dangling = build_sentences(w("وهذا هو السبب في كل شيء يحدث."))
    question = build_sentences(w("لماذا ينسى الطلاب بهذه السرعة الكبيرة؟"))
    digit = build_sentences(w("70% من الطلاب ينسون خلال يوم واحد فقط."))
    assert score._standalone_score(question) > score._standalone_score(dangling)
    assert score._standalone_score(digit) >= 0.8


def test_trim_edges_keeps_meaning_and_bounds():
    sents = build_sentences(
        w("طيب يعني الفكرة الأساسية هي أن تبدأ بخطوة صغيرة جدا اليوم.",
          start=100.0))
    trimmed = trim_edges(sents, min_sec=3.0)
    assert trimmed[0].start > 100.0  # filler gone
    assert "الفكرة" in trimmed[0].text  # meaning kept
    assert trimmed[-1].end - trimmed[0].start >= 3.0
    # teaser closer dropped when the clip stays long enough
    two = (build_sentences(w("الفكرة الأساسية هي البدء بخطوة صغيرة جدا.", start=200.0))
           + build_sentences(w("تابعونا في الحلقة القادمة يا أصدقاء.", start=210.0)))
    kept = trim_edges(two, min_sec=3.0)
    assert len(kept) == 1 and "تابعونا" not in kept[0].text
    # clean input untouched
    clean = build_sentences(w("الفكرة الأساسية هي البدء بخطوة صغيرة جدا."))
    assert trim_edges(clean, min_sec=3.0)[0].text == clean[0].text


def _lecture_words():
    parts = [
        "أهلا بكم في محاضرة اليوم عن التعلم الفعال.",
        "طيب يعني لنبدأ بالمشكلة الأساسية التي تواجه الطلاب.",
        "لماذا ينسى الطلاب بهذه السرعة الكبيرة جدا؟",
        "السبب الرئيسي هو الاعتماد على إعادة القراءة وحدها.",
        "الحل هو التكرار المتباعد بخطوات صغيرة كل يوم.",
        "الخلاصة إذن ابدأ اليوم بخطوة واحدة صغيرة فقط.",
        "تابعونا في الحلقة القادمة يا أصدقاء الأعزاء.",
    ]
    words, t = [], 200.0
    for p in parts:
        for tok in p.split():
            words.append(Word(start=t, end=t + 0.4, text=tok))
            t += 0.5
        t += 1.2
    return words


def test_expansion_grows_core_with_setup_and_conclusion():
    cands = select(_lecture_words(), limit=6)
    assert cands
    # the refined winner should reach back to the question (setup) or
    # forward to the conclusion instead of sitting on one sentence
    top = cands[0]
    assert top.duration >= 8.0
    assert ("لماذا" in top.text) or ("الخلاصة" in top.text) or ("السبب" in top.text)


def test_expansion_respects_max_sec():
    from app.config import settings

    cands = select(_lecture_words(), limit=10)
    for c in cands:
        assert c.duration <= settings.max_clip_sec + 0.01


def test_select_attaches_moment_topic_reasons():
    cands = select(_lecture_words(), limit=4)
    assert cands
    for c in cands:
        assert c.moment in score.MOMENT_AR
        assert isinstance(c.topic, str)
        assert c.reasons and len(c.reasons) <= 5
        for key in ("value", "standalone", "hook", "clarity",
                    "completeness", "emotion", "novelty", "share"):
            assert key in c.parts


def test_select_content_type_override_changes_weights():
    words = _lecture_words()
    auto = select(words, limit=6)
    pod = select(words, limit=6, content_type="podcast")
    assert auto and pod
    # same engine, different ears: the order is allowed to differ and the
    # metric set stays identical
    assert {k for k in auto[0].parts} == {k for k in pod[0].parts}
    assert select([], limit=3) == []


def test_explain_mentions_moment_kind():
    c = score.Candidate(start=0, end=10, text="t", score=0.5,
                        parts={"standalone": 0.9, "hook": 0.8,
                               "completeness": 0.9, "value": 0.7,
                               "novelty": 0.8, "emotion": 0.1},
                        moment="story")
    reasons = explain(c)
    assert any("قصة" in r for r in reasons)
    assert any("دون" in r for r in reasons)


def test_diversify_drops_same_topic_neighbours():
    def cand(start, topic, text):
        return score.Candidate(start=start, end=start + 30, text=text,
                               score=0.9, topic=topic)

    a = cand(10.0, "نوم,صحة", "النوم مهم لصحة الدماغ والذاكرة والتركيز اليوم")
    b = cand(50.0, "نوم,صحة", "النوم مهم جدا لصحة الدماغ والذاكرة والتركيز")
    c = cand(400.0, "نوم,صحة", "كلام مختلف تماما عن الاستثمار والصبر والمال")
    far = cand(500.0, "نوم,صحة", "النوم مهم لصحة الدماغ والذاكرة والتركيز اليوم")
    picked = diversify([a, b, c, far], 3)
    assert a in picked and c in picked and b not in picked
    # same topic far apart in time is fine (not a neighbour twin)
    d = cand(1000.0, "نوم,صحة", "الاستثمار يحتاج صبرا وتنويعا وحكمة كبيرة جدا")
    assert d in diversify([a, c, d], 3)


def test_classify_content_never_raises_and_falls_back(monkeypatch):
    from app.config import settings
    from app.pipeline import llm

    orig = settings.llm_provider
    object.__setattr__(settings, "llm_provider", "none")
    try:
        assert llm.classify_content([])["content_type"] == "lecture"
        info = llm.classify_content(_lecture_words())
        assert info["content_type"] in CONTENT_PROFILES
        assert set(info) == {"content_type", "audience", "goal"}
    finally:
        object.__setattr__(settings, "llm_provider", orig)


def test_api_threads_content_type_and_ignores_garbage(monkeypatch):
    from app import jobs
    from app.main import app  # noqa: F401 (ensures routes import cleanly)

    seen: dict = {}
    orig_create, orig_enqueue = jobs.create_job, jobs.enqueue
    jobs.create_job = lambda url, options=None: (seen.update(options or {}), "abcdef012345")[1]
    jobs.enqueue = lambda jid: None
    try:
        from fastapi.testclient import TestClient

        c = TestClient(app, raise_server_exceptions=False)
        r = c.post("/api/jobs", json={"url": "https://example.com/v",
                                      "content_type": "podcast"})
        assert r.status_code == 200 and seen.get("content_type") == "podcast"
        seen.clear()
        r = c.post("/api/jobs", json={"url": "https://example.com/v",
                                      "content_type": "8K-TV"})
        assert r.status_code == 200 and "content_type" not in seen
    finally:
        jobs.create_job, jobs.enqueue = orig_create, orig_enqueue
