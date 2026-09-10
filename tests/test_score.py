"""Tests for the clip selection heuristics — no GPU, no network, no ffmpeg."""

from app.pipeline.score import (
    build_sentences,
    generate_candidates,
    normalize,
    select,
    suppress_overlaps,
)
from app.pipeline.transcribe import Word


def make_words(spec: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(start=s, end=e, text=t) for t, s, e in spec]


def synth_transcript(seconds: int = 200) -> list[Word]:
    """A fake lecture: one 'sentence' per 5 seconds, 10 words each."""
    words: list[Word] = []
    t = 0.0
    phrases = [
        "لماذا يفشل أغلب الناس في تطبيق العادات الجديدة كل مرة.",
        "الفكرة الأساسية هي تقليل حجم العادة إلى أصغر خطوة ممكنة.",
        "التركيز يعتمد على البيئة أكثر من اعتماده على الإرادة نفسها.",
    ]
    i = 0
    while t < seconds:
        for tok in phrases[i % len(phrases)].split():
            words.append(Word(start=t, end=t + 0.45, text=tok))
            t += 0.5
        t += 0.8  # pause between sentences
        i += 1
    return words


def test_normalize_strips_diacritics_and_unifies_alef():
    assert normalize("الْأَمْر") == "الامر"
    assert normalize("كتابة،") == "كتابه"


def test_sentences_split_on_punctuation_and_pauses():
    words = make_words([
        ("مرحبا", 0.0, 0.4),
        ("بكم.", 0.5, 0.9),      # punctuation ends sentence
        ("اليوم", 1.0, 1.4),
        ("نتحدث", 1.5, 1.9),
        ("عنه", 5.0, 5.4),       # long gap ends sentence before it
    ])
    sentences = build_sentences(words, pause=0.65)
    assert len(sentences) == 3
    assert sentences[0].text == "مرحبا بكم."


def test_candidates_respect_duration_bounds():
    sentences = build_sentences(synth_transcript(200))
    cands = generate_candidates(sentences, min_sec=20, max_sec=60)
    assert cands, "expected at least one candidate"
    assert all(20 <= c.duration <= 60 for c in cands)
    assert cands[0].score >= cands[-1].score  # sorted best first


def test_candidates_start_and_end_on_sentence_boundaries():
    sentences = build_sentences(synth_transcript(150))
    starts = {round(s.start, 2) for s in sentences}
    ends = {round(s.end, 2) for s in sentences}
    for c in generate_candidates(sentences, min_sec=20, max_sec=60)[:20]:
        assert round(c.start, 2) in starts
        assert round(c.end, 2) in ends


def test_suppression_removes_overlapping_clips():
    picks = select(synth_transcript(300), limit=4)
    assert len(picks) <= 4
    for a, b in zip(picks, picks[1:]):
        overlap = min(a.end, b.end) - max(a.start, b.start)
        assert overlap <= 0.2 * min(a.duration, b.duration)


def test_hook_opener_scores_higher_than_filler():
    hook = build_sentences(make_words(
        [("لماذا", 0.0, 0.4), ("يفشل", 0.5, 0.9), ("الناس؟", 1.0, 1.4)]
    ))
    filler = build_sentences(make_words(
        [("و", 0.0, 0.4), ("يعني", 0.5, 0.9), ("شي", 1.0, 1.4)]
    ))
    from app.pipeline.score import score_candidate

    assert score_candidate(hook, {}).parts["hook"] > score_candidate(filler, {}).parts["hook"]


def test_empty_input_is_safe():
    assert select([]) == []
