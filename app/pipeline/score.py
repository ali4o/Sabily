"""Stage 3 — pick the clip candidates using heuristics only.

Pure Python, no models, no GPU. This is what keeps the local machine free:
the LLM never sees the whole transcript, only the top candidates this file
produces.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from app.config import settings
from app.pipeline.transcribe import Word

# Sentence enders in Arabic and Latin text.
END_PUNCT = ".؟!?،…"
HARD_END = ".؟!?…"

# Openers that make a strong first line for a short.
HOOKS = {
    "لماذا", "كيف", "ليش", "وش", "ما هو", "ما هي", "هل", "متى", "أين",
    "السر", "السبب", "الخطأ", "أهم", "أكبر", "أول", "تخيل", "انتبه",
    "الحقيقة", "المشكلة", "الفكرة", "خلاصة", "باختصار", "تذكر",
}

STOPWORDS = {
    "في", "من", "على", "إلى", "عن", "هذا", "هذه", "ذلك", "التي", "الذي",
    "أن", "إن", "كان", "كانت", "يكون", "ما", "لا", "و", "أو", "ثم", "قد",
    "كل", "بعض", "هو", "هي", "هم", "نحن", "أنا", "أنت", "يعني", "شي",
    "the", "a", "an", "is", "are", "of", "to", "and", "in", "that", "it",
}

# \w already covers Arabic letters in Python 3; the Arabic block also holds
# punctuation (،؛؟) and tatweel, so those must be stripped explicitly.
_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def normalize_token(token: str) -> str:
    t = _DIACRITICS.sub("", token)
    t = _NON_WORD.sub("", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه")
    return t.strip().lower()


# Promo talk is fluent and keyword-dense, so the old scorer liked it. It is
# also the fastest way to make a clip look stolen. Penalise it explicitly.
PROMO = {
    "اشتراك", "الاشتراك", "اشتركوا", "لايك", "اللايك", "الجرس", "تعليق",
    "تعليقاتكم", "قناتي", "القناة", "رابط", "الوصف", "الفيديو_السابق",
    "الحلقة_القادمة", "راعي", "الرعاة", "برعاية", "خصم", "كوبون",
}


@dataclass
class Sentence:
    start: float
    end: float
    words: list[Word]
    gap: float = 0.0          # silence after this sentence

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    score: float
    parts: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


def build_sentences(words: list[Word], pause: float = 0.65) -> list[Sentence]:
    """Group words into sentences using punctuation and silence gaps."""
    sentences: list[Sentence] = []
    buf: list[Word] = []
    for i, w in enumerate(words):
        buf.append(w)
        ends_punct = any(w.text.endswith(p) for p in END_PUNCT)
        gap = (words[i + 1].start - w.end) if i + 1 < len(words) else math.inf
        if ends_punct or gap >= pause:
            sentences.append(Sentence(buf[0].start, buf[-1].end, buf,
                                      gap=0.0 if gap is math.inf else gap))
            buf = []
    if buf:
        sentences.append(Sentence(buf[0].start, buf[-1].end, buf, gap=1.0))
    return sentences


def keyword_weights(sentences: list[Sentence]) -> dict[str, float]:
    """Cheap TF weighting over the whole transcript, stopwords removed."""
    counts: Counter[str] = Counter()
    for s in sentences:
        for w in s.words:
            n = normalize_token(w.text)
            if len(n) > 2 and n not in STOPWORDS:
                counts[n] += 1
    if not counts:
        return {}
    top = counts.most_common(200)
    peak = top[0][1]
    return {k: v / peak for k, v in top}


def _hook_score(text: str) -> float:
    head = " ".join(text.split()[:6])
    n_head = " ".join(normalize_token(t) for t in head.split())
    hits = sum(1 for h in HOOKS if normalize_token(h.replace(" ", "")) in n_head.replace(" ", ""))
    bonus = 0.3 if any(ch.isdigit() for ch in head) else 0.0
    return min(1.0, hits * 0.5 + bonus)


def _completeness(sentences: list[Sentence]) -> float:
    """Colloquial speech carries no punctuation, so silence is the real signal.

    A long pause after the last word means the speaker finished a thought; a
    short one means we would be cutting mid-sentence.
    """
    last = sentences[-1]
    if any(last.text.endswith(p) for p in HARD_END):
        return 1.0
    if last.gap >= 0.9:
        return 0.9
    if last.gap >= 0.55:
        return 0.7
    return 0.35


def _promo_penalty(sentences: list[Sentence]) -> float:
    """0 = clean, 1 = the clip is mostly a call to subscribe."""
    hits = total = 0
    for s in sentences:
        for w in s.words:
            total += 1
            if normalize_token(w.text) in {normalize_token(p) for p in PROMO}:
                hits += 1
    if not total:
        return 0.0
    return min(1.0, (hits / total) * 12)


def _position_penalty(start: float, intro_sec: float) -> float:
    """Openings are greetings and housekeeping, not the good part."""
    if start >= intro_sec:
        return 0.0
    return 1.0 - (start / intro_sec)


def _confidence(sentences: list[Sentence]) -> float:
    vals = [w.conf for s in sentences for w in s.words if w.conf is not None]
    return sum(vals) / len(vals) if vals else 1.0


def _density(text: str, duration: float) -> float:
    """Words per second, normalised. Too slow = dead air, too fast = rushed."""
    if duration <= 0:
        return 0.0
    wps = len(text.split()) / duration
    return max(0.0, 1.0 - abs(wps - 2.6) / 2.6)


def _keyword_score(sentences: list[Sentence], weights: dict[str, float]) -> float:
    vals = []
    for s in sentences:
        for w in s.words:
            n = normalize_token(w.text)
            if n in weights:
                vals.append(weights[n])
    if not vals:
        return 0.0
    vals.sort(reverse=True)
    top = vals[: max(5, len(vals) // 10)]
    return min(1.0, sum(top) / len(top))


def score_candidate(group: list[Sentence], weights: dict[str, float]) -> Candidate:
    start, end = group[0].start, group[-1].end
    text = " ".join(s.text for s in group)
    parts = {
        "hook": _hook_score(group[0].text),
        "keywords": _keyword_score(group, weights),
        "complete": _completeness(group),
        "density": _density(text, end - start),
        "confidence": _confidence(group),
        "promo": _promo_penalty(group),
        "intro": _position_penalty(start, settings.intro_sec),
    }
    score = (
        0.26 * parts["hook"]
        + 0.24 * parts["keywords"]
        + 0.24 * parts["complete"]
        + 0.13 * parts["density"]
        + 0.13 * parts["confidence"]
        - 0.45 * parts["promo"]
        - 0.30 * parts["intro"]
    )
    return Candidate(start=start, end=end, text=text,
                     score=round(max(0.0, score), 4), parts=parts)


def generate_candidates(
    sentences: list[Sentence],
    min_sec: float | None = None,
    max_sec: float | None = None,
) -> list[Candidate]:
    """Sliding window over sentence boundaries — cuts never land mid-word."""
    min_sec = min_sec or settings.min_clip_sec
    max_sec = max_sec or settings.max_clip_sec
    weights = keyword_weights(sentences)
    out: list[Candidate] = []
    for i in range(len(sentences)):
        group: list[Sentence] = []
        for j in range(i, len(sentences)):
            group.append(sentences[j])
            dur = group[-1].end - group[0].start
            if dur < min_sec:
                continue
            if dur > max_sec:
                break
            out.append(score_candidate(group, weights))
    return sorted(out, key=lambda c: c.score, reverse=True)


def suppress_overlaps(cands: list[Candidate], limit: int, max_overlap: float = 0.2) -> list[Candidate]:
    """Greedy non-max suppression so clips don't repeat the same moment."""
    kept: list[Candidate] = []
    for c in cands:
        clash = False
        for k in kept:
            inter = min(c.end, k.end) - max(c.start, k.start)
            if inter > 0 and inter / min(c.duration, k.duration) > max_overlap:
                clash = True
                break
        if not clash:
            kept.append(c)
        if len(kept) >= limit:
            break
    return kept


def select(words: list[Word], limit: int | None = None) -> list[Candidate]:
    """Entry point: words in, ranked non-overlapping candidates out."""
    limit = limit or settings.llm_candidates
    sentences = build_sentences(words)
    if not sentences:
        return []
    return suppress_overlaps(generate_candidates(sentences), limit)


# kept for callers and tests that imported the old name
normalize = normalize_token
