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


# --------------------------------------------------------------------------- #
# Moment engine: UNDERSTAND → SEGMENT → DISCOVER → RANK → REFINE.
#
# Lectures, lessons, podcasts and interviews are not chosen for a "visual
# shot" but for valuable, self-contained moments: an insight, a story, a
# question answered. Everything below is pure heuristics (no models, no
# GPU) so selection keeps working with LLM_PROVIDER=none; the LLM layer
# in llm.py only refines what this engine discovers.
# --------------------------------------------------------------------------- #

# Cues are matched against normalize_token() output (alef unified, teh
# marbuta folded, diacritics stripped), so write them pre-normalised.
MOMENT_CUES: dict[str, set[str]] = {
    "insight": {"السبب", "السر", "الحقيقه", "المشكله", "الخطا", "الغلطه",
                "اهم", "اكبر", "الحل", "الخلل", "اعرف", "انتبه"},
    "explanation": {"لان", "بسبب", "يعني", "مثال", "مثلا", "عشان",
                    "لذلك", "لانه", "بمعني", "يوضح", "اشرح", "خليني"},
    "statement": {"احذر", "اياك", "ابدا", "مستحيل", "يجب", "لازم",
                  "تضيع", "غلط", "خطير", "توقف", "لاتفعل"},
    "story": {"تخيل", "قصه", "مره", "حدث", "اتذكر", "كنت", "زمان",
              "موقف", "يحكي", "سنوات"},
    "question": {"لماذا", "كيف", "هل", "متي", "اين", "ماهو", "ماهي",
                 "ليش", "وش", "ماذا", "سوال"},
    "fact": {"دراسه", "ابحاث", "احصائيه", "مليون", "الغريب",
             "المثير", "اكتشف", "اثبتت", "رقم", "بالارقام"},
    "framework": {"خطوات", "خطوه", "اولا", "ثانيا", "ثالثا", "قواعد",
                  "طريقه", "نظام", "مراحل", "اطار", "نموذج"},
    "opinion": {"برايي", "اعتقد", "اظن", "وجهه", "نظري", "اري"},
    "emotion": {"مضحك", "مؤثر", "بكيت", "ضحكت", "لايصدق", "رائع",
                "مذهل", "صدمه", "مخيف", "مدهش"},
    "conclusion": {"الخلاصه", "النهايه", "النتيجه", "باختصار", "المهم",
                   "الزبده", "اذا", "الخلاصه"},
}

MOMENT_AR = {
    "insight": "فكرة محورية", "explanation": "شرح", "statement": "قول قوي",
    "story": "قصة", "question": "سؤال وجواب", "fact": "معلومة مدهشة",
    "framework": "منهجية", "opinion": "رأي", "emotion": "لحظة مؤثرة",
    "conclusion": "خلاصة", "general": "لحظة مختارة",
}

# Per-content-type weights over the 8 clip metrics. They always sum to 1:
# a podcast lives on hook and emotion, a lecture on value and completeness.
CONTENT_PROFILES: dict[str, dict[str, float]] = {
    "general": {"value": 0.20, "standalone": 0.15, "hook": 0.15,
                "clarity": 0.12, "completeness": 0.14, "emotion": 0.07,
                "novelty": 0.07, "share": 0.10},
    "lecture": {"value": 0.22, "standalone": 0.18, "hook": 0.10,
                "clarity": 0.16, "completeness": 0.14, "emotion": 0.05,
                "novelty": 0.10, "share": 0.05},
    "lesson": {"value": 0.20, "standalone": 0.20, "hook": 0.12,
               "clarity": 0.16, "completeness": 0.14, "emotion": 0.05,
               "novelty": 0.08, "share": 0.05},
    "podcast": {"value": 0.12, "standalone": 0.15, "hook": 0.20,
                "clarity": 0.10, "completeness": 0.08, "emotion": 0.15,
                "novelty": 0.08, "share": 0.12},
    "interview": {"value": 0.14, "standalone": 0.16, "hook": 0.18,
                  "clarity": 0.12, "completeness": 0.10, "emotion": 0.12,
                  "novelty": 0.08, "share": 0.10},
}

CONTENT_ALIASES = {
    "lecture": "lecture", "course": "lecture", "class": "lecture",
    "lesson": "lesson", "tutorial": "lesson", "howto": "lesson",
    "podcast": "podcast", "show": "podcast",
    "interview": "interview", "qna": "interview",
    "auto": "auto", "": "auto",
}


def normalize_content_type(v: str | None) -> str:
    """User input -> lecture|lesson|podcast|interview|general|auto."""
    if not v:
        return "auto"
    return CONTENT_ALIASES.get(str(v).strip().lower(), "general")


def _content_tokens(text: str) -> list[str]:
    toks = [normalize_token(t) for t in text.split()]
    return [t for t in toks if len(t) > 2 and t not in STOPWORDS]


def _tok_hits(text: str, cues: set[str]) -> int:
    """Token-level cue hits. Substring matching on spaceless text caused
    false positives ("ماذا" inside "لماذا"), so cues only match whole
    normalised tokens; %, ؟ and ! are read from the raw text instead
    (normalisation strips them)."""
    toks = set(normalize_token(t) for t in text.split())
    return sum(1 for c in cues if c in toks)


def moment_scores(text: str) -> dict[str, float]:
    """How strongly the text reads as each moment type (0..1, heuristic)."""
    scores = {kind: min(1.0, _tok_hits(text, cues) / 2.0)
              for kind, cues in MOMENT_CUES.items()}
    if "%" in text or "بالمئه" in "".join(normalize_token(t) for t in text.split()):
        scores["fact"] = min(1.0, scores["fact"] + 0.5)
    if "؟" in text or "?" in text:
        scores["question"] = min(1.0, scores["question"] + 0.5)
    if "!" in text:
        scores["emotion"] = min(1.0, scores["emotion"] + 0.5)
    return scores


def classify_moment(text: str) -> tuple[str, float]:
    """Best moment type + confidence.

    An explicit framing at the head ("برأيي", "الخلاصة", "لماذا") wins
    ties: how the speaker opens the thought is the strongest signal of
    what kind of moment it is.
    """
    order = ("question", "story", "statement", "fact", "insight",
             "framework", "explanation", "opinion", "emotion", "conclusion")
    scores = moment_scores(text)
    toks = [normalize_token(t) for t in text.split()[:3]]
    first, rest = (toks[0] if toks else ""), set(toks[1:])
    for kind in order:
        cues = MOMENT_CUES[kind]
        if first in cues:
            scores[kind] = min(1.0, scores[kind] + 0.6)
        elif rest & cues:
            scores[kind] = min(1.0, scores[kind] + 0.3)
    best, conf = "general", 0.0
    for kind in order:
        if scores[kind] > conf:
            best, conf = kind, scores[kind]
    return best, round(conf, 3)


@dataclass
class Topic:
    idx: int
    first: int       # first sentence index
    last: int        # last sentence index (inclusive)
    keywords: str    # "kw1,kw2,kw3" label for the UI and dedup


def segment_topics(sentences: list["Sentence"], min_len: int = 4,
                   threshold: float = 0.22) -> list[Topic]:
    """SEGMENT: split the transcript into topic blocks.

    Greedy vocabulary walk: a sentence joins the current topic while it
    shares enough content words with it, otherwise it opens a new one.
    No embeddings, no models — Jaccard over normalised tokens is enough
    to keep one idea's sentences together for the expansion step.
    """
    if not sentences:
        return []
    vocabs = [set(_content_tokens(s.text)) for s in sentences]
    bounds = [0]
    cur: set[str] = set(vocabs[0])
    for i in range(1, len(sentences)):
        v = vocabs[i]
        union = cur | v
        sim = len(cur & v) / len(union) if union else 0.0
        if sim < threshold and i - bounds[-1] >= min_len:
            bounds.append(i)
            cur = set(v)
        else:
            cur |= v
    bounds.append(len(sentences))
    topics = []
    for t, (a, b) in enumerate(zip(bounds, bounds[1:])):
        counts: Counter[str] = Counter()
        for v in vocabs[a:b]:
            counts.update(v)
        kw = ",".join(k for k, _ in counts.most_common(3)) or f"مقطع {t + 1}"
        topics.append(Topic(idx=t, first=a, last=b - 1, keywords=kw))
    return topics


def topic_of(topics: list[Topic], sent_idx: int) -> Topic | None:
    for t in topics:
        if t.first <= sent_idx <= t.last:
            return t
    return topics[-1] if topics else None


def detect_content_type(sentences: list["Sentence"], turns: list[dict] | None = None,
                        duration: float = 0.0) -> str:
    """UNDERSTAND: lecture, lesson, podcast or interview — heuristic.

    Multi-speaker + stories/emotion reads as podcast, multi-speaker +
    questions as interview, long monologues as lectures, the rest as
    lessons. Wrong guesses only shift metric weights, never drop clips.
    """
    if not sentences:
        return "general"
    n = len(sentences)
    q = sum(1 for s in sentences if "؟" in s.text or "?" in s.text
            or classify_moment(s.text)[0] == "question") / n
    story = sum(1 for s in sentences if classify_moment(s.text)[0] == "story") / n
    emo = sum(1 for s in sentences if classify_moment(s.text)[0] == "emotion") / n
    n_turns = len(turns or [])
    dur = duration or (sentences[-1].end - sentences[0].start)
    if dur >= 300 and n_turns >= 6 and (story > 0.06 or emo > 0.06 or q > 0.15):
        # turns alone prove nothing (lecturers pause too) — conversational
        # content markers must agree before leaving lecture/lesson land,
        # and short videos are always lessons (standalone matters most).
        return "podcast" if (story > 0.06 or emo > 0.06) else "interview"
    if dur >= 1500 or n >= 200:
        return "lecture"
    if q > 0.25 and n_turns >= 2:
        return "interview"
    return "lesson" if dur < 900 else "lecture"


# A clip that only makes sense inside the lecture is a bad Short, even
# with a brilliant core sentence. Openings that point outside the clip
# ("وهذا هو السبب") score low; questions and named hooks score high.
DANGLING_START = {
    "وهذا", "وهذه", "وهو", "وهي", "وهم", "ذلك", "تلك", "لهذا",
    "كما", "التي", "الذي", "الذين", "بعدها", "عشان", "كذا",
}
REFERENCE_HINT = {"قلت", "ذكرت", "تكلمنا", "سابقا", "قليل", "الحلقه", "تذكروا"}

# Leading fillers trimmed from a clip start (never from the middle — the
# meaning must not change, only "طيب... مثل ما قلت" throat-clearing goes).
FILLER_OPEN = {
    "طيب", "يعني", "ااا", "امم", "اها", "هاه", "اوكي", "حسنا",
    "تمام", "اقول", "هلا", "اهلا",
}
# A closing sentence made only of teaser/promo is cut, not kept.
TEASER_END = {"لاحقا", "القادم", "القادمه", "تابعونا", "اشتركوا", "الحلقه", "الجرس"}


def _standalone_score(group: list["Sentence"]) -> float:
    """Would a stranger understand this clip without the lecture? 0..1."""
    head = group[0].text.split()
    first = normalize_token(head[0]) if head else ""
    norm_all = {normalize_token(t) for t in group[0].text.split()}
    if first in DANGLING_START or len(norm_all & DANGLING_START) >= 2:
        # may still work if the antecedent sits inside the same clip
        body = " ".join(s.text for s in group[1:])
        if len(body.split()) >= 12:
            return 0.55
        return 0.3
    if any(ch.isdigit() for ch in group[0].text):
        return 0.9
    if _hook_score(group[0].text) >= 0.5 or "؟" in group[0].text or "?" in group[0].text:
        return 0.85
    if _tok_hits(" ".join(s.text for s in group), REFERENCE_HINT):
        return 0.55
    return 0.7 if len(group) > 1 else 0.6


def _emotion_score(text: str) -> float:
    cues = MOMENT_CUES["emotion"] | MOMENT_CUES["story"] | MOMENT_CUES["opinion"]
    hits = _tok_hits(text, cues)
    marks = text.count("!") + text.count("؟") + text.count("?")
    return min(1.0, hits * 0.4 + min(0.4, marks * 0.2))


def _novelty_score(group: list["Sentence"], weights: dict[str, float]) -> float:
    """Rare words + numbers read as new information, not filler."""
    toks = _content_tokens(" ".join(s.text for s in group))
    if not toks:
        return 0.0
    rare = sum(1 for t in toks if weights.get(t, 0.0) < 0.3)
    digits = any(ch.isdigit() for s in group for ch in s.text)
    return min(1.0, rare / len(toks) + (0.25 if digits else 0.0))


def _share_score(moment: str, hook: float, emotion: float) -> float:
    boost = 1.0 if moment in ("story", "statement", "question", "fact") else 0.3
    return min(1.0, 0.4 * hook + 0.3 * emotion + 0.3 * boost)


def trim_edges(group: list["Sentence"], min_sec: float) -> list["Sentence"]:
    """REFINE (edges): drop leading filler words and trailing teasers.

    Only the clip borders move, and only onto word starts — the spoken
    meaning is never edited, "طيب يعني" just stops opening the Short and
    "تابعونا في الحلقة القادمة" stops closing it.
    """
    if not group:
        return group
    # leading fillers: at most the first sentence's head, ≤2.5s, and the
    # clip must stay within [min_sec, max_sec] — bounds are a contract.
    first = group[0]
    toks = first.words
    cut = 0
    total_dur = group[-1].end - group[0].start
    while (cut < len(toks) - 1
           and normalize_token(toks[cut].text) in FILLER_OPEN
           and toks[cut].end - first.start < 2.5
           and total_dur - (toks[cut].end - first.start) >= min_sec):
        cut += 1
    if cut:
        first = Sentence(toks[cut].start, first.end, toks[cut:], gap=first.gap)
        group = [first, *group[1:]]
    # trailing teaser sentence: dropped only if the clip stays long enough
    if len(group) >= 2:
        tail = {normalize_token(t) for t in group[-1].text.split()}
        if tail & TEASER_END:
            kept = group[:-1]
            if kept[-1].end - kept[0].start >= min_sec:
                group = kept
    return group


def _is_setup(s: "Sentence") -> bool:
    head = {normalize_token(t) for t in s.text.split()[:8]}
    return ("؟" in s.text or "?" in s.text
            or _hook_score(" ".join(s.text.split()[:8])) >= 0.5
            or bool(head & {"المشكله", "تخيل", "السوال", "قبل", "قصه"}))


def _is_conclusion(s: "Sentence") -> bool:
    toks = {normalize_token(t) for t in s.text.split()}
    cues = MOMENT_CUES["conclusion"] | {"لذلك", "النتيجه"}
    return any(s.text.endswith(p) for p in HARD_END) and (
        bool(toks & cues) or s.gap >= 0.9)


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
    # moment engine (Stage 3b): what kind of moment this is, which topic
    # it belongs to, and why it was picked (Arabic strings for the UI).
    moment: str = "general"
    topic: str = ""
    reasons: list[str] = field(default_factory=list)

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
    short one means we would be cutting mid-sentence. A terminal punctuation
    mark is still required for a perfect score — without one the thought may
    continue after the pause.
    """
    last = sentences[-1]
    punctuated = any(last.text.endswith(p) for p in HARD_END)
    if punctuated:
        return 1.0
    if last.gap >= 0.9:
        return 0.85
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


def _audio_score(stats: dict[str, float] | None) -> float:
    """1.0 = clean speech. Demotes dead air and clipped shouting.

    None (no audio provided) is neutral 1.0 so text-only callers and
    old tests behave exactly as before.
    """
    if not stats:
        return 1.0
    return max(0.0, min(1.0,
                        1.0 - min(0.6, stats.get("silence", 0.0) * 0.8)
                        - stats.get("clipped", 0.0) * 0.5))


def score_candidate(group: list[Sentence], weights: dict[str, float],
                    audio: dict[str, float] | None = None,
                    profile: str = "general", topic: str = "") -> Candidate:
    """RANK one window of sentences.

    Eight content metrics (weighted per content type) minus the acoustic
    and spam penalties. Legacy part keys (hook/keywords/complete/density/
    confidence/audio/promo/intro) are kept so old callers and tests keep
    working; value/clarity/completeness alias them for the new scheme.
    """
    text = " ".join(s.text for s in group)
    start, end = group[0].start, group[-1].end
    moment, mconf = classify_moment(text)
    hook = _hook_score(group[0].text)
    keywords = _keyword_score(group, weights)
    complete = _completeness(group)
    density = _density(text, end - start)
    confidence = _confidence(group)
    standalone = _standalone_score(group)
    emotion = _emotion_score(text)
    novelty = _novelty_score(group, weights)
    share = _share_score(moment, hook, emotion)
    clarity = 0.7 * density + 0.3 * confidence
    parts = {
        "hook": hook,
        "keywords": keywords,
        "value": keywords,
        "complete": complete,
        "completeness": complete,
        "density": density,
        "confidence": confidence,
        "clarity": round(clarity, 4),
        "standalone": standalone,
        "emotion": emotion,
        "novelty": novelty,
        "share": share,
        "moment": mconf,
        "audio": _audio_score(audio),
        "promo": _promo_penalty(group),
        "intro": _position_penalty(start, settings.intro_sec),
    }
    w = CONTENT_PROFILES.get(profile) or CONTENT_PROFILES["general"]
    score = (
        w["value"] * parts["value"]
        + w["standalone"] * parts["standalone"]
        + w["hook"] * parts["hook"]
        + w["clarity"] * parts["clarity"]
        + w["completeness"] * parts["completeness"]
        + w["emotion"] * parts["emotion"]
        + w["novelty"] * parts["novelty"]
        + w["share"] * parts["share"]
        # NOTE: the audio term is a pure penalty — 0 when the audio is clean
        # or absent — so every score computed without audio is bit-identical
        # to previous versions.
        - 0.35 * (1.0 - parts["audio"])
        - 0.45 * parts["promo"]
        - 0.30 * parts["intro"]
    )
    return Candidate(start=start, end=end, text=text,
                     score=round(max(0.0, score), 4), parts=parts,
                     moment=moment, topic=topic)


def generate_candidates(
    sentences: list[Sentence],
    min_sec: float | None = None,
    max_sec: float | None = None,
    audio_path: "str | Path | None" = None,
    profile: str = "general",
    topics: list[Topic] | None = None,
) -> list[Candidate]:
    """DISCOVER: sliding window over sentence boundaries.

    Cuts never land mid-word. Every window is scored, then REFINE expands
    strong cores with their setup/conclusion and trims filler edges, so a
    15s peak can grow into the 35s self-contained clip around it.
    """
    min_sec = min_sec or settings.min_clip_sec
    max_sec = max_sec or settings.max_clip_sec
    weights = keyword_weights(sentences)
    audio = None
    if audio_path:
        try:
            from app.pipeline import audio_q

            audio = audio_q.load(audio_path)
        except Exception:  # noqa: BLE001 - audio probe must not fail scoring
            audio = None

    def _stats(a: int, b: int) -> dict | None:
        if not audio:
            return None
        from app.pipeline import audio_q

        return audio_q.stats_from(audio, sentences[a].start, sentences[b].end)

    base: list[tuple[int, int, Candidate]] = []
    for i in range(len(sentences)):
        group: list[Sentence] = []
        for j in range(i, len(sentences)):
            group.append(sentences[j])
            dur = group[-1].end - group[0].start
            if dur < min_sec:
                continue
            if dur > max_sec:
                break
            t = topic_of(topics or [], i)
            base.append((i, j, score_candidate(
                list(group), weights, _stats(i, j), profile,
                t.keywords if t else "")))
    refined = _refine_candidates(base, sentences, weights, audio, profile,
                                 topics or [], min_sec, max_sec)
    return sorted(refined, key=lambda c: c.score, reverse=True)


def _refine_candidates(base: list[tuple[int, int, Candidate]],
                       sentences: list[Sentence], weights: dict[str, float],
                       audio: object, profile: str, topics: list[Topic],
                       min_sec: float, max_sec: float,
                       top_n: int = 200) -> list[Candidate]:
    """REFINE: context expansion + edge trims, rescored.

    Only the top windows pay for variants (bounded work): each may grow
    one setup sentence backwards and one conclusion forwards, and loses
    filler openers / teaser closers. The best-scoring variant wins, so
    the clip starts on its hook and ends closed — start/end stay on word
    (never mid-word) boundaries.
    """
    from app.pipeline import audio_q

    def _stats(a: int, b: int) -> dict | None:
        if audio is None:
            return None
        try:
            return audio_q.stats_from(audio, sentences[a].start, sentences[b].end)
        except Exception:  # noqa: BLE001
            return None

    order = sorted(base, key=lambda t: t[2].score, reverse=True)
    rest = [c for _, _, c in order[top_n:]]
    out: list[Candidate] = list(rest)
    for i, j, cand in order[:top_n]:
        variants = [(i, j)]
        if i > 0 and _is_setup(sentences[i - 1]) \
                and sentences[j].end - sentences[i - 1].start <= max_sec:
            variants.append((i - 1, j))
        if j + 1 < len(sentences) and _is_conclusion(sentences[j + 1]) \
                and sentences[j + 1].end - sentences[i].start <= max_sec:
            variants.append((i, j + 1))
        if (i - 1, j + 1) not in variants and i > 0 and j + 1 < len(sentences) \
                and _is_setup(sentences[i - 1]) \
                and _is_conclusion(sentences[j + 1]) \
                and sentences[j + 1].end - sentences[i - 1].start <= max_sec:
            variants.append((i - 1, j + 1))
        best = cand
        for a, b in variants:
            grp = trim_edges(sentences[a:b + 1], min_sec)
            if not grp:
                continue
            t = topic_of(topics, a)
            v = score_candidate(grp, weights, _stats(a, b), profile,
                                t.keywords if t else "")
            if v.score > best.score:
                best = v
        out.append(best)
    return out


def explain(candidate: Candidate) -> list[str]:
    """Why this clip was picked — short Arabic reasons for the dashboard."""
    p = candidate.parts
    reasons = [f"نوع اللحظة: {MOMENT_AR.get(candidate.moment, 'لحظة مختارة')}"]
    if p.get("standalone", 0) >= 0.8:
        reasons.append("يمكن فهمه دون مشاهدة المحاضرة")
    if p.get("hook", 0) >= 0.7:
        reasons.append("يبدأ بخطاف واضح")
    if p.get("completeness", 0) >= 0.8:
        reasons.append("فكرة مكتملة بنهاية مغلقة")
    if p.get("value", 0) >= 0.6:
        reasons.append("معلومة قيّمة ومحورية")
    if p.get("novelty", 0) >= 0.7:
        reasons.append("معلومة جديدة غير متكررة")
    if p.get("emotion", 0) >= 0.6:
        reasons.append("لحظة مؤثرة تجذب المشاهد")
    return reasons[:5]


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


def select(words: list[Word], limit: int | None = None,
           audio_path: "str | Path | None" = None,
           content_type: str | None = None) -> list[Candidate]:
    """Entry point: words in, ranked non-overlapping candidates out.

    Full moment pipeline: UNDERSTAND (content type) → SEGMENT (topics) →
    DISCOVER (windows) → RANK (weighted metrics) → REFINE (expansion +
    edge trims). `content_type` overrides auto-detection; "auto"/None
    detects from the transcript itself.
    """
    limit = limit or settings.llm_candidates
    sentences = build_sentences(words)
    if not sentences:
        return []
    ctype = normalize_content_type(content_type)
    if ctype == "auto":
        ctype = detect_content_type(sentences, detect_turns(words),
                                    sentences[-1].end - sentences[0].start)
    if ctype not in CONTENT_PROFILES:
        ctype = "general"
    topics = segment_topics(sentences)
    cands = generate_candidates(sentences, audio_path=audio_path,
                                profile=ctype, topics=topics)
    picked = suppress_overlaps(cands, limit)
    for c in picked:
        c.reasons = explain(c)
    return picked


def _text_sim(a: str, b: str) -> float:
    """0..1 token-overlap similarity (cheap Jaccard, no new deps)."""
    import difflib

    if not a or not b:
        return 0.0
    sa = set(normalize_token(t) for t in a.split())
    sb = set(normalize_token(t) for t in b.split())
    sa.discard("")
    sb.discard("")
    if not sa or not sb:
        return 0.0
    if sa & sb:
        inter = len(sa & sb)
        return inter / max(len(sa), len(sb))
    return difflib.SequenceMatcher(None, a, b).ratio() * 0.5


def diversify(cands: list[Candidate], keep: int,
              min_gap: float = 25.0, max_sim: float = 0.65) -> list[Candidate]:
    """Greedy variety pass over an already-ranked list.

    Skips a candidate that starts within `min_gap` seconds of an accepted
    one, whose text is more than `max_sim` similar, or that shares a topic
    with a nearby accepted one (semantic dedup: one clip per idea) — then
    fills back up from the rest, so a video never yields adjacent twins.
    """
    picked: list[Candidate] = []
    for c in cands:
        if len(picked) >= keep:
            break
        clash = False
        for k in picked:
            if abs(c.start - k.start) < min_gap:
                clash = True
                break
            if _text_sim(c.text, k.text) > max_sim:
                clash = True
                break
            if (c.topic and c.topic == k.topic
                    and abs(c.start - k.start) < 120.0):
                clash = True
                break
        if not clash:
            picked.append(c)
    return picked


def detect_turns(words: list[Word], turn_sec: float | None = None) -> list[dict]:
    """Split words into speaker turns at long silences.

    Single-mic diarization-lite: a pause ≥ turn_sec starts a new turn and
    the speaker label alternates 1/2. Only a hint (pauses also end topics),
    so labels stay OFF in subtitles unless SPEAKER_LABELS is enabled.
    """
    gap = turn_sec if turn_sec and turn_sec > 0 else settings.turn_sec
    turns: list[dict] = []
    speaker = 1
    if not words:
        return turns
    start = words[0].start
    for prev, cur in zip(words, words[1:]):
        if cur.start - prev.end >= gap:
            turns.append({"start": round(start, 2), "end": round(prev.end, 2),
                          "speaker": speaker})
            speaker = 2 if speaker == 1 else 1
            start = cur.start
    turns.append({"start": round(start, 2), "end": round(words[-1].end, 2),
                  "speaker": speaker})
    return turns


# kept for callers and tests that imported the old name
normalize = normalize_token
