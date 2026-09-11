"""Stage 4 — titles, captions, and optional re-ranking.

One interface, three backends. Switching between the local model and an
external API is a single line in .env; no other file knows the difference.
If the LLM is unavailable the pipeline still produces clips using a
heuristic title/caption — it degrades, it does not fail.
"""

import json
import logging
import re
from typing import Any, Protocol

import httpx

from app.config import settings

log = logging.getLogger("sabily.llm")


def _redact(s: str) -> str:
    key = settings.llm_api_key
    return s.replace(key, "[redacted]") if key else s

SYSTEM_AR = (
    "أنت محرر محتوى قصير. تُعطى نصاً مفرغاً من مقطع فيديو، وتُخرج JSON فقط "
    "بالمفاتيح: title (عنوان جذاب أقل من 60 حرفاً يصف ما في المقطع فعلاً، "
    "بدون مبالغة أو وعود كاذبة)، summary (سطران يشرحان المقطع بلغة بسيطة)، "
    "hashtags (مصفوفة من 3 إلى 5 وسوم بدون مسافات). لا تكتب أي نص خارج JSON.\n"
    "إلزامي: قيم title و summary و hashtags كلها بالعربية. "
    "ممنوع الإنجليزية إلا للمصطلحات التقنية التي لا ترجمة شائعة لها.\n"
    "مثال على الشكل المطلوب:\n"
    '{"title":"سبب فشل العادات الجديدة","summary":"يشرح المقطع لماذا يتوقف '
    'أغلب الناس بعد أسبوع. الحل هو البدء بخطوة صغيرة جداً.",'
    '"hashtags":["#عادات","#تطوير_الذات","#إنتاجية"]}'
)
SYSTEM_EN = (
    "You are a short-form video editor. Given a transcript excerpt, return JSON only "
    "with keys: title (under 60 chars, accurate to the content, no clickbait), "
    "summary (two plain sentences), hashtags (3-5 tags, no spaces). No text outside JSON."
)

RERANK_AR = (
    "لديك مقاطع مرشحة من فيديو طويل. اختر الأفضل للنشر كفيديو قصير. "
    "فضّل المقاطع المفهومة بذاتها (تُفهم دون مشاهدة الأصل)، التي تبدأ بخطاف "
    "واضح وتنتهي بفكرة مغلقة، وتجنب تكرار نفس الفكرة. "
    "أعد JSON فقط: {\"order\": [أرقام المرشحين مرتبة من الأفضل]}. لا شيء غير ذلك."
)

CLASSIFY_AR = (
    "صنّف هذا المقتطف المفرغ من فيديو طويل. أعد JSON فقط بالمفاتيح: "
    "content_type (واحد من: lecture محاضرة، lesson درس تعليمي، "
    "podcast بودكاست، interview مقابلة)، audience (الجمهور بكلمات قليلة)، "
    "goal (education أو engagement). لا شيء غير ذلك."
)


class Provider(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OllamaProvider:
    """Local model via Ollama. Default path."""

    def _messages(self, system: str, user: str) -> list[dict]:
        # Gemma's chat template has no system role; Ollama either drops it or
        # errors out. Folding it into the user turn keeps the instructions.
        if "gemma" in settings.llm_model.lower():
            return [{"role": "user", "content": f"{system}\n\n---\n\n{user}"}]
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def complete(self, system: str, user: str) -> str:
        r = httpx.post(
            f"{settings.ollama_url}/api/chat",
            json={
                "model": settings.llm_model,
                "messages": self._messages(system, user),
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.3, "num_ctx": 4096, "num_predict": 400},
            },
            timeout=settings.llm_timeout,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]


class OpenAIProvider:
    """Any OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter, local vLLM)."""

    def complete(self, system: str, user: str) -> str:
        base = settings.llm_api_base or "https://api.openai.com/v1"
        r = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json={
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
            timeout=settings.llm_timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class GeminiProvider:
    def complete(self, system: str, user: str) -> str:
        base = settings.llm_api_base or "https://generativelanguage.googleapis.com/v1beta"
        r = httpx.post(
            f"{base}/models/{settings.llm_model}:generateContent",
            params={"key": settings.llm_api_key},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"},
            },
            timeout=settings.llm_timeout,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def get_provider() -> Provider | None:
    name = settings.llm_provider.lower()
    if name == "ollama":
        return OllamaProvider()
    if name == "openai":
        return OpenAIProvider()
    if name == "gemini":
        return GeminiProvider()
    return None


def _parse_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?|```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.S)
    return json.loads(match.group(0) if match else raw)


def _fallback_meta(text: str) -> dict[str, Any]:
    first = re.split(r"[.؟!?\n]", text.strip())[0]
    words = first.split()
    return {
        "title": " ".join(words[:9])[:60] or "مقطع مختار",
        "summary": " ".join(text.split()[:35]),
        "hashtags": ["#Shorts", "#Reels", "#مقاطع"],
    }


def generate_metadata(text: str, lang: str | None = None) -> dict[str, Any]:
    """Title + summary + hashtags for one clip. Never raises."""
    lang = lang or settings.caption_lang
    provider = get_provider()
    if provider is None:
        return _fallback_meta(text)

    excerpt = " ".join(text.split()[:400])  # keep the local model's context small
    system = SYSTEM_AR if lang == "ar" else SYSTEM_EN
    # Small models drift to English on Arabic input, so repeat the language
    # requirement next to the text itself, where it is hardest to ignore.
    if lang == "ar":
        excerpt = f"النص المفرّغ:\n{excerpt}\n\nأعد JSON بالعربية فقط."
    try:
        data = _parse_json(provider.complete(system, excerpt))
        tags = data.get("hashtags") or []
        if isinstance(tags, str):
            tags = tags.split()
        title = str(data.get("title", "")).strip()
        # a "title" that is really a paragraph is a model failure, not a title
        if len(title) > 90 or len(title.split()) > 14 or not title:
            title = _fallback_meta(text)["title"]
        return {
            "title": title[:80],
            "summary": str(data.get("summary", "")).strip(),
            "hashtags": ["#" + t.lstrip("#").replace(" ", "_") for t in tags][:5],
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM metadata failed (%s), using fallback", _redact(str(exc)))
        return _fallback_meta(text)


def classify_content(words: list[Any]) -> dict[str, str]:
    """UNDERSTAND via LLM when available, heuristics otherwise. Never raises.

    Returns {content_type, audience, goal}. The transcript head (first
    ~300 words) is enough to tell a lecture from a podcast; the full
    text never leaves the machine unless a provider is configured.
    """
    from app.pipeline import score as _score

    fallback = {"content_type": "lecture", "audience": "عام", "goal": "education"}
    try:
        sents = _score.build_sentences(words)
        if sents:
            fallback["content_type"] = _score.detect_content_type(
                sents, _score.detect_turns(words),
                sents[-1].end - sents[0].start)
    except Exception:  # noqa: BLE001 - detection must not fail the job
        pass
    provider = get_provider()
    if provider is None:
        return dict(fallback)
    excerpt = " ".join(w.text for w in words[:300])
    if not excerpt.strip():
        return dict(fallback)
    try:
        data = _parse_json(provider.complete(CLASSIFY_AR, excerpt[:2000]))
        ctype = _score.normalize_content_type(str(data.get("content_type", "")))
        if ctype in ("auto", "general"):
            ctype = fallback["content_type"]
        return {
            "content_type": ctype,
            "audience": str(data.get("audience", "")).strip()[:60] or fallback["audience"],
            "goal": "engagement" if "engag" in str(data.get("goal", "")).lower()
                    or "تفاعل" in str(data.get("goal", "")) else "education",
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM classify failed (%s), using heuristic", _redact(str(exc)))
        return dict(fallback)


def rerank(candidates: list[Any], keep: int) -> list[int]:
    """Optional. Returns indices in preferred order; falls back to heuristic order."""
    default = list(range(min(keep, len(candidates))))
    provider = get_provider()
    if provider is None or not settings.llm_rerank or len(candidates) <= keep:
        return default
    listing = "\n".join(
        f"[{i}] ({c.duration:.0f}s) [{getattr(c, 'moment', 'general')}] "
        + " ".join(c.text.split()[:120])
        for i, c in enumerate(candidates)
    )
    try:
        data = _parse_json(provider.complete(RERANK_AR, listing))
        order = [int(i) for i in data.get("order", []) if 0 <= int(i) < len(candidates)]
        seen, out = set(), []
        for i in order:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return (out + [i for i in default if i not in seen])[:keep] if out else default
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM rerank failed (%s), keeping heuristic order", _redact(str(exc)))
        return default


POLISH_AR = (
    "صحح النص المفرغ التالي من فيديو عربي: أضف علامات الترقيم العربية "
    "المناسبة (، ؛ ؟ .) وصحح الأخطاء الإملائية الواضحة فقط. لا تغيّر "
    "المعنى ولا تحذف كلمات ولا تضف شرحاً. أعد JSON فقط: {\"text\": \"...\"}."
)

TRANSLATE_AR_EN = (
    "Translate each Arabic subtitle line to simple English, one per line, "
    "same order, no numbering, no extra text. Keep technical terms as-is. "
    "Reply with a JSON array of strings only."
)


def polish_text(text: str) -> str:
    """LLM punctuation/spelling pass over one clip transcript. Never raises.

    Returns the original text when polishing is disabled, the provider is
    missing, or the model output is unusable (too short/long = drift).
    """
    if not settings.transcript_polish or not text.strip():
        return text
    provider = get_provider()
    if provider is None:
        return text
    try:
        data = _parse_json(provider.complete(POLISH_AR, text[:1500]))
        fixed = str(data.get("text", "")).strip()
        n0, n1 = len(text.split()), len(fixed.split())
        if not fixed or n1 < n0 * 0.7 or n1 > n0 * 1.4:
            return text
        return fixed
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM polish failed (%s), keeping raw", _redact(str(exc)))
        return text


def translate_lines(texts: list[str]) -> list[str]:
    """Translate subtitle lines to English, index-aligned. Never raises.

    Falls back to the Arabic originals (render stays Arabic-only) when
    bilingual output is disabled, the provider is missing, or parsing
    fails.
    """
    if not settings.subtitle_bilingual or not texts:
        return list(texts)
    provider = get_provider()
    if provider is None:
        return list(texts)
    try:
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        data = _parse_json(provider.complete(
            TRANSLATE_AR_EN, numbered + '\n\nJSON array only, e.g. ["..."].'))
        if isinstance(data, dict):  # tolerate {"lines": [...]} wrappers
            data = data.get("lines", data.get("translations", []))
        if isinstance(data, list) and len(data) == len(texts):
            return [str(x).strip() or o for x, o in zip(data, texts)]
        return list(texts)
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM translate failed (%s), Arabic only", _redact(str(exc)))
        return list(texts)


def build_caption(meta: dict[str, Any], source_url: str) -> str:
    """The caption the user copies: explanation, source link, hashtags."""
    lines = [meta.get("summary", "").strip()]
    lines.append("")
    lines.append(f"🎬 الفيديو الأصلي: {source_url}")
    tags = " ".join(meta.get("hashtags", []))
    if tags:
        lines.append("")
        lines.append(tags)
    return "\n".join(x for x in lines if x is not None).strip()
