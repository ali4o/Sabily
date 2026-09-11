"""Editable server settings backed by the .env file.

GET /api/settings describes every whitelisted key (current effective
value + where it comes from + input type). PUT /api/settings validates
and rewrites only those keys in .env, preserving comments and order.
A restart applies them (settings are read at import).

Secrets (LLM_API_KEY) are never returned in full: the descriptor carries
has_value, and submitting an empty/masked value keeps the stored one.
"""

import os
import re
from pathlib import Path
from typing import Any

from app.config import settings

_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
MASK = "••••••"

# key -> {type, group, label, default, options?, min?, max?, secret?}
SCHEMA: dict[str, dict[str, Any]] = {
    # --- server ---
    "HOST": {"type": "str", "group": "server", "label": "العنوان", "default": "127.0.0.1"},
    "PORT": {"type": "int", "group": "server", "label": "المنفذ", "default": "6767",
             "min": 1024, "max": 65535},
    # --- ingest ---
    "MAX_VIDEO_MINUTES": {"type": "int", "group": "ingest", "label": "أقصى مدة (دقيقة)",
                          "default": "180", "min": 1, "max": 600},
    "MAX_HEIGHT": {"type": "int", "group": "ingest", "label": "أقصى ارتفاع للتحميل",
                   "default": "1080", "min": 360, "max": 4320},
    "COOKIES_FILE": {"type": "path", "group": "ingest", "label": "ملف الكوكيز", "default": ""},
    "COOKIES_FROM_BROWSER": {"type": "enum", "group": "ingest", "label": "كوكيز المتصفح",
                             "default": "", "options": ["", "chrome", "edge", "firefox"]},
    # --- ASR accuracy (all local) ---
    "WHISPER_MODEL": {"type": "enum", "group": "asr", "label": "نموذج التفريغ",
                      "default": "large-v3-turbo",
                      "options": ["tiny", "base", "small", "medium",
                                  "large-v3", "large-v3-turbo"]},
    "WHISPER_ACCURACY": {"type": "enum", "group": "asr", "label": "مستوى الدقة",
                         "default": "balanced",
                         "options": ["fast", "balanced", "accurate"]},
    "WHISPER_DEVICE": {"type": "enum", "group": "asr", "label": "الجهاز",
                       "default": "cuda", "options": ["cuda", "cpu"]},
    "WHISPER_COMPUTE": {"type": "str", "group": "asr", "label": "دقة الحوسبة",
                        "default": "int8_float16"},
    "WHISPER_LANG": {"type": "str", "group": "asr", "label": "لغة التفريغ", "default": "ar"},
    "WHISPER_PROMPT": {"type": "str", "group": "asr", "label": "تلميح البداية", "default": ""},
    "WHISPER_CONDITION": {"type": "bool", "group": "asr", "label": "سياق الجمل السابقة",
                          "default": "false"},
    "WHISPER_HOTWORDS": {"type": "str", "group": "asr", "label": "كلمات مُرجحة",
                         "default": "auto"},
    "WHISPER_NO_SPEECH": {"type": "float", "group": "asr", "label": "عتبة الصمت",
                          "default": "0.6", "min": 0.0, "max": 1.0},
    "WHISPER_LOGPROB": {"type": "float", "group": "asr", "label": "عتبة الثقة",
                        "default": "-1.0", "min": -2.0, "max": 0.0},
    "WHISPER_COMPRESSION": {"type": "float", "group": "asr", "label": "عتبة التكرار",
                            "default": "2.4", "min": 1.0, "max": 4.0},
    "VAD_MIN_SILENCE": {"type": "int", "group": "asr", "label": "الصمت الفاصل (ms)",
                        "default": "400", "min": 100, "max": 2000},
    "VAD_SPEECH_PAD": {"type": "int", "group": "asr", "label": "حشو الكلام (ms)",
                       "default": "400", "min": 0, "max": 2000},
    "VAD_MAX_SPEECH": {"type": "float", "group": "asr", "label": "أقصى مقطع صوتي (s)",
                       "default": "0", "min": 0, "max": 120},
    "BEAM_SIZE": {"type": "int", "group": "asr", "label": "شعاع البحث",
                  "default": "5", "min": 1, "max": 10},
    # --- clip selection ---
    "CLIPS_PER_VIDEO": {"type": "int", "group": "selection", "label": "مقاطع لكل فيديو",
                        "default": "5", "min": 1, "max": 20},
    "MIN_CLIP_SEC": {"type": "float", "group": "selection", "label": "أقصر مقطع (s)",
                     "default": "20", "min": 5, "max": 300},
    "MAX_CLIP_SEC": {"type": "float", "group": "selection", "label": "أطول مقطع (s)",
                     "default": "75", "min": 10, "max": 600},
    "LLM_CANDIDATES": {"type": "int", "group": "selection", "label": "مرشحون للترتيب",
                       "default": "12", "min": 1, "max": 50},
    "INTRO_SEC": {"type": "float", "group": "selection", "label": "تجاهل المقدمة (s)",
                  "default": "75", "min": 0, "max": 600},
    # --- LLM ---
    "LLM_PROVIDER": {"type": "enum", "group": "llm", "label": "المزوّد",
                     "default": "ollama",
                     "options": ["ollama", "openai", "gemini", "none"]},
    "LLM_MODEL": {"type": "str", "group": "llm", "label": "النموذج", "default": "gemma2:2b"},
    "OLLAMA_URL": {"type": "str", "group": "llm", "label": "رابط Ollama",
                   "default": "http://localhost:11434"},
    "LLM_RERANK": {"type": "bool", "group": "llm", "label": "إعادة الترتيب",
                   "default": "true"},
    "LLM_TIMEOUT": {"type": "int", "group": "llm", "label": "المهلة (s)",
                    "default": "180", "min": 10, "max": 900},
    "LLM_API_KEY": {"type": "str", "group": "llm", "label": "مفتاح API",
                    "default": "", "secret": True},
    "LLM_API_BASE": {"type": "str", "group": "llm", "label": "رابط API المخصص", "default": ""},
    # --- render / appearance defaults ---
    "QUALITY": {"type": "enum", "group": "appearance", "label": "الجودة الافتراضية",
                "default": "FHD", "options": ["HD", "FHD", "QHD"]},
    "ASPECT": {"type": "enum", "group": "appearance", "label": "الاتجاه الافتراضي",
               "default": "9:16", "options": ["9:16", "1:1"]},
    "OUT_WIDTH": {"type": "int", "group": "appearance", "label": "عرض FHD",
                  "default": "1080", "min": 360, "max": 2160},
    "OUT_HEIGHT": {"type": "int", "group": "appearance", "label": "ارتفاع FHD",
                   "default": "1920", "min": 640, "max": 4320},
    "DYNAMIC_CROP": {"type": "bool", "group": "appearance", "label": "قص متحرك",
                     "default": "true"},
    "BURN_SUBTITLES": {"type": "bool", "group": "appearance", "label": "النص السفلي",
                       "default": "true"},
    "SUBTITLE_FONT": {"type": "str", "group": "appearance", "label": "خط الترجمة",
                      "default": "Cairo"},
    "CAPTION_LANG": {"type": "str", "group": "appearance", "label": "لغة الكابشن",
                     "default": "ar"},
    "CRF": {"type": "int", "group": "appearance", "label": "الجودة (CRF أصغر=أعلى)",
            "default": "20", "min": 14, "max": 32},
    "PRESET": {"type": "enum", "group": "appearance", "label": "سرعة الترميز",
               "default": "veryfast",
               "options": ["ultrafast", "superfast", "veryfast", "faster", "fast",
                           "medium", "slow"]},
    "ENCODER": {"type": "enum", "group": "appearance", "label": "المرمّز",
                "default": "auto", "options": ["auto", "nvenc", "cpu"]},
    "LOUDNORM": {"type": "bool", "group": "appearance", "label": "تطبيع الصوت",
                 "default": "true"},
    "FONTS_DIR": {"type": "path", "group": "appearance", "label": "مجلد الخطوط",
                  "default": "assets/fonts"},
    "TERMS_FILE": {"type": "path", "group": "appearance", "label": "ملف القاموس",
                   "default": "assets/terms.json"},
    "BRAND_WATERMARK": {"type": "bool", "group": "appearance", "label": "الشعار",
                        "default": "true"},
    "BRAND_TEXT": {"type": "str", "group": "appearance", "label": "نص الشعار الاحتياطي",
                   "default": "سبيلي"},
    "BRAND_LOGO": {"type": "path", "group": "appearance", "label": "صورة الشعار",
                   "default": "assets/brand/sabily-logo.png"},
    "BRAND_LOGO_SCALE": {"type": "float", "group": "appearance", "label": "حجم الشعار",
                         "default": "0.24", "min": 0.08, "max": 0.5},
    "BRAND_POS": {"type": "enum", "group": "appearance", "label": "موضع الشعار",
                  "default": "tr",
                  "options": ["tl", "tc", "tr", "bl", "bc", "br"]},
    "SOURCE_TAG": {"type": "bool", "group": "appearance", "label": "الهاشتاق",
                   "default": "true"},
    "SOURCE_POS": {"type": "enum", "group": "appearance", "label": "موضع الهاشتاق",
                   "default": "tl",
                   "options": ["tl", "tc", "tr", "bl", "bc", "br"]},
    "KEEP_SOURCE": {"type": "bool", "group": "appearance", "label": "الاحتفاظ بالأصل",
                    "default": "true"},
    "FACE_BLUR": {"type": "int", "group": "tracking", "label": "ضبابية الخلفية",
                  "default": "40", "min": 5, "max": 100},
    "FACE_SAMPLE_FPS": {"type": "float", "group": "tracking", "label": "إطارات التحليل",
                        "default": "2.0", "min": 0.5, "max": 8},
    "FACE_MIN_COVERAGE": {"type": "float", "group": "tracking", "label": "أدنى تغطية للوجه",
                          "default": "0.25", "min": 0.0, "max": 1.0},
    "FACE_HOLD_SEC": {"type": "float", "group": "tracking", "label": "جسر الالتفات (s)",
                      "default": "1.5", "min": 0.0, "max": 10},
    "FACE_PROFILE": {"type": "bool", "group": "tracking", "label": "كشف الوجه الجانبي",
                     "default": "true"},
    "FACE_CENTER_BIAS": {"type": "float", "group": "tracking", "label": "انحياز الوسط",
                         "default": "0.3", "min": 0.0, "max": 0.9},
    "SUBTITLE_POS": {"type": "enum", "group": "appearance", "label": "موضع الترجمة",
                     "default": "bottom", "options": ["bottom", "middle", "top"]},
    "SUBTITLE_STYLE": {"type": "enum", "group": "appearance", "label": "نمط الترجمة",
                       "default": "line", "options": ["line", "karaoke"]},
    "SPEAKER_LABELS": {"type": "bool", "group": "appearance", "label": "تسمية المتحدثين",
                       "default": "false"},
    "SUBTITLE_BILINGUAL": {"type": "bool", "group": "appearance", "label": "ترجمة ثنائية",
                           "default": "false"},
    "TRANSCRIPT_POLISH": {"type": "bool", "group": "asr", "label": "تصحيح النص",
                          "default": "true"},
    "INTRO_CARD": {"type": "bool", "group": "appearance", "label": "بطاقة بداية",
                   "default": "false"},
    "OUTRO_CARD": {"type": "bool", "group": "appearance", "label": "بطاقة نهاية",
                   "default": "false"},
    "ZOOM_PUNCH": {"type": "bool", "group": "appearance", "label": "تكبير نابض",
                   "default": "false"},
    "TURN_SEC": {"type": "float", "group": "asr", "label": "فاصل المتحدث (s)",
                 "default": "1.5", "min": 0.5, "max": 5},
    "CLEANUP_DAYS": {"type": "int", "group": "ingest", "label": "تنظيف المصادر (يوم)",
                     "default": "0", "min": 0, "max": 365},
}

GROUP_LABELS = {
    "server": "الخادم", "ingest": "التحميل", "asr": "التفريغ الصوتي",
    "selection": "اختيار المقاطع", "llm": "النموذج اللغوي", "appearance": "المظهر والمخرجات",
    "tracking": "تتبع الوجه",
}


def env_path(base: Path | None = None) -> Path:
    return (base or settings.base_dir) / ".env"


def _read_file(path: Path) -> dict[str, str]:
    vals: dict[str, str] = {}
    if not path.exists():
        return vals
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        if _KEY_RE.fullmatch(k):
            vals[k] = v.strip().strip("'\"")
    return vals


def describe(base: Path | None = None) -> dict[str, Any]:
    """Every editable key: effective value + source + input metadata."""
    path = env_path(base)
    file_vals = _read_file(path)
    out: dict[str, Any] = {}
    for key, meta in SCHEMA.items():
        if key in file_vals:
            out[key] = {**meta, "value": file_vals[key], "source": "env_file"}
        elif key in os.environ:
            out[key] = {**meta, "value": os.environ[key], "source": "environment"}
        else:
            out[key] = {**meta, "value": meta["default"], "source": "default"}
        if meta.get("secret") and out[key]["value"]:
            out[key]["value"] = ""
            out[key]["has_value"] = True
        else:
            out[key]["has_value"] = bool(out[key]["value"]) if meta.get("secret") else None
    return {"groups": GROUP_LABELS, "keys": out, "needs_restart": True}


def _coerce(key: str, meta: dict[str, Any], raw: Any) -> str:
    if raw is None:
        raise ValueError("قيمة فارغة")
    s = str(raw).strip()
    t = meta["type"]
    if t == "bool":
        low = s.lower()
        if low in ("1", "true", "yes", "on"):
            return "true"
        if low in ("0", "false", "no", "off"):
            return "false"
        raise ValueError(f"{key}: قيمة منطقية غير صالحة")
    if t == "int":
        try:
            v = int(s)
        except ValueError:
            raise ValueError(f"{key}: عدد صحيح مطلوب") from None
        if "min" in meta and v < meta["min"]:
            raise ValueError(f"{key}: أقل من {meta['min']}")
        if "max" in meta and v > meta["max"]:
            raise ValueError(f"{key}: أكبر من {meta['max']}")
        return str(v)
    if t == "float":
        try:
            v = float(s)
        except ValueError:
            raise ValueError(f"{key}: عدد مطلوب") from None
        if "min" in meta and v < meta["min"]:
            raise ValueError(f"{key}: أقل من {meta['min']}")
        if "max" in meta and v > meta["max"]:
            raise ValueError(f"{key}: أكبر من {meta['max']}")
        return str(v)
    if t == "enum":
        for opt in meta["options"]:
            if s == opt or (opt and s.lower() == opt.lower()):
                return opt
        raise ValueError(f"{key}: اختر من {meta['options']}")
    # str / path
    if "\n" in s or "\r" in s:
        raise ValueError(f"{key}: سطر واحد فقط")
    if len(s) > 2000:
        raise ValueError(f"{key}: أطول من اللازم")
    return s


def save(updates: dict[str, Any], base: Path | None = None) -> dict[str, Any]:
    """Validate and persist whitelisted keys to .env. Unknown keys rejected."""
    unknown = [k for k in updates if k not in SCHEMA]
    if unknown:
        raise ValueError(f"مفاتيح غير معروفة: {sorted(unknown)}")
    path = env_path(base)
    file_vals = _read_file(path)
    coerced: dict[str, str] = {}
    for key, raw in updates.items():
        meta = SCHEMA[key]
        if meta.get("secret") and str(raw or "").strip() in ("", MASK):
            continue  # masked/empty secret = keep the stored one
        coerced[key] = _coerce(key, meta, raw)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen: set[str] = set()
    out_lines: list[str] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.partition("=")[0].strip()
            if k in coerced:
                out_lines.append(f"{k}={coerced[k]}")
                seen.add(k)
                continue
        out_lines.append(raw_line)
    for key, val in coerced.items():
        if key not in seen:
            out_lines.append(f"{key}={val}")
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return {"saved": sorted(coerced), "needs_restart": True}
