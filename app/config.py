"""Sabily configuration. All tunables live here, values come from .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return default


def _path(key: str, default: str) -> Path:
    v = os.getenv(key, default)
    p = Path(v)
    return p if p.is_absolute() else BASE_DIR / v


def _bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    # --- server ---
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = _int("PORT", 6767)

    # --- paths ---
    base_dir: Path = BASE_DIR
    work_dir: Path = _path("WORK_DIR", "data")
    outputs_dir: Path = _path("OUTPUTS_DIR", "outputs")
    db_path: Path = _path("DB_PATH", "data/sabily.db")
    ffmpeg: str = os.getenv("FFMPEG_BIN", "ffmpeg")
    ffprobe: str = os.getenv("FFPROBE_BIN", "ffprobe")

    # --- ingest ---
    max_video_minutes: int = _int("MAX_VIDEO_MINUTES", 180)
    max_height: int = _int("MAX_HEIGHT", 1080)
    cookies_file: str = os.getenv("COOKIES_FILE", "")
    cookies_from_browser: str = os.getenv("COOKIES_FROM_BROWSER", "")

    # --- ASR ---
    whisper_model: str = os.getenv("WHISPER_MODEL", "large-v3-turbo")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cuda")
    whisper_compute: str = os.getenv("WHISPER_COMPUTE", "int8_float16")
    whisper_lang: str = os.getenv("WHISPER_LANG", "ar")
    whisper_prompt: str = os.getenv("WHISPER_PROMPT", "")
    beam_size: int = _int("BEAM_SIZE", 5)

    # --- ASR accuracy (all local, applied after download, before scoring) ---
    whisper_accuracy: str = os.getenv("WHISPER_ACCURACY", "balanced")  # fast|balanced|accurate
    whisper_condition: bool = _bool("WHISPER_CONDITION", False)
    whisper_hotwords: str = os.getenv("WHISPER_HOTWORDS", "auto")  # auto|off|custom words
    whisper_no_speech: float = _float("WHISPER_NO_SPEECH", 0.6)
    whisper_logprob: float = _float("WHISPER_LOGPROB", -1.0)
    whisper_compression: float = _float("WHISPER_COMPRESSION", 2.4)
    vad_min_silence: int = _int("VAD_MIN_SILENCE", 400)
    vad_speech_pad: int = _int("VAD_SPEECH_PAD", 400)
    vad_max_speech: float = _float("VAD_MAX_SPEECH", 0)  # 0 = silero default

    # --- clip selection ---
    clips_per_video: int = _int("CLIPS_PER_VIDEO", 5)
    min_clip_sec: float = _float("MIN_CLIP_SEC", 20.0)
    max_clip_sec: float = _float("MAX_CLIP_SEC", 75.0)
    llm_candidates: int = _int("LLM_CANDIDATES", 12)
    intro_sec: float = _float("INTRO_SEC", 75.0)   # opening stretch to avoid
    content_type: str = os.getenv("CONTENT_TYPE", "auto")  # auto|lecture|lesson|podcast|interview

    # --- LLM ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")  # ollama|gemini|openai|none
    llm_model: str = os.getenv("LLM_MODEL", "gemma2:2b")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_api_base: str = os.getenv("LLM_API_BASE", "")
    llm_timeout: int = _int("LLM_TIMEOUT", 180)
    llm_rerank: bool = _bool("LLM_RERANK", True)

    # --- render ---
    out_width: int = _int("OUT_WIDTH", 1080)
    out_height: int = _int("OUT_HEIGHT", 1920)
    aspect: str = os.getenv("ASPECT", "9:16")  # 9:16 | 1:1
    quality: str = os.getenv("QUALITY", "FHD")  # HD|FHD|QHD (QHD = 2K)

    # --- subtitle style ---
    subtitle_pos: str = os.getenv("SUBTITLE_POS", "bottom")  # bottom|middle|top
    subtitle_style: str = os.getenv("SUBTITLE_STYLE", "line")  # line|karaoke
    speaker_labels: bool = _bool("SPEAKER_LABELS", False)
    subtitle_bilingual: bool = _bool("SUBTITLE_BILINGUAL", False)
    turn_sec: float = _float("TURN_SEC", 1.5)

    # --- transcript polish (local LLM, graceful fallback) ---
    transcript_polish: bool = _bool("TRANSCRIPT_POLISH", True)

    # --- bumpers / motion ---
    intro_card: bool = _bool("INTRO_CARD", False)
    outro_card: bool = _bool("OUTRO_CARD", False)
    zoom_punch: bool = _bool("ZOOM_PUNCH", False)

    # --- housekeeping ---
    cleanup_days: int = _int("CLEANUP_DAYS", 0)  # 0 = off

    # --- face tracking (all CPU, all local) ---
    face_sample_fps: float = _float("FACE_SAMPLE_FPS", 2.0)
    face_min_coverage: float = _float("FACE_MIN_COVERAGE", 0.25)
    face_hold_sec: float = _float("FACE_HOLD_SEC", 1.5)
    face_profile: bool = _bool("FACE_PROFILE", True)
    face_blur: int = _int("FACE_BLUR", 40)
    face_center_bias: float = _float("FACE_CENTER_BIAS", 0.3)
    dynamic_crop: bool = _bool("DYNAMIC_CROP", True)
    burn_subtitles: bool = _bool("BURN_SUBTITLES", True)
    subtitle_font: str = os.getenv("SUBTITLE_FONT", "Cairo")
    fonts_dir: Path = _path("FONTS_DIR", "assets/fonts")
    terms_file: Path = _path("TERMS_FILE", "assets/terms.json")
    crf: int = _int("CRF", 20)
    preset: str = os.getenv("PRESET", "veryfast")
    encoder: str = os.getenv("ENCODER", "auto")    # auto|nvenc|cpu
    loudnorm: bool = _bool("LOUDNORM", True)
    caption_lang: str = os.getenv("CAPTION_LANG", "ar")

    # --- on-video attribution ---
    brand_watermark: bool = _bool("BRAND_WATERMARK", True)
    brand_text: str = os.getenv("BRAND_TEXT", "سبيلي")
    brand_logo: Path = _path("BRAND_LOGO", "assets/brand/sabily-logo.png")
    brand_logo_scale: float = _float("BRAND_LOGO_SCALE", 0.24)  # logo width as fraction of output width
    source_tag: bool = _bool("SOURCE_TAG", True)
    brand_pos: str = os.getenv("BRAND_POS", "tr")     # tl|tc|tr|bl|bc|br
    source_pos: str = os.getenv("SOURCE_POS", "tl")
    keep_source: bool = _bool("KEEP_SOURCE", True)    # required for re-editing

    def ensure_dirs(self) -> None:
        for d in (self.work_dir, self.outputs_dir, self.db_path.parent):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def use_brand_logo(self) -> bool:
        """Image watermark when enabled and the file exists.

        Text fallback (Brand ASS line) is used otherwise, so deleting
        the PNG never breaks a render — it just restores the old look.
        """
        try:
            return bool(self.brand_watermark) and self.brand_logo.exists()
        except OSError:
            return False

    @property
    def target_size(self) -> tuple[int, int]:
        return resolve_target(self.quality, self.aspect,
                              self.out_width, self.out_height)


# --- output quality presets ---
# HD  = 720p  (720x1280)  — light files, fast render
# FHD = 1080p (1080x1920) — default, matches most phone screens
# QHD = 2K    (1440x2560) — sharpest, bigger files / slower render
QUALITY_PRESETS: dict[str, dict[str, tuple[int, int]]] = {
    "HD": {"9:16": (720, 1280), "1:1": (720, 720)},
    "FHD": {"9:16": (1080, 1920), "1:1": (1080, 1080)},
    "QHD": {"9:16": (1440, 2560), "1:1": (1440, 1440)},
}

# minimum source height worth fetching per quality (avoids upscaling
# a 720p source to 2K when the publisher offers 1440p/2160p).
QUALITY_SOURCE_HEIGHT: dict[str, int] = {"HD": 720, "FHD": 1080, "QHD": 1440}

# user-facing aliases (API / CLI / web)
QUALITY_ALIASES: dict[str, str] = {
    "HD": "HD", "720": "HD", "720P": "HD",
    "FHD": "FHD", "1080": "FHD", "1080P": "FHD",
    "QHD": "QHD", "2K": "QHD", "1440": "QHD", "1440P": "QHD",
}


def normalize_quality(v: str | None, fallback: str = "FHD") -> str:
    """User input -> HD|FHD|QHD. Never raises; unknown falls back."""
    if not v:
        return fallback if fallback in QUALITY_PRESETS else "FHD"
    q = QUALITY_ALIASES.get(str(v).strip().upper(), "")
    return q or (fallback if fallback in QUALITY_PRESETS else "FHD")


def resolve_target(quality: str | None, aspect: str | None,
                   out_w: int = 1080, out_h: int = 1920) -> tuple[int, int]:
    """(quality, aspect) -> (w, h). FHD honours custom OUT_WIDTH/OUT_HEIGHT."""
    q = normalize_quality(quality)
    a = aspect if aspect in ("9:16", "1:1") else "9:16"
    if q == "FHD":
        return (min(out_w, out_h), min(out_w, out_h)) if a == "1:1" \
            else (out_w, out_h)
    return QUALITY_PRESETS[q][a]


settings = Settings()
