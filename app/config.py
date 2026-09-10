"""Sabily configuration. All tunables live here, values come from .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def _bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    # --- server ---
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = _int("PORT", 6767)

    # --- paths ---
    base_dir: Path = BASE_DIR
    work_dir: Path = BASE_DIR / os.getenv("WORK_DIR", "data")
    outputs_dir: Path = BASE_DIR / os.getenv("OUTPUTS_DIR", "outputs")
    db_path: Path = BASE_DIR / os.getenv("DB_PATH", "data/sabily.db")
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

    # --- clip selection ---
    clips_per_video: int = _int("CLIPS_PER_VIDEO", 5)
    min_clip_sec: float = _float("MIN_CLIP_SEC", 20.0)
    max_clip_sec: float = _float("MAX_CLIP_SEC", 75.0)
    llm_candidates: int = _int("LLM_CANDIDATES", 12)
    intro_sec: float = _float("INTRO_SEC", 75.0)   # opening stretch to avoid

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
    dynamic_crop: bool = _bool("DYNAMIC_CROP", True)
    burn_subtitles: bool = _bool("BURN_SUBTITLES", True)
    subtitle_font: str = os.getenv("SUBTITLE_FONT", "Cairo")
    fonts_dir: Path = BASE_DIR / os.getenv("FONTS_DIR", "assets/fonts")
    terms_file: Path = BASE_DIR / os.getenv("TERMS_FILE", "assets/terms.json")
    crf: int = _int("CRF", 20)
    preset: str = os.getenv("PRESET", "veryfast")
    encoder: str = os.getenv("ENCODER", "auto")    # auto|nvenc|cpu
    loudnorm: bool = _bool("LOUDNORM", True)
    caption_lang: str = os.getenv("CAPTION_LANG", "ar")

    # --- on-video attribution ---
    brand_watermark: bool = _bool("BRAND_WATERMARK", True)
    brand_text: str = os.getenv("BRAND_TEXT", "سبيلي")
    source_tag: bool = _bool("SOURCE_TAG", True)
    brand_pos: str = os.getenv("BRAND_POS", "tr")     # tr|tl|br|bl
    source_pos: str = os.getenv("SOURCE_POS", "tl")
    keep_source: bool = _bool("KEEP_SOURCE", True)    # required for re-editing

    def ensure_dirs(self) -> None:
        for d in (self.work_dir, self.outputs_dir, self.db_path.parent):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def target_size(self) -> tuple[int, int]:
        if self.aspect == "1:1":
            return (1080, 1080)
        return (self.out_width, self.out_height)


settings = Settings()
