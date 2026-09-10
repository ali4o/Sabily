"""Stage 1 — fetch the source video and a 16kHz mono WAV for ASR."""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.config import settings

log = logging.getLogger("sabily.download")


@dataclass
class Source:
    video_path: Path
    audio_path: Path
    title: str
    duration: float
    url: str
    webpage_url: str


def _fmt() -> str:
    h = settings.max_height
    return f"bestvideo[height<={h}][ext=mp4]+bestaudio[ext=m4a]/best[height<={h}]/best"


# YouTube rejects some clients with 403 depending on the video and the day.
# Trying a few in order costs nothing and fixes most refusals without cookies.
PLAYER_CLIENTS = ["web_safari", "android", "ios", "tv", "web"]


def _explain(err: str) -> str:
    """Turn a yt-dlp wall of text into one line that says what to do."""
    low = err.lower()
    if "403" in err or "forbidden" in low:
        return (
            "يوتيوب رفض التحميل (403). جرّب بالترتيب: "
            "1) تأكد أنك داخل الـ venv وأن yt-dlp محدّث "
            "(python -m pip install -U yt-dlp) "
            "2) ضع COOKIES_FROM_BROWSER=chrome في .env"
        )
    if "sign in" in low or "bot" in low or "cookies" in low:
        return "الفيديو يطلب تسجيل دخول. ضع COOKIES_FROM_BROWSER=chrome في .env"
    if "private" in low or "unavailable" in low:
        return "الفيديو خاص أو محذوف أو محجوب في منطقتك"
    if "age" in low and "restrict" in low:
        return "الفيديو مقيّد بالعمر ويحتاج كوكيز حساب مسجّل"
    return f"فشل التحميل: {err.strip().splitlines()[-1][:200]}"


def _base_opts(job_dir: Path) -> dict:
    opts = {
        "format": _fmt(),
        "outtmpl": str(job_dir / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "concurrent_fragment_downloads": 4,
    }
    if settings.cookies_file:
        opts["cookiefile"] = settings.cookies_file
    if settings.cookies_from_browser:
        opts["cookiesfrombrowser"] = (settings.cookies_from_browser,)
    return opts


def fetch(url: str, job_dir: Path) -> Source:
    """Download the video, then extract audio.

    Tries each player client in turn; raises a single readable error if all
    of them fail, instead of letting a yt-dlp traceback escape.
    """
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    job_dir.mkdir(parents=True, exist_ok=True)
    info = None
    last_error = ""

    for client in PLAYER_CLIENTS:
        opts = _base_opts(job_dir)
        opts["extractor_args"] = {"youtube": {"player_client": [client]}}
        try:
            with YoutubeDL(opts) as ydl:
                probe_info = ydl.extract_info(url, download=False)
                duration = float(probe_info.get("duration") or 0)
                if duration > settings.max_video_minutes * 60:
                    raise ValueError(
                        f"الفيديو {duration/60:.0f} دقيقة، "
                        f"والحد الأقصى {settings.max_video_minutes}"
                    )
                info = ydl.extract_info(url, download=True)
                video_path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
            log.info("downloaded via player_client=%s", client)
            break
        except ValueError:
            raise
        except DownloadError as exc:
            last_error = str(exc)
            log.warning("player_client=%s failed: %s", client, last_error.splitlines()[-1][:120])
            continue

    if info is None:
        raise RuntimeError(_explain(last_error))

    if not video_path.exists():  # merge may keep the original container
        found = list(job_dir.glob("source.*"))
        if not found:
            raise FileNotFoundError("لم يتم العثور على الملف بعد التحميل")
        video_path = found[0]

    audio_path = job_dir / "audio.wav"
    extract_audio(video_path, audio_path)

    return Source(
        video_path=video_path,
        audio_path=audio_path,
        title=info.get("title") or "",
        duration=float(info.get("duration") or 0),
        url=url,
        webpage_url=info.get("webpage_url") or url,
    )


def extract_audio(video: Path, out: Path) -> Path:
    cmd = [
        settings.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def timestamped_url(url: str, start_sec: float) -> str:
    """Append a start time so the caption links to the exact moment."""
    t = int(max(0, start_sec))
    if "youtube.com" in url or "youtu.be" in url:
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}t={t}s"
    if "vimeo.com" in url:
        return f"{url}#t={t}s"
    return url
