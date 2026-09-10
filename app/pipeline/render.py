"""Stage 7 — one ffmpeg pass per clip: cut → crop → scale → subtitles → mp4."""

import logging
import subprocess
from pathlib import Path

from app.config import settings

log = logging.getLogger("sabily.render")


def _escape_for_filter(path: Path) -> str:
    """Windows paths and colons need escaping inside a filtergraph."""
    p = str(path.resolve()).replace("\\", "/")
    return p.replace(":", "\\:").replace("'", "\\'")


def subtitles_filter(ass_path: Path) -> str:
    """Build the ass filter for burned-in subtitles.

    Two deliberate choices, both learned the hard way on Windows:

    * the `ass` filter, not `subtitles` — only `ass` exposes `shaping`.
      With `shaping=simple` libass converts lam-alef to the legacy
      presentation form U+FEFB, which modern fonts don't carry, and the
      ligature renders as an empty box. `complex` forces the HarfBuzz path.
    * `fontsdir` is still passed, but note it is IGNORED by the directwrite
      font provider (the default on Windows). There, the font named in the
      ASS style must be installed system-wide or libass silently substitutes
      another face. Check with:
          ffmpeg -v verbose ... 2>&1 | Select-String "fontselect"
    """
    f = f"ass='{_escape_for_filter(ass_path)}':shaping=complex"
    fonts = settings.fonts_dir
    if fonts.exists() and any(fonts.glob("*.tt*")):
        f += f":fontsdir='{_escape_for_filter(fonts)}'"
    return f


def _has_nvenc() -> bool:
    try:
        out = subprocess.run([settings.ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=30).stdout
        return "h264_nvenc" in out
    except Exception:  # noqa: BLE001
        return False


def _video_args(use_gpu: bool) -> list[str]:
    if use_gpu:
        # NVENC on this class of card is roughly 3x faster than x264 and frees
        # the CPU for the next clip's face pass. cq is the quality knob here.
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                "-cq", str(settings.crf), "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf)]


def render_clip(
    source: Path,
    out_path: Path,
    start: float,
    end: float,
    crop: dict,
    ass_path: Path | None = None,
) -> Path:
    w, h = settings.target_size
    chain = [
        f"crop={crop['w']}:{crop['h']}:{crop['x_expr']}:{crop['y']}",
        f"scale={w}:{h}:flags=bicubic",
        "setsar=1",
    ]
    if ass_path and settings.burn_subtitles:
        chain.append(subtitles_filter(ass_path))

    audio = ["-c:a", "aac", "-b:a", "160k", "-ac", "2", "-ar", "48000"]
    if settings.loudnorm:
        # Social platforms normalise to about -14 LUFS on playback. Doing it
        # here means the clip sounds the same as everything else in the feed
        # instead of being quietly turned down.
        audio = ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"] + audio

    def build(use_gpu: bool) -> list[str]:
        return [
            settings.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, start):.3f}",
            "-to", f"{end:.3f}",
            "-i", str(source),
            "-vf", ",".join(chain),
            *_video_args(use_gpu),
            "-pix_fmt", "yuv420p",
            *audio,
            "-movflags", "+faststart",
            str(out_path),
        ]

    want_gpu = settings.encoder == "nvenc" or (settings.encoder == "auto" and _has_nvenc())
    log.info("rendering %s (%.1fs-%.1fs, crop=%s, encoder=%s)",
             out_path.name, start, end, crop["mode"], "nvenc" if want_gpu else "x264")
    proc = subprocess.run(build(want_gpu), capture_output=True, text=True)
    if proc.returncode != 0 and want_gpu:
        log.warning("nvenc failed (%s), falling back to x264", proc.stderr[-160:].strip())
        proc = subprocess.run(build(False), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-400:]}")
    return out_path
