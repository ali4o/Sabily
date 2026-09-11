"""Shared ffprobe helpers."""

import json
import subprocess
from pathlib import Path

from app.config import settings


def probe(video: Path) -> dict:
    cmd = [
        settings.ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json", str(video),
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed for {video}: {(exc.stderr or '')[-300:]}") from exc
    except OSError as exc:
        raise RuntimeError(f"ffprobe failed for {video}: {exc}") from exc
    try:
        data = json.loads(out)
    except ValueError as exc:
        raise RuntimeError(f"ffprobe failed for {video}: bad output") from exc
    stream = (data.get("streams") or [{}])[0]
    num_s, _, den_s = (stream.get("r_frame_rate") or "25/1").partition("/")
    try:
        num = float(num_s or 25)
        den = float(den_s or 1)
        fps = num / den if den else 25.0
    except (ValueError, ZeroDivisionError):
        fps = 25.0
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if not width or not height:
        raise RuntimeError(f"تعذّر قراءة أبعاد الفيديو: {video}")
    return {
        "width": width,
        "height": height,
        "fps": fps or 25.0,
        "duration": float((data.get("format") or {}).get("duration") or 0),
    }


def crop_size(width: int, height: int, aspect: str) -> tuple[int, int]:
    """Largest crop of the given aspect that fits inside the source frame."""
    ratio = 1.0 if aspect == "1:1" else 9 / 16
    cw = min(width, int(height * ratio))
    ch = min(height, int(cw / ratio))
    # even numbers keep libx264 happy
    return (cw - cw % 2, ch - ch % 2)
