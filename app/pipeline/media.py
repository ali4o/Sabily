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
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    num, _, den = (stream.get("r_frame_rate") or "25/1").partition("/")
    fps = float(num) / float(den or 1)
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
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
