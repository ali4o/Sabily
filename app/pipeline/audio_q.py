"""Audio-quality features for scoring (stdlib only, no new deps).

The wav is read ONCE per job (load) and sliced per candidate window
(stats_from), so scoring hundreds of candidates stays cheap. Probing
never raises: on any error the scorer falls back to text-only.

What it catches: dead air / mumbling (low RMS, high silence share) and
clipped shouting (samples pinned at max). Music beds can only be hinted
at — the penalty demotes, never vetoes.
"""

import logging
import struct
import wave
from pathlib import Path

try:
    import audioop  # deprecated since 3.12, gone in 3.13
except ImportError:  # pragma: no cover - future Pythons
    audioop = None  # type: ignore[assignment]

log = logging.getLogger("sabily.audioq")

SILENCE_RMS = 120      # below this the mic is basically off
SILENCE_BUCKET = 0.1   # seconds per silence probe bucket


def _rms(chunk: bytes) -> float:
    if audioop is not None:
        try:
            return float(audioop.rms(chunk, 2))
        except audioop.error:
            return 0.0
    vals = struct.unpack("<%dh" % (len(chunk) // 2), chunk)
    return (sum(v * v for v in vals) / max(1, len(vals))) ** 0.5


def _peak(chunk: bytes) -> int:
    if audioop is not None:
        try:
            return audioop.max(chunk, 2)
        except audioop.error:
            return 0
    vals = struct.unpack("<%dh" % (len(chunk) // 2), chunk)
    return max([abs(v) for v in vals] + [0])


def load(path: "str | Path") -> tuple[bytes, int] | None:
    """Raw 16-bit mono frames + sample rate, or None on any error."""
    try:
        with wave.open(str(path), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                log.debug("energy probe needs 16k mono wav, got %s", path)
                return None
            return w.readframes(w.getnframes()), w.getframerate()
    except Exception as exc:  # noqa: BLE001
        log.debug("energy probe failed (%s)", exc)
        return None


def stats_from(audio: tuple[bytes, int] | None, start: float, end: float) -> dict | None:
    """Summary stats for one candidate window, or None when unavailable."""
    if not audio or end <= start:
        return None
    frames, rate = audio
    s = int(start * rate) * 2
    chunk = frames[s:int(end * rate) * 2]
    if len(chunk) < 100:
        return None
    rms, maxed = _rms(chunk), _peak(chunk)
    step = max(2, int(2 * rate * SILENCE_BUCKET))
    buckets = max(1, len(chunk) // step)
    silent = sum(1 for i in range(buckets)
                 if _rms(chunk[i * step:(i + 1) * step]) < SILENCE_RMS)
    return {"silence": silent / buckets,
            "clipped": 1.0 if maxed >= 32760 else 0.0,
            "rms": float(rms)}


def energy_map(path: "str | Path", start: float, end: float,
               bucket: float = 0.5) -> dict[int, float]:
    """RMS energy per `bucket`-second slot over [start, end). Empty on error."""
    audio = load(path)
    if not audio:
        return {}
    frames, rate = audio
    width = max(1, int(rate * bucket))
    out: dict[int, float] = {}
    for i in range(max(1, int((end - start) / bucket))):
        s = int((start + i * bucket) * rate) * 2
        chunk = frames[s:s + width * 2]
        if len(chunk) < 2:
            continue
        out[i] = _rms(chunk)
    return out
