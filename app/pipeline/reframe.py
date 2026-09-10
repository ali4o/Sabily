"""Stage 5 — decide where to crop so the speaker stays in frame.

Detection runs on CPU with OpenCV only (no mediapipe/jax): Haar cascade by
default, or YuNet if the ONNX model is present in data/models/ — YuNet is
noticeably better on profile/tilted faces and is a drop-in upgrade.

Samples 2 frames per second instead of every frame, and produces an ffmpeg
crop expression rather than writing frames to disk.
"""

import logging
from functools import lru_cache
from pathlib import Path

from app.config import settings
from app.pipeline.media import crop_size

log = logging.getLogger("sabily.reframe")

SAMPLE_FPS = 2.0
EMA_ALPHA = 0.25          # lower = smoother, slower camera
MAX_KEYFRAMES = 20        # ffmpeg expressions get fragile beyond this
MOVE_THRESHOLD = 0.04     # ignore jitter under 4% of frame width
YUNET_PATH = settings.work_dir / "models" / "face_detection_yunet_2023mar.onnx"


@lru_cache(maxsize=1)
def _detector():
    """Return (kind, object). Cached — building a cascade per clip is wasteful."""
    import cv2

    if YUNET_PATH.exists():
        try:
            det = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.6, 0.3, 5000)
            log.info("face detector: YuNet")
            return ("yunet", det)
        except Exception as exc:  # noqa: BLE001
            log.warning("YuNet load failed (%s), falling back to Haar", exc)

    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise RuntimeError("Haar cascade not found in this OpenCV build")
    log.info("face detector: Haar cascade")
    return ("haar", cascade)


def _detect_center(frame) -> float | None:
    """Largest face in the frame → x center as a 0..1 fraction. None if no face."""
    import cv2

    kind, det = _detector()
    h, w = frame.shape[:2]

    if kind == "yunet":
        det.setInputSize((w, h))
        _, faces = det.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: f[2] * f[3])
        return float(best[0] + best[2] / 2) / w

    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    boxes = det.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=6, minSize=(int(h * 0.08), int(h * 0.08))
    )
    if len(boxes) == 0:
        return None
    x, _, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
    return float(x + bw / 2) / w


def _sample_face_centers(video: Path, start: float, end: float) -> list[tuple[float, float]]:
    """Return [(t_relative, x_center_normalised)] for the clip window."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []

    points: list[tuple[float, float]] = []
    step = 1.0 / SAMPLE_FPS
    t = start
    try:
        while t < end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            center = _detect_center(frame)
            if center is not None:
                points.append((t - start, center))
            t += step
    finally:
        cap.release()
    return points


def _smooth(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    value = points[0][1]
    for t, x in points:
        value = EMA_ALPHA * x + (1 - EMA_ALPHA) * value
        out.append((t, value))
    return out


def _thin(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Keep only meaningful moves, capped at MAX_KEYFRAMES."""
    kept = [points[0]]
    for t, x in points[1:]:
        if abs(x - kept[-1][1]) >= MOVE_THRESHOLD:
            kept.append((t, x))
    if len(kept) > MAX_KEYFRAMES:
        stride = len(kept) / MAX_KEYFRAMES
        kept = [kept[int(i * stride)] for i in range(MAX_KEYFRAMES)]
    return kept


def _x_for(center: float, width: int, cw: int) -> int:
    x = int(center * width - cw / 2)
    return max(0, min(width - cw, x))


def build_crop(video: Path, start: float, end: float, width: int, height: int) -> dict:
    """Return {w, h, x_expr, y, mode} ready to drop into an ffmpeg crop filter."""
    cw, ch = crop_size(width, height, settings.aspect)
    y = max(0, (height - ch) // 2)
    centered = {"w": cw, "h": ch, "x_expr": str((width - cw) // 2), "y": y, "mode": "center"}

    if cw >= width:  # source is already portrait/square, nothing to choose
        return {**centered, "x_expr": "0", "mode": "full"}

    try:
        points = _sample_face_centers(video, start, end)
    except Exception as exc:  # noqa: BLE001 - never fail a render over tracking
        log.warning("face tracking unavailable (%s), centering", exc)
        return centered

    if not points:
        return centered

    smoothed = _thin(_smooth(points))
    if not settings.dynamic_crop or len(smoothed) == 1:
        xs = sorted(p[1] for p in smoothed)
        median = xs[len(xs) // 2]
        return {**centered, "x_expr": str(_x_for(median, width, cw)), "mode": "static"}

    # piecewise: x = if(lt(t,t1), x0, if(lt(t,t2), x1, ... ))
    expr = str(_x_for(smoothed[-1][1], width, cw))
    for t, center in reversed(smoothed[:-1]):
        expr = f"if(lt(t,{t:.2f}),{_x_for(center, width, cw)},{expr})"
    return {**centered, "x_expr": expr, "mode": "dynamic"}
