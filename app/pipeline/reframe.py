"""Stage 5 — decide where to crop so the speaker stays in frame.

Detection runs on CPU with OpenCV only (no mediapipe/jax): YuNet if its
ONNX model is present in data/models/, else Haar frontal + profile
cascades (the profile pass catches side faces the frontal pass misses).

Samples a few frames per second and produces an ffmpeg crop expression
rather than writing frames to disk. Brief look-aways are bridged
(FACE_HOLD_SEC); when too few frames contain a face the whole clip
renders as a full-frame fit over a blurred fill (mode "fit") instead
of a blind center slice.
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


def _yunet_path() -> Path:
    return settings.work_dir / "models" / "face_detection_yunet_2023mar.onnx"


@lru_cache(maxsize=1)
def _detector():
    """Return (kind, object). Cached — building a cascade per clip is wasteful."""
    import cv2

    yunet = _yunet_path()
    if yunet.exists():
        try:
            det = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), 0.6, 0.3, 5000)
            log.info("face detector: YuNet")
            return ("yunet", det)
        except Exception as exc:  # noqa: BLE001
            log.warning("YuNet load failed (%s), falling back to Haar", exc)

    frontal = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    if frontal.empty():
        raise RuntimeError("Haar cascade not found in this OpenCV build")
    profile = None
    if settings.face_profile:
        profile = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"))
        if profile.empty():
            profile = None  # side-view pass simply stays off
    log.info("face detector: Haar frontal%s", " + profile" if profile else "")
    return ("haar", (frontal, profile))


def _haar_boxes(det, gray, h: int):
    frontal, profile = det
    boxes = frontal.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=6, minSize=(int(h * 0.08), int(h * 0.08))
    )
    if len(boxes) or profile is None:
        return boxes
    # second opinion for side faces: the frame and its mirror
    import cv2

    for img, mirror in ((gray, False), (cv2.flip(gray, 1), True)):
        found = profile.detectMultiScale(
            img, scaleFactor=1.15, minNeighbors=6,
            minSize=(int(h * 0.08), int(h * 0.08)))
        if len(found):
            if mirror:
                w = gray.shape[1]
                found = [(w - x - bw, y, bw, bh) for x, y, bw, bh in found]
            return found
    return boxes


def _pick_best(cands: list[tuple[float, float]]) -> float | None:
    """Pick one face: largest, biased toward frame center.

    Single-speaker podcasts usually center the host while posters or
    guests sit at the edges — pure largest-area often locks onto a
    background poster. FACE_CENTER_BIAS (0..0.9) discounts edge faces.
    """
    if not cands:
        return None
    try:
        bias = min(0.9, max(0.0, float(settings.face_center_bias)))
    except (TypeError, ValueError):
        bias = 0.0
    best, best_key = cands[0][0], -1.0
    for cx, area in cands:
        off = min(1.0, abs(cx - 0.5) * 2.0)
        key = area * (1.0 - bias * off)
        if key > best_key:
            best, best_key = cx, key
    return best


def _detect_center(frame) -> float | None:
    """Best face in the frame → x center as a 0..1 fraction. None if no face."""
    kind, det = _detector()
    h, w = frame.shape[:2]

    if kind == "yunet":
        det.setInputSize((w, h))
        _, faces = det.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        return _pick_best([(float(f[0] + f[2] / 2) / w, float(f[2] * f[3]))
                           for f in faces])

    import cv2

    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    boxes = _haar_boxes(det, gray, h)
    if len(boxes) == 0:
        return None
    return _pick_best([(float(x + bw / 2) / w, float(bw * bh))
                       for x, _, bw, bh in boxes])


def _sample_face_centers(
    video: Path, start: float, end: float
) -> list[tuple[float, float | None]]:
    """One entry per sample: (t_relative, x_center_normalised | None).

    Misses stay in the list as None so the caller can measure coverage
    and bridge short gaps instead of going blind between detections.
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []

    fps = max(0.5, settings.face_sample_fps)
    points: list[tuple[float, float | None]] = []
    step = 1.0 / fps
    t = start
    try:
        while t < end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            try:
                center = _detect_center(frame)
            except Exception:  # noqa: BLE001 - one bad frame must not kill sampling
                log.debug("detection failed on one frame", exc_info=True)
                center = None
            points.append((t - start, center))
            t += step
    finally:
        cap.release()
    return points


def _fill_gaps(
    samples: list[tuple[float, float | None]], hold_sec: float
) -> tuple[list[tuple[float, float]], float]:
    """Bridge short misses with the last known position.

    Returns (filled points, coverage 0..1). A miss longer than hold_sec
    breaks the track there instead of dragging a stale position along.
    """
    if not samples:
        return [], 0.0
    hits = sum(1 for _, x in samples if x is not None)
    filled: list[tuple[float, float]] = []
    last_x: float | None = None
    last_t = 0.0
    for t, x in samples:
        if x is not None:
            last_x, last_t = x, t
            filled.append((t, x))
        elif last_x is not None and t - last_t <= hold_sec:
            filled.append((t, last_x))
    return filled, hits / len(samples)


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


def manual_crop(width: int, height: int, aspect: str | None,
                x_center: float, scale: float = 1.0) -> dict:
    """User-locked crop from the preview popup: guaranteed, no sampling.

    x_center 0..1 across the frame, scale 0.4..1 of the max fitting box.
    Always returns mode "manual" — the video IS cropped, never fit/blind.
    """
    from app.pipeline.media import crop_size as _cs

    asp = aspect if aspect in ("9:16", "1:1") else "9:16"
    cw, ch = _cs(width, height, asp)
    s = min(1.0, max(0.4, float(scale)))
    w = max(2, int(cw * s) // 2 * 2)
    ratio = 1.0 if asp == "1:1" else 9 / 16
    h = max(2, min(ch, int(w / ratio)) // 2 * 2)
    cx = min(1.0, max(0.0, float(x_center)))
    x = max(0, min(width - w, int(cx * width - w / 2)))
    y = max(0, (height - h) // 2)
    return {"w": w, "h": h, "x_expr": str(x), "y": y, "mode": "manual"}


def build_crop(video: Path, start: float, end: float, width: int, height: int,
               aspect: str | None = None) -> dict:
    """Return {w, h, x_expr, y, mode} ready to drop into an ffmpeg crop filter.

    Modes: full (source already fits) | static | dynamic (face tracked) |
    center (tracking failed, blind middle slice) | fit (too few faces:
    the whole frame fitted over a blurred fill — never a blind slice).
    """
    cw, ch = crop_size(width, height, aspect or settings.aspect)
    y = max(0, (height - ch) // 2)
    centered = {"w": cw, "h": ch, "x_expr": str((width - cw) // 2), "y": y, "mode": "center"}

    if cw >= width:  # source is already portrait/square, nothing to choose
        return {**centered, "x_expr": "0", "mode": "full"}

    try:
        samples = _sample_face_centers(video, start, end)
    except Exception as exc:  # noqa: BLE001 - never fail a render over tracking
        log.warning("face tracking unavailable (%s), centering", exc)
        return centered

    points, coverage = _fill_gaps(samples, settings.face_hold_sec)
    if coverage < settings.face_min_coverage:
        log.info("face coverage %.0f%% < %.0f%% — full-frame fit",
                 coverage * 100, settings.face_min_coverage * 100)
        return {**centered, "x_expr": "0", "mode": "fit"}

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
    return {**centered, "x_expr": expr, "mode": "dynamic",
            "zoom_times": [round(t, 2) for t, _ in smoothed]}
