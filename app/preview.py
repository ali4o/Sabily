"""Crop-calibration previews: one real frame before a job starts.

POST /api/preview {url} downloads the source once (same pipeline fetcher),
probes it, extracts a SINGLE middle frame and suggests a starting crop box.
The dashboard popup lets the user measure the cut on that real frame;
POST /api/jobs {crop_x, crop_scale} then locks that crop.

Sessions live in memory + data/previews/<id>/ on disk; entries older
than a day are swept at startup.
"""

import json
import logging
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import settings

log = logging.getLogger("sabily.preview")

FRACTIONS = (0.08, 0.24, 0.40, 0.56, 0.72, 0.88)
# the preview spread: 5 real frames across the timeline (past the intro,
# covering the body of the video). n=1 callers get the middle one only.
PREVIEW_FRACTIONS = (0.12, 0.30, 0.48, 0.66, 0.84)
MIDDLE_FRACTION = 0.48
FRAME_W = 640
MAX_AGE_SEC = 24 * 3600
MAX_SESSIONS = 5  # newest preview sessions kept; older ones (with their
# ~hundreds-of-MB source videos) are deleted to stop disk creep.
MAX_SAMPLES = 3  # sample renders kept per session; older ones pruned.
SAMPLE_SEC = 3.0  # true-preview clip length: same ffmpeg chain, tiny wait
SAMPLE_TEXT = "نستخدم تقنية AP36 في خط الإنتاج - الجزء 3"

_lock = threading.Lock()
_sessions: dict[str, dict[str, Any]] = {}


def preview_dir(pid: str | None = None) -> Path:
    base = settings.work_dir / "previews"
    return base / pid if pid else base


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _extract_frames(video: Path, duration: float, outdir: Path, n: int = 5) -> list[str]:
    """Five real frames spread over the timeline by default (n=1 gives
    the middle one only). Small jpgs — fast to extract, enough to judge
    the cut before committing to a full job."""
    names = []
    if n == 1:
        fracs = (MIDDLE_FRACTION,)
    else:
        fracs = PREVIEW_FRACTIONS[:max(1, min(n, len(PREVIEW_FRACTIONS)))]
    for i, f in enumerate(fracs):
        name = f"frame_{i:02d}.jpg"
        cmd = [settings.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{max(0.0, duration * f):.1f}", "-i", str(video),
               "-frames:v", "1", "-vf", f"scale={FRAME_W}:-1",
               "-q:v", "4", str(outdir / name)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode == 0 and (outdir / name).exists():
            names.append(name)
    return names


def _suggest(video: Path) -> dict[str, float]:
    """Starting box: face center on the middle frame, else frame center."""
    try:
        import cv2

        from app.pipeline import reframe

        cap = cv2.VideoCapture(str(video))
        if cap.isOpened():
            try:
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if n > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
                ok, frame = cap.read()
                if ok and frame is not None:
                    cx = reframe._detect_center(frame)
                    if cx is not None:
                        return {"x": round(cx, 3), "scale": 1.0, "face": True}
            finally:
                cap.release()
    except Exception as exc:  # noqa: BLE001 - suggestion is best-effort
        log.debug("preview suggest failed (%s)", exc)
    return {"x": 0.5, "scale": 1.0, "face": False}


def _run(pid: str, url: str, quality: str | None) -> None:
    from app.pipeline import download
    from app.pipeline.media import probe

    d = preview_dir(pid)
    try:
        src = download.fetch(url, d, quality)
        info = probe(src.video_path)
        names = _extract_frames(src.video_path, src.duration or 0.0, d)
        if not names:
            raise RuntimeError("تعذّر استخراج فريمات المعاينة")
        suggest = _suggest(src.video_path)
        tag = download.as_hashtag(src.channel)
        with _lock:
            st = _sessions.get(pid, {})
            st.update(status="ready", width=info["width"], height=info["height"],
                      duration=round(src.duration or 0.0, 1),
                      frames=[f"/preview/{pid}/{n}" for n in names],
                      suggest=suggest,
                      title=src.title, error="",
                      # NOTE: internal only — stripped from the API response
                      # (see main.get_preview). Needed for true-preview renders.
                      video=str(src.video_path), tag=tag)
            _sessions[pid] = st
        # NOTE 2026-09-12: sessions also live on disk — a server restart
        # wipes memory, and without this every pending preview/modal would
        # poll a 404 forever with a broken image and no message.
        meta = {"width": info["width"], "height": info["height"],
                "duration": round(src.duration or 0.0, 1),
                "frames": names, "suggest": suggest,
                "title": src.title or "", "video": str(src.video_path),
                "tag": tag}
        try:
            (d / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        except OSError:
            log.warning("preview %s: could not write meta.json", pid)
            _sessions[pid] = st
    except Exception as exc:  # noqa: BLE001 - surfaced via polling
        log.warning("preview %s failed (%s)", pid, exc)
        with _lock:
            st = _sessions.get(pid, {})
            st.update(status="error", error=str(exc)[:300])
            _sessions[pid] = st


def start_preview(url: str, quality: str | None = None) -> str:
    """Queue a preview session in a daemon thread; poll get_preview()."""
    cleanup()  # opportunistic: the server may run for weeks without restart
    pid = _new_id()
    preview_dir(pid).mkdir(parents=True, exist_ok=True)
    with _lock:
        _sessions[pid] = {"status": "working", "url": url, "created_at": time.time()}
        # cap the session count — each one holds a full source video.
        if len(_sessions) > MAX_SESSIONS:
            olds = sorted(_sessions.items(), key=lambda kv: kv[1].get("created_at", 0))
            for old_pid, _ in olds[:len(_sessions) - MAX_SESSIONS]:
                _sessions.pop(old_pid, None)
                import shutil as _sh
                _sh.rmtree(preview_dir(old_pid), ignore_errors=True)
    th = threading.Thread(target=_run, args=(pid, url, quality), daemon=True,
                          name=f"sabily-preview-{pid}")
    th.start()
    return pid


def _load_from_disk(pid: str) -> dict[str, Any] | None:
    """Rebuild a ready session from data/previews/<pid>/meta.json.

    Lets previews survive a server restart: frames + source video are
    already on disk, only the in-memory entry is gone.
    """
    d = preview_dir(pid)
    try:
        raw = (d / "meta.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        names = [n for n in (data.get("frames") or [])
                 if isinstance(n, str) and (d / Path(n).name).is_file()]
        if not names:
            return None
        st = {"status": "ready", "url": str(data.get("url") or ""),
              "width": int(data["width"]), "height": int(data["height"]),
              "duration": float(data.get("duration") or 0.0),
              "frames": [f"/preview/{pid}/{Path(n).name}" for n in names],
              "suggest": data.get("suggest") or {"x": 0.5, "scale": 1.0},
              "title": str(data.get("title") or ""), "error": "",
              "video": str(data.get("video") or ""),
              "tag": str(data.get("tag") or ""),
              "created_at": (d / "meta.json").stat().st_mtime}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    with _lock:
        _sessions[pid] = st
    return dict(st)


def get_preview(pid: str) -> dict[str, Any] | None:
    with _lock:
        st = _sessions.get(pid)
        if st:
            return dict(st)
    return _load_from_disk(pid)


def render_sample(
    pid: str,
    quality: str | None = None,
    aspect: str | None = None,
    crop_x: float | None = None,
    crop_scale: float = 1.0,
    brand_pos: str | None = None,
    source_pos: str | None = None,
    show_brand: bool | None = None,
    show_source: bool | None = None,
    burn: bool | None = None,
    frame_scale: float | None = None,
    fill_mode: str | None = None,
    sample_text: str = "",
) -> dict[str, Any]:
    """Render a short TRUE preview clip with the real production chain.

    Same functions the job runner calls, in the same order: manual_crop
    (locked) or build_crop (auto face tracking) → write_ass (real Arabic
    shaping path, real logo/hashtag styles) → render_clip (same ffmpeg
    filters, encoder fallback, verify_output). What the <video> plays is
    therefore what a job with the same options produces for that segment.

    `None` flags fall back to the server .env defaults — exactly like
    runner.process — so parity holds in both directions.
    """
    from app.config import normalize_quality, resolve_target
    from app.pipeline import reframe, render, subtitle
    from app.pipeline.media import probe
    from app.pipeline.transcribe import Word

    with _lock:
        st = _sessions.get(pid)
        st = dict(st) if st else None
    if st is None:
        st = _load_from_disk(pid)  # survive restarts (frames stay on disk)
    if not st:
        raise FileNotFoundError("لا توجد معاينة بهذا المعرف")
    if st.get("status") != "ready":
        raise RuntimeError("المعاينة غير جاهزة بعد — انتظر اكتمال الفريمات")
    video = Path(str(st.get("video") or ""))
    try:
        # meta.json is disk-controlled: the source must stay inside the
        # work tree (same bar rerender uses) — never an arbitrary path.
        if not video.resolve().is_relative_to(settings.work_dir.resolve()):
            raise RuntimeError("مسار الفيديو غير صالح — أعد بدء المعاينة")
    except RuntimeError:
        raise
    except Exception:  # noqa: BLE001 - resolve must not crash the endpoint
        raise RuntimeError("مسار الفيديو غير صالح — أعد بدء المعاينة")
    if not video.is_file():
        raise RuntimeError("الفيديو الأصلي غير موجود — أعد بدء المعاينة")

    asp = aspect if aspect in ("9:16", "1:1") else settings.aspect
    if asp not in ("9:16", "1:1"):
        asp = "9:16"
    qual = normalize_quality(quality, settings.quality)
    tw, th = resolve_target(qual, asp, settings.out_width, settings.out_height)
    w, h = int(st["width"]), int(st["height"])
    dur = float(st.get("duration") or 0.0)
    if dur <= 0:
        info = probe(video)
        dur, w, h = float(info.get("duration") or 0.0), info["width"], info["height"]
    if dur < 1.0:
        raise RuntimeError("الفيديو قصير جداً للمعاينة")

    span = min(SAMPLE_SEC, dur)
    t0 = min(max(0.0, dur * 0.4), max(0.0, dur - span))
    t1 = t0 + span

    if crop_x is None:
        # auto path — literally the runner's call (face track or fit)
        crop = reframe.build_crop(video, t0, t1, w, h, asp)
    else:
        crop = reframe.manual_crop(w, h, asp, crop_x, crop_scale)

    text = (sample_text or "").strip() or SAMPLE_TEXT
    toks = text.split()
    per = span / max(1, len(toks))
    words = [Word(start=t0 + i * per, end=t0 + (i + 1) * per, text=t)
             for i, t in enumerate(toks)]

    d = preview_dir(pid)
    with _lock:
        n = int(st.get("sample_n", 0)) + 1
        _sessions[pid]["sample_n"] = n
    ass = subtitle.write_ass(
        words, t0, t1, d / f"sample_{n:02d}.ass",
        source_tag=str(st.get("tag") or ""),
        target_size=(tw, th),
        show_brand=show_brand, show_source=show_source, burn=burn,
        brand_pos=brand_pos, source_pos=source_pos,
    )
    out = d / f"sample_{n:02d}.mp4"
    render.render_clip(video, out, t0, t1, crop, ass,
                       target_size=(tw, th),
                       brand_pos=brand_pos,
                       show_logo=show_brand, burn_subtitles=burn,
                       frame_scale=frame_scale, fill_mode=fill_mode)
    _prune_samples(d)
    got = probe(out)
    fill = "black" if str(fill_mode or "").lower() == "black" else "blur"
    return {
        "video": f"/preview/{pid}/{out.name}",
        "width": got["width"], "height": got["height"],
        "target_size": [tw, th],
        "start": round(t0, 2), "end": round(t1, 2),
        "quality": qual, "aspect": asp,
        "frame_scale": frame_scale if frame_scale else 1.0,
        "fill_mode": fill,
        "crop": {k: crop.get(k) for k in ("w", "h", "x_expr", "y", "mode")},
    }


def _prune_samples(d: Path, keep: int = MAX_SAMPLES) -> int:
    """Keep only the newest `keep` sample renders per session.

    Every true-preview render writes an mp4 (+ass); without pruning a
    tweaking user accumulates dozens of megabytes per session.
    """
    try:
        mps = sorted(d.glob("sample_*.mp4"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return 0
    removed = 0
    for old in mps[:-keep] if len(mps) > keep else []:
        try:
            old.unlink(missing_ok=True)
            old.with_suffix(".ass").unlink(missing_ok=True)
            removed += 1
        except OSError:
            continue
    return removed


def cleanup(max_age_sec: int = MAX_AGE_SEC) -> int:
    """Drop preview sessions (memory + disk) older than the window."""
    now = time.time()
    removed = 0
    with _lock:
        olds = [pid for pid, st in _sessions.items()
                if now - float(st.get("created_at", now)) > max_age_sec]
        for pid in olds:
            _sessions.pop(pid, None)
    import shutil

    base = preview_dir()
    if base.exists():
        for child in base.iterdir():
            try:
                if child.is_dir() and now - child.stat().st_mtime > max_age_sec:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
    return removed + len(olds)
