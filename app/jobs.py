"""Job store + single-worker queue.

One job at a time on purpose: the machine has 4GB VRAM, and running two
transcriptions in parallel is the fastest way to hit OOM.
"""

import json
import logging
import queue
import sqlite3
import threading
import time
import uuid
from typing import Any, Callable, Optional

from app.config import settings

log = logging.getLogger("sabily.jobs")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    status       TEXT NOT NULL,           -- queued|running|done|error|cancelled
    stage        TEXT DEFAULT '',
    progress     INTEGER DEFAULT 0,
    title        TEXT DEFAULT '',
    error        TEXT DEFAULT '',
    options      TEXT DEFAULT '{}',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS clips (
    id           TEXT PRIMARY KEY,
    job_id       TEXT NOT NULL,
    idx          INTEGER NOT NULL,
    start_sec    REAL NOT NULL,
    end_sec      REAL NOT NULL,
    score        REAL DEFAULT 0,
    title        TEXT DEFAULT '',
    caption      TEXT DEFAULT '',
    source_url   TEXT DEFAULT '',
    video_path   TEXT DEFAULT '',
    meta         TEXT DEFAULT '{}',
    created_at   REAL NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_clips_job ON clips(job_id);
CREATE TABLE IF NOT EXISTS render_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id       TEXT NOT NULL,
    quality      TEXT DEFAULT '',
    seconds      REAL DEFAULT 0,
    ms           INTEGER DEFAULT 0,
    mb           REAL DEFAULT 0,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_render_job ON render_log(job_id);
"""

_lock = threading.Lock()

_JOB_COLS = {"status", "stage", "progress", "title", "error", "options", "updated_at"}
_CLIP_COLS = {"title", "caption", "score", "video_path", "meta", "start_sec", "end_sec", "idx", "source_url"}


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    settings.ensure_dirs()
    with connect() as conn:
        conn.executescript(SCHEMA)
    # any job left "running" from a crash is not actually running.
    # "queued" jobs stay queued so they still run after a restart.
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='error', error='interrupted' WHERE status='running'"
        )
        conn.commit()


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #

def create_job(url: str, options: dict[str, Any] | None = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    with _lock, connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id,url,status,stage,progress,options,created_at,updated_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (job_id, url, "queued", "في الانتظار", 0, json.dumps(options or {}), now, now),
        )
        conn.commit()
    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    unknown = set(fields) - _JOB_COLS
    if unknown:
        raise ValueError(f"unknown field: {sorted(unknown)}")
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    with _lock, connect() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))
        conn.commit()


def add_clip(job_id: str, clip: dict[str, Any]) -> str:
    clip_id = uuid.uuid4().hex[:12]
    with _lock, connect() as conn:
        conn.execute(
            "INSERT INTO clips (id,job_id,idx,start_sec,end_sec,score,title,caption,"
            "source_url,video_path,meta,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                clip_id,
                job_id,
                clip["idx"],
                clip["start_sec"],
                clip["end_sec"],
                clip.get("score", 0),
                clip.get("title", ""),
                clip.get("caption", ""),
                clip.get("source_url", ""),
                clip.get("video_path", ""),
                json.dumps(clip.get("meta", {}), ensure_ascii=False),
                time.time(),
            ),
        )
        conn.commit()
    return clip_id


def get_clip(clip_id: str) -> Optional[dict[str, Any]]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["meta"] = json.loads(d.get("meta") or "{}")
    return d


def update_clip(clip_id: str, **fields: Any) -> None:
    if "meta" in fields and isinstance(fields["meta"], dict):
        fields["meta"] = json.dumps(fields["meta"], ensure_ascii=False)
    if not fields:
        return
    unknown = set(fields) - _CLIP_COLS
    if unknown:
        raise ValueError(f"unknown field: {sorted(unknown)}")
    cols = ", ".join(f"{k}=?" for k in fields)
    with _lock, connect() as conn:
        conn.execute(f"UPDATE clips SET {cols} WHERE id=?", (*fields.values(), clip_id))
        conn.commit()


def delete_clip(clip_id: str) -> None:
    with _lock, connect() as conn:
        conn.execute("DELETE FROM clips WHERE id=?", (clip_id,))
        conn.commit()


def delete_all_clips() -> None:
    with _lock, connect() as conn:
        conn.execute("DELETE FROM clips")
        conn.commit()


def delete_clips_by_job(job_id: str) -> None:
    with _lock, connect() as conn:
        conn.execute("DELETE FROM clips WHERE job_id=?", (job_id,))
        conn.commit()


def cancel_job(job_id: str) -> None:
    update_job(job_id, status="cancelled", stage="ملغي")


def log_render(job_id: str, quality: str, seconds: float, ms: int, mb: float) -> None:
    """One row per rendered clip: quality techniques comparison log."""
    with _lock, connect() as conn:
        conn.execute(
            "INSERT INTO render_log (job_id,quality,seconds,ms,mb,created_at)"
            " VALUES (?,?,?,?,?,?)",
            (job_id, quality, seconds, ms, mb, time.time()),
        )
        conn.commit()


def list_renders(limit: int = 20) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM render_log ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def cleanup_old_sources(days: int) -> int:
    """Delete work/<job> source dirs for finished jobs older than N days.

    Outputs stay. 0/negative disables. Job ids are validated before any
    deletion — never trust a path built from DB text blindly.
    """
    import re

    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    with connect() as conn:
        rows = conn.execute(
            "SELECT id FROM jobs WHERE status IN ('done','error','cancelled')"
            " AND updated_at < ?", (cutoff,)).fetchall()
    base = settings.work_dir.resolve()
    for r in rows:
        jid = r["id"]
        if not re.fullmatch(r"[a-f0-9]{12}", jid or ""):
            continue
        target = (settings.work_dir / jid).resolve()
        try:
            if target.parent == base and target.is_dir():
                import shutil

                shutil.rmtree(target, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    if removed:
        log.info("cleanup removed %d old source dirs", removed)
    return removed


def sweep_temp(audio_days: int = 7, bumper_days: int = 30) -> dict[str, int]:
    """Delete regenerable temp files that otherwise creep forever.

    - work/<job>/audio_<clip>.m4a: on-demand listening cuts, re-cut in ms.
    - work/<job>/bumper_*.mp4: branded cards, rebuilt when missing.
    Deliverables (outputs/), sources (needed for re-render) and transcript
    caches are never touched here.
    """
    import time as _time

    now = _time.time()
    done = {"audio": 0, "bumper": 0}
    try:
        base = settings.work_dir.resolve()
    except OSError:
        return done
    if not base.is_dir():
        return done
    jobs_root = base.resolve()
    for path in list(base.rglob("audio_*.m4a")) + list(base.rglob("bumper_*.mp4")):
        try:
            if not path.is_file() or not path.resolve().is_relative_to(jobs_root):
                continue
            age_days = (now - path.stat().st_mtime) / 86400
            limit = bumper_days if path.name.startswith("bumper_") else audio_days
            if age_days > limit:
                path.unlink(missing_ok=True)
                done["audio" if limit == audio_days else "bumper"] += 1
        except OSError:
            continue
    if sum(done.values()):
        log.info("swept temp files: %s", done)
    return done


# --------------------------------------------------------------------------- #
# reads
# --------------------------------------------------------------------------- #

def get_job(job_id: str) -> Optional[dict[str, Any]]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def list_clips(job_id: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM clips"
    args: tuple = ()
    if job_id:
        sql += " WHERE job_id=?"
        args = (job_id,)
    sql += " ORDER BY created_at DESC, idx ASC"
    with connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["meta"] = json.loads(d.get("meta") or "{}")
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #

_q: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None

# --------------------------------------------------------------------------- #
# log capture — so the dashboard can show what the pipeline is doing right now
# --------------------------------------------------------------------------- #

MAX_LINES = 500
_buffers: dict[str, list[str]] = {}
_current: str | None = None
_log_lock = threading.Lock()


class _JobLogHandler(logging.Handler):
    """Route sabily.* log records into the running job's buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        cur = _current
        if cur is None:
            return
        with _log_lock:
            buf = _buffers.setdefault(cur, [])
            try:
                stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
                buf.append(f"{stamp}  {record.levelname[:4]:<4} {record.getMessage()}")
            except Exception:  # noqa: BLE001 - logging must never raise
                return
            if len(buf) > MAX_LINES:
                del buf[: len(buf) - MAX_LINES]


def attach_log_capture() -> None:
    root = logging.getLogger()
    # the handler's own level is not enough: a logger whose effective level is
    # WARNING never hands INFO records to any handler.
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    if not any(isinstance(h, _JobLogHandler) for h in root.handlers):
        root.addHandler(_JobLogHandler(level=logging.INFO))


def log_lines(job_id: str, after: int = 0) -> tuple[list[str], int]:
    with _log_lock:
        buf = list(_buffers.get(job_id, []))
    after = max(0, min(after, len(buf)))
    return buf[after:], len(buf)


def _loop(handler: Callable[[str], None]) -> None:
    global _current
    while True:
        job_id = _q.get()
        _current = job_id
        with _log_lock:
            _buffers.setdefault(job_id, [])
        try:
            handler(job_id)
        except Exception as exc:  # noqa: BLE001 - worker must never die
            log.exception("job %s failed", job_id)
            update_job(job_id, status="error", error=str(exc)[:500], stage="فشل")
        finally:
            _current = None
            _q.task_done()
            # buffers are in-memory only (volatile across restarts); cap them
            # so a long-lived server never grows without bound.
            with _log_lock:
                if len(_buffers) > 20:
                    for k in list(_buffers):
                        if k != job_id:
                            _buffers.pop(k, None)
                            break


def start_worker(handler: Callable[[str], None]) -> None:
    global _worker
    if _worker and _worker.is_alive():
        return
    _worker = threading.Thread(target=_loop, args=(handler,), daemon=True, name="sabily-worker")
    _worker.start()


def enqueue(job_id: str) -> None:
    _q.put(job_id)
