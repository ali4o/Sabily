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
"""

_lock = threading.Lock()


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
    # any job left "running" from a crash is not actually running
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='error', error='interrupted' WHERE status IN ('running','queued')"
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


class _JobLogHandler(logging.Handler):
    """Route sabily.* log records into the running job's buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        if _current is None:
            return
        buf = _buffers.setdefault(_current, [])
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
    buf = _buffers.get(job_id, [])
    after = max(0, min(after, len(buf)))
    return buf[after:], len(buf)


def _loop(handler: Callable[[str], None]) -> None:
    global _current
    while True:
        job_id = _q.get()
        _current = job_id
        _buffers.setdefault(job_id, [])
        try:
            handler(job_id)
        except Exception as exc:  # noqa: BLE001 - worker must never die
            log.exception("job %s failed", job_id)
            update_job(job_id, status="error", error=str(exc)[:500], stage="فشل")
        finally:
            _current = None
            _q.task_done()


def start_worker(handler: Callable[[str], None]) -> None:
    global _worker
    if _worker and _worker.is_alive():
        return
    _worker = threading.Thread(target=_loop, args=(handler,), daemon=True, name="sabily-worker")
    _worker.start()


def enqueue(job_id: str) -> None:
    _q.put(job_id)
