"""Sabily — local dashboard + API on http://127.0.0.1:6767

No auth: the server binds to localhost only and stores nothing personal.
If it is ever exposed beyond this machine, add auth first.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app import jobs
from app.config import settings
from app.pipeline.runner import process

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("sabily")

app = FastAPI(title="Sabily", version="0.1.0")

_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def _require_id(v: str) -> str:
    if not _ID_RE.fullmatch(v or ""):
        raise HTTPException(404, "لا يوجد job بهذا المعرف")
    return v


def _require_clip_id(v: str) -> str:
    if not _ID_RE.fullmatch(v or ""):
        raise HTTPException(404, "المقطع غير موجود")
    return v


class JobRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2000)
    clips: int = Field(default=0, ge=0, le=20)
    lang: str = Field(default="", max_length=5)
    aspect: str = Field(default="", max_length=8)
    quality: str = Field(default="", max_length=8)
    content_type: str = Field(default="", max_length=16)
    brand_watermark: bool | None = None
    show_source: bool | None = None
    burn_subtitles: bool | None = None
    intro_card: bool | None = None
    outro_card: bool | None = None
    crop_x: float | None = Field(default=None, ge=0.0, le=1.0)
    crop_scale: float | None = Field(default=None, ge=0.4, le=1.0)
    frame_scale: float | None = Field(default=None, ge=0.4, le=1.0)
    fill_mode: str | None = Field(default=None, pattern="^(blur|black)$")
    keep_source: bool | None = None


def _startup() -> None:
    if settings.host not in _LOOPBACK_HOSTS:
        log.error("HOST=%s is not loopback — refusing without auth (see main docstring)", settings.host)
        raise SystemExit(1)
    settings.ensure_dirs()
    jobs.init_db()
    jobs.attach_log_capture()
    jobs.start_worker(process)
    try:
        jobs.cleanup_old_sources(settings.cleanup_days)
    except Exception:  # noqa: BLE001 - housekeeping must not block startup
        log.warning("old-source cleanup failed", exc_info=True)
    try:
        from app import preview as _pv

        _pv.preview_dir().mkdir(parents=True, exist_ok=True)
        _pv.cleanup()
    except Exception:  # noqa: BLE001
        log.warning("preview cleanup failed", exc_info=True)
    try:
        swept = jobs.sweep_temp()
        if sum(swept.values()):
            log.info("startup temp sweep: %s", swept)
    except Exception:  # noqa: BLE001 - housekeeping must not block startup
        log.warning("temp sweep failed", exc_info=True)
    # _q is in-memory: re-queue jobs that stayed "queued" across a restart,
    # otherwise they would stall forever (init_db preserves them by design).
    try:
        for j in jobs.list_jobs(1000):
            if j.get("status") == "queued":
                jobs.enqueue(j["id"])
    except Exception:  # noqa: BLE001 - startup must not die on re-queue
        log.warning("could not re-queue pending jobs", exc_info=True)
    log.info("Sabily ready on http://%s:%s", settings.host, settings.port)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup()
    yield


app.router.lifespan_context = lifespan


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

@app.post("/api/jobs")
def create_job(req: JobRequest) -> dict:
    if not req.url.startswith(("http://", "https://")):
        raise HTTPException(400, "الرابط غير صالح")
    d = req.model_dump()
    options: dict = {}
    if d.get("clips"):
        options["clips"] = d["clips"]
    if d.get("lang"):
        options["lang"] = d["lang"]
    if d.get("aspect"):
        options["aspect"] = d["aspect"]
    if d.get("quality"):
        from app.config import QUALITY_ALIASES
        q = QUALITY_ALIASES.get(str(d["quality"]).strip().upper(), "")
        if q:
            options["quality"] = q
    if d.get("content_type"):
        from app.pipeline.score import normalize_content_type
        ct = normalize_content_type(d["content_type"])
        # unknown values are ignored (same contract as quality)
        if ct in ("auto", "lecture", "lesson", "podcast", "interview"):
            options["content_type"] = ct
    # None = not specified → fall back to settings.keep_source in runner.
    # Explicit True/False is preserved (fixes falsy-drop where False was lost).
    if d.get("keep_source") is not None:
        options["keep_source"] = bool(d["keep_source"])
    # appearance flags per job (absent = global .env default in runner)
    for key in ("brand_watermark", "show_source", "burn_subtitles",
                "intro_card", "outro_card"):
        if d.get(key) is not None:
            options[key] = bool(d[key])
    # manual crop lock from the preview popup (absent = auto tracking)
    if d.get("crop_x") is not None:
        options["crop_lock"] = {
            "x": float(d["crop_x"]),
            "scale": float(d.get("crop_scale") or 1.0),
        }
    # shrink-in-output with edge fill (absent/1.0 = full-bleed)
    if d.get("frame_scale") is not None:
        options["frame_scale"] = float(d["frame_scale"])
    if d.get("fill_mode") is not None:
        options["fill_mode"] = d["fill_mode"]
    job_id = jobs.create_job(req.url, options)
    jobs.enqueue(job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs")
def get_jobs(limit: int = Query(20, ge=1, le=100)) -> list[dict]:
    return jobs.list_jobs(limit)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    _require_id(job_id)
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "لا يوجد job بهذا المعرف")
    job["clips"] = jobs.list_clips(job_id)
    return job


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    """Server-sent events: poll the DB and push status to the dashboard."""
    _require_id(job_id)

    async def stream():
        last, cursor = None, 0
        while True:
            job = jobs.get_job(job_id)
            if not job:
                yield 'event: error\ndata: {"error":"not found"}\n\n'
                return
            lines, cursor = jobs.log_lines(job_id, cursor)
            payload = {
                "status": job["status"],
                "stage": job["stage"],
                "progress": job["progress"],
                "error": job["error"],
            }
            if lines:
                yield f"data: {json.dumps({**payload, 'log': lines}, ensure_ascii=False)}\n\n"
                last = payload
            elif payload != last:
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last = payload
            if job["status"] in ("done", "error", "cancelled"):
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.7)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/log")
def get_log(job_id: str, after: int = 0) -> dict:
    _require_id(job_id)
    lines, cursor = jobs.log_lines(job_id, after)
    return {"lines": lines, "cursor": cursor}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Cancel a queued/running job. Finished jobs report already-done."""
    _require_id(job_id)
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "لا يوجد job بهذا المعرف")
    if job.get("status") in ("done", "error", "cancelled"):
        return {"ok": True, "already": job.get("status")}
    jobs.cancel_job(job_id)
    return {"ok": True, "status": "cancelled"}


@app.get("/api/jobs/{job_id}/zip")
def download_zip(job_id: str) -> FileResponse:
    """All clips of one job (mp4 + json) as a single zip download."""
    import zipfile

    _require_id(job_id)
    clips = jobs.list_clips(job_id)
    if not clips:
        raise HTTPException(404, "لا توجد مقاطع لهذه المهمة")
    out = (settings.outputs_dir / job_id / "sabily_all.zip").resolve()
    if not out.is_relative_to(settings.outputs_dir.resolve()):
        raise HTTPException(400, "مسار غير صالح")
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for c in clips:
                for suffix in (".mp4", ".json"):
                    name = Path(str(c.get("video_path", ""))).with_suffix(suffix).name
                    if not name or name in ("", ".mp4", ".json"):
                        continue
                    src = (settings.outputs_dir / job_id / name).resolve()
                    if src.is_relative_to((settings.outputs_dir / job_id).resolve()) \
                            and src.is_file():
                        zf.write(src, name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"تعذّر بناء ZIP: {exc}") from exc
    return FileResponse(out, filename=f"sabily_{job_id}.zip")


@app.get("/api/clips/{clip_id}/audio")
def clip_audio(clip_id: str) -> FileResponse:
    """Audio-only cut of one clip: quick listening check without video."""
    import subprocess

    _require_clip_id(clip_id)
    clip = jobs.get_clip(clip_id)
    if not clip:
        raise HTTPException(404, "المقطع غير موجود")
    meta = clip["meta"]
    raw_source = meta.get("source_video", "")
    source = Path(raw_source) if raw_source else Path("")
    if not source.is_absolute():
        source = (settings.base_dir / source).resolve()
    else:
        source = source.resolve()
    if not raw_source or not source.is_file():
        raise HTTPException(409, "الفيديو الأصلي حُذف — لا توجد معاينة صوتية.")
    try:
        if not source.is_relative_to(settings.work_dir.resolve()):
            raise HTTPException(400, "مسار غير صالح")
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "مسار غير صالح")
    work = (settings.work_dir / clip["job_id"]).resolve()
    if not work.is_relative_to(settings.work_dir.resolve()):
        raise HTTPException(400, "مسار غير صالح")
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"audio_{clip_id}.m4a"
    if not out.exists():
        cmd = [settings.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{max(0.0, clip['start_sec']):.3f}",
               "-to", f"{clip['end_sec']:.3f}", "-i", str(source),
               "-vn", "-c:a", "aac", "-b:a", "128k", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise HTTPException(500, "تعذّر قص الصوت")
    return FileResponse(out, media_type="audio/mp4", filename=f"clip_preview.m4a")


@app.get("/api/clips")
def get_clips(job_id: str | None = None) -> list[dict]:
    if job_id is not None and not _ID_RE.fullmatch(job_id):
        return []
    return jobs.list_clips(job_id)


class ClipLine(BaseModel):
    model_config = ConfigDict(extra="allow")  # preserve kara/en timings on edit
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(max_length=200)

    @model_validator(mode="after")
    def _check_order(self) -> "ClipLine":
        if self.end <= self.start:
            raise ValueError("end must be > start")
        return self


class ClipEdit(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    caption: str | None = Field(default=None, max_length=4000)
    brand_text: str | None = Field(default=None, max_length=40)
    brand_pos: str | None = Field(default=None, pattern="^(tl|tc|tr|bl|bc|br)$")
    brand_watermark: bool | None = None
    source_tag: str | None = Field(default=None, max_length=80)
    source_pos: str | None = Field(default=None, pattern="^(tl|tc|tr|bl|bc|br)$")
    show_source: bool | None = None
    burn_subtitles: bool | None = None
    frame_scale: float | None = Field(default=None, ge=0.4, le=1.0)
    fill_mode: str | None = Field(default=None, pattern="^(blur|black)$")
    lines: list[ClipLine] | None = Field(default=None, max_length=200)


@app.patch("/api/clips/{clip_id}")
def edit_clip(clip_id: str, edit: ClipEdit) -> dict:
    """Metadata-only edit. Instant — no re-render, no ffmpeg."""
    _require_clip_id(clip_id)
    clip = jobs.get_clip(clip_id)
    if not clip:
        raise HTTPException(404, "المقطع غير موجود")
    meta = clip["meta"]
    fields: dict = {}
    if edit.title is not None:
        fields["title"] = edit.title
    if edit.caption is not None:
        fields["caption"] = edit.caption
    for key in ("brand_text", "brand_pos", "source_tag", "source_pos",
                "brand_watermark", "show_source", "burn_subtitles",
                "frame_scale", "fill_mode"):
        val = getattr(edit, key)
        if val is not None:
            meta[key] = val
    learned: dict = {}
    if edit.lines is not None:
        from app.pipeline import normalize

        edit_lines = [line.model_dump() for line in edit.lines]
        learned = normalize.learn(meta.get("lines") or [], edit_lines)
        meta["lines"] = edit_lines
    fields["meta"] = meta
    jobs.update_clip(clip_id, **fields)
    result = jobs.get_clip(clip_id) or {}
    result["learned"] = learned
    return result


@app.post("/api/clips/{clip_id}/rerender")
def rerender_clip(clip_id: str) -> dict:
    """Rebuild the mp4 from the saved edits. Needs the source video on disk."""
    from app.pipeline import render, subtitle

    _require_clip_id(clip_id)
    clip = jobs.get_clip(clip_id)
    if not clip:
        raise HTTPException(404, "المقطع غير موجود")
    meta = clip["meta"]
    raw_source = meta.get("source_video", "")
    source = Path(raw_source) if raw_source else Path("")
    if not source.is_absolute():
        source = (settings.base_dir / source).resolve()
    else:
        source = source.resolve()
    if not raw_source or not source.is_file():
        raise HTTPException(
            409, "الفيديو الأصلي حُذف — لا يمكن إعادة الرندر. أعد معالجة الرابط."
        )
    # source must live under work_dir (defense-in-depth: meta is DB-controlled).
    try:
        if not source.is_relative_to(settings.work_dir.resolve()):
            raise HTTPException(400, "مسار غير صالح")
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 - resolve/is_relative_to must not crash rerender
        raise HTTPException(400, "مسار غير صالح")

    out_path = (settings.outputs_dir / clip["video_path"]).resolve()
    if not out_path.is_relative_to(settings.outputs_dir.resolve()):
        raise HTTPException(400, "مسار غير صالح")
    work = (settings.work_dir / clip["job_id"]).resolve()
    if not work.is_relative_to(settings.work_dir.resolve()):
        raise HTTPException(400, "مسار غير صالح")
    work.mkdir(parents=True, exist_ok=True)
    from app.config import normalize_quality, resolve_target
    asp = meta.get("aspect") or settings.aspect
    if asp not in ("9:16", "1:1"):
        asp = settings.aspect if settings.aspect in ("9:16", "1:1") else "9:16"
    qual = normalize_quality(meta.get("quality"), settings.quality)
    tw, th = resolve_target(qual, asp, settings.out_width, settings.out_height)
    # appearance flags stored per clip (old clips fall back to globals)
    show_brand = meta.get("brand_watermark") if "brand_watermark" in meta \
        else settings.brand_watermark
    show_source = meta.get("show_source") if "show_source" in meta \
        else settings.source_tag
    burn = meta.get("burn_subtitles") if "burn_subtitles" in meta \
        else settings.burn_subtitles
    meta_lines = meta.get("lines") or []
    en_lines = [str(r.get("en", "")) for r in meta_lines] or None
    crop = meta.get("crop") or {"w": tw, "h": th, "x_expr": "0",
                                "y": 0, "mode": "center"}
    ass = subtitle.write_ass(
        [], clip["start_sec"], clip["end_sec"], work / f"edit_{clip_id}.ass",
        source_tag=meta.get("source_tag", ""),
        lines=meta_lines,
        brand_pos=meta.get("brand_pos"),
        source_pos=meta.get("source_pos"),
        brand_text=meta.get("brand_text"),
        target_size=(tw, th),
        show_brand=show_brand, show_source=show_source, burn=burn,
        en_lines=en_lines, turns=meta.get("turns"),
    )
    try:
        render.render_clip(source, out_path, clip["start_sec"], clip["end_sec"],
                           crop, ass,
                           target_size=(tw, th),
                           brand_pos=meta.get("brand_pos"),
                           show_logo=show_brand, burn_subtitles=burn,
                           zoom_times=crop.get("zoom_times"),
                           frame_scale=meta.get("frame_scale"),
                           fill_mode=meta.get("fill_mode"),
                           bumpers={"intro": meta.get("intro_card", False),
                                    "outro": meta.get("outro_card", False),
                                    "title": meta.get("source_title", ""),
                                    "workdir": work})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"فشل الرندر: {exc}") from exc
    meta["rendered_at"] = time.time()
    jobs.update_clip(clip_id, meta=meta)
    return {"ok": True, "video_path": clip["video_path"]}


@app.delete("/api/jobs/{job_id}/source")
def delete_source(job_id: str) -> dict:
    """Free disk space. After this the job's clips can no longer be re-rendered."""
    _require_id(job_id)
    target = (settings.work_dir / job_id).resolve()
    base = settings.work_dir.resolve()
    if target.parent != base:
        raise HTTPException(400, "مسار غير صالح")
    shutil.rmtree(target, ignore_errors=True)
    return {"ok": True}


@app.delete("/api/clips/{clip_id}")
def remove_clip(clip_id: str) -> dict:
    _require_clip_id(clip_id)
    jobs.delete_clip(clip_id)
    return {"ok": True}


@app.delete("/api/clips")
def remove_all_clips(job_id: str | None = None) -> dict:
    if not job_id:
        raise HTTPException(400, "job_id مطلوب")
    _require_id(job_id)
    jobs.delete_clips_by_job(job_id)
    return {"ok": True}


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "whisper_model": settings.whisper_model,
        "device": settings.whisper_device,
        "quality": settings.quality,
        "target_size": list(settings.target_size),
    }


@app.get("/api/settings")
def get_settings() -> dict:
    """Editable .env-backed settings with input metadata for the UI."""
    from app import settings_store

    return settings_store.describe()


@app.put("/api/settings")
def put_settings(body: dict) -> dict:
    """Validate and persist whitelisted keys to .env (restart to apply)."""
    from app import settings_store

    if not isinstance(body, dict) or not body:
        raise HTTPException(400, "أرسل قاموساً من المفاتيح والقيم")
    if len(body) > len(settings_store.SCHEMA):
        raise HTTPException(400, "طلب كبير")
    try:
        return settings_store.save(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/stats")
def get_stats() -> dict:
    """Dashboard numbers: jobs, clips, rendered time, disk usage."""
    all_jobs = jobs.list_jobs(1000)
    by_status: dict[str, int] = {}
    for j in all_jobs:
        by_status[j.get("status", "?")] = by_status.get(j.get("status", "?"), 0) + 1
    clips = jobs.list_clips()
    rendered_sec = 0.0
    for c in clips:
        try:
            rendered_sec += max(0.0, float(c.get("end_sec", 0)) - float(c.get("start_sec", 0)))
        except (TypeError, ValueError):
            continue

    def _mb(p: Path) -> float:
        try:
            if p.is_file():
                return p.stat().st_size / 1e6
            if p.is_dir():
                return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
        except OSError:
            pass
        return 0.0

    recent = []
    for j in all_jobs[:10]:
        try:
            n = len(jobs.list_clips(j["id"]))
        except Exception:  # noqa: BLE001 - one bad job must not break stats
            n = 0
        recent.append({"id": j["id"], "title": j.get("title", ""),
                       "status": j.get("status", ""), "stage": j.get("stage", ""),
                       "error": (j.get("error") or "")[:120], "clips": n,
                       "created_at": j.get("created_at", 0)})
    try:
        renders = jobs.list_renders(10)
    except Exception:  # noqa: BLE001 - old DBs without the table still work
        renders = []
    return {
        "jobs_total": len(all_jobs),
        "by_status": by_status,
        "clips_total": len(clips),
        "rendered_min": round(rendered_sec / 60, 1),
        "renders": renders,
        "disk_mb": {"outputs": round(_mb(settings.outputs_dir), 1),
                    "work": round(_mb(settings.work_dir), 1),
                    "db": round(_mb(settings.db_path), 1)},
        "recent": recent,
    }


@app.post("/api/preview")
def start_preview(req: JobRequest) -> dict:
    """Fetch a video and extract sample frames for crop calibration.

    Returns immediately; poll GET /api/preview/{id} until status=ready.
    Only downloads — no transcription, no GPU.
    """
    if not req.url.startswith(("http://", "https://")):
        raise HTTPException(400, "الرابط غير صالح")
    # pass the requested quality so the preview source matches what the
    # job would download (and HD previews download faster).
    pid = _preview_mod.start_preview(req.url, req.quality or None)
    return {"preview_id": pid, "status": "working"}


@app.get("/api/preview/{preview_id}")
def get_preview(preview_id: str) -> dict:
    if not _ID_RE.fullmatch(preview_id or ""):
        raise HTTPException(404, "لا توجد معاينة بهذا المعرف")
    st = _preview_mod.get_preview(preview_id)
    if not st:
        raise HTTPException(404, "لا توجد معاينة بهذا المعرف")
    st.pop("video", None)  # internal absolute path — never leave the server
    return {"preview_id": preview_id, **st}


class PreviewRenderRequest(BaseModel):
    quality: str = Field(default="", max_length=8)
    aspect: str = Field(default="", max_length=8)
    crop_x: float | None = Field(default=None, ge=0.0, le=1.0)
    crop_scale: float | None = Field(default=None, ge=0.4, le=1.0)
    brand_pos: str | None = Field(default=None, pattern="^(tl|tc|tr|bl|bc|br)$")
    source_pos: str | None = Field(default=None, pattern="^(tl|tc|tr|bl|bc|br)$")
    brand_watermark: bool | None = None
    show_source: bool | None = None
    burn_subtitles: bool | None = None
    frame_scale: float | None = Field(default=None, ge=0.4, le=1.0)
    fill_mode: str | None = Field(default=None, pattern="^(blur|black)$")
    sample_text: str = Field(default="", max_length=200)


@app.post("/api/preview/{preview_id}/render")
def render_preview(preview_id: str, req: PreviewRenderRequest) -> dict:
    """True preview: a short mp4 rendered by the real production chain.

    Same manual_crop/build_crop → write_ass → render_clip calls the job
    runner makes, so the <video> shows exactly what a job with the same
    options produces for that segment. crop_x=None     replays the automatic
    face-tracking path (including fit mode).
    """
    if not _ID_RE.fullmatch(preview_id or ""):
        raise HTTPException(404, "لا توجد معاينة بهذا المعرف")
    try:
        return _preview_mod.render_sample(
            preview_id,
            quality=req.quality or None,
            aspect=req.aspect or None,
            crop_x=req.crop_x,
            crop_scale=req.crop_scale if req.crop_scale is not None else 1.0,
            brand_pos=req.brand_pos,
            source_pos=req.source_pos,
            show_brand=req.brand_watermark,
            show_source=req.show_source,
            burn=req.burn_subtitles,
            frame_scale=req.frame_scale,
            fill_mode=req.fill_mode,
            sample_text=req.sample_text,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


def _exit_later(code: int = 0, delay: float = 0.8) -> None:
    """Terminate this worker after the HTTP response has been flushed."""
    import threading

    threading.Timer(delay, lambda: os._exit(code)).start()


@app.post("/api/server/shutdown")
def server_shutdown() -> dict:
    """Stop the server process (dashboard power button, localhost only)."""
    log.warning("shutdown requested via dashboard")
    _exit_later(0)
    return {"ok": True, "action": "shutdown"}


@app.post("/api/server/restart")
def server_restart() -> dict:
    """Restart: spawn a fresh detached server, then stop this process."""
    import subprocess
    import sys

    log.warning("restart requested via dashboard")
    try:
        subprocess.Popen(
            [sys.executable, "-m", "app.main"],
            cwd=str(settings.base_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            close_fds=True,
        )
    except Exception as exc:  # noqa: BLE001 - spawn failure must not kill us
        raise HTTPException(500, f"تعذّر إعادة التشغيل: {exc}") from exc
    _exit_later(0)
    return {"ok": True, "action": "restart"}


# --------------------------------------------------------------------------- #
# static
# --------------------------------------------------------------------------- #

settings.ensure_dirs()
app.mount("/outputs", StaticFiles(directory=settings.outputs_dir), name="outputs")

WEB_DIR = settings.base_dir / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

BRAND_DIR = settings.base_dir / "assets" / "brand"
if BRAND_DIR.exists():
    app.mount("/brand", StaticFiles(directory=BRAND_DIR), name="brand")

from app import preview as _preview_mod  # noqa: E402 - after mounts, before routes

_preview_mod.preview_dir().mkdir(parents=True, exist_ok=True)
app.mount("/preview", StaticFiles(directory=_preview_mod.preview_dir()), name="preview")


@app.get("/")
def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(404, "الواجهة غير مثبتة بعد — ضع index.html داخل web/")
    return FileResponse(page)


def main() -> None:
    import uvicorn

    if settings.host not in _LOOPBACK_HOSTS:
        log.error("HOST=%s is not loopback — refusing without auth (see main docstring)", settings.host)
        raise SystemExit(1)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
