"""Sabily — local dashboard + API on http://127.0.0.1:6767

No auth: the server binds to localhost only and stores nothing personal.
If it is ever exposed beyond this machine, add auth first.
"""

import asyncio
import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import jobs
from app.config import settings
from app.pipeline.runner import process

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("sabily")

app = FastAPI(title="Sabily", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class JobRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2000)
    clips: int = Field(default=0, ge=0, le=20)
    lang: str = Field(default="", max_length=5)
    aspect: str = Field(default="", max_length=8)
    keep_source: bool = False


@app.on_event("startup")
def _startup() -> None:
    settings.ensure_dirs()
    jobs.init_db()
    jobs.start_worker(process)
    log.info("Sabily ready on http://%s:%s", settings.host, settings.port)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

@app.post("/api/jobs")
def create_job(req: JobRequest) -> dict:
    if not req.url.startswith(("http://", "https://")):
        raise HTTPException(400, "الرابط غير صالح")
    options = {k: v for k, v in req.model_dump().items() if k != "url" and v}
    job_id = jobs.create_job(req.url, options)
    jobs.enqueue(job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs")
def get_jobs(limit: int = 20) -> list[dict]:
    return jobs.list_jobs(limit)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "لا يوجد job بهذا المعرف")
    job["clips"] = jobs.list_clips(job_id)
    return job


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """Server-sent events: poll the DB and push status to the dashboard."""

    async def stream():
        last = None
        while True:
            job = jobs.get_job(job_id)
            if not job:
                yield 'event: error\ndata: {"error":"not found"}\n\n'
                return
            payload = {
                "status": job["status"],
                "stage": job["stage"],
                "progress": job["progress"],
                "error": job["error"],
            }
            if payload != last:
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last = payload
            if job["status"] in ("done", "error", "cancelled"):
                return
            await asyncio.sleep(0.7)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/clips")
def get_clips(job_id: str | None = None) -> list[dict]:
    return jobs.list_clips(job_id)


@app.delete("/api/clips/{clip_id}")
def remove_clip(clip_id: str) -> dict:
    jobs.delete_clip(clip_id)
    return {"ok": True}


@app.delete("/api/clips")
def remove_all_clips() -> dict:
    jobs.delete_all_clips()
    return {"ok": True}


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "whisper_model": settings.whisper_model,
        "device": settings.whisper_device,
    }


# --------------------------------------------------------------------------- #
# static
# --------------------------------------------------------------------------- #

settings.ensure_dirs()
app.mount("/outputs", StaticFiles(directory=settings.outputs_dir), name="outputs")

WEB_DIR = settings.base_dir / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(404, "الواجهة غير مثبتة بعد — ضع index.html داخل web/")
    return FileResponse(page)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
