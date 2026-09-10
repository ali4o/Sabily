"""The orchestrator: URL in, finished clips + metadata out.

Stage order is deliberate. Whisper owns the GPU alone, is fully released,
and only then does the LLM run. MediaPipe and ffmpeg are CPU-side.
"""

import json
import logging
import shutil
from pathlib import Path

from app import jobs
from app.config import settings
from app.pipeline import download, llm, normalize, reframe, render, score, subtitle
from app.pipeline.media import probe
from app.pipeline.transcribe import Word
from app.pipeline import transcribe as asr

log = logging.getLogger("sabily.runner")


def _stage(job_id: str, stage: str, progress: int) -> None:
    jobs.update_job(job_id, stage=stage, progress=progress, status="running")
    log.info("[%s] %s (%d%%)", job_id, stage, progress)


def process(job_id: str) -> None:
    job = jobs.get_job(job_id)
    if not job:
        return
    options = json.loads(job.get("options") or "{}")
    url = job["url"]
    work = settings.work_dir / job_id
    out_dir = settings.outputs_dir / job_id
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1 — download ------------------------------------------------------
        _stage(job_id, "جاري تحميل الفيديو", 5)
        src = download.fetch(url, work)
        jobs.update_job(job_id, title=src.title)
        info = probe(src.video_path)

        # 2 — ASR (GPU, exclusive) -----------------------------------------
        _stage(job_id, "تفريغ الصوت وتحديد التوقيتات", 20)
        words: list[Word] = asr.run(
            src.audio_path,
            cache=work / "transcript.json",
            on_progress=lambda f: jobs.update_job(job_id, progress=20 + int(f * 35)),
        )
        if not words:
            raise RuntimeError("لم يتم استخراج أي كلام من الفيديو")
        words = normalize.apply(words)

        # 3 — heuristic selection (CPU, cheap) ------------------------------
        _stage(job_id, "اختيار أفضل اللحظات", 58)
        wanted = int(options.get("clips", settings.clips_per_video))
        candidates = score.select(words, limit=max(wanted, settings.llm_candidates))
        if not candidates:
            raise RuntimeError("لم يتم العثور على مقاطع مناسبة")

        # 4 — LLM: rerank + metadata ---------------------------------------
        _stage(job_id, "توليد العناوين والكابشن", 64)
        order = llm.rerank(candidates, wanted)
        chosen = [candidates[i] for i in order][:wanted]

        # 5 — render one clip at a time ------------------------------------
        for n, cand in enumerate(chosen, start=1):
            _stage(job_id, f"مونتاج المقطع {n} من {len(chosen)}", 70 + int(25 * n / len(chosen)))

            crop = reframe.build_crop(
                src.video_path, cand.start, cand.end, info["width"], info["height"]
            )
            tag = download.as_hashtag(src.channel)
            ass = subtitle.write_ass(
                words, cand.start, cand.end, work / f"clip_{n:02d}.ass",
                source_tag=tag,
            )
            mp4 = out_dir / f"clip_{n:02d}.mp4"
            render.render_clip(src.video_path, mp4, cand.start, cand.end, crop, ass)

            meta = llm.generate_metadata(cand.text, options.get("lang"))
            source_url = download.timestamped_url(src.webpage_url, cand.start)
            caption = llm.build_caption(meta, source_url)

            record = {
                "idx": n,
                "start_sec": round(cand.start, 2),
                "end_sec": round(cand.end, 2),
                "score": cand.score,
                "title": meta["title"],
                "caption": caption,
                "source_url": source_url,
                "video_path": f"{job_id}/{mp4.name}",
                "meta": {
                    "duration": round(cand.duration, 1),
                    "hashtags": meta.get("hashtags", []),
                    "crop_mode": crop["mode"],
                    "source_tag": tag,
                    "brand_text": settings.brand_text,
                    "brand_pos": settings.brand_pos,
                    "source_pos": settings.source_pos,
                    "crop": crop,
                    "source_video": str(src.video_path),
                    "lines": [
                        {"start": round(max(0.0, s - cand.start), 2),
                         "end": round(min(cand.end - cand.start, e - cand.start), 2),
                         "text": tx}
                        for s, e, tx in subtitle.group_lines(
                            [w for w in words
                             if w.start >= cand.start - 0.2 and w.end <= cand.end + 0.2]
                        )
                    ],
                    "score_parts": cand.parts,
                    "source_title": src.title,
                    "transcript": cand.text,
                },
            }
            jobs.add_clip(job_id, record)
            (out_dir / f"clip_{n:02d}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        jobs.update_job(job_id, status="done", stage="اكتمل", progress=100)

    except Exception as exc:  # noqa: BLE001
        stage = (jobs.get_job(job_id) or {}).get("stage", "?")
        log.error("[%s] فشل عند مرحلة: %s — %s", job_id, stage, exc)
        log.debug("traceback", exc_info=True)
        jobs.update_job(job_id, status="error", error=str(exc)[:500], stage="فشل")
    finally:
        # the source stays by default: re-rendering an edited clip needs it.
        # remove it from the dashboard, or set KEEP_SOURCE=false in .env.
        if not (options.get("keep_source") or settings.keep_source):
            shutil.rmtree(work, ignore_errors=True)
