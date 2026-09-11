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


def _opt_flag(options: dict, key: str, default: bool) -> bool:
    """Per-job boolean with global fallback (absent key = .env default)."""
    return options.get(key) if key in options else default


def process(job_id: str) -> None:
    job = jobs.get_job(job_id)
    if not job:
        return
    if job.get("status") == "cancelled":
        log.info("[%s] أُلغيت قبل البدء", job_id)
        return
    options = json.loads(job.get("options") or "{}")
    url = job["url"]
    work = settings.work_dir / job_id
    out_dir = settings.outputs_dir / job_id
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 0 — resolve per-job output spec (quality + aspect) -----------------
        from app.config import normalize_quality, resolve_target
        aspect = options.get("aspect") or settings.aspect
        if aspect not in ("9:16", "1:1"):
            aspect = settings.aspect if settings.aspect in ("9:16", "1:1") else "9:16"
        quality = normalize_quality(options.get("quality"), settings.quality)
        target_w, target_h = resolve_target(quality, aspect,
                                            settings.out_width, settings.out_height)
        target_size = (target_w, target_h)
        # appearance flags per job (absent = global .env default)
        show_brand = options.get("brand_watermark") if "brand_watermark" in options \
            else settings.brand_watermark
        show_source = options.get("show_source") if "show_source" in options \
            else settings.source_tag
        burn = options.get("burn_subtitles") if "burn_subtitles" in options \
            else settings.burn_subtitles
        # frame shrink inside the output (absent/1.0 = full-bleed classic)
        try:
            _fs = float(options.get("frame_scale", 1.0))
            frame_scale = _fs if 0.4 <= _fs <= 1.0 else 1.0
        except (TypeError, ValueError):
            frame_scale = 1.0
        fill_mode = "black" if str(options.get("fill_mode") or "").lower() == "black" \
            else "blur"

        # 1 — download ------------------------------------------------------
        _stage(job_id, "جاري تحميل الفيديو", 5)
        src = download.fetch(url, work, quality)
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
        wanted = max(1, min(20, wanted))
        from app.pipeline.score import normalize_content_type
        ctype = normalize_content_type(
            options.get("content_type") or settings.content_type)
        content_info: dict = {"content_type": ctype, "audience": "",
                              "goal": "education"}
        if ctype == "auto":
            # one small LLM call per job when a provider exists, otherwise
            # pure heuristics — selection itself never needs the LLM.
            content_info = llm.classify_content(words)
            ctype = content_info.get("content_type", "lecture")
        candidates = score.select(words, limit=max(wanted, settings.llm_candidates),
                                  audio_path=src.audio_path, content_type=ctype)
        if not candidates:
            raise RuntimeError("لم يتم العثور على مقاطع مناسبة")
        turns = score.detect_turns(words)

        # 4 — LLM: rerank + metadata ---------------------------------------
        _stage(job_id, "توليد العناوين والكابشن", 64)
        order = llm.rerank(candidates, wanted)
        ranked = [candidates[i] for i in order]
        seen_ids = {id(c) for c in ranked}
        pool = ranked + [c for c in candidates[:wanted * 2] if id(c) not in seen_ids]
        chosen = score.diversify(pool, wanted) or ranked[:wanted]
        if not chosen:
            raise RuntimeError("لم يتم العثور على مقاطع مناسبة")

        # 5 — render one clip at a time ------------------------------------
        for n, cand in enumerate(chosen, start=1):
            cur = jobs.get_job(job_id)
            if cur and cur.get("status") == "cancelled":
                log.info("[%s] أُلغي أثناء المونتاج عند المقطع %d", job_id, n)
                return
            _stage(job_id, f"مونتاج المقطع {n} من {len(chosen)}", 70 + int(25 * n / len(chosen)))

            lock = options.get("crop_lock") or {}
            if isinstance(lock.get("x"), (int, float)):
                # user-measured cut from the preview popup: guaranteed crop,
                # no face sampling at all (fast + deterministic).
                crop = reframe.manual_crop(
                    info["width"], info["height"], aspect,
                    lock["x"], lock.get("scale", 1.0))
            else:
                crop = reframe.build_crop(
                    src.video_path, cand.start, cand.end, info["width"], info["height"],
                    aspect,
                )
            tag = download.as_hashtag(src.channel)
            polished = llm.polish_text(cand.text)
            clip_words = [w for w in words
                          if w.start >= cand.start - 0.2 and w.end <= cand.end + 0.2]
            rows = [
                {"start": round(max(0.0, s - cand.start), 2),
                 "end": round(min(cand.end - cand.start, e - cand.start), 2),
                 "text": tx,
                 "kara": [[w, d] for w, d in parts]}
                for (s, e, parts), (_, _, tx) in zip(
                    subtitle.group_karaoke(clip_words),
                    subtitle.group_lines(clip_words))
            ]
            en_list = None
            if settings.subtitle_bilingual and rows:
                en_list = llm.translate_lines([r["text"] for r in rows])
                for r, e in zip(rows, en_list):
                    r["en"] = e
            ass = subtitle.write_ass(
                words, cand.start, cand.end, work / f"clip_{n:02d}.ass",
                source_tag=tag, target_size=target_size,
                show_brand=show_brand, show_source=show_source, burn=burn,
                lines=[{k: r[k] for k in ("start", "end", "text", "kara") if k in r}
                       for r in rows],
                en_lines=en_list, turns=turns,
            )
            mp4 = out_dir / f"clip_{n:02d}.mp4"
            import time as _time
            _t0 = _time.perf_counter()
            render.render_clip(src.video_path, mp4, cand.start, cand.end, crop, ass,
                               target_size=target_size,
                               brand_pos=settings.brand_pos,
                               show_logo=show_brand, burn_subtitles=burn,
                               zoom_times=crop.get("zoom_times"),
                               frame_scale=frame_scale,
                               fill_mode=fill_mode,
                               bumpers={"intro": _opt_flag(options, "intro_card",
                                                           settings.intro_card),
                                        "outro": _opt_flag(options, "outro_card",
                                                           settings.outro_card),
                                        "title": src.title or tag,
                                        "workdir": work})
            _ms = int((_time.perf_counter() - _t0) * 1000)
            try:
                jobs.log_render(job_id, quality, round(cand.duration, 1), _ms,
                                round(mp4.stat().st_size / 1e6, 1) if mp4.exists() else 0.0)
            except Exception:  # noqa: BLE001 - stats must never fail a job
                pass

            meta = llm.generate_metadata(polished, options.get("lang"))
            source_url = download.timestamped_url(src.webpage_url, cand.start)
            caption = llm.build_caption(meta, source_url)
            reasons = list(getattr(cand, "reasons", []) or [])
            if len(chosen) > 1:
                reasons.append("لا يتكرر مع المقاطع الأعلى ترتيبًا")

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
                    "aspect": aspect,
                    "quality": quality,
                    "target_size": [target_w, target_h],
                    "hashtags": meta.get("hashtags", []),
                    "content_type": ctype,
                    "content_audience": content_info.get("audience", ""),
                    "moment": getattr(cand, "moment", "general"),
                    "topic": getattr(cand, "topic", ""),
                    "reasons": reasons,
                    "crop_mode": crop["mode"],
                    "source_tag": tag,
                    "brand_text": settings.brand_text,
                    "brand_pos": settings.brand_pos,
                    "source_pos": settings.source_pos,
                    "brand_watermark": bool(show_brand),
                    "show_source": bool(show_source),
                    "burn_subtitles": bool(burn),
                    "frame_scale": frame_scale,
                    "fill_mode": fill_mode,
                    "intro_card": _opt_flag(options, "intro_card", settings.intro_card),
                    "outro_card": _opt_flag(options, "outro_card", settings.outro_card),
                    "crop": crop,
                    "crop_lock": lock if isinstance(lock.get("x"), (int, float)) else None,
                    "source_video": str(src.video_path),
                    "lines": rows,
                    "turns": [t for t in turns
                              if cand.start - 1.0 <= t["start"] <= cand.end + 1.0],
                    "score_parts": cand.parts,
                    "source_title": src.title,
                    "transcript": cand.text,
                    "polished": polished,
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
        # options always carries an explicit bool (see main.create_job);
        # old jobs without the key fall back to the global default.
        keep = options.get("keep_source") if "keep_source" in options else settings.keep_source
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
