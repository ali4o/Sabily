"""Run the whole pipeline on one URL without the web UI.

    python -m scripts.cli "https://www.youtube.com/watch?v=..." --clips 3
"""

import argparse
import logging
import sys

from app import jobs
from app.config import settings
from app.pipeline.runner import process


def main() -> int:
    parser = argparse.ArgumentParser(prog="sabily")
    parser.add_argument("url")
    parser.add_argument("--clips", type=int, default=settings.clips_per_video)
    parser.add_argument("--lang", default=settings.caption_lang)
    parser.add_argument("--quality", default="",
                        help="HD (720p) | FHD (1080p, default) | QHD (2K 1440p)")
    parser.add_argument("--no-logo", action="store_true", help="بدون علامة سبيلي")
    parser.add_argument("--no-tag", action="store_true", help="بدون هاشتاق المصدر")
    parser.add_argument("--no-subs", action="store_true", help="بدون النص السفلي")
    parser.add_argument("--keep-source", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")
    logging.getLogger("sabily.runner").propagate = True
    jobs.init_db()

    from app.config import QUALITY_ALIASES
    options: dict = {"clips": args.clips, "lang": args.lang,
                     "keep_source": args.keep_source}
    q = QUALITY_ALIASES.get(str(args.quality or "").strip().upper(), "")
    if q:
        options["quality"] = q
    if args.no_logo:
        options["brand_watermark"] = False
    if args.no_tag:
        options["show_source"] = False
    if args.no_subs:
        options["burn_subtitles"] = False
    job_id = jobs.create_job(args.url, options)
    process(job_id)

    job = jobs.get_job(job_id) or {}
    if job.get("status") != "done":
        print(f"فشل: {job.get('error')}")
        return 1

    for clip in jobs.list_clips(job_id):
        print("\n" + "=" * 60)
        print(f"[{clip['idx']}] {clip['title']}  ({clip['start_sec']:.0f}s → {clip['end_sec']:.0f}s)")
        print(settings.outputs_dir / clip["video_path"])
        print("-" * 60)
        print(clip["caption"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
