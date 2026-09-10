"""Sabily doctor — one script that checks everything and tells you what's broken.

    python -m scripts.doctor                    # full local check
    python -m scripts.doctor --url "<video>"    # also probe a real URL (no download)
    python -m scripts.doctor --quick            # skip the slow model checks

Every check is isolated: one failure never stops the rest. The output is meant
to be copy-pasted when asking for help, so it prints versions and paths, not
just pass/fail.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from app.config import settings

OK, BAD, WARN, INFO = "[ OK ]", "[FAIL]", "[WARN]", "[ .. ]"
results: list[tuple[str, str]] = []


def report(status: str, name: str, detail: str = "") -> None:
    results.append((status, name))
    line = f"{status} {name}"
    if detail:
        line += f"\n       {detail}"
    print(line, flush=True)


def section(title: str) -> None:
    print(f"\n{'─' * 62}\n  {title}\n{'─' * 62}", flush=True)


def run(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, "command not found"
    except subprocess.TimeoutExpired:
        return 124, "timeout"


# --------------------------------------------------------------------------- #
# 1. environment
# --------------------------------------------------------------------------- #

def check_environment() -> None:
    section("1. البيئة")
    print(f"{INFO} {platform.system()} {platform.release()} | Python {platform.python_version()}")
    print(f"{INFO} مسار المشروع: {settings.base_dir}")

    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        report(OK, "يعمل داخل venv", sys.prefix)
    else:
        report(BAD, "خارج الـ venv — سيستخدم حزم النظام القديمة",
               "شغّل: .\\.venv\\Scripts\\Activate.ps1")

    for var in ("HF_HOME", "OLLAMA_MODELS", "PIP_CACHE_DIR"):
        val = os.getenv(var)
        report(OK if val else WARN, f"متغيّر {var}", val or "غير مضبوط (سيستخدم C:)")

    for drive in {settings.base_dir.anchor, Path(os.getenv("HF_HOME", str(Path.home()))).anchor}:
        try:
            free = shutil.disk_usage(drive).free / 1e9
            report(OK if free > 10 else WARN, f"مساحة حرة على {drive}", f"{free:.1f} GB")
        except Exception as exc:  # noqa: BLE001
            report(WARN, f"مساحة {drive}", str(exc))


def check_code_freshness() -> None:
    """Catch the classic failure: new scripts dropped next to old app/ files."""
    section("2. تطابق ملفات المشروع")
    expected = [
        ("app.config", "settings", "fonts_dir"),
        ("app.pipeline.render", "subtitles_filter", None),
        ("app.pipeline.download", "PLAYER_CLIENTS", None),
        ("app.pipeline.llm", "SYSTEM_AR", None),
    ]
    import importlib

    stale, skipped = [], []
    for module, attr, sub in expected:
        try:
            m = importlib.import_module(module)
            obj = getattr(m, attr)
            if sub and not hasattr(obj, sub):
                raise AttributeError(f"{attr}.{sub}")
        except ModuleNotFoundError as exc:
            # a missing third-party package is a dependency problem, not stale code
            if exc.name and not exc.name.startswith("app"):
                skipped.append(f"{module} (تنقص الحزمة {exc.name})")
            else:
                stale.append(f"{module}.{attr} (مفقود)")
        except Exception as exc:  # noqa: BLE001
            stale.append(f"{module}.{attr} ({exc.__class__.__name__})")
    for s in skipped:
        report(WARN, "تعذّر الفحص", s)
    if stale:
        report(BAD, "ملفات قديمة لم تُحدَّث",
               ", ".join(stale) + "\n       فُك آخر zip فوق مجلد المشروع كاملاً، لا ملفاً ملفاً")
    else:
        report(OK, "كل الملفات محدّثة ومتوافقة")


def check_packages() -> None:
    section("3. الحزم")
    required = ["fastapi", "uvicorn", "pydantic", "dotenv", "httpx",
                "yt_dlp", "faster_whisper", "cv2", "numpy"]
    import importlib

    for mod in required:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            if not ver:
                from importlib.metadata import version as _v
                dist = {"dotenv": "python-dotenv", "yt_dlp": "yt-dlp",
                        "cv2": "opencv-python-headless"}.get(mod, mod)
                try:
                    ver = _v(dist)
                except Exception:  # noqa: BLE001
                    ver = "?"
            report(OK, f"{mod}", f"{ver} — {getattr(m, '__file__', '')}")
        except Exception as exc:  # noqa: BLE001
            report(BAD, f"{mod}", f"غير مثبّت ({exc})")


# --------------------------------------------------------------------------- #
# 2. ffmpeg + fonts
# --------------------------------------------------------------------------- #

def check_ffmpeg() -> None:
    section("4. ffmpeg والخطوط")
    code, out = run([settings.ffmpeg, "-version"])
    if code != 0:
        report(BAD, "ffmpeg", f"{settings.ffmpeg} غير موجود — ثبّته أو صحّح FFMPEG_BIN في .env")
        return
    first = out.splitlines()[0] if out else ""
    report(OK, "ffmpeg موجود", first[:90])

    for flag in ("libass", "libfribidi", "libharfbuzz", "libx264"):
        present = f"--enable-{flag}" in out
        report(OK if present else BAD, f"ffmpeg يدعم {flag}",
               "" if present else "بناء ناقص — نزّل build كامل (BtbN / Gyan full)")

    code, _ = run([settings.ffprobe, "-version"])
    report(OK if code == 0 else BAD, "ffprobe موجود")

    fonts = list(settings.fonts_dir.glob("*.tt*")) if settings.fonts_dir.exists() else []
    report(OK if fonts else WARN, f"خطوط في {settings.fonts_dir.name}/",
           ", ".join(f.name for f in fonts) or "فارغ (مهم فقط على لينكس؛ ويندوز يتجاهله)")


def check_arabic_render() -> None:
    """The decisive test: does a real Arabic line survive the whole render path?"""
    section("5. رسم النص العربي")
    from app.pipeline.render import subtitles_filter
    from app.pipeline.subtitle import HEADER, _ts

    settings.ensure_dirs()
    ass = settings.work_dir / "doctor.ass"
    png = settings.work_dir / "doctor_arabic.png"
    samples = ["السلام عليكم ورحمة الله", "لا إله إلا الله", "تقنية AP36 في الخط رقم 3"]
    body = [f"Dialogue: 0,{_ts(0)},{_ts(5)},Sabily,0,0,{150 + i * 180},,{s}"
            for i, s in enumerate(samples)]
    ass.write_text(
        HEADER.format(w=1080, h=1920, font=settings.subtitle_font,
                      size=64, outline=3, margin=200) + "\n".join(body) + "\n",
        encoding="utf-8",
    )

    cmd = [settings.ffmpeg, "-y", "-hide_banner", "-v", "verbose",
           "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1",
           "-vf", subtitles_filter(ass), "-frames:v", "1", str(png)]
    code, out = run(cmd)
    if code != 0:
        report(BAD, "رسم الترجمة", out[-300:])
        return

    report(OK, "ffmpeg رسم الملف", str(png))

    # which font did libass actually pick? this is what catches silent fallback
    picks = [l.strip() for l in out.splitlines() if "fontselect" in l]
    if picks:
        wanted = settings.subtitle_font.lower()
        got = picks[0]
        matched = wanted.replace(" ", "") in got.lower().replace(" ", "").split("->")[-1]
        report(OK if matched else BAD, "libass استخدم الخط المطلوب", got[:110]
               + ("" if matched else f"\n       المطلوب '{settings.subtitle_font}' — ثبّته على النظام (Install for all users)"))
    else:
        report(WARN, "لم يُطبع سطر fontselect", "لا يمكن التأكد أي خط استُخدم")

    print(f"{INFO} افتح الصورة وتأكد بعينك: الاتصال، الاتجاه، وعدم وجود مربعات □")
    print(f"       start {png}")


# --------------------------------------------------------------------------- #
# 3. models
# --------------------------------------------------------------------------- #

def check_gpu() -> None:
    section("6. كرت الشاشة")
    code, out = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                     "--format=csv,noheader"])
    if code == 0 and out.strip():
        report(OK, "GPU", out.strip().splitlines()[0])
    else:
        report(WARN, "nvidia-smi غير متاح", "سيعمل Whisper على المعالج (أبطأ بكثير)")

    try:
        import ctranslate2

        devices = ctranslate2.get_cuda_device_count()
        report(OK if devices else WARN, "CUDA متاح لـ ctranslate2", f"{devices} جهاز")
        if devices == 0 and settings.whisper_device == "cuda":
            print("       WHISPER_DEVICE=cuda لكن لا يوجد CUDA — سيسقط تلقائياً إلى CPU")
    except Exception as exc:  # noqa: BLE001
        report(WARN, "ctranslate2", str(exc))

    try:
        from app.pipeline.transcribe import cuda_libs_present

        if settings.whisper_device == "cuda":
            ok = cuda_libs_present()
            report(OK if ok else BAD, "مكتبات cuBLAS/cuDNN",
                   "" if ok else "ناقصة — شغّل: python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
                                 "\n       (بدونها سيسقط Whisper تلقائياً إلى المعالج وسيصير أبطأ بكثير)")
    except Exception as exc:  # noqa: BLE001
        report(WARN, "فحص مكتبات CUDA", str(exc))


def check_whisper(quick: bool) -> None:
    section("7. Whisper")
    if quick:
        print(f"{INFO} تم التخطي (--quick)")
        return
    settings.ensure_dirs()
    wav = settings.work_dir / "doctor_tone.wav"
    code, _ = run([settings.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                   "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
                   "-ar", "16000", "-ac", "1", str(wav)])
    if code != 0:
        report(BAD, "تعذّر توليد ملف صوتي للاختبار")
        return

    print(f"{INFO} تحميل {settings.whisper_model} (قد ينزّل ~1.6GB أول مرة)...")
    try:
        from app.pipeline import transcribe

        t0 = time.time()
        words = transcribe.run(wav)
        report(OK, "Whisper يعمل",
               f"{settings.whisper_model} على {settings.whisper_device} "
               f"— {time.time()-t0:.1f}s، {len(words)} كلمة من نغمة صامتة")
    except Exception as exc:  # noqa: BLE001
        report(BAD, "Whisper فشل", str(exc)[:300])


def check_llm(quick: bool) -> None:
    section("8. النموذج اللغوي")
    print(f"{INFO} المزوّد: {settings.llm_provider} | الموديل: {settings.llm_model}")

    if settings.llm_provider == "ollama":
        import httpx

        try:
            r = httpx.get(f"{settings.ollama_url}/api/tags", timeout=10)
            names = [m["name"] for m in r.json().get("models", [])]
            report(OK, "Ollama يعمل", ", ".join(names) or "لا توجد موديلات")
            if settings.llm_model not in names:
                report(BAD, f"الموديل {settings.llm_model} غير موجود",
                       f"شغّل: ollama pull {settings.llm_model}")
                return
        except Exception as exc:  # noqa: BLE001
            report(BAD, "Ollama لا يستجيب", f"{settings.ollama_url} — {exc}")
            return
    elif settings.llm_provider in ("openai", "gemini") and not settings.llm_api_key:
        report(BAD, "LLM_API_KEY فارغ", "مطلوب مع المزوّد الخارجي")
        return

    if quick:
        print(f"{INFO} تم تخطي اختبار التوليد (--quick)")
        return

    from app.pipeline import llm

    sample = ("لماذا يفشل أغلب الناس في بناء العادات؟ السبب أنهم يبدأون بهدف كبير "
              "ثم يتوقفون بعد أسبوع. الحل هو تصغير العادة إلى خطوة واحدة سهلة.")
    t0 = time.time()
    meta = llm.generate_metadata(sample)
    took = time.time() - t0
    fallback = meta.get("hashtags") == ["#Shorts", "#Reels", "#مقاطع"]
    blob = f"{meta.get('title','')} {meta.get('summary','')}"
    arabic = sum(1 for c in blob if "\u0600" <= c <= "\u06FF")
    if settings.caption_lang == "ar" and arabic < len(blob.strip()) * 0.3:
        report(BAD, "النموذج ردّ بالإنجليزية رغم أن CAPTION_LANG=ar",
               "جرّب موديلاً أكبر: LLM_MODEL=gemma4:e2b")
    report(WARN if fallback else OK, "توليد العنوان والكابشن",
           f"{took:.1f}s — {json.dumps(meta, ensure_ascii=False)}"
           + ("\n       هذه مخرجات heuristic لا النموذج — راجع الـ log أعلاه" if fallback else ""))
    print(f"{INFO} الكابشن النهائي:\n" +
          "\n".join("       " + l for l in
                    llm.build_caption(meta, "https://youtu.be/xxxx?t=42").splitlines()))


# --------------------------------------------------------------------------- #
# 4. selection logic + storage
# --------------------------------------------------------------------------- #

def check_selection() -> None:
    section("9. منطق اختيار المقاطع")
    from app.pipeline.score import select
    from app.pipeline.transcribe import Word

    words, t = [], 0.0
    phrases = [
        "لماذا يفشل أغلب الناس في تطبيق العادات الجديدة كل مرة.",
        "الفكرة الأساسية هي تقليل حجم العادة إلى أصغر خطوة ممكنة.",
        "التركيز يعتمد على البيئة أكثر من اعتماده على الإرادة نفسها.",
    ]
    i = 0
    while t < 240:
        for tok in phrases[i % 3].split():
            words.append(Word(start=t, end=t + 0.45, text=tok))
            t += 0.5
        t += 0.8
        i += 1

    picks = select(words, limit=3)
    if not picks:
        report(BAD, "لم يُنتج أي مرشح من نص اصطناعي")
        return
    report(OK, f"أنتج {len(picks)} مرشحاً",
           " | ".join(f"{c.start:.0f}-{c.end:.0f}s ({c.score:.2f})" for c in picks))
    bad = [c for c in picks if not (settings.min_clip_sec <= c.duration <= settings.max_clip_sec)]
    report(OK if not bad else BAD, "كل المرشحين داخل حدود المدة",
           f"{settings.min_clip_sec}-{settings.max_clip_sec}s")


def check_storage() -> None:
    section("10. قاعدة البيانات والمخرجات")
    from app import jobs

    try:
        jobs.init_db()
        report(OK, "قاعدة البيانات", str(settings.db_path))
    except Exception as exc:  # noqa: BLE001
        report(BAD, "قاعدة البيانات", str(exc))
        return

    all_jobs = jobs.list_jobs(50)
    done = [j for j in all_jobs if j["status"] == "done"]
    failed = [j for j in all_jobs if j["status"] == "error"]
    report(OK, "المهام المسجّلة", f"{len(all_jobs)} إجمالاً | {len(done)} ناجحة | {len(failed)} فاشلة")
    for j in failed[:3]:
        print(f"       ✗ {j['id']} — مرحلة '{j['stage']}': {j['error'][:110]}")

    clips = jobs.list_clips()
    report(OK if clips else WARN, "المقاطع في القاعدة", f"{len(clips)} مقطعاً")
    for c in clips[:5]:
        path = settings.outputs_dir / c["video_path"]
        size = f"{path.stat().st_size/1e6:.1f}MB" if path.exists() else "الملف مفقود!"
        print(f"       • [{c['idx']}] {c['title'][:44]} — {c['end_sec']-c['start_sec']:.0f}s — {size}")

    if settings.outputs_dir.exists():
        files = sorted(settings.outputs_dir.rglob("*.mp4"))
        report(OK if files else WARN, "ملفات mp4 على القرص", f"{len(files)} ملفاً")
        for f in files[:8]:
            print(f"       • {f.relative_to(settings.outputs_dir)} — {f.stat().st_size/1e6:.1f}MB")


def check_url(url: str) -> None:
    section("11. فحص الرابط (بدون تحميل)")
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    from app.pipeline.download import PLAYER_CLIENTS, _base_opts, _explain, timestamped_url

    for client in PLAYER_CLIENTS:
        opts = _base_opts(settings.work_dir / "doctor")
        opts["extractor_args"] = {"youtube": {"player_client": [client]}}
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
            mins = float(info.get("duration") or 0) / 60
            report(OK, f"الرابط يعمل عبر player_client={client}",
                   f"{info.get('title', '')[:60]} — {mins:.1f} دقيقة")
            print(f"       رابط موقوت: {timestamped_url(info.get('webpage_url') or url, 90)}")
            if mins > settings.max_video_minutes:
                report(WARN, "أطول من الحد المسموح", f"{settings.max_video_minutes} دقيقة")
            return
        except DownloadError as exc:
            print(f"{INFO} player_client={client} فشل")
            last = str(exc)
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
    report(BAD, "كل العملاء فشلوا", _explain(last))


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(prog="sabily-doctor")
    ap.add_argument("--url", default="", help="افحص رابط فيديو بدون تحميله")
    ap.add_argument("--quick", action="store_true", help="تخطَّ تحميل النماذج")
    args = ap.parse_args()

    print("=" * 62)
    print("  Sabily — فحص شامل")
    print("=" * 62)

    for fn in (check_environment, check_code_freshness, check_packages, check_ffmpeg,
               check_arabic_render, check_gpu):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            report(BAD, fn.__name__, f"انهار الفحص نفسه: {exc}")

    for fn, arg in ((check_whisper, args.quick), (check_llm, args.quick)):
        try:
            fn(arg)
        except Exception as exc:  # noqa: BLE001
            report(BAD, fn.__name__, f"انهار الفحص نفسه: {exc}")

    for fn in (check_selection, check_storage):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            report(BAD, fn.__name__, f"انهار الفحص نفسه: {exc}")

    if args.url:
        try:
            check_url(args.url)
        except Exception as exc:  # noqa: BLE001
            report(BAD, "check_url", str(exc))

    section("الخلاصة")
    fails = [n for s, n in results if s == BAD]
    warns = [n for s, n in results if s == WARN]
    print(f"  نجح: {sum(1 for s, _ in results if s == OK)} | "
          f"تحذير: {len(warns)} | فشل: {len(fails)}")
    for n in fails:
        print(f"  ✗ {n}")
    for n in warns:
        print(f"  ! {n}")
    if not fails:
        print("\n  كل الفحوص الحرجة مرّت. تبقّى فحص الصورة بعينك.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
