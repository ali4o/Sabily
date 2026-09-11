"""Gate 1 — does this ffmpeg build render Arabic subtitles correctly?

Run once before trusting any output:
    python -m scripts.gate_arabic

Produces data/gate_arabic.png. Open it and check three things:
  1. letters are connected (شكل متصل), not isolated: "السلام" not "ا ل س ل ا م"
  2. word order runs right-to-left
  3. the mixed line keeps "AP36" left-to-right inside the Arabic sentence

If any of those fail, the ffmpeg build lacks libass/fribidi/harfbuzz —
install a full build (Windows: Gyan.FFmpeg full, Linux: distro ffmpeg).
"""

import subprocess
import sys
from pathlib import Path

from app.config import settings
from app.pipeline.render import subtitles_filter
from app.pipeline.subtitle import _ts, build_header

SAMPLES = [
    "السلام عليكم ورحمة الله وبركاته",
    "هذه المقاطع تُولَّد تلقائياً بواسطة سبيلي",
    "نستخدم تقنية AP36 في خط الإنتاج رقم 3",
]


def main() -> int:
    settings.ensure_dirs()
    ass = settings.work_dir / "gate_arabic.ass"
    png = settings.work_dir / "gate_arabic.png"

    body = [
        f"Dialogue: 0,{_ts(0)},{_ts(5)},Sabily,0,0,{120 + i * 160},,{text}"
        for i, text in enumerate(SAMPLES)
    ]
    ass.write_text(
        build_header(1080, 1920, settings.subtitle_font)
        + "\n".join(body) + "\n",
        encoding="utf-8",
    )

    filt = subtitles_filter(ass)
    cmd = [
        settings.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1",
        "-vf", filt, "-frames:v", "1", str(png),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("FAIL — ffmpeg could not burn the subtitles:\n", proc.stderr[-600:])
        return 1

    fonts = list(settings.fonts_dir.glob("*.tt*")) if settings.fonts_dir.exists() else []
    print(f"fontsdir: {settings.fonts_dir} ({len(fonts)} خط)")
    print(f"OK — رُسم الملف: {png}")
    print("افتح الصورة وتأكد من الاتصال والاتجاه قبل المتابعة.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
