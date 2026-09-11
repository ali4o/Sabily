"""Build a transparent Sabily logo PNG from the bundled Cairo font.

The Arabic wordmark is rendered through libass (the same shaper the
subtitles use), so letters join correctly — PIL alone cannot shape Arabic
(no raqm here). A transparent lavfi canvas + ASS overlay gives us a real
RGBA cutout; PIL then only draws straight rectangles (no text shaping):

    white wordmark (سبيلي)
    ──  ──   <- two white rules with a green tick between them

Usage:
    python -m scripts.build_logo            # writes assets/brand/sabily-logo.png
    python -m scripts.build_logo --force    # rebuild even if it exists

Replace the PNG with your own artwork any time — any RGBA PNG works.
A black-background JPG also works: the renderer keys black out
automatically (see render.overlay_logo_filter).
"""

import argparse
import subprocess
import sys
from pathlib import Path

from app.config import settings

LOGO_DIR = settings.base_dir / "assets" / "brand"
LOGO_PNG = LOGO_DIR / "sabily-logo.png"

CANVAS_W, CANVAS_H = 1200, 520
WORD_FONTSIZE = 190
GREEN = (61, 179, 131, 255)
WHITE = (255, 255, 255, 255)


def _ass_text() -> str:
    # centered white wordmark; thin dark outline keeps it readable
    # over bright video without changing the logo look on dark scenes.
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {CANVAS_W}\n"
        f"PlayResY: {CANVAS_H}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        "\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
        "BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Logo,{settings.subtitle_font},{WORD_FONTSIZE},"
        "&H00FFFFFF,&H99000000,&H00000000,-1,0,1,3,0,5,40,40,40,1\n"
        "\n[Events]\n"
        "Format: Layer, Start, End, Style, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:01.00,Logo,0,0,0,,سبيلي\n"
    )


def _render_wordmark(work: Path) -> Path:
    # NOTE: opaque black canvas, not transparent: this ffmpeg build drops
    # ASS alpha on a transparent lavfi source, so we luminance-key the
    # black out in PIL instead (same trick the renderer uses for JPG logos).
    ass = work / "logo.ass"
    raw = work / "logo_raw.png"
    ass.write_text(_ass_text(), encoding="utf-8")
    esc = str(ass.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    cmd = [
        settings.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=black:s={CANVAS_W}x{CANVAS_H}:d=1",
        "-vf", f"ass='{esc}':shaping=complex",
        "-frames:v", "1", str(raw),
    ]
    subprocess.run(cmd, check=True)
    return raw


def _key_black_to_transparent(raw: Path) -> "object":
    """Luminance key: near-black -> transparent, with soft edges.

    Same algorithm documents the renderer's JPG fallback: white wordmark
    and the green tick survive, antialiased edges fade smoothly.
    """
    from PIL import Image

    im = Image.open(raw).convert("RGB")
    lum = im.convert("L")
    # alpha ramps 0 below 8 -> 255 at/above 60
    alpha = lum.point(lambda v: 0 if v <= 8 else (255 if v >= 60 else int((v - 8) * 255 / 52)))
    bbox = alpha.point(lambda v: 255 if v > 12 else 0).getbbox()
    if not bbox:
        raise RuntimeError("logo render came out empty — check the Cairo font")
    im.putalpha(alpha)
    pad = 24
    x0 = max(0, bbox[0] - pad)
    y0 = max(0, bbox[1] - pad)
    x1 = min(im.width, bbox[2] + pad)
    y1 = min(im.height, bbox[3] + pad)
    return im.crop((x0, y0, x1, y1))


def _add_accent(wordmark: "object") -> "object":
    """Two white rules + green tick under the word, like the reference logo."""
    from PIL import Image, ImageDraw

    w, h = wordmark.size
    rule_h = max(6, h // 28)
    gap = rule_h * 3
    accent_h = rule_h * 3
    canvas = Image.new("RGBA", (w, h + gap + accent_h), (0, 0, 0, 0))
    canvas.alpha_composite(wordmark, (0, 0))
    d = ImageDraw.Draw(canvas)
    y = h + gap
    tick_w = max(10, w // 42)
    side = (w - tick_w * 3) // 2
    # white rules
    d.rounded_rectangle([0, y + accent_h // 3, side, y + accent_h // 3 + rule_h],
                        radius=rule_h // 2, fill=WHITE)
    d.rounded_rectangle([w - side, y + accent_h // 3, w, y + accent_h // 3 + rule_h],
                        radius=rule_h // 2, fill=WHITE)
    # green tick
    cx = w // 2
    d.rounded_rectangle([cx - tick_w // 2, y, cx + tick_w // 2, y + accent_h],
                        radius=tick_w // 2, fill=GREEN)
    return canvas


def build_logo(force: bool = False) -> Path:
    if LOGO_PNG.exists() and not force:
        return LOGO_PNG
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    work = settings.work_dir
    work.mkdir(parents=True, exist_ok=True)
    raw = _render_wordmark(work)
    logo = _add_accent(_key_black_to_transparent(raw))
    # cap width so the file stays small; renderer scales per quality anyway
    if logo.width > 900:
        from PIL import Image

        logo = logo.resize((900, int(logo.height * 900 / logo.width)), Image.LANCZOS)
    logo.save(LOGO_PNG)
    try:
        raw.unlink(missing_ok=True)
        (work / "logo.ass").unlink(missing_ok=True)
    except OSError:
        pass
    return LOGO_PNG


def main() -> int:
    ap = argparse.ArgumentParser(prog="sabily-build-logo")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    try:
        out = build_logo(args.force)
    except Exception as exc:  # noqa: BLE001 - report, don't traceback
        print(f"FAIL — {exc}")
        return 1
    print(f"OK — {out} ({out.stat().st_size // 1024}KB)")
    print("لاستبداله بصورتك: ضع PNG شفافاً بنفس الاسم، أو JPG بخلفية سوداء (تُزال تلقائياً).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
