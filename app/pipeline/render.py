"""Stage 7 — one ffmpeg pass per clip: cut → crop → scale → subtitles → mp4."""

import logging
import subprocess
from pathlib import Path

from app.config import settings

log = logging.getLogger("sabily.render")


def _escape_x(x_expr: str) -> str:
    """Quote dynamic crop x expressions so filtergraph commas/colons survive.

    Static values (plain numbers) are returned unchanged to preserve
    existing behavior. Dynamic expressions from reframe contain `,`/`:`/`'`
    (e.g. if(lt(t,...),...)) and must be single-quoted per ffmpeg filter
    escaping rules.
    """
    if any(c in x_expr for c in (",", ":", "'")):
        return "'" + x_expr.replace("'", "\\'") + "'"
    return x_expr


def _escape_for_filter(path: Path) -> str:
    """Windows paths and colons need escaping inside a filtergraph."""
    p = str(path.resolve()).replace("\\", "/")
    return p.replace(":", "\\:").replace("'", "\\'")


def subtitles_filter(ass_path: Path) -> str:
    """Build the ass filter for burned-in subtitles.

    Two deliberate choices, both learned the hard way on Windows:

    * the `ass` filter, not `subtitles` — only `ass` exposes `shaping`.
      With `shaping=simple` libass converts lam-alef to the legacy
      presentation form U+FEFB, which modern fonts don't carry, and the
      ligature renders as an empty box. `complex` forces the HarfBuzz path.
    * `fontsdir` is still passed, but note it is IGNORED by the directwrite
      font provider (the default on Windows). There, the font named in the
      ASS style must be installed system-wide or libass silently substitutes
      another face. Check with:
          ffmpeg -v verbose ... 2>&1 | Select-String "fontselect"
    """
    f = f"ass='{_escape_for_filter(ass_path)}':shaping=complex"
    fonts = settings.fonts_dir
    if fonts.exists() and any(fonts.glob("*.tt*")):
        f += f":fontsdir='{_escape_for_filter(fonts)}'"
    return f


LOGO_POS = {
    # (x_expr, y_expr) in overlay coords: W/H = output, w/h = logo
    "tl": ("{padx}", "{pady}"),
    "tc": ("(W-w)/2", "{pady}"),
    "tr": ("W-w-{padx}", "{pady}"),
    "bl": ("{padx}", "H-h-{pady}"),
    "bc": ("(W-w)/2", "H-h-{pady}"),
    "br": ("W-w-{padx}", "H-h-{pady}"),
}


def overlay_logo_filter(target_w: int, target_h: int,
                        brand_pos: str | None = None,
                        show: bool | None = None) -> str | None:
    """ffmpeg filter chunk overlaying the transparent brand logo.

    Returns None when the image watermark is disabled or the PNG is
    missing (caller falls back to the Brand ASS text line).
    `colorkey` makes black-background JPGs work too: pure-black pixels
    turn transparent while the white wordmark + green tick survive.
    `show=None` follows the global BRAND_WATERMARK switch.
    """
    eff = settings.use_brand_logo if show is None else (
        bool(show) and settings.brand_logo.exists())
    if not eff:
        return None
    scale = min(0.5, max(0.08, settings.brand_logo_scale))
    logo_w = max(64, int(target_w * scale))
    pos = brand_pos if brand_pos in LOGO_POS else settings.brand_pos
    if pos not in LOGO_POS:
        pos = "tr"
    padx = int(target_w * 0.045)
    pady = int(target_h * 0.035)
    x, y = LOGO_POS[pos]
    x = x.format(padx=padx)
    y = y.format(pady=pady)
    esc = _escape_for_filter(settings.brand_logo)
    # NOTE: no loop= on movie — an infinite secondary stream plus
    # overlay's default eof_action=repeat never terminates. A still
    # holds for the whole output on its own (and must NOT use
    # shortest=1, which would cut the video to the still's 1 frame).
    movie = (f"movie='{esc}',format=rgba,"
             f"scale={logo_w}:-1,colorkey=0x000000:0.28:0.15[wm]")
    return f"[v0];{movie};[v0][wm]overlay={x}:{y}"


def full_frame_filter(target_w: int, target_h: int, blur: int | None = None) -> str:
    """Fit the whole frame over a blurred fill (mode "fit").

    Used when too few frames contain a face: instead of a blind center
    slice, the viewer sees the entire picture — sharp, centered — over
    the same picture enlarged and blurred. Ends with the [v0] label so
    the logo/subtitle chaining below works unchanged.

    NOTE: no "[0:v]" input prefix — this chain runs via -vf (single
    implicit input). The prefix made ffmpeg count 2 inputs/2 outputs
    and fail fit renders without a logo (found live 2026-09-12).
    """
    sigma = blur if blur and blur > 0 else max(1, settings.face_blur)
    return (
        "split=2[bg0][fg0];"
        f"[bg0]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h},gblur=sigma={sigma}[bg];"
        f"[fg0]scale={target_w}:-2:flags=bicubic[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]"
    )


def zoom_filter(times: list[float]) -> str:
    """Subtle alternating zoom (1.0 ↔ 1.06) per tracked segment.

    `times` are clip-relative breakpoints (first ≈ 0). Empty unless the
    camera actually moves between 2+ positions — static shots stay still.
    """
    pts = [round(float(t), 2) for t in times]
    if len(pts) < 2:
        return ""
    zooms = [1.0 if i % 2 == 0 else 1.06 for i in range(len(pts))]
    expr = f"{zooms[-1]}"
    for t, z in zip(reversed(pts[1:]), reversed(zooms[:-1])):
        expr = f"if(lt(t,{t:.2f}),{z},{expr})"
    return expr


def build_bumper(kind: str, w: int, h: int, text: str, workdir: Path,
                 use_gpu: bool, duration: float = 1.5) -> Path | None:
    """1.5s branded card (intro/outro). None on any failure — never fatal."""
    from app.pipeline import subtitle as _sub

    try:
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / f"bumper_{kind}_{w}x{h}.mp4"
        if out.exists():
            return out
        ass = workdir / f"bumper_{kind}_{w}x{h}.ass"
        _sub.write_card_ass(text, w, h, ass, duration)
        vf = subtitles_filter(ass)
        logo = overlay_logo_filter(w, h, "br")
        if logo:
            vf = "null[v0]" + logo[len("[v0]"):] + "," + vf

        def _cmd(gpu: bool) -> list[str]:
            return [
                settings.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", f"color=c=0x14483C:s={w}x{h}:d={duration}:r=30",
                "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={duration}",
                "-vf", vf, *_video_args(gpu),
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                "-shortest", str(out),
            ]

        proc = subprocess.run(_cmd(use_gpu), capture_output=True, text=True)
        if proc.returncode != 0 and use_gpu:
            log.warning("bumper %s nvenc failed, falling back to x264", kind)
            proc = subprocess.run(_cmd(False), capture_output=True, text=True)
        if proc.returncode != 0:
            log.warning("bumper %s failed (%s)", kind, proc.stderr[-160:].strip())
            return None
        return out if out.exists() else None
    except Exception as exc:  # noqa: BLE001 - bumpers are decoration
        log.warning("bumper %s failed (%s)", kind, exc)
        return None


def concat_parts(parts: list[Path], out_path: Path) -> bool:
    """Join same-codec segments with stream copy (no re-encode)."""
    import tempfile

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8") as f:
            for p in parts:
                f.write(f"file '{p.resolve()}'\n")
            lst = f.name
        cmd = [settings.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "concat", "-safe", "0", "-i", lst,
               "-c", "copy", "-movflags", "+faststart", str(out_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        Path(lst).unlink(missing_ok=True)
        return proc.returncode == 0
    except Exception as exc:  # noqa: BLE001
        log.warning("concat failed (%s)", exc)
        return False


def _has_nvenc() -> bool:
    try:
        out = subprocess.run([settings.ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=30).stdout
        return "h264_nvenc" in out
    except Exception:  # noqa: BLE001
        return False


def _video_args(use_gpu: bool) -> list[str]:
    if use_gpu:
        # NVENC on this class of card is roughly 3x faster than x264 and frees
        # the CPU for the next clip's face pass. cq is the quality knob here.
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                "-cq", str(settings.crf), "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf)]


def glass_fill_chain(crop: dict, target_w: int, target_h: int,
                     frame_scale: float, blur: int | None = None) -> str:
    """Shrunk frame over a blurred glossy fill — never black edges.

    The crop region is scaled to frame_scale (0.4..1) of the output and
    centered over the same picture enlarged + blurred. Ends with [v0] so
    logo chaining works exactly like the fit path (stripped when no logo).
    No "[0:v]" prefix: this runs via -vf with a single implicit input.
    """
    sigma = blur if blur and blur > 0 else max(1, settings.face_blur)
    s = min(0.99, max(0.4, float(frame_scale)))
    fw = max(2, int(target_w * s) // 2 * 2)
    fh = max(2, int(target_h * s) // 2 * 2)
    return (
        "split=2[srcbg][srccrop];"
        f"[srcbg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h},gblur=sigma={sigma}[bg];"
        f"[srccrop]crop={crop['w']}:{crop['h']}:{_escape_x(str(crop['x_expr']))}:{crop['y']},"
        f"scale={fw}:{fh}:flags=bicubic[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]"
    )


def black_fill_chain(crop: dict, target_w: int, target_h: int,
                     frame_scale: float) -> str:
    """Shrunk frame over matte black bars (cinematic letterbox look).

    Same geometry as glass_fill_chain, but the freed edges stay pure
    black instead of a blurred fill. Unlabeled chain (no [v0]) so logo
    chaining works exactly like the classic crop path.
    """
    s = min(0.99, max(0.4, float(frame_scale)))
    fw = max(2, int(target_w * s) // 2 * 2)
    fh = max(2, int(target_h * s) // 2 * 2)
    return ",".join([
        f"crop={crop['w']}:{crop['h']}:{_escape_x(str(crop['x_expr']))}:{crop['y']}",
        f"scale={fw}:{fh}:flags=bicubic",
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black",
        "setsar=1",
    ])


def render_clip(
    source: Path,
    out_path: Path,
    start: float,
    end: float,
    crop: dict,
    ass_path: Path | None = None,
    target_size: tuple | None = None,
    brand_pos: str | None = None,
    show_logo: bool | None = None,
    burn_subtitles: bool | None = None,
    zoom_times: list[float] | None = None,
    bumpers: dict | None = None,
    frame_scale: float | None = None,
    fill_mode: str | None = None,
) -> Path:
    """Cut → crop → scale → zoom → logo → subtitles → mp4 (+bumpers).

    `show_logo` / `burn_subtitles` (None = global settings) let a single
    job or clip override the appearance without touching .env.
    `frame_scale` 0.4..1 shrinks the picture inside the output; `fill_mode`
    decides what the freed edges become: "blur" (default — blurred glossy
    backdrop) or "black" (matte cinematic bars). None/1.0 keeps the classic
    full-bleed look. Both ignored in "fit" mode, which already fills the
    frame with its own blurred backdrop.
    `zoom_times` (clip-relative breakpoints) adds a subtle alternating
    zoom when ZOOM_PUNCH is on. `bumpers` {intro,outro,title,workdir}
    concats branded cards around the clip (decorative: any failure
    keeps the bare clip).
    """
    w, h = target_size if target_size else settings.target_size
    shrink: float | None = None
    if frame_scale is not None:
        try:
            v = float(frame_scale)
            shrink = v if 0.4 <= v < 1.0 else None
        except (TypeError, ValueError):
            shrink = None
    fill = "black" if str(fill_mode or "").strip().lower() == "black" else "blur"
    if crop.get("mode") == "fit":
        base = full_frame_filter(w, h)  # already ends with [v0]
    elif shrink is not None and crop.get("mode") != "fit":
        base = black_fill_chain(crop, w, h, shrink) if fill == "black" \
            else glass_fill_chain(crop, w, h, shrink)  # ends with [v0]
    else:
        base = ",".join([
            f"crop={crop['w']}:{crop['h']}:{_escape_x(str(crop['x_expr']))}:{crop['y']}",
            f"scale={w}:{h}:flags=bicubic",
            "setsar=1",
        ])
    if zoom_times and settings.zoom_punch and crop.get("mode") == "dynamic" \
            and not base.endswith("[v0]"):
        zexpr = zoom_filter(zoom_times)
        if zexpr:
            base += f",scale=iw*({zexpr}):ih*({zexpr}),crop={w}:{h}"
    logo = overlay_logo_filter(w, h, brand_pos, show_logo)
    if logo:
        # "[v0];movie=…[wm];[v0][wm]overlay=…" — label the base output
        # so the movie stream can join it, then continue chaining.
        # (the fit chain already ends with [v0]; the crop chain needs it.)
        vf = base + logo[len("[v0]"):] if base.endswith("[v0]") \
            else base + "[v0]" + logo[len("[v0]"):]
    elif base.endswith("[v0]"):
        # NOTE 2026-09-12: fit chain without a logo must not keep the
        # trailing label — "-vf ...[v0]" is not a valid simple filtergraph
        # (live failure: 2 inputs/2 outputs). Strip it; subtitles append
        # with "," below exactly like the crop path.
        vf = base[:-len("[v0]")]
    else:
        vf = base
    eff_burn = settings.burn_subtitles if burn_subtitles is None else burn_subtitles
    if ass_path and eff_burn:
        vf = vf + "," + subtitles_filter(ass_path)

    audio = ["-c:a", "aac", "-b:a", "160k", "-ac", "2", "-ar", "48000"]
    if settings.loudnorm:
        # Social platforms normalise to about -14 LUFS on playback. Doing it
        # here means the clip sounds the same as everything else in the feed
        # instead of being quietly turned down.
        audio = ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"] + audio

    def build(use_gpu: bool) -> list[str]:
        return [
            settings.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, start):.3f}",
            "-to", f"{end:.3f}",
            "-i", str(source),
            "-vf", vf,
            *_video_args(use_gpu),
            "-pix_fmt", "yuv420p",
            *audio,
            "-movflags", "+faststart",
            str(out_path),
        ]

    want_gpu = settings.encoder == "nvenc" or (settings.encoder == "auto" and _has_nvenc())
    log.info("rendering %s (%.1fs-%.1fs, crop=%s, %dx%d, encoder=%s)",
             out_path.name, start, end, crop["mode"], w, h,
             "nvenc" if want_gpu else "x264")
    proc = subprocess.run(build(want_gpu), capture_output=True, text=True)
    used_gpu = want_gpu
    if proc.returncode != 0 and want_gpu:
        log.warning("nvenc failed (%s), falling back to x264", proc.stderr[-160:].strip())
        proc = subprocess.run(build(False), capture_output=True, text=True)
        used_gpu = False
    if proc.returncode != 0:
        log.debug("ffmpeg -vf failed: %s", vf)
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-400:]}")
    if bumpers and (bumpers.get("intro") or bumpers.get("outro")):
        _attach_bumpers(out_path, w, h, used_gpu, bumpers)
    verify_output(out_path, (w, h))
    return out_path


def _attach_bumpers(out_path: Path, w: int, h: int, use_gpu: bool, bumpers: dict) -> None:
    """Concat branded intro/outro cards around the finished clip.

    Decorative by contract: any failure logs a warning and keeps the
    bare clip — a bumper must never fail a job.
    """
    import shutil

    try:
        workdir = Path(bumpers.get("workdir") or out_path.parent)
        title = str(bumpers.get("title") or settings.brand_text)
        parts: list[Path] = []
        if bumpers.get("intro"):
            card = build_bumper("intro", w, h, title, workdir, use_gpu)
            if card:
                parts.append(card)
        parts.append(out_path)
        if bumpers.get("outro"):
            card = build_bumper("outro", w, h, "تابعنا للمزيد", workdir, use_gpu)
            if card:
                parts.append(card)
        if len(parts) < 2:
            return
        tmp = out_path.with_name(out_path.stem + ".bump.mp4")
        if concat_parts(parts, tmp):
            shutil.move(str(tmp), str(out_path))
        else:
            tmp.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 - bumpers are decoration
        log.warning("bumpers skipped (%s)", exc)


def verify_output(out_path: Path, target_size: tuple[int, int]) -> None:
    """Guarantee the mp4 is exactly the selected quality — nothing else.

    ffprobe is milliseconds; a mismatch means the scale chain did not
    apply (wrong quality would reach the user silently otherwise).
    """
    from app.pipeline.media import probe

    tw, th = int(target_size[0]), int(target_size[1])
    try:
        info = probe(out_path)
    except Exception as exc:  # noqa: BLE001 - unreadable output is a failure
        raise RuntimeError(f"تعذّر التحقق من المقطع: {exc}") from exc
    if int(info["width"]) != tw or int(info["height"]) != th:
        raise RuntimeError(
            f"أبعاد المقطع {info['width']}x{info['height']} "
            f"لا تطابق الجودة المختارة {tw}x{th}"
        )
