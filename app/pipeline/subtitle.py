"""Stage 6 — burned-in subtitles as .ass (libass handles Arabic shaping/bidi).

Requires an ffmpeg built with libass + fribidi + harfbuzz. Run
`python -m scripts.gate_arabic` once to confirm before trusting output.
"""

import re
from pathlib import Path

from app.config import settings
from app.pipeline.transcribe import Word

MAX_CHARS = 32
MAX_WORDS = 6
MAX_LINE_SEC = 4.0
MIN_LINE_SEC = 1.1
CHARS_PER_SEC = 15.0      # comfortable Arabic reading speed

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sabily,{font},{size},&H00FFFFFF,&H00101010,&H80000000,-1,0,1,{outline},2,{subalign},60,60,{margin},1
Style: Brand,{font},{corner},&H00FFFFFF,&H00101010,&H80000000,-1,0,1,{cornerline},1,{balign},{cpad},{cpad},{cvert},1
Style: Source,{font},{corner},&H00E8E8E8,&H00101010,&H80000000,0,0,1,{cornerline},1,{salign},{cpad},{cpad},{cvert},1

[Events]
Format: Layer, Start, End, Style, MarginL, MarginR, MarginV, Effect, Text
"""


def _ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def group_lines(words: list[Word]) -> list[tuple[float, float, str]]:
    lines: list[tuple[float, float, str]] = []
    buf: list[Word] = []
    for w in words:
        buf.append(w)
        text = " ".join(x.text for x in buf)
        span = buf[-1].end - buf[0].start
        ends = w.text.endswith((".", "؟", "!", "?", "،", "…"))
        if len(buf) >= MAX_WORDS or len(text) >= MAX_CHARS or span >= MAX_LINE_SEC or ends:
            lines.append((buf[0].start, buf[-1].end, text))
            buf = []
    if buf:
        lines.append((buf[0].start, buf[-1].end, " ".join(x.text for x in buf)))

    # A line timed to the speech alone can flash by faster than anyone reads.
    # Stretch short lines into the silence that follows, never into the next line.
    out: list[tuple[float, float, str]] = []
    for i, (start, end, text) in enumerate(lines):
        needed = max(MIN_LINE_SEC, len(text) / CHARS_PER_SEC)
        limit = lines[i + 1][0] if i + 1 < len(lines) else end + needed
        out.append((start, min(max(end, start + needed), limit), text))
    return out


ALIGN = {"tl": 7, "tc": 8, "tr": 9, "bl": 1, "bc": 2, "br": 3}


def _safe_tag(s: str) -> str:
    t = (s or "").replace("\r", " ").replace("\n", " ").replace("{", "(").replace("}", ")")
    return t.replace("\\N", " ").replace("\\n", " ")


# Bidi isolates so Latin/digits inside RTL lines keep their order
# (AP36, 3, Keyframe) without reordering the Arabic around them.
LRI, PDI = "\u2066", "\u2069"

# hyphen/dash variants that Whisper and keyboards mix freely:
# U+2010‐ U+2011‑ U+2012‒ U+2013– U+2014— U+2015― U+2212−
_DASHES = "‐‑‒–—―−"
_LATIN_RUN = re.compile(r"[@#]?[A-Za-z0-9]+(?:[._\-/][A-Za-z0-9]+)*")
_AR_PUNCT_BEFORE = ".,،؛؟!?:;,)}\\]]"


def mix_ar_en(text: str) -> str:
    """Display mixing for Arabic↔Latin subtitles.

    - Unifies dash variants (—, –, ‐ …) to a single hyphen.
    - Hyphens inside words/numbers (AP-36) stay glued; standalone
      dashes get breathing spaces.
    - Collapses stray whitespace, glues spaces before punctuation.
    - Wraps Latin/digit runs in bidi isolates so libass/HarfBuzz
      keeps "تقنية AP36 في" ordered instead of flipping it.
    Idempotent: existing isolates are stripped first.
    """
    t = (text or "").replace(LRI, "").replace(PDI, "")
    t = re.sub(f"[{_DASHES}]+", "-", t)
    t = re.sub(r"\s+", " ", t).strip()
    # every hyphen becomes a spaced separator first…
    t = re.sub(r"\s*-\s*", " - ", t)
    # …then Latin/digit compounds re-glue: AP - 36 -> AP-36,
    # while Arabic separators keep breathing room: الإنتاج - الجزء.
    t = re.sub(r"(?<=[A-Za-z0-9]) - (?=[A-Za-z0-9])", "-", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(f"\\s+([{_AR_PUNCT_BEFORE}])", r"\1", t)
    t = _LATIN_RUN.sub(lambda m: f"{LRI}{m.group(0)}{PDI}", t)
    return t


SUB_ALIGN = {"bottom": 2, "middle": 5, "top": 8}


def resolve_sub_align(pos: str | None = None) -> int:
    """Bottom/center/top subtitle position → ASS alignment number."""
    pos = pos or settings.subtitle_pos
    return SUB_ALIGN.get(pos, SUB_ALIGN.get(settings.subtitle_pos, 2))


def group_karaoke(words: list[Word]) -> list[tuple[float, float, list[tuple[str, int]]]]:
    """Group like group_lines but keep per-word karaoke durations (centiseconds)."""
    out = []
    for start, end, text in group_lines(words):
        toks = text.split()
        total = max(1, len("".join(toks)))
        budget = max(len(toks), int(round((end - start) * 100)))
        parts = []
        for tok in toks:
            dur = max(1, int(round(budget * len(tok) / total)))
            parts.append((tok, dur))
        out.append((start, end, parts))
    return out


def _even_karaoke(text: str, duration: float) -> list[tuple[str, int]]:
    toks = text.split() or [text]
    budget = max(len(toks), int(round(max(0.2, duration) * 100)))
    total = max(1, len("".join(toks)))
    return [(tok, max(1, int(round(budget * len(tok) / total)))) for tok in toks]


def _kara_text(parts: list[tuple[str, int]]) -> str:
    return " ".join(f"{{\\kf{d}}}{_safe_tag(mix_ar_en(w))}" for w, d in parts)


def label_turns(rows: list[dict], turns: list[dict], clip_start: float) -> list[dict]:
    """Prefix the first line of each new speaker turn with المتحدث N:.

    Operates on clip-relative rows ([{start,end,text}]); the opening turn
    is left bare. Pure helper, unit-testable.
    """
    if not rows or not turns:
        return rows
    out = [dict(r) for r in rows]
    for t in turns[1:]:
        rel = t["start"] - clip_start
        for r in out:
            if r["end"] > rel and not r["text"].startswith("المتحدث"):
                r["text"] = f"المتحدث {t['speaker']}: " + r["text"]
                break
    return out


def write_card_ass(text: str, w: int, h: int, out: Path,
                   duration: float = 1.5) -> Path:
    """Centered title card burned onto a bumper clip (intro/outro).

    Middle alignment + larger type so the channel card reads instantly.
    """
    safe = _safe_tag(mix_ar_en(text))
    header = HEADER.format(**{**header_params(w, h), "subalign": 5,
                              "size": int(h * 0.055)})
    body = f"Dialogue: 0,{_ts(0)},{_ts(duration)},Sabily,0,0,0,,{safe}"
    out.write_text(header + body + "\n", encoding="utf-8")
    return out


def header_params(
    w: int,
    h: int,
    font: str | None = None,
    brand_pos: str | None = None,
    source_pos: str | None = None,
    sub_align: int | None = None,
) -> dict:
    """Return every placeholder HEADER needs, using the write_ass formulas."""
    return {
        "w": w,
        "h": h,
        "font": font or settings.subtitle_font,
        "subalign": sub_align or 2,
        "size": int(h * 0.042),
        "outline": max(2, int(h * 0.0025)),
        "margin": int(h * 0.16),
        "corner": int(h * 0.03),
        "cornerline": max(2, int(h * 0.0018)),
        "cpad": int(w * 0.045),
        "cvert": int(h * 0.035),
        "balign": ALIGN.get(brand_pos or settings.brand_pos, 9),
        "salign": ALIGN.get(source_pos or settings.source_pos, 7),
    }


def build_header(
    w: int,
    h: int,
    font: str | None = None,
    brand_pos: str | None = None,
    source_pos: str | None = None,
    sub_align: int | None = None,
) -> str:
    """Render HEADER with all placeholders supplied — never a partial .format."""
    return HEADER.format(**header_params(w, h, font, brand_pos, source_pos, sub_align))


def write_ass(
    words: list[Word],
    clip_start: float,
    clip_end: float,
    out: Path,
    source_tag: str = "",
    lines: list[dict] | None = None,
    brand_pos: str | None = None,
    source_pos: str | None = None,
    brand_text: str | None = None,
    target_size: tuple | None = None,
    show_brand: bool | None = None,
    show_source: bool | None = None,
    burn: bool | None = None,
    style: str | None = None,
    pos: str | None = None,
    en_lines: list[str] | None = None,
    turns: list[dict] | None = None,
) -> Path | None:
    """Write subtitles for one clip, timed relative to the clip start.

    Also lays the attribution row across the top: the brand mark top-right,
    the source hashtag top-left. Both are plain ASS dialogue lines with their
    own alignment, so they cost nothing extra at render time.

    Appearance flags (None = global settings default):
      show_brand  — Brand text line (skipped anyway when the logo image
                    overlay is active, to avoid double branding).
      show_source — source hashtag text line.
      burn        — bottom subtitle lines. When everything is off the
                    function returns None (nothing to burn).
      style       — "line" (default) or "karaoke" word-by-word highlight.
      pos         — "bottom" (default) | "middle" | "top" subtitle position.
      en_lines    — optional English translations aligned 1:1 with the
                    subtitle rows (second line per dialogue).
      turns       — speaker turns for optional المتحدث N: prefixes.
    """
    if lines is None:
        inside = [w for w in words if w.start >= clip_start - 0.2 and w.end <= clip_end + 0.2]
        if not inside and not source_tag:
            return None
        grouped = group_lines(inside)
        rows = [{"start": s, "end": e, "text": tx} for s, e, tx in grouped]
    else:
        # already relative to the clip, coming back from the editor
        rows = [{"start": clip_start + r["start"], "end": clip_start + r["end"],
                 "text": r["text"]} for r in lines]

    w, h = target_size if target_size else settings.target_size
    logo_active = settings.use_brand_logo
    eff_burn = settings.burn_subtitles if burn is None else burn
    eff_brand = settings.brand_watermark if show_brand is None else show_brand
    eff_source = settings.source_tag if show_source is None else show_source
    eff_style = style or settings.subtitle_style
    if eff_style not in ("line", "karaoke"):
        eff_style = "line"
    sub_align = resolve_sub_align(pos)
    if settings.speaker_labels and turns:
        rows = label_turns(rows, turns, clip_start)
    kara = eff_style == "karaoke"
    # karaoke timings for the live path come from real word timestamps;
    # the editor path (lines=) reuses stored per-word timings when present.
    kara_map: dict[int, list[tuple[str, int]]] = {}
    if kara and lines is None:
        for idx, (_, _, parts) in enumerate(group_karaoke(
                [w for w in words if w.start >= clip_start - 0.2
                 and w.end <= clip_end + 0.2])):
            kara_map[idx] = parts
    body = []
    if eff_burn:
        for idx, row in enumerate(rows):
            start, end, text = row["start"], row["end"], row["text"]
            s = max(0.0, start - clip_start)
            e = min(clip_end - clip_start, end - clip_start)
            if e <= s:
                continue
            if kara:
                parts = kara_map.get(idx)
                if parts is None:
                    if isinstance(row.get("kara"), list) and row["kara"]:
                        parts = [(str(p[0]), max(1, int(p[1])))
                                 for p in row["kara"] if len(p) >= 2]
                    else:
                        parts = _even_karaoke(text, e - s)
                line = _kara_text(parts)
            else:
                line = _safe_tag(mix_ar_en(text))
            if en_lines and idx < len(en_lines) and str(en_lines[idx]).strip():
                line += "\\N" + _safe_tag(str(en_lines[idx]).strip())
            body.append(f"Dialogue: 0,{_ts(s)},{_ts(e)},Sabily,0,0,0,,{line}")

    span = max(0.0, clip_end - clip_start)
    brand = settings.brand_text if brand_text is None else brand_text
    # image watermark replaces the Brand text line (no double branding);
    # the source hashtag stays as text, sized (corner) to match the logo.
    if eff_brand and brand and not logo_active:
        body.insert(0, f"Dialogue: 0,{_ts(0)},{_ts(span)},Brand,0,0,0,,{_safe_tag(mix_ar_en(brand))}")
    if eff_source and source_tag:
        body.insert(0, f"Dialogue: 0,{_ts(0)},{_ts(span)},Source,0,0,0,,{_safe_tag(mix_ar_en(source_tag))}")

    if not body:
        return None
    header = build_header(w, h, brand_pos=brand_pos, source_pos=source_pos,
                          sub_align=sub_align)
    out.write_text(header + "\n".join(body) + "\n", encoding="utf-8")
    return out
