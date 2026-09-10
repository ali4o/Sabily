"""Stage 6 — burned-in subtitles as .ass (libass handles Arabic shaping/bidi).

Requires an ffmpeg built with libass + fribidi + harfbuzz. Run
`python -m scripts.gate_arabic` once to confirm before trusting output.
"""

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
Style: Sabily,{font},{size},&H00FFFFFF,&H00101010,&H80000000,-1,0,1,{outline},2,2,60,60,{margin},1
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
) -> Path | None:
    """Write subtitles for one clip, timed relative to the clip start.

    Also lays the attribution row across the top: the brand mark top-right,
    the source hashtag top-left. Both are plain ASS dialogue lines with their
    own alignment, so they cost nothing extra at render time.
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

    w, h = settings.target_size
    body = []
    for row in rows:
        start, end, text = row["start"], row["end"], row["text"]
        s = max(0.0, start - clip_start)
        e = min(clip_end - clip_start, end - clip_start)
        if e <= s:
            continue
        safe = text.replace("\n", " ").replace("{", "(").replace("}", ")")
        body.append(f"Dialogue: 0,{_ts(s)},{_ts(e)},Sabily,0,0,0,,{safe}")

    span = max(0.0, clip_end - clip_start)
    brand = settings.brand_text if brand_text is None else brand_text
    if settings.brand_watermark and brand:
        body.insert(0, f"Dialogue: 0,{_ts(0)},{_ts(span)},Brand,0,0,0,,{brand}")
    if settings.source_tag and source_tag:
        body.insert(0, f"Dialogue: 0,{_ts(0)},{_ts(span)},Source,0,0,0,,{source_tag}")

    header = HEADER.format(
        w=w, h=h,
        font=settings.subtitle_font,
        size=int(h * 0.042),
        outline=max(2, int(h * 0.0025)),
        margin=int(h * 0.16),
        corner=int(h * 0.026),
        cornerline=max(2, int(h * 0.0018)),
        cpad=int(w * 0.045),
        cvert=int(h * 0.035),
        balign=ALIGN.get(brand_pos or settings.brand_pos, 9),
        salign=ALIGN.get(source_pos or settings.source_pos, 7),
    )
    out.write_text(header + "\n".join(body) + "\n", encoding="utf-8")
    return out
