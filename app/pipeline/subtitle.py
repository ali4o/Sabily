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
    return lines


def write_ass(words: list[Word], clip_start: float, clip_end: float, out: Path) -> Path | None:
    """Write subtitles for one clip, timed relative to the clip start."""
    inside = [w for w in words if w.start >= clip_start - 0.2 and w.end <= clip_end + 0.2]
    if not inside:
        return None

    w, h = settings.target_size
    body = []
    for start, end, text in group_lines(inside):
        s = max(0.0, start - clip_start)
        e = min(clip_end - clip_start, end - clip_start)
        if e <= s:
            continue
        safe = text.replace("\n", " ").replace("{", "(").replace("}", ")")
        body.append(f"Dialogue: 0,{_ts(s)},{_ts(e)},Sabily,0,0,0,,{safe}")

    header = HEADER.format(
        w=w, h=h,
        font=settings.subtitle_font,
        size=int(h * 0.042),
        outline=max(2, int(h * 0.0025)),
        margin=int(h * 0.16),
    )
    out.write_text(header + "\n".join(body) + "\n", encoding="utf-8")
    return out
