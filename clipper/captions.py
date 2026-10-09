"""Build animated "word-pop" captions as an ASS subtitle file.

Words are grouped into short chunks (TikTok style). Within a chunk the word
being spoken is highlighted and slightly enlarged, the rest stay white.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Word

WIDTH, HEIGHT = 1080, 1920


@dataclass
class CaptionStyle:
    font: str = "DejaVu Sans"
    size: int = 84
    highlight: str = "00FFFF"  # BGR hex as ASS expects: 00FFFF = yellow
    outline: int = 6
    words_per_chunk: int = 3
    max_chunk_seconds: float = 1.6
    uppercase: bool = True
    margin_v: int = 520  # distance from bottom edge
    hook_size: int = 76
    hook_seconds: float = 3.0


def chunk_words(words: list[Word], max_words: int, max_seconds: float) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        if current and (
            len(current) >= max_words
            or w.end - current[0].start > max_seconds
            or w.start - current[-1].end > 0.6  # pause in speech
            or current[-1].text[-1:] in ".?!"
        ):
            chunks.append(current)
            current = []
        current.append(w)
    if current:
        chunks.append(current)
    return chunks


def ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass(
    words: list[Word],
    clip_start: float,
    clip_end: float,
    hook: str | None = None,
    style: CaptionStyle | None = None,
) -> str:
    """Return ASS file contents. Word times are absolute; output is clip-relative."""
    st = style or CaptionStyle()
    duration = clip_end - clip_start

    rel = [
        Word(start=max(0.0, w.start - clip_start), end=min(duration, w.end - clip_start), text=w.text)
        for w in words
        if w.end > clip_start and w.start < clip_end
    ]

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {WIDTH}",
        f"PlayResY: {HEIGHT}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Caption,{st.font},{st.size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,{st.outline},2,2,60,60,{st.margin_v},1",
        f"Style: Hook,{st.font},{st.hook_size},&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,"
        f"-1,0,0,0,100,100,0,0,3,18,0,8,80,80,260,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    if hook:
        text = _escape(hook.upper() if st.uppercase else hook)
        end = min(st.hook_seconds, duration)
        lines.append(
            f"Dialogue: 1,{ass_time(0)},{ass_time(end)},Hook,,0,0,0,,"
            f"{{\\fad(150,250)}}{text}"
        )

    for chunk in chunk_words(rel, st.words_per_chunk, st.max_chunk_seconds):
        for i, active in enumerate(chunk):
            start = active.start
            # Hold each highlight until the next word starts so captions don't flicker.
            end = chunk[i + 1].start if i + 1 < len(chunk) else active.end
            if end <= start:
                continue
            parts = []
            for j, w in enumerate(chunk):
                word = _escape(w.text.upper() if st.uppercase else w.text)
                if j == i:
                    parts.append(f"{{\\c&H{st.highlight}&\\fscx112\\fscy112}}{word}{{\\r}}")
                else:
                    parts.append(word)
            lines.append(
                f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Caption,,0,0,0,,{' '.join(parts)}"
            )

    return "\n".join(lines) + "\n"
