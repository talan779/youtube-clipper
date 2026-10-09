"""Speech-to-text with word-level timestamps (faster-whisper, runs locally)."""

from __future__ import annotations

from pathlib import Path

from .models import Segment, Transcript, Word


def transcribe(
    video: Path,
    model_size: str = "small",
    device: str = "auto",
    language: str | None = None,
) -> Transcript:
    from faster_whisper import WhisperModel

    compute_type = "int8" if device in ("cpu", "auto") else "float16"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    raw_segments, info = model.transcribe(
        str(video),
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )

    segments: list[Segment] = []
    for seg in raw_segments:
        words = [
            Word(start=round(w.start, 3), end=round(w.end, 3), text=w.word.strip())
            for w in (seg.words or [])
            if w.word.strip()
        ]
        segments.append(
            Segment(start=round(seg.start, 3), end=round(seg.end, 3), text=seg.text.strip(), words=words)
        )
    return Transcript(language=info.language, segments=segments)
