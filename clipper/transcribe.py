"""Speech-to-text with word-level timestamps (faster-whisper, runs locally)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .models import Segment, Transcript, Word


SAMPLE_RATE = 16000


def load_audio(video: Path):
    """Decode the audio track to 16 kHz mono float32 with ffmpeg.

    faster-whisper's own decoder relies on PyAV, whose API changes between
    releases (e.g. `metadata_errors` was removed), so we use ffmpeg directly.
    """
    import numpy as np

    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
    ]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH; install it and open a new terminal.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg couldn't read the audio: {e.stderr.decode(errors='replace')}")
    if not raw:
        raise RuntimeError(f"No audio track found in {video}")
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


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
        load_audio(video),
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
