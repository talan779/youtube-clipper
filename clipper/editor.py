"""Cut, reframe to 9:16, burn in captions and normalize audio with ffmpeg."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .captions import HEIGHT, WIDTH, CaptionStyle, build_ass
from .models import Clip, Transcript

REFRAME_MODES = ("auto", "face", "center", "blur")


def probe(video: Path) -> tuple[int, int, float]:
    """Return (width, height, duration_seconds) of the first video stream."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:format=duration",
            "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"]), float(data["format"]["duration"])


def detect_face_center(video: Path, start: float, end: float, samples: int = 12) -> float | None:
    """Median horizontal face position (0..1) across the clip, or None if no faces / no OpenCV."""
    try:
        import cv2
    except ImportError:
        return None

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(video))
    centers: list[float] = []
    try:
        step = (end - start) / max(1, samples)
        for i in range(samples):
            cap.set(cv2.CAP_PROP_POS_MSEC, (start + step * (i + 0.5)) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces):
                x, _, w, _ = max(faces, key=lambda f: f[2] * f[3])  # largest face = speaker
                centers.append((x + w / 2) / frame.shape[1])
    finally:
        cap.release()

    if len(centers) < max(2, samples // 4):
        return None
    centers.sort()
    return centers[len(centers) // 2]


def build_video_filter(
    src_w: int, src_h: int, mode: str, face_x: float | None, ass_name: str
) -> str:
    target_ratio = WIDTH / HEIGHT
    if src_w / src_h <= target_ratio + 0.01:
        # Already vertical (or narrower): fit and pad.
        base = (
            f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"
        )
    elif mode == "blur" or (mode == "auto" and face_x is None):
        base = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={WIDTH}:{HEIGHT},gblur=sigma=40,eq=brightness=-0.08[bgb];"
            f"[fg]scale={WIDTH}:-2[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2"
        )
    else:
        crop_w = int(src_h * target_ratio) // 2 * 2
        center = face_x if (face_x is not None and mode in ("auto", "face")) else 0.5
        x = int(min(max(center * src_w - crop_w / 2, 0), src_w - crop_w))
        base = f"[0:v]crop={crop_w}:{src_h}:{x}:0,scale={WIDTH}:{HEIGHT}"
    return f"{base},setsar=1,ass={ass_name}[v]"


def render_clip(
    source: Path,
    clip: Clip,
    transcript: Transcript,
    out_path: Path,
    mode: str = "auto",
    style: CaptionStyle | None = None,
    show_hook: bool = True,
) -> Path:
    if mode not in REFRAME_MODES:
        raise ValueError(f"mode must be one of {REFRAME_MODES}")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")

    source = source.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src_w, src_h, _ = probe(source)
    face_x = detect_face_center(source, clip.start, clip.end) if mode in ("auto", "face") else None

    ass = build_ass(
        transcript.words_between(clip.start, clip.end),
        clip.start,
        clip.end,
        hook=clip.hook if show_hook else None,
        style=style,
    )

    with tempfile.TemporaryDirectory() as tmp:
        # Run ffmpeg inside tmp so the ass filter gets a plain relative filename
        # (avoids filter-graph escaping of ':' and '\' in absolute paths).
        (Path(tmp) / "subs.ass").write_text(ass, encoding="utf-8")
        vf = build_video_filter(src_w, src_h, mode, face_x, "subs.ass")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{clip.start:.3f}", "-i", str(source), "-t", f"{clip.duration:.3f}",
            "-filter_complex", vf,
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-r", "30",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-movflags", "+faststart",
            str(out_path),
        ]
        result = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    return out_path


def slugify(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "clip"
