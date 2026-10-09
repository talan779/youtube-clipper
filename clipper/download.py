"""Download a YouTube (or any yt-dlp supported) video."""

from __future__ import annotations

import json
from pathlib import Path


def download(url: str, out_dir: Path, max_height: int = 1080) -> Path:
    """Download `url` into `out_dir` as source.mp4 and return its path.

    Also writes metadata.json (title, channel, duration, ...) next to it.
    """
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "source.mp4"
    opts = {
        "format": (
            f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
            f"b[height<={max_height}][ext=mp4]/bv*[height<={max_height}]+ba/b"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    meta = {
        k: info.get(k)
        for k in ("id", "title", "channel", "uploader", "duration", "webpage_url", "description")
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    if not target.exists():
        # yt-dlp may have kept a different container; pick up whatever it wrote.
        candidates = sorted(out_dir.glob("source.*"))
        candidates = [c for c in candidates if c.suffix not in (".json", ".part")]
        if not candidates:
            raise RuntimeError(f"yt-dlp finished but no video file found in {out_dir}")
        return candidates[0]
    return target
