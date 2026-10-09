"""Publishers for short-form platforms.

Each publisher exposes `publish(video_path, clip, privacy) -> str` returning the
platform's id or URL for the new post.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..models import Clip

PLATFORMS = ("youtube", "tiktok", "instagram")


def get_publisher(platform: str) -> Callable[[Path, Clip, str], str]:
    if platform == "youtube":
        from .youtube import publish
    elif platform == "tiktok":
        from .tiktok import publish
    elif platform == "instagram":
        from .instagram import publish
    else:
        raise ValueError(f"Unknown platform {platform!r}; choose from {PLATFORMS}")
    return publish


def caption_text(clip: Clip, limit: int | None = None) -> str:
    tags = " ".join(f"#{t}" for t in clip.hashtags)
    text = f"{clip.description}\n\n{tags}".strip()
    return text[:limit] if limit else text
