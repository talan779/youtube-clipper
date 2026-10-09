"""Post to TikTok via the Content Posting API (Direct Post, file upload).

Setup: register an app at developers.tiktok.com with the Content Posting API and
the `video.publish` scope, complete OAuth for your account, and export the user
access token as TIKTOK_ACCESS_TOKEN. Until TikTok audits your app, posts are
forced to SELF_ONLY (private) visibility.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from ..models import Clip
from . import caption_text

API = "https://open.tiktokapis.com/v2"
PRIVACY = {"private": "SELF_ONLY", "unlisted": "MUTUAL_FOLLOW_FRIENDS", "public": "PUBLIC_TO_EVERYONE"}
MIN_CHUNK = 5 * 1024 * 1024
MAX_SINGLE = 64 * 1024 * 1024
CHUNK = 10 * 1024 * 1024


def _token() -> str:
    token = os.environ.get("TIKTOK_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Set TIKTOK_ACCESS_TOKEN to a user access token with video.publish scope.")
    return token


def _check(resp: requests.Response) -> dict:
    data = resp.json()
    err = data.get("error", {})
    if resp.status_code >= 400 or err.get("code") not in (None, "ok"):
        raise RuntimeError(f"TikTok API error {resp.status_code}: {err or data}")
    return data.get("data", {})


def plan_chunks(size: int) -> tuple[int, int]:
    """Return (chunk_size, total_chunk_count) per TikTok's upload rules."""
    if size <= MAX_SINGLE:
        return size, 1
    count = size // CHUNK  # the last chunk absorbs the remainder
    return CHUNK, count


def publish(video: Path, clip: Clip, privacy: str = "private") -> str:
    headers = {"Authorization": f"Bearer {_token()}", "Content-Type": "application/json; charset=UTF-8"}
    size = video.stat().st_size
    chunk_size, count = plan_chunks(size)

    init = _check(
        requests.post(
            f"{API}/post/publish/video/init/",
            headers=headers,
            json={
                "post_info": {
                    "title": f"{clip.title}\n\n{caption_text(clip)}"[:2200],
                    "privacy_level": PRIVACY.get(privacy, "SELF_ONLY"),
                    "disable_comment": False,
                    "disable_duet": False,
                    "disable_stitch": False,
                },
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": size,
                    "chunk_size": chunk_size,
                    "total_chunk_count": count,
                },
            },
            timeout=60,
        )
    )
    publish_id, upload_url = init["publish_id"], init["upload_url"]

    with video.open("rb") as f:
        for i in range(count):
            start = i * chunk_size
            end = size - 1 if i == count - 1 else start + chunk_size - 1
            f.seek(start)
            body = f.read(end - start + 1)
            resp = requests.put(
                upload_url,
                data=body,
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Length": str(len(body)),
                    "Content-Range": f"bytes {start}-{end}/{size}",
                },
                timeout=300,
            )
            if resp.status_code not in (200, 201, 206):
                raise RuntimeError(f"TikTok chunk upload failed ({resp.status_code}): {resp.text}")

    for _ in range(60):
        status = _check(
            requests.post(
                f"{API}/post/publish/status/fetch/",
                headers=headers,
                json={"publish_id": publish_id},
                timeout=30,
            )
        )
        state = status.get("status")
        if state == "PUBLISH_COMPLETE":
            return publish_id
        if state == "FAILED":
            raise RuntimeError(f"TikTok publish failed: {status.get('fail_reason')}")
        time.sleep(5)
    return publish_id  # still processing; TikTok will finish it server-side
