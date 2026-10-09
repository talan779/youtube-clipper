"""Post an Instagram Reel via the Instagram Graph API (resumable upload).

Setup: an Instagram Business or Creator account linked to a Facebook Page, a Meta
app with `instagram_content_publish`, and a long-lived access token. Export
INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_USER_ID (the IG user id, not the page id).
Instagram has no private posting mode, so this refuses to run unless privacy is "public".
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from ..models import Clip
from . import caption_text

GRAPH_VERSION = os.environ.get("INSTAGRAM_GRAPH_VERSION", "v21.0")
GRAPH = f"https://graph.facebook.com/{GRAPH_VERSION}"
RUPLOAD = f"https://rupload.facebook.com/ig-api-upload/{GRAPH_VERSION}"


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} to post to Instagram.")
    return value


def _check(resp: requests.Response) -> dict:
    data = resp.json()
    if resp.status_code >= 400 or "error" in data:
        raise RuntimeError(f"Instagram API error {resp.status_code}: {data.get('error', data)}")
    return data


def publish(video: Path, clip: Clip, privacy: str = "private") -> str:
    if privacy != "public":
        raise RuntimeError("Instagram Reels are always public; re-run with --privacy public to post there.")
    token, user_id = _env("INSTAGRAM_ACCESS_TOKEN"), _env("INSTAGRAM_USER_ID")

    container = _check(
        requests.post(
            f"{GRAPH}/{user_id}/media",
            data={
                "media_type": "REELS",
                "upload_type": "resumable",
                "caption": caption_text(clip, limit=2200),
                "share_to_feed": "true",
                "access_token": token,
            },
            timeout=60,
        )
    )["id"]

    size = video.stat().st_size
    with video.open("rb") as f:
        _check(
            requests.post(
                f"{RUPLOAD}/{container}",
                headers={"Authorization": f"OAuth {token}", "offset": "0", "file_size": str(size)},
                data=f,
                timeout=600,
            )
        )

    for _ in range(60):
        status = _check(
            requests.get(
                f"{GRAPH}/{container}",
                params={"fields": "status_code,status", "access_token": token},
                timeout=30,
            )
        )
        if status.get("status_code") == "FINISHED":
            break
        if status.get("status_code") in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"Instagram processing failed: {status.get('status')}")
        time.sleep(5)
    else:
        raise RuntimeError("Instagram is still processing the video after 5 minutes; try publishing later.")

    media = _check(
        requests.post(
            f"{GRAPH}/{user_id}/media_publish",
            data={"creation_id": container, "access_token": token},
            timeout=60,
        )
    )
    return media["id"]
