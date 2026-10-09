"""Upload to YouTube Shorts via the YouTube Data API v3.

Setup: create an OAuth 2.0 "Desktop app" client in Google Cloud Console with the
YouTube Data API v3 enabled, download it as client_secret.json, and point
YOUTUBE_CLIENT_SECRETS at it. The first upload opens a browser for consent and
caches the token in YOUTUBE_TOKEN_FILE (default: ~/.config/clipper/youtube_token.json).
"""

from __future__ import annotations

import os
from pathlib import Path

from ..models import Clip
from . import caption_text

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
PRIVACY = {"private": "private", "unlisted": "unlisted", "public": "public"}


def _credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_file = Path(
        os.environ.get("YOUTUBE_TOKEN_FILE", "~/.config/clipper/youtube_token.json")
    ).expanduser()
    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        secrets = os.environ.get("YOUTUBE_CLIENT_SECRETS", "client_secret.json")
        if not Path(secrets).exists():
            raise RuntimeError(
                f"YouTube OAuth client file not found at {secrets!r}; set YOUTUBE_CLIENT_SECRETS."
            )
        flow = InstalledAppFlow.from_client_secrets_file(secrets, SCOPES)
        creds = flow.run_local_server(port=0)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(creds.to_json())
    return creds


def publish(video: Path, clip: Clip, privacy: str = "private") -> str:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    youtube = build("youtube", "v3", credentials=_credentials())
    title = clip.title if "#shorts" in clip.title.lower() else f"{clip.title} #Shorts"
    body = {
        "snippet": {
            "title": title[:100],
            "description": caption_text(clip, limit=5000),
            "tags": clip.hashtags[:15],
            "categoryId": "22",  # People & Blogs
        },
        "status": {
            "privacyStatus": PRIVACY.get(privacy, "private"),
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(video), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        _, response = request.next_chunk()
    return f"https://youtube.com/shorts/{response['id']}"
