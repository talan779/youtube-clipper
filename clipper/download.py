"""Download a YouTube (or any yt-dlp supported) video."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BROWSERS = ("firefox", "edge", "chrome", "brave", "opera", "vivaldi")
BOT_CHECK = re.compile(r"sign in to confirm|not a bot|cookies", re.I)

COOKIE_HELP = (
    "YouTube is asking the downloader to prove it isn't a bot. To fix it, give the clipper "
    "your YouTube login cookies:\n"
    "1. In Chrome, install the extension \"Get cookies.txt LOCALLY\".\n"
    "2. Open youtube.com (signed in), click the extension and click Export.\n"
    f"3. Move the downloaded file (it ends in cookies.txt) into {REPO_ROOT}\n"
    "Then try again. (Keep that file private: it logs into your YouTube account.)"
)


def cookie_file() -> Path | None:
    """cookies.txt (or e.g. www.youtube.com_cookies.txt from the browser extension)."""
    for folder in dict.fromkeys((Path.cwd(), REPO_ROOT)):
        found = sorted(folder.glob("*cookies*.txt"), key=lambda p: p.name != "cookies.txt")
        for p in found:
            if p.is_file() and p.stat().st_size:
                return p
    return None


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def download(url: str, out_dir: Path, max_height: int = 1080) -> Path:
    """Download `url` into `out_dir` as source.mp4 and return its path.

    Also writes metadata.json (title, channel, duration, ...) next to it. If
    YouTube demands a sign-in ("confirm you're not a bot"), retries with a
    cookies.txt in the clipper folder, then with cookies from installed browsers.
    """
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "source.mp4"
    base = {
        "format": (
            f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
            f"b[height<={max_height}][ext=mp4]/bv*[height<={max_height}]+ba/b"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    attempts: list[tuple[str, dict]] = [("", {})]
    cf = cookie_file()
    if cf:
        attempts.insert(0, (f"using {cf.name}", {"cookiefile": str(cf)}))
    attempts += [(f"using {b} cookies", {"cookiesfrombrowser": (b,)}) for b in BROWSERS]

    info, last_err = None, None
    for i, (label, extra) in enumerate(attempts):
        if i and not (last_err and BOT_CHECK.search(str(last_err))):
            break  # only fall back for the sign-in / bot check
        if label:
            _say(f"  retrying {label}…")
        try:
            with yt_dlp.YoutubeDL({**base, **extra}) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except Exception as e:  # yt-dlp DownloadError, cookie-database errors, ...
            if extra.get("cookiesfrombrowser") and not BOT_CHECK.search(str(e)):
                last_err = last_err or e  # browser missing/locked: try the next one
                continue
            last_err = e
    if info is None:
        if last_err and BOT_CHECK.search(str(last_err)):
            raise RuntimeError(COOKIE_HELP) from last_err
        raise RuntimeError(re.sub(r"\x1b\[[0-9;]*m", "", str(last_err))) from last_err

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
