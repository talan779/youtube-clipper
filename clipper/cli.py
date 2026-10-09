"""Command-line entry point.

A "project" is a folder holding everything for one source video:

    workspace/<id>/
        source.mp4        downloaded or copied video
        metadata.json     title etc. (when downloaded)
        transcript.json   word-timestamped transcript
        claude_prompt.txt copy-paste mode: the prompt to paste into the Claude app
        claude_reply.txt  copy-paste mode: Claude's answer, pasted back in
        clips.json        clips chosen by Claude (edit by hand if you like)
        renders/          finished vertical clips
        posted.json       what has been posted where

Each step skips work whose output already exists, so you can re-run, tweak
clips.json, delete a render, etc. Use --force to redo a step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .models import Transcript, load_clips, save_clips

DEFAULT_WORKSPACE = Path("workspace")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def project_dir_for(source: str, workspace: Path) -> Path:
    if Path(source).exists():
        name = Path(source).stem
    else:
        m = re.search(r"(?:v=|youtu\.be/|shorts/|live/)([A-Za-z0-9_-]{11})", source)
        name = m.group(1) if m else hashlib.sha1(source.encode()).hexdigest()[:11]
    return workspace / re.sub(r"[^A-Za-z0-9_-]+", "-", name)


def find_source(project: Path) -> Path:
    for p in sorted(project.glob("source.*")):
        if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov"):
            return p
    raise SystemExit(f"No source video in {project}; run `clipper download` first.")


# --- steps -----------------------------------------------------------------


def step_download(source: str, project: Path, force: bool = False) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    existing = [p for p in project.glob("source.*") if p.suffix != ".part"]
    if existing and not force:
        log(f"✓ source already present: {existing[0]}")
        return existing[0]
    if Path(source).exists():
        dest = project / f"source{Path(source).suffix.lower()}"
        shutil.copy2(source, dest)
        log(f"✓ copied local file → {dest}")
        return dest
    from .download import download

    log(f"↓ downloading {source}")
    path = download(source, project)
    log(f"✓ downloaded → {path}")
    return path


def step_transcribe(project: Path, model: str, language: str | None, force: bool = False) -> Transcript:
    out = project / "transcript.json"
    if out.exists() and not force:
        log("✓ transcript already present")
        return Transcript.load(out)
    from .transcribe import transcribe

    log(f"✎ transcribing with whisper '{model}' (this can take a while on CPU)")
    transcript = transcribe(find_source(project), model_size=model, language=language)
    transcript.save(out)
    log(f"✓ transcript: {len(transcript.segments)} segments, language={transcript.language}")
    return transcript


def use_manual_mode(flag: bool) -> bool:
    """Copy-paste mode when asked for, or automatically when there's no API key."""
    return flag or not os.environ.get("ANTHROPIC_API_KEY")


def _copy_to_clipboard(text: str) -> bool:
    try:
        if sys.platform == "win32":
            subprocess.run(["clip"], input=text.encode("utf-16-le"), check=True)
        elif sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
        else:
            return False
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _open_file(path: Path) -> None:
    try:
        if sys.platform == "win32":
            subprocess.Popen(["notepad.exe", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-t", str(path)])
    except OSError:
        pass


def manual_find(project: Path, transcript: Transcript, num: int, min_s: float, max_s: float,
                title: str | None, force: bool = False) -> list[dict]:
    """Have the user run the prompt in the Claude app and paste the answer back."""
    from .highlights import build_manual_prompt, parse_reply

    prompt_path = project / "claude_prompt.txt"
    reply_path = project / "claude_reply.txt"
    if force and reply_path.exists():
        reply_path.unlink()

    def read_reply() -> list[dict] | None:
        if reply_path.exists() and reply_path.read_text(encoding="utf-8-sig").strip():
            return parse_reply(reply_path.read_text(encoding="utf-8-sig"))
        return None

    raw = read_reply()
    if raw is not None:
        log(f"✓ using Claude's reply from {reply_path}")
        return raw

    prompt_path.write_text(build_manual_prompt(transcript, num, min_s, max_s, title), encoding="utf-8")
    reply_path.write_text("", encoding="utf-8")
    copied = _copy_to_clipboard(prompt_path.read_text(encoding="utf-8"))

    log("")
    log("★ COPY-PASTE MODE (free, no API key)")
    if copied:
        log("  1. The prompt is already COPIED. Open the Claude app, start a new chat,")
        log("     press Ctrl+V and send it.")
    else:
        log(f"  1. Open {prompt_path}, copy all of it (Ctrl+A, Ctrl+C), paste it into")
        log("     a new Claude chat and send it.")
    log("  2. Copy Claude's whole reply (the copy button under it).")
    log("  3. Notepad is opening claude_reply.txt: paste the reply there, press")
    log("     Ctrl+S to save, and close Notepad.")
    log("  4. Come back here and press Enter.")
    log("")
    _open_file(reply_path)

    if not sys.stdin.isatty():
        raise SystemExit(f"Paste Claude's reply into {reply_path}, then run this command again.")
    while True:
        input("  Press Enter once you've saved the reply... ")
        try:
            raw = read_reply()
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            log(f"✗ couldn't read that reply ({e}). Make sure you pasted Claude's whole answer, save, try again.")
            continue
        if raw is None:
            log(f"✗ {reply_path.name} is still empty. Paste the reply, press Ctrl+S, then Enter.")
            continue
        return raw


def step_find(project: Path, num: int, min_s: float, max_s: float, force: bool = False,
              manual: bool = False):
    out = project / "clips.json"
    if out.exists() and not force:
        log("✓ clips.json already present")
        return load_clips(out)
    from .highlights import find_highlights, postprocess

    transcript = Transcript.load(project / "transcript.json")
    title = None
    meta = project / "metadata.json"
    if meta.exists():
        title = json.loads(meta.read_text()).get("title")
    if use_manual_mode(manual):
        raw = manual_find(project, transcript, num, min_s, max_s, title, force)
        clips = postprocess(raw, transcript, num, min_s, max_s)
        if not clips:
            raise SystemExit("None of Claude's clips matched the transcript; delete claude_reply.txt and retry.")
    else:
        log(f"★ asking Claude for the {num} most viral moments")
        clips = find_highlights(transcript, num, min_s, max_s, video_title=title)
    save_clips(clips, out)
    for i, c in enumerate(clips, 1):
        log(f"  {i}. [{c.virality_score:>3}] {c.start:7.1f}s–{c.end:7.1f}s  {c.title}")
    return clips


def step_render(project: Path, mode: str, show_hook: bool, force: bool = False) -> list[Path]:
    from .editor import render_clip, slugify

    source = find_source(project)
    transcript = Transcript.load(project / "transcript.json")
    clips = load_clips(project / "clips.json")
    outputs = []
    for i, clip in enumerate(clips, 1):
        out = project / "renders" / f"{i:02d}-{slugify(clip.title)}.mp4"
        if out.exists() and not force:
            log(f"✓ already rendered: {out.name}")
        else:
            log(f"▶ rendering {i}/{len(clips)}: {clip.title}")
            render_clip(source, clip, transcript, out, mode=mode, show_hook=show_hook)
        outputs.append(out)
    log(f"✓ {len(outputs)} clips in {project / 'renders'}")
    return outputs


def step_post(project: Path, platforms: list[str], privacy: str, only: list[int] | None) -> None:
    from .publish import get_publisher

    clips = load_clips(project / "clips.json")
    renders = sorted((project / "renders").glob("*.mp4"))
    ledger_path = project / "posted.json"
    ledger: dict = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}

    for i, clip in enumerate(clips, 1):
        if only and i not in only:
            continue
        video = next((r for r in renders if r.name.startswith(f"{i:02d}-")), None)
        if video is None:
            log(f"✗ clip {i} not rendered yet; skipping")
            continue
        for platform in platforms:
            key = f"{video.name}:{platform}"
            if key in ledger:
                log(f"✓ {video.name} already on {platform}: {ledger[key]}")
                continue
            log(f"⇪ posting {video.name} to {platform} ({privacy})")
            try:
                result = get_publisher(platform)(video, clip, privacy)
            except Exception as e:  # keep going with the other platforms/clips
                log(f"✗ {platform} failed: {e}")
                continue
            ledger[key] = result
            ledger_path.write_text(json.dumps(ledger, indent=2))
            log(f"✓ posted: {result}")


# --- CLI -------------------------------------------------------------------


def _parse_platforms(value: str) -> list[str]:
    from .publish import PLATFORMS

    items = [p.strip().lower() for p in value.split(",") if p.strip()]
    bad = [p for p in items if p not in PLATFORMS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown platform(s) {bad}; choose from {PLATFORMS}")
    return items


def _parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    from .editor import REFRAME_MODES

    p = argparse.ArgumentParser(prog="clipper", description=__doc__.split("\n")[0])
    p.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    sub = p.add_subparsers(dest="command", required=True)

    def add_find_opts(sp):
        sp.add_argument("-n", "--clips", type=int, default=5, help="number of clips to make")
        sp.add_argument("--min", dest="min_s", type=float, default=20, help="min clip seconds")
        sp.add_argument("--max", dest="max_s", type=float, default=60, help="max clip seconds")
        sp.add_argument("--manual", action="store_true",
                        help="copy-paste with the Claude app instead of the API "
                             "(automatic when ANTHROPIC_API_KEY isn't set)")

    def add_transcribe_opts(sp):
        sp.add_argument("--whisper-model", default="small", help="tiny/base/small/medium/large-v3")
        sp.add_argument("--language", default=None, help="force language code, e.g. en")

    def add_render_opts(sp):
        sp.add_argument("--mode", choices=REFRAME_MODES, default="auto",
                        help="auto: follow face if found, else blurred background")
        sp.add_argument("--no-hook", action="store_true", help="don't overlay the hook text")

    def add_post_opts(sp, flag: str):
        sp.add_argument(flag, type=_parse_platforms, default=[],
                        help="comma-separated: youtube,tiktok,instagram")
        sp.add_argument("--privacy", choices=("private", "unlisted", "public"), default="private")

    run = sub.add_parser("run", help="do everything: download → transcribe → find → render [→ post]")
    run.add_argument("source", help="YouTube URL or local video file")
    run.add_argument("--force", action="store_true", help="redo every step")
    add_transcribe_opts(run)
    add_find_opts(run)
    add_render_opts(run)
    add_post_opts(run, "--post")

    dl = sub.add_parser("download", help="download a video into a new project")
    dl.add_argument("source")
    dl.add_argument("--force", action="store_true")

    for name, help_ in (
        ("transcribe", "transcribe a project's video"),
        ("find", "pick viral clips with Claude"),
        ("render", "render clips as captioned vertical videos"),
    ):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("project", type=Path)
        sp.add_argument("--force", action="store_true")
        {"transcribe": add_transcribe_opts, "find": add_find_opts, "render": add_render_opts}[name](sp)

    post = sub.add_parser("post", help="upload rendered clips")
    post.add_argument("project", type=Path)
    add_post_opts(post, "--to")
    post.add_argument("--only", type=_parse_ints, default=None, help="clip numbers, e.g. 1,3")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        project = project_dir_for(args.source, args.workspace)
        step_download(args.source, project, args.force)
        step_transcribe(project, args.whisper_model, args.language, args.force)
        step_find(project, args.clips, args.min_s, args.max_s, args.force, args.manual)
        step_render(project, args.mode, not args.no_hook, args.force)
        if args.post:
            step_post(project, args.post, args.privacy, None)
        print(project)
    elif args.command == "download":
        project = project_dir_for(args.source, args.workspace)
        step_download(args.source, project, args.force)
        print(project)
    elif args.command == "transcribe":
        step_transcribe(args.project, args.whisper_model, args.language, args.force)
    elif args.command == "find":
        step_find(args.project, args.clips, args.min_s, args.max_s, args.force, args.manual)
    elif args.command == "render":
        step_render(args.project, args.mode, not args.no_hook, args.force)
    elif args.command == "post":
        if not args.to:
            raise SystemExit("--to is required, e.g. --to youtube,tiktok")
        step_post(args.project, args.to, args.privacy, args.only)


if __name__ == "__main__":
    main()
