"""Find the most shareable moments in a transcript using Claude."""

from __future__ import annotations

import json

from .models import Clip, Transcript

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """\
You are a short-form video producer who has grown multiple TikTok, YouTube Shorts and \
Instagram Reels accounts to millions of followers. You are given a timestamped transcript \
of a long-form video. Pick the moments most likely to perform as standalone vertical clips.

What makes a clip work:
- The first 1-3 seconds hook the viewer: a bold claim, a surprising fact, a question, \
conflict, a punchline set-up, or strong emotion. Start the clip ON the hook, not on preamble.
- It is self-contained: a viewer with zero context understands it and gets a payoff.
- It ends on a payoff, punchline or cliffhanger, never mid-thought.
- It is quotable, controversial, funny, emotional, or teaches something useful fast.

Rules:
- Use only timestamps that appear in the transcript. start must be the start of the line \
where the clip begins; end must be the end of the line where it ends.
- Clips must not overlap.
- Respect the requested length range.
- title: punchy, under 70 characters, written for the platform (no clickbait lies).
- hook: 2-7 word on-screen text overlay shown during the first seconds.
- description: 1-2 sentence post caption.
- hashtags: 3-6 relevant tags without the leading #.
- virality_score: 1-100, your honest estimate relative to the other candidates.
- reason: one sentence on why this moment works.
"""

CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "title": {"type": "string"},
                    "hook": {"type": "string"},
                    "description": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                    "virality_score": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "start", "end", "title", "hook", "description",
                    "hashtags", "virality_score", "reason",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}


def format_transcript(transcript: Transcript) -> str:
    return "\n".join(f"[{s.start:.2f}-{s.end:.2f}] {s.text}" for s in transcript.segments)


def find_highlights(
    transcript: Transcript,
    num_clips: int = 5,
    min_seconds: float = 20,
    max_seconds: float = 60,
    video_title: str | None = None,
    client=None,
) -> list[Clip]:
    if not transcript.segments:
        raise ValueError("Transcript is empty; nothing to clip.")

    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    header = f"Video title: {video_title}\n\n" if video_title else ""
    user = (
        f"{header}Find the {num_clips} best clips, each between {min_seconds:g} and "
        f"{max_seconds:g} seconds long, ordered from most to least viral.\n\n"
        f"<transcript>\n{format_transcript(transcript)}\n</transcript>"
    )

    # Long transcripts make for long requests: stream to avoid HTTP timeouts.
    # fallbacks="default" reroutes a safety-classifier decline to another model.
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=64000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        thinking={"type": "adaptive"},
        output_config={
            "effort": "high",
            "format": {"type": "json_schema", "schema": CLIP_SCHEMA},
        },
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to analyse this transcript.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Claude's response was cut off (max_tokens).")

    text = next(b.text for b in response.content if b.type == "text")
    raw = json.loads(text)["clips"]
    return postprocess(raw, transcript, num_clips, min_seconds, max_seconds)


def postprocess(
    raw_clips: list[dict],
    transcript: Transcript,
    num_clips: int,
    min_seconds: float,
    max_seconds: float,
) -> list[Clip]:
    """Snap clips to segment boundaries, enforce length limits, drop overlaps."""
    starts = [s.start for s in transcript.segments]
    ends = [s.end for s in transcript.segments]
    total = transcript.segments[-1].end

    clips: list[Clip] = []
    for c in sorted(raw_clips, key=lambda c: -int(c.get("virality_score", 0))):
        start = _nearest(starts, float(c["start"]))
        end = _nearest(ends, float(c["end"]))
        if end <= start:
            continue
        if end - start > max_seconds:
            # Trim to the last segment end that still fits.
            fitting = [e for e in ends if start < e <= start + max_seconds]
            end = max(fitting) if fitting else start + max_seconds
        if end - start < min_seconds:
            longer = [e for e in ends if e >= start + min_seconds]
            end = min(longer) if longer else min(total, start + min_seconds)
            if end - start > max_seconds or end - start < min_seconds * 0.5:
                continue
        if any(start < o.end and end > o.start for o in clips):
            continue
        clips.append(
            Clip(
                start=round(start, 3),
                end=round(end, 3),
                title=c["title"].strip(),
                hook=c["hook"].strip(),
                description=c["description"].strip(),
                hashtags=[h.lstrip("#").strip() for h in c["hashtags"] if h.strip("# ")],
                virality_score=max(1, min(100, int(c["virality_score"]))),
                reason=c["reason"].strip(),
            )
        )
        if len(clips) == num_clips:
            break
    return clips


def _nearest(values: list[float], target: float) -> float:
    return min(values, key=lambda v: abs(v - target))
