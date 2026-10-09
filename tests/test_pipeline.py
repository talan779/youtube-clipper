import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from clipper import highlights
from clipper.captions import ass_time, build_ass, chunk_words
from clipper.cli import main, project_dir_for
from clipper.editor import build_video_filter, probe, render_clip
from clipper.models import Clip, Segment, Transcript, Word, load_clips
from clipper.publish.tiktok import plan_chunks


def make_transcript(n_segments=30, seg_len=4.0):
    segments = []
    for i in range(n_segments):
        start = i * seg_len
        words = [
            Word(start=start + j * 0.5, end=start + j * 0.5 + 0.4, text=f"word{i}_{j}")
            for j in range(int(seg_len / 0.5))
        ]
        segments.append(Segment(start=start, end=start + seg_len, text=" ".join(w.text for w in words), words=words))
    return Transcript(language="en", segments=segments)


def raw_clip(start, end, score=50, **kw):
    return {
        "start": start, "end": end, "title": kw.get("title", "T"), "hook": "Hook",
        "description": "D", "hashtags": ["#a", "b"], "virality_score": score, "reason": "R",
    }


# --- captions ----------------------------------------------------------------


def test_ass_time():
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(61.234) == "0:01:01.23"
    assert ass_time(3725.5) == "1:02:05.50"


def test_chunk_words_breaks_on_count_pause_and_punctuation():
    words = [Word(0, 0.3, "a"), Word(0.3, 0.6, "b"), Word(0.6, 0.9, "c."), Word(0.9, 1.2, "d"),
             Word(3.0, 3.3, "e")]
    chunks = chunk_words(words, max_words=5, max_seconds=5)
    assert [[w.text for w in c] for c in chunks] == [["a", "b", "c."], ["d"], ["e"]]


def test_build_ass_is_clip_relative_and_has_hook():
    t = make_transcript()
    ass = build_ass(t.words_between(8, 16), 8, 16, hook="watch this {now}")
    assert "Style: Hook" in ass
    assert "WATCH THIS (NOW)" in ass  # braces escaped so they can't inject ASS tags
    dialogue = [l for l in ass.splitlines() if l.startswith("Dialogue: 0")]
    assert dialogue
    first_start = dialogue[0].split(",")[1]
    assert first_start == "0:00:00.00"
    assert all(l.split(",")[2] <= "0:00:08.00" for l in dialogue)


# --- highlights --------------------------------------------------------------


def test_postprocess_snaps_trims_and_dedupes():
    t = make_transcript()
    clips = highlights.postprocess(
        [
            raw_clip(9.1, 31.7, score=90),     # snaps to 8–32
            raw_clip(10, 30, score=80),        # overlaps the first -> dropped
            raw_clip(40, 200, score=70),       # too long -> trimmed to <= 60s
            raw_clip(100, 104, score=60),      # too short -> extended to >= 20s
        ],
        t, num_clips=5, min_seconds=20, max_seconds=60,
    )
    assert [(c.start, c.end) for c in clips] == [(8, 32), (40, 100), (100, 120)]
    assert clips[0].hashtags == ["a", "b"]


def test_find_highlights_calls_claude_with_structured_output():
    t = make_transcript()
    payload = json.dumps({"clips": [raw_clip(0, 24, 88, title="Best bit")]})
    captured = {}

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return SimpleNamespace(stop_reason="end_turn",
                                   content=[SimpleNamespace(type="text", text=payload)])

    def stream(**kwargs):
        captured.update(kwargs)
        return FakeStream()

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    clips = highlights.find_highlights(t, num_clips=1, client=client, video_title="My video")
    assert clips[0].title == "Best bit"
    assert captured["model"] == "claude-opus-5-5"
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert "[0.00-4.00]" in captured["messages"][0]["content"]
    assert "My video" in captured["messages"][0]["content"]


# --- editor ------------------------------------------------------------------


def test_filter_modes():
    assert "gblur" in build_video_filter(1920, 1080, "auto", None, "s.ass")
    crop = build_video_filter(1920, 1080, "auto", 0.9, "s.ass")
    assert "crop=606:1080:1314:0" in crop  # clamped to right edge
    assert "pad=" in build_video_filter(1080, 1920, "auto", None, "s.ass")


needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "in.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=40",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=40",
         "-shortest", "-c:v", "libx264", "-c:a", "aac", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
@pytest.mark.parametrize("mode", ["blur", "center"])
def test_render_clip_produces_vertical_video(sample_video, tmp_path, mode):
    t = make_transcript(n_segments=10)
    clip = Clip(4, 16, "Test", "Hook text", "d", ["x"], 50, "r")
    out = render_clip(sample_video, clip, t, tmp_path / f"{mode}.mp4", mode=mode)
    w, h, dur = probe(out)
    assert (w, h) == (1080, 1920)
    assert abs(dur - 12) < 0.5


@needs_ffmpeg
def test_cli_render_from_local_file(sample_video, tmp_path):
    ws = tmp_path / "ws"
    project = project_dir_for(str(sample_video), ws)
    main(["--workspace", str(ws), "download", str(sample_video)])
    make_transcript(n_segments=10).save(project / "transcript.json")
    (project / "clips.json").write_text(json.dumps([
        {"start": 0, "end": 8, "title": "Hello World!", "hook": "h", "description": "d",
         "hashtags": [], "virality_score": 1, "reason": "r"}
    ]))
    main(["render", str(project), "--mode", "blur"])
    assert (project / "renders" / "01-hello-world.mp4").exists()
    assert load_clips(project / "clips.json")[0].title == "Hello World!"


# --- misc --------------------------------------------------------------------


def test_project_dir_for_youtube_urls(tmp_path):
    assert project_dir_for("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=1", tmp_path).name == "dQw4w9WgXcQ"
    assert project_dir_for("https://youtu.be/dQw4w9WgXcQ", tmp_path).name == "dQw4w9WgXcQ"


def test_tiktok_chunk_plan():
    assert plan_chunks(3_000_000) == (3_000_000, 1)
    size, count = plan_chunks(100 * 1024 * 1024)
    assert size == 10 * 1024 * 1024 and count == 10
