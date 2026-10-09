# youtube-clipper

Turn a long-form YouTube video into short vertical clips, then post them.

```
YouTube URL ─► download ─► transcribe ─► find viral moments ─► edit ─► post
               (yt-dlp)   (Whisper,      (Claude picks hooks,  (9:16,    (YouTube Shorts,
                           word timings)  titles, hashtags)     captions,  TikTok,
                                                                 loudness)  Instagram Reels)
```

## What it does

1. **Clip**: downloads the video, transcribes it locally with word-level timestamps, and has
   Claude choose the moments most likely to go viral. Each clip starts on a hook, makes sense
   without context, and ends on a payoff. Claude also writes a title, an on-screen hook line,
   a caption, hashtags and a virality score for each clip.
2. **Edit**: renders each clip at 1080×1920 30fps H.264. The steps:
   - **Reframing**: crops to the speaker's face when one is found (needs OpenCV), and otherwise
     puts the full frame over a blurred fill.
   - **Captions**: animated word-by-word captions, with the spoken word highlighted.
   - **Hook banner**: the hook line shows for the first 3 seconds.
   - **Audio**: loudness is normalized to −14 LUFS, the level the short-form platforms target.
3. **Post**: uploads to YouTube Shorts, TikTok and/or Instagram Reels. Posts are **private by
   default**.

## Install

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/) on your PATH.

```bash
pip install -e ".[all]"          # core + face tracking + YouTube upload
export ANTHROPIC_API_KEY=sk-ant-...
```

## Use

```bash
# Everything in one go: 5 clips of 20–60s, rendered into workspace/<video-id>/renders/
clipper run "https://www.youtube.com/watch?v=VIDEO_ID"

# ...and post them (private uploads so you can review first)
clipper run "https://youtu.be/VIDEO_ID" --post youtube,tiktok

# Local files work too
clipper run my_podcast.mp4 -n 8 --min 15 --max 45
```

Or one step at a time. Every step skips work that is already done, so you can review the
clips Claude chose before rendering or posting:

```bash
clipper download "https://youtu.be/VIDEO_ID"        # → workspace/VIDEO_ID
clipper transcribe workspace/VIDEO_ID --whisper-model medium
clipper find       workspace/VIDEO_ID -n 6
#   edit workspace/VIDEO_ID/clips.json by hand if you want: change times, titles, hooks...
clipper render     workspace/VIDEO_ID --mode auto   # auto | face | center | blur
clipper post       workspace/VIDEO_ID --to youtube,tiktok --only 1,3 --privacy public
```

Use `--force` to redo a step. `posted.json` records every upload, so you won't post the same
clip twice.

## Platform setup

| Platform | What you need |
|---|---|
| **YouTube Shorts** | In Google Cloud, enable *YouTube Data API v3* and create an OAuth client of type *Desktop app*. Download the JSON and set `YOUTUBE_CLIENT_SECRETS=/path/client_secret.json`. Your first upload opens a browser so you can sign in. |
| **TikTok** | Create an app at developers.tiktok.com with the *Content Posting API* (`video.publish` scope), sign in through OAuth, and set `TIKTOK_ACCESS_TOKEN`. Until TikTok audits the app, posts can only be private. |
| **Instagram Reels** | You need a Business or Creator account linked to a Facebook Page, and a Meta app with `instagram_content_publish`. Set `INSTAGRAM_ACCESS_TOKEN` and `INSTAGRAM_USER_ID`. Reels are always public, so this platform needs `--privacy public`. |

## Tips

- On CPU, the `small` Whisper model is a reasonable trade-off. With a GPU, use
  `--whisper-model large-v3` for the best caption accuracy.
- The `virality_score` is Claude's estimate relative to the other clips from the same video.
  It ranks the clips; it doesn't predict views.
- Repost only content you own or have permission to use. Clipping other creators' videos can
  get your accounts struck.

## Development

```bash
pip install -e ".[dev]" && pytest
```
