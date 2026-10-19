"""Pick where the streamer's facecam is, so the "split" layout can put it on top.

`pick_facecam` opens a small page in the browser showing frames from the video;
you drag a box around the facecam and press Save. The box is stored in
<project>/facecam.json as fractions of the frame (x, y, w, h), so it works at
any resolution and can be reused for every clip of that video.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FACECAM_FILE = "facecam.json"


def load_facecam(project: Path) -> tuple[float, float, float, float] | None:
    path = project / FACECAM_FILE
    if not path.exists():
        return None
    box = json.loads(path.read_text())["box"]
    return validate_box(box)


def save_facecam(project: Path, box) -> tuple[float, float, float, float]:
    box = validate_box(box)
    (project / FACECAM_FILE).write_text(json.dumps({"box": list(box)}, indent=2))
    return box


def validate_box(box) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in box)
    x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
    w, h = min(w, 1.0 - x), min(h, 1.0 - y)
    if w < 0.03 or h < 0.03:
        raise ValueError("Facecam box is too small; drag a bigger box around the facecam.")
    return (round(x, 4), round(y, 4), round(w, 4), round(h, 4))


def extract_frames(source: Path, times: list[float], out_dir: Path) -> list[Path]:
    frames = []
    for i, t in enumerate(times):
        out = out_dir / f"{i}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(source),
             "-frames:v", "1", "-vf", "scale=1280:-2", "-q:v", "3", str(out)],
            check=False, capture_output=True,
        )
        if out.exists() and out.stat().st_size:
            frames.append(out)
    if not frames:
        raise RuntimeError(f"Couldn't grab any frames from {source}")
    return frames


def frame_times(project: Path, duration: float) -> list[float]:
    clips_path = project / "clips.json"
    if clips_path.exists():
        clips = json.loads(clips_path.read_text())
        times = [(c["start"] + c["end"]) / 2 for c in clips][:6]
        if times:
            return times
    return [duration * f for f in (0.2, 0.4, 0.6, 0.8)]


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Pick the facecam</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #111; color: #eee; font: 16px system-ui, sans-serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 20px 16px 40px; }
  h1 { font-size: 22px; margin: 0 0 6px; }
  p { margin: 0 0 14px; color: #aaa; }
  #stage { position: relative; user-select: none; touch-action: none; cursor: crosshair;
           border-radius: 8px; overflow: hidden; background: #000; }
  #stage img { display: block; width: 100%; pointer-events: none; }
  #box { position: absolute; border: 3px solid #ffd400; background: rgba(255,212,0,.15);
         box-shadow: 0 0 0 9999px rgba(0,0,0,.45); display: none; }
  .bar { display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap; align-items: center; }
  button { font: inherit; padding: 10px 18px; border-radius: 8px; border: 0; cursor: pointer;
           background: #333; color: #eee; }
  #save { background: #ffd400; color: #111; font-weight: 700; }
  #save:disabled { opacity: .4; cursor: default; }
  #msg { color: #aaa; }
</style></head>
<body><main>
  <h1>Drag a box around the facecam</h1>
  <p>Click and drag over the streamer's camera. Use "Another frame" if the cam is hidden in this one.</p>
  <div id="stage"><img id="img" alt="video frame"><div id="box"></div></div>
  <div class="bar">
    <button id="next">Another frame</button>
    <button id="save" disabled>Save facecam</button>
    <span id="msg"></span>
  </div>
</main>
<script>
const N = __NFRAMES__;
let frame = 0, box = __BOX__, start = null;
const img = document.getElementById('img'), stage = document.getElementById('stage');
const el = document.getElementById('box'), save = document.getElementById('save');
const msg = document.getElementById('msg');
function show() {
  if (!box) { el.style.display = 'none'; save.disabled = true; return; }
  el.style.display = 'block';
  el.style.left = box[0] * 100 + '%'; el.style.top = box[1] * 100 + '%';
  el.style.width = box[2] * 100 + '%'; el.style.height = box[3] * 100 + '%';
  save.disabled = !(box[2] > 0.03 && box[3] > 0.03);
}
function pos(e) {
  const r = stage.getBoundingClientRect();
  return [Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1),
          Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1)];
}
stage.addEventListener('pointerdown', e => { start = pos(e); stage.setPointerCapture(e.pointerId); });
stage.addEventListener('pointermove', e => {
  if (!start) return;
  const p = pos(e);
  box = [Math.min(start[0], p[0]), Math.min(start[1], p[1]),
         Math.abs(p[0] - start[0]), Math.abs(p[1] - start[1])];
  show();
});
stage.addEventListener('pointerup', () => { start = null; });
document.getElementById('next').onclick = () => {
  frame = (frame + 1) % N; img.src = '/frame/' + frame + '.jpg';
};
save.onclick = async () => {
  save.disabled = true; msg.textContent = 'Saving...';
  const r = await fetch('/save', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                  body: JSON.stringify({box})});
  const j = await r.json();
  if (j.ok) {
    document.querySelector('main').innerHTML =
      '<h1>Saved!</h1><p>You can close this tab and go back to the command window.</p>';
  } else { msg.textContent = j.error; save.disabled = false; }
};
img.src = '/frame/0.jpg';
show();
</script></body></html>
"""


def pick_facecam(source: Path, project: Path, duration: float) -> tuple[float, float, float, float]:
    """Open the browser picker and block until the user saves a box."""
    existing = load_facecam(project)
    result: dict = {}
    done = threading.Event()

    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_frames(source, frame_times(project, duration), Path(tmp))
        page = (PAGE.replace("__NFRAMES__", str(len(frames)))
                    .replace("__BOX__", json.dumps(list(existing)) if existing else "null"))

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the terminal quiet
                pass

            def _send(self, code: int, body: bytes, ctype: str):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    return self._send(200, page.encode(), "text/html; charset=utf-8")
                if self.path.startswith("/frame/"):
                    name = self.path.rsplit("/", 1)[-1]
                    match = [f for f in frames if f.name == name]
                    if match:
                        return self._send(200, match[0].read_bytes(), "image/jpeg")
                self._send(404, b"not found", "text/plain")

            def do_POST(self):
                if self.path != "/save":
                    return self._send(404, b"not found", "text/plain")
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    box = save_facecam(project, json.loads(self.rfile.read(length))["box"])
                except (ValueError, KeyError, TypeError) as e:
                    return self._send(200, json.dumps({"ok": False, "error": str(e)}).encode(),
                                      "application/json")
                result["box"] = box
                self._send(200, b'{"ok": true}', "application/json")
                done.set()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"\n★ Pick the facecam: a page is opening in your browser.\n"
              f"  If it doesn't open, paste this into Chrome: {url}\n"
              f"  Drag a box around the facecam and click Save.\n", file=sys.stderr, flush=True)
        webbrowser.open(url)
        try:
            while not done.wait(0.5):
                pass
        finally:
            server.shutdown()
            server.server_close()
    print(f"✓ facecam saved to {project / FACECAM_FILE}", file=sys.stderr, flush=True)
    return result["box"]
