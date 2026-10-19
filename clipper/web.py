"""A local web page for the clipper: paste a link, click a button, get clips.

Run `clipper ui` (or double-click "Start Clipper.bat"). It serves a page on
http://127.0.0.1:8765 that walks through the same steps as `clipper run`:
download → transcribe → pick clips (API or copy-paste with the Claude app) →
facecam box (streamer layout) → render, then shows the finished clips.

Only listens on 127.0.0.1, so nothing is reachable from other machines.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import cli
from .models import Transcript, load_clips, save_clips

BUSY_STAGES = ("download", "transcribe", "find", "render")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class Cancelled(Exception):
    pass


class Job:
    """One link being turned into clips, run on a background thread."""

    def __init__(self, url: str, workspace: Path, clips: int, layout: str, captions: bool):
        self.url, self.workspace = url, workspace
        self.n, self.layout, self.captions = clips, layout, captions
        self.project = cli.project_dir_for(url, workspace)
        self.lock = threading.Lock()
        self.stage, self.progress, self.error = "download", None, None
        self.render_done = self.render_total = 0
        self.log: list[str] = []
        self.prompt = None
        self.reply_text, self.reply_error = None, None
        self.reply_event = threading.Event()
        self.frames: list[Path] = []
        self.box, self.box_error = None, None
        self.box_event = threading.Event()
        self.cancelled = False

    # -- state ------------------------------------------------------------
    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def on_log(self, msg: str):
        with self.lock:
            self.log = (self.log + [msg])[-40:]
            m = re.search(r"rendering (\d+)/(\d+)", msg)
            if m:
                self.render_done, self.render_total = int(m.group(1)) - 1, int(m.group(2))

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "stage": self.stage, "progress": self.progress, "error": self.error,
                "render_done": self.render_done, "render_total": self.render_total,
                "log": self.log[-12:], "reply_error": self.reply_error,
                "frames": len(self.frames), "box_error": self.box_error,
                "project": self.project.name, "url": self.url, "layout": self.layout,
            }

    def cancel(self):
        self.cancelled = True
        self.reply_event.set()
        self.box_event.set()

    # -- pipeline ---------------------------------------------------------
    def run(self):
        cli.LOG_SINKS.append(self.on_log)
        try:
            cli.step_download(self.url, self.project)
            self.set(stage="transcribe", progress=0.0)
            cli.step_transcribe(self.project, "small", None,
                                progress=lambda f: self.set(progress=round(f, 3)))
            self.set(stage="find", progress=None)
            self._find()
            if self.layout == "split":
                from .facecam import load_facecam
                if load_facecam(self.project) is None:
                    self._pick_facecam()
            total = len(load_clips(self.project / "clips.json"))
            self.set(stage="render", render_done=0, render_total=total)
            cli.step_render(self.project, self.layout, True, force=True, captions=self.captions)
            self.set(stage="done", render_done=total)
        except Cancelled:
            self.set(stage="idle")
        except BaseException as e:  # SystemExit from the CLI helpers too
            msg = str(e) or e.__class__.__name__
            if "Permission denied" in msg:
                msg = ("Windows wouldn't let me overwrite a clip. Close any video player "
                       "that has one of these clips open, then click Get clips again.")
            self.set(stage="error", error=msg)
        finally:
            cli.LOG_SINKS.remove(self.on_log)

    def _find(self):
        from .highlights import build_manual_prompt, parse_reply, postprocess

        out = self.project / "clips.json"
        if out.exists():
            cli.log("✓ clips.json already present")
            return
        if os.environ.get("ANTHROPIC_API_KEY"):
            cli.step_find(self.project, self.n, 20, 60)
            return
        transcript = Transcript.load(self.project / "transcript.json")
        meta = self.project / "metadata.json"
        title = json.loads(meta.read_text()).get("title") if meta.exists() else None
        prompt = build_manual_prompt(transcript, self.n, 20, 60, title)
        (self.project / "claude_prompt.txt").write_text(prompt, encoding="utf-8")
        self.set(stage="waiting_reply", prompt=prompt, reply_error=None)
        while True:
            self.reply_event.wait()
            self.reply_event.clear()
            if self.cancelled:
                raise Cancelled
            try:
                raw = parse_reply(self.reply_text or "")
                clips = postprocess(raw, transcript, self.n, 20, 60)
                if not clips:
                    raise ValueError("none of those clips match this video; paste the reply "
                                     "for this video's prompt")
            except (ValueError, KeyError, TypeError) as e:
                self.set(reply_error=f"Couldn't read that reply: {e}")
                continue
            (self.project / "claude_reply.txt").write_text(self.reply_text, encoding="utf-8")
            save_clips(clips, out)
            cli.log(f"✓ {len(clips)} clips picked")
            self.set(stage="find", prompt=None, reply_error=None)
            return

    def _pick_facecam(self):
        from .editor import probe
        from .facecam import extract_frames, frame_times, save_facecam

        source = cli.find_source(self.project)
        _, _, duration = probe(source)
        frame_dir = self.project / "_frames"
        frame_dir.mkdir(exist_ok=True)
        frames = extract_frames(source, frame_times(self.project, duration), frame_dir)
        self.set(stage="facecam", frames=frames, box_error=None)
        while True:
            self.box_event.wait()
            self.box_event.clear()
            if self.cancelled:
                raise Cancelled
            try:
                save_facecam(self.project, self.box)
            except (ValueError, TypeError) as e:
                self.set(box_error=str(e))
                continue
            self.set(stage="render", box_error=None)
            return


def list_projects(workspace: Path) -> list[dict]:
    out = []
    if not workspace.is_dir():
        return out
    for d in workspace.iterdir():
        renders = sorted((d / "renders").glob("*.mp4")) if (d / "renders").is_dir() else []
        if not renders:
            continue
        clips = load_clips(d / "clips.json") if (d / "clips.json").exists() else []
        meta = d / "metadata.json"
        title = (json.loads(meta.read_text()).get("title") if meta.exists() else None) or d.name
        items = []
        for r in renders:
            m = re.match(r"(\d+)-", r.name)
            c = clips[int(m.group(1)) - 1] if m and 0 < int(m.group(1)) <= len(clips) else None
            tags = " ".join(f"#{h}" for h in (c.hashtags if c else []))
            items.append({
                "file": r.name,
                "url": f"/media/{d.name}/{r.name}?v={int(r.stat().st_mtime)}",
                "title": c.title if c else r.stem,
                "score": c.virality_score if c else None,
                "caption": ((c.description + " " + tags).strip() if c else ""),
            })
        newest = max(r.stat().st_mtime for r in renders)
        out.append({"id": d.name, "title": title, "clips": items, "updated": newest})
    out.sort(key=lambda p: -p["updated"])
    return out


class App:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.job: Job | None = None
        self.lock = threading.Lock()

    def state(self) -> dict:
        snap = self.job.snapshot() if self.job else {"stage": "idle"}
        snap["has_api_key"] = bool(os.environ.get("ANTHROPIC_API_KEY"))
        return snap

    def start(self, body: dict) -> dict:
        url = str(body.get("url", "")).strip()
        if not url:
            raise ValueError("Paste a YouTube link first.")
        if not re.match(r"^https?://", url) and not Path(url).exists():
            raise ValueError("That doesn't look like a link. Copy it from the YouTube address bar.")
        layout = body.get("layout", "split")
        if layout not in ("split", "blur", "auto"):
            raise ValueError("Unknown layout")
        clips = max(1, min(10, int(body.get("clips", 5))))
        with self.lock:
            if self.job and self.job.stage in BUSY_STAGES:
                raise ValueError("Already working on a video. Wait for it to finish.")
            if self.job:
                self.job.cancel()
            self.job = Job(url, self.workspace, clips, layout, bool(body.get("captions", True)))
            threading.Thread(target=self.job.run, daemon=True).start()
        return {"ok": True}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clipper</title>
<style>
:root { color-scheme: dark; --bg:#0c0c10; --card:#16161d; --line:#262632; --text:#ededf2;
  --muted:#9a9aab; --accent:#7c5cff; --accent2:#a48bff; --ok:#3ccf8e; --bad:#ff5c7a; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 32px 16px 80px; }
header { display:flex; align-items:center; gap:10px; margin-bottom: 28px; }
.logo { width:34px; height:34px; border-radius:10px; background:linear-gradient(135deg,var(--accent),#ff5cc8);
  display:grid; place-items:center; font-weight:800; }
header h1 { font-size:20px; margin:0; letter-spacing:-.01em; }
.hero { text-align:center; margin: 10px 0 26px; }
.hero h2 { font-size: 30px; margin:0 0 6px; letter-spacing:-.02em; }
.hero p { color:var(--muted); margin:0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:20px; margin-bottom:18px; }
.input { display:flex; gap:8px; background:#0f0f15; border:1px solid var(--line); border-radius:12px; padding:6px; }
.input input { flex:1; min-width:0; background:none; border:0; color:var(--text); font:inherit; padding:10px 10px; outline:none; }
button, .btn { font:inherit; border:0; border-radius:10px; padding:10px 18px; cursor:pointer; color:#fff;
  background:var(--accent); font-weight:650; text-decoration:none; display:inline-block; white-space:nowrap; }
button:hover, .btn:hover { background:var(--accent2); }
button:disabled { opacity:.45; cursor:default; background:var(--accent); }
.ghost { background:#24242e; color:var(--text); font-weight:550; }
.ghost:hover { background:#2e2e3a; }
.small { padding:7px 12px; font-size:13px; }
.opts { display:flex; flex-wrap:wrap; gap:14px 22px; margin-top:14px; color:var(--muted); font-size:14px; align-items:center; }
.opts label { display:flex; align-items:center; gap:8px; }
select { font:inherit; background:#0f0f15; color:var(--text); border:1px solid var(--line); border-radius:8px; padding:6px 8px; }
.steps { list-style:none; margin:0; padding:0; display:grid; gap:10px; }
.steps li { display:flex; align-items:center; gap:12px; color:var(--muted); }
.dot { width:22px; height:22px; border-radius:50%; border:2px solid var(--line); flex:none; display:grid; place-items:center; font-size:12px; }
.steps li.done { color:var(--text); } .steps li.done .dot { background:var(--ok); border-color:var(--ok); color:#0c0c10; }
.steps li.active { color:var(--text); font-weight:600; }
.steps li.active .dot { border-color:var(--accent); border-top-color:transparent; animation:spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.bar { height:6px; background:#0f0f15; border-radius:6px; overflow:hidden; margin-top:14px; }
.bar > div { height:100%; background:linear-gradient(90deg,var(--accent),#ff5cc8); width:0; transition:width .4s; }
.hint { color:var(--muted); font-size:13px; margin:10px 0 0; }
details { margin-top:12px; color:var(--muted); font-size:13px; }
pre { white-space:pre-wrap; background:#0f0f15; border-radius:8px; padding:10px; margin:8px 0 0; max-height:180px; overflow:auto; }
h3 { margin:0 0 12px; font-size:17px; }
ol.how { margin:0 0 14px; padding-left:20px; display:grid; gap:10px; }
ol.how li::marker { color:var(--accent2); font-weight:700; }
textarea { width:100%; min-height:130px; background:#0f0f15; color:var(--text); border:1px solid var(--line);
  border-radius:10px; padding:10px; font:13px/1.4 ui-monospace,Consolas,monospace; resize:vertical; }
.err { color:var(--bad); margin:8px 0 0; font-size:14px; }
.errcard { border-color:#5a2433; background:#1d1116; }
#stage { position:relative; user-select:none; touch-action:none; cursor:crosshair; border-radius:10px; overflow:hidden; background:#000; }
#stage img { display:block; width:100%; pointer-events:none; }
#box { position:absolute; border:3px solid #ffd400; background:rgba(255,212,0,.12); box-shadow:0 0 0 9999px rgba(0,0,0,.45); display:none; }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.proj { margin-top:26px; }
.proj-head { display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:12px; }
.proj-head h3 { margin:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:14px; }
.clip { background:var(--card); border:1px solid var(--line); border-radius:14px; overflow:hidden; display:flex; flex-direction:column; }
.clip video { width:100%; aspect-ratio:9/16; background:#000; display:block; }
.clip .meta { padding:10px 12px 12px; display:grid; gap:8px; }
.clip b { font-size:14px; line-height:1.3; }
.score { background:#24242e; color:var(--accent2); font-weight:700; font-size:12px; border-radius:6px; padding:2px 6px; }
.cap { color:var(--muted); font-size:12px; margin:0; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }
[hidden] { display:none !important; }
@media (max-width: 560px) { .hero h2 { font-size:24px; } .input { flex-wrap:wrap; } .input input { flex-basis:100%; } }
</style></head>
<body><main>
<header><div class="logo">✂</div><h1>Clipper</h1></header>

<div class="hero"><h2>Turn a video into clips</h2><p>Paste a YouTube link. Get vertical clips with captions.</p></div>

<section class="card">
  <div class="input">
    <input id="url" placeholder="Paste a YouTube link here…" autocomplete="off">
    <button class="ghost" id="paste">Paste</button>
    <button id="go">Get clips</button>
  </div>
  <div class="opts">
    <label>Clips <select id="clips"><option>3</option><option selected>5</option><option>8</option></select></label>
    <label>Layout <select id="layout">
      <option value="split" selected>Streamer (facecam on top)</option>
      <option value="blur">Full video</option>
      <option value="auto">Auto (follow face)</option></select></label>
    <label><input type="checkbox" id="captions" checked> Word captions</label>
  </div>
</section>

<section class="card errcard" id="errcard" hidden><b>Something went wrong</b><p class="err" id="errtext"></p></section>

<section class="card" id="progress" hidden>
  <ul class="steps" id="steps"></ul>
  <div class="bar" id="barwrap" hidden><div id="bar"></div></div>
  <p class="hint" id="hint"></p>
  <details><summary>Details</summary><pre id="log"></pre></details>
</section>

<section class="card" id="reply" hidden>
  <h3>Let Claude pick the best moments</h3>
  <ol class="how">
    <li><button class="small" id="copyprompt">Copy prompt</button></li>
    <li><a class="btn ghost small" href="https://claude.ai/new" target="_blank" rel="noopener">Open Claude ↗</a>
        &nbsp;Start a new chat, paste with Ctrl+V and send.</li>
    <li>Click the copy button on Claude's answer, then paste it here:</li>
  </ol>
  <textarea id="replytext" placeholder="Paste Claude's reply here…"></textarea>
  <p class="err" id="replyerr" hidden></p>
  <div class="row" style="margin-top:10px"><button id="sendreply">Use these clips</button></div>
</section>

<section class="card" id="facecam" hidden>
  <h3>Drag a box around the facecam</h3>
  <p class="hint" style="margin:0 0 12px">You only do this once per video.</p>
  <div id="stage"><img id="frame" alt="video frame"><div id="box"></div></div>
  <p class="err" id="boxerr" hidden></p>
  <div class="row" style="margin-top:12px"><button class="ghost" id="nextframe">Another frame</button>
    <button id="savebox" disabled>Save facecam</button></div>
</section>

<section id="results" hidden><div id="projects"></div></section>
</main>
<script>
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let state = {stage:'idle'}, lastStage = null, promptText = '', frame = 0, box = null, drag = null, timer;
const BUSY = ['download','transcribe','find','render'];
const STEPS = [['download','Download video'],['transcribe','Transcribe speech'],['find','Pick the best moments'],['render','Edit clips']];
const IDX = {download:0, transcribe:1, find:2, waiting_reply:2, facecam:3, render:3, done:4};

async function api(path, body) {
  const opt = body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)};
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ('Error ' + r.status));
  return j;
}
function showErr(msg) { $('#errcard').hidden = !msg; $('#errtext').textContent = msg || ''; }

$('#paste').onclick = async () => {
  try { $('#url').value = (await navigator.clipboard.readText()).trim(); } catch (e) { $('#url').focus(); }
};
$('#url').addEventListener('keydown', e => { if (e.key === 'Enter') $('#go').click(); });
$('#go').onclick = async () => {
  showErr('');
  try {
    await api('/api/start', {url:$('#url').value, clips:+$('#clips').value, layout:$('#layout').value, captions:$('#captions').checked});
    $('#replytext').value = ''; poll();
  } catch (e) { showErr(e.message); }
};

$('#copyprompt').onclick = async () => {
  try { await navigator.clipboard.writeText(promptText); }
  catch (e) { const t = document.createElement('textarea'); t.value = promptText; document.body.append(t); t.select(); document.execCommand('copy'); t.remove(); }
  $('#copyprompt').textContent = 'Copied ✓'; setTimeout(() => $('#copyprompt').textContent = 'Copy prompt', 2500);
};
$('#sendreply').onclick = async () => {
  try { await api('/api/reply', {text:$('#replytext').value}); poll(); } catch (e) { showErr(e.message); }
};

function drawBox() {
  const el = $('#box');
  if (!box) { el.style.display = 'none'; $('#savebox').disabled = true; return; }
  el.style.display = 'block';
  el.style.left = box[0]*100+'%'; el.style.top = box[1]*100+'%'; el.style.width = box[2]*100+'%'; el.style.height = box[3]*100+'%';
  $('#savebox').disabled = !(box[2] > .03 && box[3] > .03);
}
function pos(e) { const r = $('#stage').getBoundingClientRect();
  return [Math.min(Math.max((e.clientX-r.left)/r.width,0),1), Math.min(Math.max((e.clientY-r.top)/r.height,0),1)]; }
$('#stage').addEventListener('pointerdown', e => { drag = pos(e); $('#stage').setPointerCapture(e.pointerId); });
$('#stage').addEventListener('pointermove', e => { if (!drag) return; const p = pos(e);
  box = [Math.min(drag[0],p[0]), Math.min(drag[1],p[1]), Math.abs(p[0]-drag[0]), Math.abs(p[1]-drag[1])]; drawBox(); });
$('#stage').addEventListener('pointerup', () => drag = null);
$('#nextframe').onclick = () => { frame = (frame + 1) % Math.max(1, state.frames); $('#frame').src = '/api/frame/' + frame + '?t=' + Date.now(); };
$('#savebox').onclick = async () => { try { await api('/api/facecam', {box}); poll(); } catch (e) { showErr(e.message); } };

document.addEventListener('click', async e => {
  const open = e.target.closest('[data-open]'), copy = e.target.closest('[data-copy]');
  if (open) api('/api/open', {project: open.dataset.open}).catch(err => showErr(err.message));
  if (copy) { try { await navigator.clipboard.writeText(copy.dataset.copy); copy.textContent = 'Copied ✓'; } catch (err) {} }
});

async function loadProjects() {
  const {projects} = await api('/api/projects');
  $('#results').hidden = !projects.length;
  $('#projects').innerHTML = projects.map(p => `
    <section class="proj"><div class="proj-head"><h3>${esc(p.title)}</h3>
      <button class="ghost small" data-open="${esc(p.id)}">Open folder</button></div>
      <div class="grid">${p.clips.map(c => `
        <article class="clip"><video src="${esc(c.url)}" controls preload="metadata" playsinline></video>
          <div class="meta"><div class="row">${c.score ? `<span class="score">${c.score}</span>` : ''}<b>${esc(c.title)}</b></div>
            ${c.caption ? `<p class="cap">${esc(c.caption)}</p>` : ''}
            <div class="row"><a class="btn small" href="${esc(c.url)}&download=1">Download</a>
              ${c.caption ? `<button class="ghost small" data-copy="${esc(c.caption)}">Copy caption</button>` : ''}</div>
          </div></article>`).join('')}</div></section>`).join('');
}

async function render() {
  const s = state, st = s.stage, busy = BUSY.includes(st);
  $('#go').disabled = busy; $('#go').textContent = busy ? 'Working…' : 'Get clips';
  showErr(st === 'error' ? s.error : '');
  $('#progress').hidden = st === 'idle';
  const idx = IDX[st] ?? (st === 'error' ? -1 : 0);
  $('#steps').innerHTML = STEPS.map(([key, label], i) => {
    let cls = i < idx ? 'done' : (i === idx ? 'active' : '');
    if (st === 'waiting_reply' || st === 'facecam') cls = i < idx ? 'done' : (i === idx ? 'active' : '');
    let extra = '';
    if (i === idx && key === 'transcribe' && s.progress != null) extra = ` · ${Math.round(s.progress*100)}%`;
    if (i === idx && key === 'render' && st === 'render' && s.render_total) extra = ` · clip ${Math.min(s.render_done+1, s.render_total)} of ${s.render_total}`;
    if (i === idx && st === 'waiting_reply') extra = ' · waiting for you';
    if (i === idx && st === 'facecam') extra = ' · waiting for you';
    return `<li class="${cls}"><span class="dot">${cls === 'done' ? '✓' : ''}</span>${label}${extra}</li>`;
  }).join('');
  let pct = null;
  if (st === 'transcribe' && s.progress != null) pct = s.progress;
  if (st === 'render' && s.render_total) pct = s.render_done / s.render_total;
  if (st === 'done') pct = 1;
  $('#barwrap').hidden = pct == null; if (pct != null) $('#bar').style.width = (pct*100) + '%';
  $('#hint').textContent = {download:'Downloading the video…', transcribe:'Turning speech into text. This is the slow part — a few minutes is normal.',
    find:'Picking clips…', render:'Cutting and editing your clips. Close any open clip videos so they can be replaced.',
    done:'Done! Your clips are below.'}[st] || '';
  $('#log').textContent = (s.log || []).join('\n');

  $('#reply').hidden = st !== 'waiting_reply';
  if (st === 'waiting_reply' && lastStage !== 'waiting_reply') promptText = (await api('/api/prompt')).prompt || '';
  $('#replyerr').hidden = !s.reply_error; $('#replyerr').textContent = s.reply_error || '';

  $('#facecam').hidden = st !== 'facecam';
  if (st === 'facecam' && lastStage !== 'facecam') { frame = 0; box = null; drawBox(); $('#frame').src = '/api/frame/0?t=' + Date.now(); }
  $('#boxerr').hidden = !s.box_error; $('#boxerr').textContent = s.box_error || '';

  if (lastStage === null || (st === 'done' && lastStage !== 'done')) loadProjects();
  if (st === 'done' && lastStage && lastStage !== 'done') $('#results').scrollIntoView({behavior:'smooth'});
  lastStage = st;
}
async function poll() {
  clearTimeout(timer);
  try { state = await api('/api/state'); await render(); } catch (e) {}
  timer = setTimeout(poll, BUSY.includes(state.stage) || ['waiting_reply','facecam'].includes(state.stage) ? 1000 : 5000);
}
poll();
</script></body></html>
"""


def make_handler(app: App, port_ref: dict):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _allowed(self) -> bool:
            # Block other websites / DNS rebinding from driving the local server.
            host = self.headers.get("Host", "")
            ok_hosts = {f"127.0.0.1:{port_ref['port']}", f"localhost:{port_ref['port']}"}
            origin = self.headers.get("Origin")
            if host not in ok_hosts:
                return False
            return origin is None or origin in {f"http://{h}" for h in ok_hosts}

        def _json(self, code: int, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _bytes(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path, ctype: str, download: bool):
            size = path.stat().st_size
            start, end = 0, size - 1
            rng = self.headers.get("Range", "")
            m = re.match(r"bytes=(\d*)-(\d*)$", rng.split(",")[0].strip())
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            try:
                with path.open("rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (ConnectionError, BrokenPipeError):
                pass

        def do_GET(self):
            if not self._allowed():
                return self._json(403, {"error": "forbidden"})
            url = urlparse(self.path)
            path = unquote(url.path)
            job = app.job
            if path in ("/", "/index.html"):
                return self._bytes(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._json(200, app.state())
            if path == "/api/projects":
                return self._json(200, {"projects": list_projects(app.workspace)})
            if path == "/api/prompt":
                return self._json(200, {"prompt": job.prompt if job else None})
            m = re.match(r"^/api/frame/(\d+)$", path)
            if m and job and int(m.group(1)) < len(job.frames):
                return self._bytes(200, job.frames[int(m.group(1))].read_bytes(), "image/jpeg")
            m = re.match(r"^/media/([^/]+)/([^/]+)$", path)
            if m and NAME_RE.match(m.group(1)) and NAME_RE.match(m.group(2)):
                f = (app.workspace / m.group(1) / "renders" / m.group(2)).resolve()
                root = app.workspace.resolve()
                if f.is_file() and root in f.parents and f.suffix == ".mp4":
                    return self._file(f, "video/mp4", "download=1" in url.query)
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._allowed():
                return self._json(403, {"error": "forbidden"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._json(400, {"error": "bad request"})
            path, job = urlparse(self.path).path, app.job
            try:
                if path == "/api/start":
                    return self._json(200, app.start(body))
                if path == "/api/reply":
                    if not job or job.stage != "waiting_reply":
                        raise ValueError("Not waiting for a reply right now.")
                    if not str(body.get("text", "")).strip():
                        raise ValueError("Paste Claude's reply into the box first.")
                    job.set(reply_text=str(body["text"]), reply_error=None)
                    job.reply_event.set()
                    return self._json(200, {"ok": True})
                if path == "/api/facecam":
                    if not job or job.stage != "facecam":
                        raise ValueError("Not picking a facecam right now.")
                    job.set(box=body.get("box"), box_error=None)
                    job.box_event.set()
                    return self._json(200, {"ok": True})
                if path == "/api/open":
                    name = str(body.get("project", ""))
                    d = app.workspace / name / "renders"
                    if not NAME_RE.match(name) or not d.is_dir():
                        raise ValueError("Folder not found.")
                    if sys.platform == "win32":
                        os.startfile(str(d.resolve()))  # type: ignore[attr-defined]
                    return self._json(200, {"ok": True})
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            self._json(404, {"error": "not found"})

    return Handler


def make_server(workspace: Path, port: int = 8765) -> ThreadingHTTPServer:
    app = App(workspace)
    port_ref = {"port": port}
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, port_ref))
    except OSError:  # port taken: let the OS pick one
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app, port_ref))
    port_ref["port"] = server.server_address[1]
    server.daemon_threads = True
    server.app = app  # type: ignore[attr-defined]
    return server


def serve(workspace: Path, port: int = 8765, open_browser: bool = True) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    server = make_server(workspace, port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"\n✂  Clipper is running at {url}\n   Keep this window open while you use it. "
          f"Close it (or press Ctrl+C) to stop.\n", file=sys.stderr, flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
