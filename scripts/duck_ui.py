#!/usr/bin/env python3
"""Web control panel for the microduck sim — buttons instead of terminal keys.

Serves http://localhost:8231 — drive buttons are hold-to-move (mousedown sends
repeats, mouseup sends stop), action buttons are one-shot. Everything writes
single keystrokes into /tmp/duckkeys.fifo (the sim's pty stdin).
"""
import http.server
import os

FIFO = "/tmp/duckkeys.fifo"
PORT = 8231

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>duck control</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { background:#141412; color:#eceae2; font-family:-apple-system,sans-serif;
         display:flex; flex-direction:column; align-items:center; padding-top:28px;
         -webkit-user-select:none; user-select:none; }
  h1 { font-size:20px; margin:0 0 4px; } .sub{color:#9c9a8e;font-size:12px;margin-bottom:24px}
  .pad { display:grid; grid-template-columns:repeat(3,72px); gap:10px; margin-bottom:22px; }
  button { background:#1d1d1a; color:#eceae2; border:1.5px solid #4a483f; border-radius:12px;
           font-size:22px; height:64px; cursor:pointer; touch-action:none; }
  button:active, button.held { background:#f0a832; color:#141412; border-color:#f0a832; }
  .acts { display:grid; grid-template-columns:repeat(3,96px); gap:10px; }
  .acts button { font-size:14px; font-weight:700; }
  .grab { border-color:#f0a832; color:#f0a832; } .jaw { border-color:#f87171; color:#f87171; }
</style></head><body>
<h1>🦆 duck control</h1>
<div class="sub">hold drive buttons to move · release to stop</div>
<div class="pad">
  <span></span><button data-k="w" data-hold="1">▲</button><span></span>
  <button data-k="a" data-hold="1">◀</button>
  <button data-k=" " >■</button>
  <button data-k="d" data-hold="1">▶</button>
  <span></span><button data-k="s" data-hold="1">▼</button><span></span>
</div>
<div class="acts">
  <button class="grab" data-k="g">GRAB</button>
  <button class="jaw" data-k="m">JAW</button>
  <button data-k="y">SIT</button>
  <button data-k="r">ROLL</button>
  <button data-k="k">KICK L</button>
  <button data-k="l">KICK R</button>
</div>
<script>
let timers = {};
async function send(k){ await fetch('/key?k='+encodeURIComponent(k)); }
document.querySelectorAll('button').forEach(b => {
  const k = b.dataset.k, hold = b.dataset.hold === '1';
  const start = e => { e.preventDefault(); b.classList.add('held');
    if (hold) { send(k); timers[k] = setInterval(() => send(k), 120); } else send(k); };
  const stop = e => { b.classList.remove('held');
    if (hold && timers[k]) { clearInterval(timers[k]); delete timers[k]; send(' '); } };
  b.addEventListener('pointerdown', start);
  b.addEventListener('pointerup', stop);
  b.addEventListener('pointerleave', stop);
});
</script></body></html>"""


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/key"):
            k = self.path.split("k=")[1]
            k = {"%20": " "}.get(k, k)[:1]
            with open(FIFO, "wb") as f:
                f.write(k.encode())
            self.send_response(204); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(PAGE.encode())

    def log_message(self, *a):
        pass


http.server.HTTPServer(("127.0.0.1", PORT), H).serve_forever()
