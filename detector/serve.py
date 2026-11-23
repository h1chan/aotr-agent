"""Разметка титанов: локальный сервер и страница с кликами.

  python3 detector/serve.py
  http://127.0.0.1:8765

ЛКМ — точка на титане, ПКМ — убрать последнюю, Enter — сохранить и дальше,
N — титанов нет. Модель подсказывает точки, их можно принять или стереть.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).parent
FRAMES = HERE / "frames"
LABELS = HERE / "labels.jsonl"
MODEL = HERE / "model2.pt"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>AoTR labeling</title>
<style>
html,body{margin:0;height:100%;background:#161616;color:#ddd;font:14px system-ui}
#bar{display:flex;gap:10px;align-items:center;padding:8px 12px;background:#222}
button{background:#2d6cdf;color:#fff;border:0;border-radius:6px;padding:8px 12px;cursor:pointer}
button.gray{background:#3d3d3d}
#stage{height:calc(100% - 52px);display:flex;align-items:center;justify-content:center}
canvas{cursor:crosshair;background:#000}
#info{margin-left:auto;white-space:nowrap}
</style></head><body>
<div id="bar">
  <button id="save">Сохранить (Enter)</button>
  <button id="none" class="gray">Титанов нет (N)</button>
  <button id="undo" class="gray">Убрать точку (ПКМ)</button>
  <button id="reload" class="gray">Обновить</button>
  <button id="conf" class="gray" style="display:none">Спорные</button>
  <label style="display:flex;gap:6px;align-items:center"><input type="checkbox" id="sug"> подсказки</label>
  <span style="opacity:.7">ЛКМ точка · ПКМ убрать · Enter сохранить · N пусто</span>
  <span id="info"></span>
</div>
<div id="stage"><canvas id="cv"></canvas></div>
<script>
var frames = [];
var labels = {};
var pos = 0;
var points = [];
var image = new Image();
var k = 1;
var reviewMap = {};
var reviewMode = false;
var saving = false;
var cv = document.getElementById('cv');
var cx = cv.getContext('2d');
var infoEl = document.getElementById('info');

function info() {
  var done = Object.keys(labels).length;
  infoEl.textContent = 'кадр ' + (pos + 1) + '/' + frames.length +
    ' · размечено ' + done + '/' + frames.length +
    ' · точек ' + points.length;
}

function draw() {
  cx.clearRect(0, 0, cv.width, cv.height);
  cx.drawImage(image, 0, 0, cv.width, cv.height);
  cx.strokeStyle = '#00ff66';
  cx.lineWidth = 2;
  for (var i = 0; i < points.length; i++) {
    var x = points[i].x * k;
    var y = points[i].y * k;
    cx.beginPath(); cx.arc(x, y, 8, 0, 6.284); cx.stroke();
    cx.beginPath();
    cx.moveTo(x - 12, y); cx.lineTo(x + 12, y);
    cx.moveTo(x, y - 12); cx.lineTo(x, y + 12);
    cx.stroke();
  }
}

function show() {
  if (!frames.length) { infoEl.textContent = 'нет списка кадров'; return; }
  var f = frames[pos];
  image.onload = function () {
    var maxW = window.innerWidth - 40;
    var maxH = window.innerHeight - 110;
    k = Math.min(maxW / image.width, maxH / image.height, 1);
    if (!(k > 0.05)) { k = 0.5; }
    cv.width = Math.round(image.width * k);
    cv.height = Math.round(image.height * k);
    points = (labels[f.i] || reviewMap[f.i] || []).slice();
    draw();
    info();
    if (!(f.i in labels) && document.getElementById('sug').checked) {
      fetch('/api/predict?i=' + f.i).then(function (r) { return r.json(); })
        .then(function (d) {
          if (d.points && d.points.length && !(f.i in labels)) {
            points = d.points;
            draw();
            info();
          }
        });
    }
  };
  image.onerror = function () {
    infoEl.textContent = 'кадр не загрузился: ' + f.name;
  };
  image.src = '/frames/' + f.name + '?v=' + Date.now();
}

function nextFrame() {
  if (reviewMode) {
    if (pos + 1 < frames.length) { pos += 1; points = []; show(); return; }
    reviewMode = false;
    infoEl.textContent = 'спорные проверены';
    return;
  }
  for (var step = 1; step <= frames.length; step++) {
    var j = (pos + step) % frames.length;
    if (!(frames[j].i in labels)) { pos = j; points = []; show(); return; }
  }
  infoEl.textContent = 'все ' + frames.length + ' кадров размечены';
}

function save() {
  if (saving) { return; }
  saving = true;
  var f = frames[pos];
  var body = JSON.stringify({ i: f.i, points: points });
  fetch('/api/label', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body })
    .then(function () {
      labels[f.i] = points.slice();
      saving = false;
      info();
      nextFrame();
    });
}

function start() {
  labels = {};
  points = [];
  pos = 0;
  fetch('/api/frames?t=' + Date.now())
    .then(function (r) { return r.json(); })
    .then(function (list) {
      frames = list;
      return fetch('/api/labels?t=' + Date.now());
    })
    .then(function (r) { return r.json(); })
    .then(function (rows) {
      for (var i = 0; i < rows.length; i++) { labels[rows[i].i] = rows[i].points; }
      var first = 0;
      for (var j = 0; j < frames.length; j++) {
        if (!(frames[j].i in labels)) { first = j; break; }
      }
      pos = first;
      show();
    });
}

cv.addEventListener('contextmenu', function (e) {
  e.preventDefault();
  points.pop();
  draw();
  info();
});

cv.addEventListener('click', function (e) {
  var r = cv.getBoundingClientRect();
  points.push({
    x: Math.round((e.clientX - r.left) / k),
    y: Math.round((e.clientY - r.top) / k)
  });
  draw();
  info();
});

document.getElementById('save').onclick = save;
document.getElementById('none').onclick = function () { points = []; save(); };
document.getElementById('undo').onclick = function () { points.pop(); draw(); info(); };
document.getElementById('reload').onclick = start;
document.getElementById('conf').onclick = function () {
  fetch('/api/conflicts').then(function (r) { return r.json(); }).then(function (rows) {
    if (!rows.length) { infoEl.textContent = 'спорных нет'; return; }
    var byId = {};
    for (var i = 0; i < frames.length; i++) { byId[frames[i].i] = frames[i]; }
    var list = [];
    for (var j = 0; j < rows.length; j++) {
      if (byId[rows[j].i]) {
        list.push(byId[rows[j].i]);
        reviewMap[rows[j].i] = rows[j].points;
      }
    }
    frames = list;
    pos = 0;
    reviewMode = true;
    points = [];
    show();
  });
};

window.addEventListener('keydown', function (e) {
  if (e.repeat) { return; }
  if (e.key === 'Enter') { save(); }
  if (e.key === 'n' || e.key === 'N') { points = []; save(); }
  if (e.key === 'Backspace') { e.preventDefault(); points.pop(); draw(); info(); }
});

start();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._html(PAGE.encode())
        elif path == "/api/frames":
            self._json(json.loads((FRAMES / "index.json").read_text()))
        elif path == "/api/labels":
            latest = {}
            if LABELS.exists():
                for line in LABELS.open():
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        latest[row["i"]] = row
            self._json(list(latest.values()))
        elif path == "/api/conflicts":
            hist = {}
            if LABELS.exists():
                for line in LABELS.open():
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    hist.setdefault(row["i"], []).append(row["points"])
            out = []
            for i, variants in hist.items():
                keys = {tuple(sorted((p["x"], p["y"]) for p in pts)) for pts in variants}
                if len(keys) > 1:
                    best = max(variants, key=len)
                    out.append({"i": i, "points": best})
            self._json(sorted(out, key=lambda r: r["i"]))
        elif path == "/api/predict":
            i = int(parse_qs(urlparse(self.path).query).get("i", ["-1"])[0])
            self._json(self._predict(i))
        elif path.startswith("/frames/"):
            file = FRAMES / Path(path).name
            if file.exists():
                data = file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def _predict(self, i: int) -> dict:
        if not MODEL.exists() or i < 0:
            return {"points": []}
        try:
            import torch  # noqa: F401
            from infer import load_model, predict
            from train import decode

            index = json.loads((FRAMES / "index.json").read_text())
            item = next((f for f in index if f["i"] == i), None)
            if not item:
                return {"points": []}
            model, ck = load_model(MODEL)
            rgb = decode(FRAMES / item["name"])
            pts, _ = predict(model, ck, rgb, 0.3)
            return {"points": [{"x": x, "y": y} for x, y, _ in pts]}
        except Exception as exc:
            return {"points": [], "error": str(exc)}

    def do_POST(self) -> None:
        if urlparse(self.path).path == "/api/label":
            length = int(self.headers.get("Content-Length", 0))
            row = json.loads(self.rfile.read(length))
            with LABELS.open("a") as f:
                f.write(json.dumps({"i": row["i"], "points": row["points"]}) + "\n")
            self._json({"ok": True})
        else:
            self.send_error(404)

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
