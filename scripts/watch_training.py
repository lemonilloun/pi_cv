#!/usr/bin/env python3
"""Live training dashboard for the pdfserver fine-tune, served on the laptop.

Ultralytics writes `results.csv` one row per epoch plus a set of PNGs. Both
live on the GPU host, which has no web server and is only reachable over SSH.
This serves a page on 127.0.0.1 that pulls those files on demand, so the run
can be watched from a browser without touching the training process.

Deliberately stdlib-only and read-only: nothing here can disturb a run that
has hours invested in it. Charts are server-rendered SVG rather than a JS
library, so there is nothing to install and nothing to load from a CDN.

    ./scripts/watch_training.sh
    # then open http://127.0.0.1:8090/

Port 8090 avoids 8080 (the robot panel) and 8765 (the Pi TCP link).
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_HOST = "pdfserver"
DEFAULT_RUN = "~/cv_research/pi_cv_train/runs/segment/runs/indoor_yolo11l"
DEFAULT_LOG = "~/cv_research/pi_cv_train/train.log"
# The remote is polled at most this often no matter how eagerly the browser
# refreshes; an ssh round trip per chart redraw would be silly.
CACHE_TTL_S = 15.0

# (csv column, label, panel) — ultralytics' exact header names.
SERIES = [
    ("metrics/mAP50(M)", "mask mAP50", "quality"),
    ("metrics/mAP50-95(M)", "mask mAP50-95", "quality"),
    ("metrics/mAP50(B)", "box mAP50", "quality"),
    ("metrics/mAP50-95(B)", "box mAP50-95", "quality"),
    ("metrics/precision(M)", "mask precision", "pr"),
    ("metrics/recall(M)", "mask recall", "pr"),
    ("train/box_loss", "box loss", "loss"),
    ("train/seg_loss", "seg loss", "loss"),
    ("train/cls_loss", "cls loss", "loss"),
    ("val/box_loss", "val box loss", "loss"),
    ("val/seg_loss", "val seg loss", "loss"),
]

COLORS = ["#2f81f7", "#3fb950", "#d29922", "#f85149", "#a371f7",
          "#39c5cf", "#db6d28", "#8b949e"]


class Remote:
    """Cached read-only access to the run directory over ssh."""

    def __init__(self, host: str, run_dir: str, log_path: str) -> None:
        self.host = host
        self.run_dir = run_dir
        self.log_path = log_path
        self._cache: dict[str, tuple[float, object]] = {}

    def _cached(self, key: str, producer):
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit and now - hit[0] < CACHE_TTL_S:
            return hit[1]
        try:
            value = producer()
        except Exception as exc:  # noqa: BLE001 - surfaced in the page, not raised
            value = {"error": f"{type(exc).__name__}: {exc}"}
        self._cache[key] = (now, value)
        return value

    def _ssh(self, command: str, binary: bool = False):
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.host, command],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", "replace")[:300] or "ssh failed")
        return result.stdout if binary else result.stdout.decode("utf-8", "replace")

    def results(self) -> object:
        def load():
            text = self._ssh(f"cat {self.run_dir}/results.csv")
            rows = list(csv.DictReader(io.StringIO(text)))
            # Ultralytics appends a fresh header on resume; drop repeats so a
            # resumed run does not draw a spike back to epoch 1.
            return [r for r in rows if r.get("epoch", "").strip().isdigit()]
        return self._cached("results", load)

    def status(self) -> object:
        def load():
            # The last progress line carries the in-flight epoch, which
            # results.csv only learns about once the epoch finishes.
            #
            # Two things make this fiddly and both were found the hard way:
            # ultralytics separates progress updates with CR (not LF), so the
            # whole epoch is one physical line until \r is TRANSLATED — not
            # deleted, which merely glues it together — and every update is
            # prefixed with an ANSI erase-line escape, so the epoch counter is
            # not at the start of the line until the escapes are stripped.
            text = self._ssh(
                f"tr '\\r' '\\n' < {self.log_path} "
                f"| sed 's/\\x1b\\[[0-9;]*[A-Za-z]//g' "
                f"| grep -E '^ *[0-9]+/[0-9]+ +[0-9.]+G' | tail -1")
            alive = self._ssh(
                "pgrep -f 'train_seg.py' >/dev/null && echo yes || echo no").strip()
            gpu = self._ssh(
                "nvidia-smi --query-gpu=index,memory.used,memory.total,"
                "utilization.gpu,temperature.gpu --format=csv,noheader")
            return {"line": text.strip()[:200], "alive": alive == "yes", "gpu": gpu.strip()}
        return self._cached("status", load)

    def image(self, name: str) -> bytes:
        def load():
            return self._ssh(f"cat {self.run_dir}/{name} | base64", binary=False)
        import base64
        data = self._cached(f"img:{name}", load)
        if isinstance(data, dict):
            raise RuntimeError(data.get("error", "unavailable"))
        return base64.b64decode(data)

    def image_list(self) -> object:
        def load():
            text = self._ssh(
                f"ls {self.run_dir} 2>/dev/null | grep -E '\\.(png|jpg)$' || true")
            return [n for n in text.split() if n]
        return self._cached("imglist", load)


def _floats(rows, column):
    out = []
    for row in rows:
        raw = (row.get(column) or "").strip()
        try:
            out.append(float(raw))
        except ValueError:
            out.append(None)
    return out


def svg_chart(rows, columns, title, height=260, width=760):
    """Server-rendered multi-series line chart. No JS, no dependencies."""
    epochs = [float(r["epoch"]) for r in rows]
    series = [(label, _floats(rows, col)) for col, label, _ in columns]
    values = [v for _, ys in series for v in ys if v is not None]
    if not epochs or not values:
        return f'<div class="card"><h3>{html.escape(title)}</h3><p class="muted">no data yet</p></div>'

    pad_l, pad_r, pad_t, pad_b = 54, 130, 16, 30
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    x_min, x_max = min(epochs), max(epochs)
    y_min, y_max = min(values), max(values)
    if y_max == y_min:
        y_max = y_min + 1e-6
    span = y_max - y_min
    y_min -= span * 0.08
    y_max += span * 0.08

    def px(e):
        return pad_l + (0 if x_max == x_min else (e - x_min) / (x_max - x_min)) * plot_w

    def py(v):
        return pad_t + (1 - (v - y_min) / (y_max - y_min)) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    # horizontal gridlines + y labels
    for i in range(5):
        v = y_min + (y_max - y_min) * i / 4
        y = py(v)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
                     f'class="grid"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" class="axis" '
                     f'text-anchor="end">{v:.3f}</text>')
    # x labels at the ends
    parts.append(f'<text x="{pad_l}" y="{height - 8}" class="axis">{int(x_min)}</text>')
    parts.append(f'<text x="{pad_l + plot_w}" y="{height - 8}" class="axis" '
                 f'text-anchor="end">epoch {int(x_max)}</text>')

    for idx, (label, ys) in enumerate(series):
        color = COLORS[idx % len(COLORS)]
        points = [(px(e), py(v)) for e, v in zip(epochs, ys) if v is not None]
        if not points:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                     for i, (x, y) in enumerate(points))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        last = points[-1]
        parts.append(f'<circle cx="{last[0]:.1f}" cy="{last[1]:.1f}" r="3" fill="{color}"/>')
        ly = pad_t + 16 + idx * 19
        parts.append(f'<line x1="{pad_l + plot_w + 14}" y1="{ly - 4}" '
                     f'x2="{pad_l + plot_w + 32}" y2="{ly - 4}" stroke="{color}" '
                     f'stroke-width="2"/>')
        value = next((v for v in reversed(ys) if v is not None), None)
        text = f"{label} {value:.4f}" if value is not None else label
        parts.append(f'<text x="{pad_l + plot_w + 38}" y="{ly}" class="legend">'
                     f'{html.escape(text)}</text>')
    parts.append("</svg>")
    return (f'<div class="card"><h3>{html.escape(title)}</h3>'
            + "".join(parts) + "</div>")


def render_page(remote: Remote, refresh: int) -> str:
    rows = remote.results()
    status = remote.status()
    error = None
    if isinstance(rows, dict):
        error = rows.get("error")
        rows = []

    done = len(rows)
    total = 0
    in_flight = ""
    gpu_rows = []
    if isinstance(status, dict) and not status.get("error"):
        in_flight = status.get("line", "")
        tokens = in_flight.split()
        if tokens and "/" in tokens[0]:
            try:
                total = int(tokens[0].split("/")[1])
            except (ValueError, IndexError):
                total = 0
        for line in (status.get("gpu") or "").splitlines():
            gpu_rows.append([c.strip() for c in line.split(",")])
    alive = bool(isinstance(status, dict) and status.get("alive"))

    eta = ""
    if rows and total:
        try:
            elapsed = float(rows[-1]["time"])
            per_epoch = elapsed / max(1, done)
            remaining = per_epoch * max(0, total - done)
            eta = (f"{per_epoch / 60:.1f} min/epoch · "
                   f"~{remaining / 3600:.1f} h left · "
                   f"{elapsed / 3600:.1f} h elapsed")
        except (ValueError, KeyError):
            eta = ""

    best = ""
    if rows:
        maps = [(v, i + 1) for i, v in enumerate(_floats(rows, "metrics/mAP50-95(M)"))
                if v is not None]
        if maps:
            top = max(maps)
            best = f"best mask mAP50-95 {top[0]:.4f} at epoch {top[1]}"

    charts = []
    for panel, title in (("quality", "Quality (higher is better)"),
                         ("pr", "Precision / recall"),
                         ("loss", "Losses (lower is better)")):
        cols = [(c, l, p) for c, l, p in SERIES if p == panel]
        charts.append(svg_chart(rows, cols, title))

    images = remote.image_list()
    image_html = ""
    if isinstance(images, list) and images:
        # results.png is ultralytics' own summary; val_batch*_pred are the
        # actual segmentations on held-out rooms, which is the only view that
        # answers "does this look right" rather than "is the number rising".
        preferred = [n for n in images if "val_batch" in n and "pred" in n]
        preferred += [n for n in images if n == "results.png"]
        preferred += [n for n in images if n not in preferred][:4]
        image_html = "".join(
            f'<div class="card"><h3>{html.escape(name)}</h3>'
            f'<img src="/img/{html.escape(name)}?t={int(time.time() // 30)}" '
            f'alt="{html.escape(name)}"></div>'
            for name in preferred[:8])

    banner = ""
    if error:
        banner = f'<div class="card err">cannot read results.csv — {html.escape(error)}</div>'
    elif not alive:
        banner = ('<div class="card warn">training process not found — it has either '
                  'finished or died. Check train.log on the host.</div>')

    # GPU 0 belongs to another user's vLLM: it holds its whole allocation and
    # sits at 0% between requests, so those numbers are SUPPOSED to look
    # frozen. Saying so on the page stops that reading as a broken panel.
    # GPU 1's memory is equally static once PyTorch's caching allocator has
    # grabbed it — utilisation and temperature are the columns that move.
    owner = {"0": "neighbour's vLLM — idle, holds memory", "1": "our training"}
    gpu_html = "".join(
        f"<tr><td>GPU {html.escape(r[0])}</td><td>{html.escape(r[1])} / {html.escape(r[2])}</td>"
        f"<td>{html.escape(r[3])}</td><td>{html.escape(r[4])}&deg;C</td>"
        f"<td class='muted'>{html.escape(owner.get(r[0], ''))}</td></tr>"
        for r in gpu_rows if len(r) >= 5)
    fetched = time.strftime("%H:%M:%S")

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh}">
<title>yolo11l-seg · indoor fine-tune</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif;
         margin: 0; padding: 24px; background: #0d1117; color: #e6edf3; }}
  @media (prefers-color-scheme: light) {{
    body {{ background: #f6f8fa; color: #1f2328; }}
    .card {{ background: #fff; border-color: #d0d7de; }}
    .grid {{ stroke: #d8dee4; }} .axis, .muted {{ fill: #656d76; color: #656d76; }}
    .legend {{ fill: #1f2328; }}
  }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  .sub {{ color: #8b949e; margin-bottom: 18px; }}
  .cards {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fit,minmax(420px,1fr)); }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 14px 16px; }}
  .card h3 {{ font-size: 13px; margin: 0 0 10px; font-weight: 600; }}
  .chart {{ width: 100%; height: auto; }}
  .grid {{ stroke: #30363d; stroke-width: 1; }}
  .axis {{ font-size: 10px; fill: #8b949e; }}
  .legend {{ font-size: 11px; fill: #e6edf3; }}
  .muted {{ color: #8b949e; }}
  .err {{ border-color: #f85149; }} .warn {{ border-color: #d29922; }}
  .stats {{ display: flex; gap: 26px; flex-wrap: wrap; margin-bottom: 18px; }}
  .stat b {{ display: block; font-size: 20px; }}
  .stat span {{ color: #8b949e; font-size: 12px; }}
  table {{ border-collapse: collapse; font-size: 12px; }}
  td {{ padding: 2px 14px 2px 0; }}
  img {{ width: 100%; border-radius: 6px; }}
  code {{ background: rgba(110,118,129,.2); padding: 1px 5px; border-radius: 4px; }}
</style></head><body>
<h1>yolo11l-seg — indoor fine-tune (ADE20K, 31 classes)</h1>
<div class="sub">{'running' if alive else 'not running'} ·
  <b>fetched {fetched}</b> · refreshes every {refresh}s · {html.escape(best)}
  <br><span class="muted">If the clock above stops advancing the page is stale,
  not the run — reload with Cmd-Shift-R.</span></div>
{banner}
<div class="stats">
  <div class="stat"><b>{done}{f' / {total}' if total else ''}</b><span>epochs done</span></div>
  <div class="stat"><b>{html.escape(eta.split(' · ')[0]) if eta else '—'}</b><span>speed</span></div>
  <div class="stat"><b>{html.escape(eta.split(' · ')[1]) if ' · ' in eta else '—'}</b><span>remaining</span></div>
</div>
<div class="cards">{''.join(charts)}
  <div class="card"><h3>Host</h3><table>{gpu_html}</table>
    <p class="muted" style="margin-top:10px">in flight: <code>{html.escape(in_flight or '—')}</code></p>
  </div>
{image_html}</div>
</body></html>"""


def make_handler(remote: Remote, refresh: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/img/"):
                name = self.path[5:].split("?")[0]
                if "/" in name or ".." in name:
                    self.send_error(400)
                    return
                try:
                    blob = remote.image(name)
                except Exception:  # noqa: BLE001
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type",
                                 "image/png" if name.endswith(".png") else "image/jpeg")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)
                return
            if self.path.startswith("/api"):
                rows = remote.results()
                body = json.dumps(rows if isinstance(rows, list) else [rows]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = render_page(remote, refresh).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            # Without this the browser happily serves its cached copy on every
            # meta-refresh, and the page sits frozen while the server is
            # returning fresh numbers. There is no validator to revalidate
            # against, so heuristic caching wins unless it is forbidden.
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=DEFAULT_HOST, help="ssh alias of the GPU box")
    parser.add_argument("--run-dir", default=DEFAULT_RUN)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--refresh", type=int, default=60)
    args = parser.parse_args()

    remote = Remote(args.host, args.run_dir, args.log)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(remote, args.refresh))
    print(f"training dashboard: http://127.0.0.1:{args.port}/   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
