#!/usr/bin/env python3
"""
RoboCar — сервер с автообнаружением.

Запуск:  python robocar_server.py
Браузер: http://localhost:8080

ESP находит этот сервер сама: она шлёт UDP-broadcast с общим токеном,
сервер отвечает своим адресом. Прописывать IP нигде не нужно.

Порты:
  5001/udp — обнаружение
  5000/tcp — канал команд
  8080/tcp — веб-интерфейс

ВАЖНО: токен — это метка "свой сервер", а не защита. Трафик открытый,
любой в той же сети может подключиться к машинке.
"""

import http.server
import json
import socket
import socketserver
import sys
import threading
import time
import urllib.parse

# ==================== НАСТРОЙКИ ====================
SECRET = "ROBOCAR-2026"     # должен совпадать с SECRET в скетче

DISCOVERY_PORT = 5001
TCP_PORT       = 5000
WEB_PORT       = 8080

BEACON_INTERVAL = 2.0       # как часто рассылать маяк
REPEAT_INTERVAL = 0.15      # автоповтор команд, чтобы не сработал failsafe
# ===================================================

PROBE = f"ROBOCAR?{SECRET}"
REPLY = f"ROBOCAR!{SECRET}:{TCP_PORT}"

LINK = None
LAST_LINES = []


# ============================================================
#                    СОЕДИНЕНИЕ С ESP
# ============================================================

class Link:
    def __init__(self, conn, addr):
        self.conn = conn
        self.addr = addr
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.repeat_cmd = None
        self.ping_sent_at = None
        self.last_hb = ""

    def send(self, cmd: str) -> bool:
        with self.lock:
            try:
                self.conn.sendall((cmd + "\n").encode())
                return True
            except OSError:
                self.stop.set()
                return False

    def reader(self):
        buf = b""
        while not self.stop.is_set():
            try:
                data = self.conn.recv(1024)
            except OSError:
                break
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue

                if text.startswith("HB "):
                    self.last_hb = text
                LAST_LINES.append(text)
                del LAST_LINES[:-40]

                if text == "PONG" and self.ping_sent_at is not None:
                    ms = (time.monotonic() - self.ping_sent_at) * 1000
                    self.ping_sent_at = None
                    print(f"\r  < PONG  {ms:.1f} мс\n> ", end="", flush=True)
                    continue

                print(f"\r  < {text}\n> ", end="", flush=True)

        self.stop.set()
        print("\r[ESP отключился]              \n> ", end="", flush=True)

    def repeater(self):
        while not self.stop.is_set():
            if self.repeat_cmd:
                self.send(self.repeat_cmd)
            time.sleep(REPEAT_INTERVAL)


# ============================================================
#                   ОБНАРУЖЕНИЕ (UDP)
# ============================================================

def discovery_service():
    """Отвечает на запросы ESP и периодически рассылает маяк."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    try:
        s.bind(("0.0.0.0", DISCOVERY_PORT))
    except OSError as e:
        print(f"[discovery] не смог занять UDP {DISCOVERY_PORT}: {e}")
        return
    s.settimeout(0.5)

    reply = REPLY.encode()
    last_beacon = 0.0

    while True:
        try:
            data, addr = s.recvfrom(256)
            if data.decode("utf-8", errors="ignore").strip() == PROBE:
                s.sendto(reply, addr)
                print(f"\r[запрос от {addr[0]} — ответил]\n> ", end="", flush=True)
        except socket.timeout:
            pass
        except OSError:
            pass

        now = time.time()
        if now - last_beacon >= BEACON_INTERVAL:
            last_beacon = now
            for target in ("255.255.255.255", broadcast_addr()):
                if not target:
                    continue
                try:
                    s.sendto(reply, (target, DISCOVERY_PORT))
                except OSError:
                    pass


def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def broadcast_addr() -> str:
    """Грубая оценка широковещательного адреса подсети (/24)."""
    ip = local_ip()
    parts = ip.split(".")
    if len(parts) != 4:
        return ""
    return ".".join(parts[:3] + ["255"])


# ============================================================
#                     ВЕБ-СТРАНИЦА
# ============================================================

PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>RoboCar</title>
<style>
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  body {
    margin: 0; padding: 20px;
    background: #14161a; color: #e6e8ec;
    font: 15px/1.4 ui-monospace, Menlo, Consolas, monospace;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; user-select: none;
  }
  h1 { font-size: 17px; font-weight: 600; margin: 0 0 4px; }
  .sub { color: #6f7580; font-size: 12px; margin-bottom: 20px; }
  .pad {
    display: grid; gap: 8px;
    grid-template-columns: repeat(3, 74px);
    grid-template-rows: repeat(2, 74px);
    margin-bottom: 22px;
  }
  .key {
    display: flex; align-items: center; justify-content: center;
    background: #1e2229; border: 1px solid #2c313a; border-radius: 10px;
    font-size: 20px; font-weight: 600; color: #99a0ac; cursor: pointer;
    transition: background .06s, color .06s, border-color .06s;
  }
  .key.on { background: #2f6f4f; border-color: #3d8f66; color: #eafff2; }
  .key.wide { grid-column: 2; }
  .row { display: flex; align-items: center; gap: 12px; width: 260px; margin-bottom: 14px; }
  .row label { color: #6f7580; font-size: 12px; width: 62px; }
  input[type=range] { flex: 1; accent-color: #3d8f66; }
  .val { width: 34px; text-align: right; }
  button.stop {
    width: 260px; padding: 12px; margin-bottom: 20px;
    background: #7a2b2b; border: 1px solid #a03a3a; border-radius: 10px;
    color: #ffecec; font: inherit; font-weight: 600; cursor: pointer;
  }
  button.stop:active { background: #963333; }
  .stat {
    width: 260px; padding: 10px 12px;
    background: #1a1d23; border: 1px solid #262b33; border-radius: 8px;
    font-size: 12px; color: #8b929d; line-height: 1.7; word-break: break-all;
  }
  .stat b { color: #e6e8ec; font-weight: 600; }
  .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; }
  .dot.ok { background: #4ea87a; }
  .dot.no { background: #a04545; }
</style>
</head>
<body>

<h1>RoboCar</h1>
<div class="sub">WASD или стрелки &nbsp;·&nbsp; пробел — стоп</div>

<div class="pad">
  <div></div>
  <div class="key wide" data-k="w">W</div>
  <div></div>
  <div class="key" data-k="a">A</div>
  <div class="key" data-k="s">S</div>
  <div class="key" data-k="d">D</div>
</div>

<div class="row">
  <label>скорость</label>
  <input id="spd" type="range" min="40" max="255" value="160">
  <span class="val" id="spdv">160</span>
</div>

<button class="stop" id="stopbtn">СТОП</button>

<div class="stat">
  <div><span class="dot no" id="dot"></span><span id="conn">ждём ESP</span></div>
  <div>команда: <b id="cmd">0 / 0</b></div>
  <div id="hb">—</div>
</div>

<script>
const keys = new Set();
let speed = 160;
const tiles = {};

document.querySelectorAll('.key').forEach(el => {
  const k = el.dataset.k;
  tiles[k] = el;
  el.addEventListener('pointerdown',   e => { e.preventDefault(); press(k); });
  el.addEventListener('pointerup',     e => { e.preventDefault(); release(k); });
  el.addEventListener('pointerleave',  () => release(k));
  el.addEventListener('pointercancel', () => release(k));
});

function press(k)   { keys.add(k);    tiles[k] && tiles[k].classList.add('on'); }
function release(k) { keys.delete(k); tiles[k] && tiles[k].classList.remove('on'); }
function releaseAll() { [...keys].forEach(release); }

const MAP = {
  'w':'w','a':'a','s':'s','d':'d',
  'ц':'w','ф':'a','ы':'s','в':'d',
  'arrowup':'w','arrowleft':'a','arrowdown':'s','arrowright':'d'
};

addEventListener('keydown', e => {
  if (e.repeat) return;
  const k = MAP[e.key.toLowerCase()];
  if (k) { e.preventDefault(); press(k); }
  if (e.key === ' ') { e.preventDefault(); releaseAll(); }
});
addEventListener('keyup', e => {
  const k = MAP[e.key.toLowerCase()];
  if (k) { e.preventDefault(); release(k); }
});
addEventListener('blur', releaseAll);
document.addEventListener('visibilitychange', () => { if (document.hidden) releaseAll(); });

const spd = document.getElementById('spd');
spd.addEventListener('input', () => {
  speed = +spd.value;
  document.getElementById('spdv').textContent = speed;
});
document.getElementById('stopbtn')
        .addEventListener('pointerdown', e => { e.preventDefault(); releaseAll(); });

function mix() {
  let thr = 0, str = 0;
  if (keys.has('w')) thr += 1;
  if (keys.has('s')) thr -= 1;
  if (keys.has('d')) str += 1;
  if (keys.has('a')) str -= 1;
  if (thr === 0 && str === 0) return [0, 0];
  let l = thr + str, r = thr - str;
  const m = Math.max(Math.abs(l), Math.abs(r), 1);
  return [Math.round(l / m * speed), Math.round(r / m * speed)];
}

let last = null;
setInterval(() => {
  const [l, r] = mix();
  document.getElementById('cmd').textContent = l + ' / ' + r;
  const key = l + ',' + r;
  if (key !== last || l !== 0 || r !== 0) {
    last = key;
    fetch('/drive?l=' + l + '&r=' + r).catch(() => {});
  }
}, 100);

setInterval(async () => {
  try {
    const s = await (await fetch('/status')).json();
    document.getElementById('dot').className = 'dot ' + (s.connected ? 'ok' : 'no');
    document.getElementById('conn').textContent =
      s.connected ? ('ESP на связи — ' + s.addr) : 'ждём ESP';
    document.getElementById('hb').textContent = s.hb || '—';
  } catch (e) {}
}, 1000);
</script>
</body>
</html>
"""


# ============================================================
#                      HTTP-СЕРВЕР
# ============================================================

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _reply(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)

        if u.path == "/":
            self._reply(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            return

        if u.path == "/drive":
            q = urllib.parse.parse_qs(u.query)
            try:
                l = max(-255, min(255, int(float(q.get("l", ["0"])[0]))))
                r = max(-255, min(255, int(float(q.get("r", ["0"])[0]))))
            except ValueError:
                l = r = 0

            if LINK and not LINK.stop.is_set():
                if l == 0 and r == 0:
                    if LINK.repeat_cmd is not None:
                        LINK.repeat_cmd = None
                        LINK.send("STOP")
                else:
                    LINK.repeat_cmd = f"M {l} {r}"
                    LINK.send(f"M {l} {r}")

            self._reply(200, "application/json", b'{"ok":true}')
            return

        if u.path == "/status":
            connected = bool(LINK and not LINK.stop.is_set())
            body = json.dumps({
                "connected": connected,
                "addr": LINK.addr[0] if connected else "",
                "hb": LINK.last_hb if LINK else "",
            }).encode()
            self._reply(200, "application/json", body)
            return

        self._reply(404, "text/plain", b"not found")


class ThreadedHTTP(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ============================================================
#                        ЗАПУСК
# ============================================================

def tcp_acceptor(srv):
    global LINK
    while True:
        try:
            conn, addr = srv.accept()
        except OSError:
            break
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"\r[ESP подключился: {addr[0]}:{addr[1]}]\n> ", end="", flush=True)

        if LINK:
            LINK.stop.set()
        LINK = Link(conn, addr)
        threading.Thread(target=LINK.reader,   daemon=True).start()
        threading.Thread(target=LINK.repeater, daemon=True).start()


def main():
    global LINK

    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        tcp.bind(("0.0.0.0", TCP_PORT))
    except OSError as e:
        print(f"Порт {TCP_PORT} занят: {e}")
        print("Закрой старый сервер:  taskkill /F /IM python.exe")
        sys.exit(1)
    tcp.listen(1)

    threading.Thread(target=tcp_acceptor, args=(tcp,), daemon=True).start()
    threading.Thread(target=discovery_service, daemon=True).start()

    web = ThreadedHTTP(("0.0.0.0", WEB_PORT), Handler)
    threading.Thread(target=web.serve_forever, daemon=True).start()

    ip = local_ip()
    print("=" * 52)
    print(f"  Этот компьютер:  {ip}")
    print(f"  Обнаружение:     UDP {DISCOVERY_PORT}  (токен {SECRET})")
    print(f"  Канал команд:    TCP {TCP_PORT}")
    print(f"  Браузер:         http://localhost:{WEB_PORT}")
    print(f"  С телефона:      http://{ip}:{WEB_PORT}")
    print("=" * 52)
    print("ESP найдёт сервер сама. Прописывать IP не нужно.")
    print("Команды: test, state, max 80, ping, stop, quit")

    try:
        while True:
            try:
                raw = input("> ").strip()
            except EOFError:
                break
            if not raw:
                continue
            low = raw.lower()

            if low in ("quit", "exit"):
                break
            if not LINK or LINK.stop.is_set():
                print("[ESP не подключён]")
                continue

            if low == "ping":
                LINK.ping_sent_at = time.monotonic()
            if raw.upper().startswith("M "):
                LINK.repeat_cmd = raw
            elif low in ("stop", "x", "brake"):
                LINK.repeat_cmd = None
                raw = "STOP" if low == "x" else raw.upper()

            LINK.send(raw)

    except KeyboardInterrupt:
        pass
    finally:
        if LINK:
            LINK.repeat_cmd = None
            LINK.send("STOP")
        print("\nСервер остановлен.")


if __name__ == "__main__":
    main()
