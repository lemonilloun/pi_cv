"""HTTP control panel and MJPEG preview server.

Serves the latest received camera/CV frame as an MJPEG stream plus a JSON
control API used by the web panel to switch session client modes and read
telemetry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from http import server
import logging
from pathlib import Path
import socketserver
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse


logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_MODES = {"idle", "stream", "depth", "yolo", "pipeline"}


@dataclass(frozen=True)
class PreviewFrame:
    data: bytes
    metadata: dict[str, Any]


class LatestFrameStore:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: PreviewFrame | None = None
        self._sequence = 0

    def update(self, data: bytes, metadata: dict[str, Any]) -> None:
        with self._condition:
            self._sequence += 1
            self._frame = PreviewFrame(data=data, metadata=metadata)
            self._condition.notify_all()

    def wait_for_next(self, last_sequence: int, timeout: float = 10.0) -> tuple[int, PreviewFrame | None]:
        with self._condition:
            self._condition.wait_for(lambda: self._sequence != last_sequence, timeout=timeout)
            return self._sequence, self._frame

    def latest(self) -> tuple[int, PreviewFrame | None]:
        with self._condition:
            return self._sequence, self._frame


class MjpegPreviewServer:
    def __init__(
        self,
        host: str,
        port: int,
        frame_store: LatestFrameStore,
        registry: Any | None = None,
        telemetry_store: Any | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.frame_store = frame_store
        self._httpd = _ThreadingHttpServer(
            (host, port),
            _make_handler(frame_store, registry, telemetry_store),
        )
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        logger.info("MJPEG preview listening on http://%s:%s/", self.host, self.port)

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


class _ThreadingHttpServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _make_handler(
    frame_store: LatestFrameStore,
    registry: Any | None = None,
    telemetry_store: Any | None = None,
) -> type[server.BaseHTTPRequestHandler]:
    class StreamingHandler(server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._serve_index()
                return
            if parsed.path == "/stream.mjpg":
                self._serve_stream()
                return
            if parsed.path == "/latest.jpg":
                self._serve_latest_jpeg()
                return
            if parsed.path == "/api/status":
                self._serve_status()
                return
            if parsed.path == "/api/telemetry":
                self._serve_telemetry(parsed.query)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/mode":
                self._handle_mode_post()
                return
            self._send_json({"error": "not found"}, status=404)

        def log_message(self, format: str, *args: object) -> None:
            logger.debug("Control panel HTTP: " + format, *args)

        def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
            content = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _read_json_body(self) -> dict[str, Any] | None:
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (TypeError, ValueError):
                return None
            if length <= 0 or length > 65536:
                return None
            try:
                body = self.rfile.read(length)
                data = json.loads(body)
            except (OSError, ValueError):
                return None
            return data if isinstance(data, dict) else None

        def _serve_status(self) -> None:
            devices: list[dict[str, Any]] = []
            if registry is not None:
                for handle in registry.list_clients():
                    entry = handle.to_status_dict()
                    if telemetry_store is not None:
                        entry["telemetry"] = telemetry_store.latest(handle.device_id)
                    devices.append(entry)

            _, frame = frame_store.latest()
            frame_meta = frame.metadata if frame is not None else None
            self._send_json({"devices": devices, "latest_frame": frame_meta})

        def _serve_telemetry(self, query: str) -> None:
            if telemetry_store is None:
                self._send_json({"devices": {}})
                return

            params = parse_qs(query)
            try:
                seconds = float(params.get("seconds", ["120"])[0])
            except ValueError:
                seconds = 120.0
            seconds = max(1.0, min(3600.0, seconds))

            device_ids = params.get("device_id") or telemetry_store.device_ids()
            history = {
                device_id: telemetry_store.recent(device_id, seconds)
                for device_id in device_ids
            }
            self._send_json({"devices": history, "seconds": seconds})

        def _handle_mode_post(self) -> None:
            if registry is None:
                self._send_json({"error": "session control is not enabled"}, status=503)
                return

            body = self._read_json_body()
            if body is None:
                self._send_json({"error": "invalid JSON body"}, status=400)
                return

            mode = str(body.get("mode", "")).strip().lower()
            if mode not in SESSION_MODES:
                self._send_json(
                    {"error": f"invalid mode: {mode!r}", "valid_modes": sorted(SESSION_MODES)},
                    status=400,
                )
                return

            device_id = body.get("device_id")
            device_id = str(device_id) if device_id else None

            from mac_server.registry import CommandDispatchError

            try:
                result = registry.send_command(device_id=device_id, action="set_mode", mode=mode)
            except CommandDispatchError as exc:
                self._send_json({"error": str(exc)}, status=409)
                return
            self._send_json({"ok": True, "command": result})

        def _serve_index(self) -> None:
            index_path = STATIC_DIR / "index.html"
            try:
                content = index_path.read_bytes()
            except OSError:
                content = _preview_page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _serve_latest_jpeg(self) -> None:
            _, frame = frame_store.latest()
            if frame is None:
                self.send_error(404, "No camera stream frame received yet")
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(frame.data)))
            self.end_headers()
            self.wfile.write(frame.data)

        def _serve_stream(self) -> None:
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()

            sequence = -1
            try:
                while True:
                    sequence, frame = frame_store.wait_for_next(sequence)
                    if frame is None:
                        continue

                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame.data)))
                    self.end_headers()
                    self.wfile.write(frame.data)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                logger.info("MJPEG preview client disconnected")
            except Exception as exc:
                logger.warning("MJPEG preview client removed: %s", exc)

    return StreamingHandler


def _preview_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Raspberry Pi Camera Stream</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
    main { min-height: 100vh; display: grid; place-items: center; }
    img { max-width: 100vw; max-height: 100vh; object-fit: contain; }
  </style>
</head>
<body>
  <main>
    <img src="/stream.mjpg" alt="Raspberry Pi camera stream">
  </main>
</body>
</html>
"""
