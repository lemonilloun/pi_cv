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


class FrameStoreHub:
    """Named LatestFrameStores: 'pi' (frames from the Pi), 'depth' (server
    computed heatmaps), 'map' (rendered room map). Stores are created lazily."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stores: dict[str, LatestFrameStore] = {}

    def get(self, name: str) -> LatestFrameStore:
        with self._lock:
            store = self._stores.get(name)
            if store is None:
                store = LatestFrameStore()
                self._stores[name] = store
            return store

    def peek(self, name: str) -> LatestFrameStore | None:
        with self._lock:
            return self._stores.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return list(self._stores.keys())


class MjpegPreviewServer:
    def __init__(
        self,
        host: str,
        port: int,
        frame_hub: FrameStoreHub,
        registry: Any | None = None,
        telemetry_store: Any | None = None,
        cv_status_provider: Any | None = None,
        scan_controller: Any | None = None,
        room_store: Any | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.frame_hub = frame_hub
        self._httpd = _ThreadingHttpServer(
            (host, port),
            _make_handler(
                frame_hub,
                registry,
                telemetry_store,
                cv_status_provider,
                scan_controller,
                room_store,
            ),
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
    frame_hub: FrameStoreHub,
    registry: Any | None = None,
    telemetry_store: Any | None = None,
    cv_status_provider: Any | None = None,
    scan_controller: Any | None = None,
    room_store: Any | None = None,
) -> type[server.BaseHTTPRequestHandler]:
    class StreamingHandler(server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._serve_index()
                return
            if parsed.path == "/stream.mjpg":
                self._serve_stream(parsed.query)
                return
            if parsed.path == "/latest.jpg":
                self._serve_latest_jpeg(parsed.query)
                return
            if parsed.path == "/api/status":
                self._serve_status()
                return
            if parsed.path == "/api/telemetry":
                self._serve_telemetry(parsed.query)
                return
            if parsed.path == "/api/rooms":
                self._serve_rooms()
                return
            if parsed.path == "/api/scan/status":
                self._serve_scan_status()
                return
            if parsed.path == "/api/objects":
                self._serve_objects()
                return
            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/mode":
                self._handle_mode_post()
                return
            if parsed.path == "/api/rooms":
                self._handle_rooms_post()
                return
            if parsed.path == "/api/scan/start":
                self._handle_scan_start()
                return
            if parsed.path == "/api/scan/stop":
                self._handle_scan_stop()
                return
            self._send_json({"error": "not found"}, status=404)

        def _view_store(self, query: str) -> LatestFrameStore | None:
            params = parse_qs(query)
            view = params.get("view", ["pi"])[0]
            if view == "pi":
                return frame_hub.get("pi")
            return frame_hub.peek(view)

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

            _, frame = frame_hub.get("pi").latest()
            frame_meta = frame.metadata if frame is not None else None

            views: dict[str, Any] = {}
            for name in frame_hub.names():
                store = frame_hub.peek(name)
                if store is None:
                    continue
                _, view_frame = store.latest()
                views[name] = view_frame.metadata if view_frame is not None else None

            payload: dict[str, Any] = {
                "devices": devices,
                "latest_frame": frame_meta,
                "views": views,
            }
            if cv_status_provider is not None:
                try:
                    payload["server_cv"] = cv_status_provider()
                except Exception as exc:  # a broken provider must not kill /api/status
                    payload["server_cv"] = {"state": "error", "error": str(exc)}
            else:
                payload["server_cv"] = {"state": "disabled"}
            if scan_controller is not None:
                try:
                    payload["scan"] = scan_controller.status()
                except Exception as exc:
                    payload["scan"] = {"active": False, "error": str(exc)}
            self._send_json(payload)

        def _serve_objects(self) -> None:
            """Fast-poll endpoint for the live radar view — decoupled from
            /api/status so its ~5 Hz cadence doesn't grow the 1 Hz payload
            the telemetry charts and mode buttons already poll."""
            if cv_status_provider is None:
                self._send_json({"error": "server CV is not enabled"}, status=503)
                return
            try:
                status = cv_status_provider()
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
                return
            self._send_json(status.get("objects") or {"objects": []})

        def _serve_rooms(self) -> None:
            if room_store is None:
                self._send_json({"error": "mapping is not enabled"}, status=503)
                return
            self._send_json({"rooms": room_store.list_rooms()})

        def _handle_rooms_post(self) -> None:
            if room_store is None:
                self._send_json({"error": "mapping is not enabled"}, status=503)
                return
            body = self._read_json_body()
            if body is None or not str(body.get("name", "")).strip():
                self._send_json({"error": "JSON body with non-empty 'name' is required"}, status=400)
                return
            try:
                room = room_store.create_room(body)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json({"ok": True, "room": room})

        def _serve_scan_status(self) -> None:
            if scan_controller is None:
                self._send_json({"active": False, "error": "mapping is not enabled"}, status=503)
                return
            self._send_json(scan_controller.status())

        def _handle_scan_start(self) -> None:
            if scan_controller is None:
                self._send_json({"error": "mapping is not enabled"}, status=503)
                return
            body = self._read_json_body()
            if body is None:
                self._send_json({"error": "invalid JSON body"}, status=400)
                return
            from mac_server.mapping.scanner import ScanError

            try:
                result = scan_controller.start_scan(
                    room_id=str(body.get("room_id", "")),
                    direction=str(body.get("direction", "")),
                    lateral_offset_m=body.get("lateral_offset_m"),
                )
            except ScanError as exc:
                self._send_json({"error": str(exc)}, status=409)
                return
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json({"ok": True, **result})

        def _handle_scan_stop(self) -> None:
            if scan_controller is None:
                self._send_json({"error": "mapping is not enabled"}, status=503)
                return
            from mac_server.mapping.scanner import ScanError

            try:
                result = scan_controller.stop_scan()
            except ScanError as exc:
                self._send_json({"error": str(exc)}, status=409)
                return
            self._send_json({"ok": True, **result})

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

        def _serve_latest_jpeg(self, query: str = "") -> None:
            store = self._view_store(query)
            if store is None:
                self.send_error(404, "Unknown view")
                return
            _, frame = store.latest()
            if frame is None:
                self.send_error(404, "No frame received yet for this view")
                return

            self.send_response(200)
            self.send_header(
                "Content-Type", str(frame.metadata.get("content_type", "image/jpeg"))
            )
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(frame.data)))
            self.end_headers()
            self.wfile.write(frame.data)

        def _serve_stream(self, query: str = "") -> None:
            store = self._view_store(query)
            if store is None:
                self.send_error(404, "Unknown view")
                return
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()

            sequence = -1
            try:
                while True:
                    sequence, frame = store.wait_for_next(sequence)
                    if frame is None:
                        continue

                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header(
                        "Content-Type", str(frame.metadata.get("content_type", "image/jpeg"))
                    )
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
