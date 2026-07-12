"""HTTP MJPEG preview server for the latest received camera stream frame."""

from __future__ import annotations

from dataclasses import dataclass
from http import server
import logging
import socketserver
import threading
from typing import Any


logger = logging.getLogger(__name__)


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
    def __init__(self, host: str, port: int, frame_store: LatestFrameStore) -> None:
        self.host = host
        self.port = port
        self.frame_store = frame_store
        self._httpd = _ThreadingHttpServer((host, port), _make_handler(frame_store))
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


def _make_handler(frame_store: LatestFrameStore) -> type[server.BaseHTTPRequestHandler]:
    class StreamingHandler(server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._serve_index()
                return
            if self.path == "/stream.mjpg":
                self._serve_stream()
                return
            if self.path == "/latest.jpg":
                self._serve_latest_jpeg()
                return
            self.send_error(404)

        def log_message(self, format: str, *args: object) -> None:
            logger.debug("MJPEG preview: " + format, *args)

        def _serve_index(self) -> None:
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
