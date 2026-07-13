"""Persistent session runtime for the Raspberry Pi client.

One long-lived TCP connection to the MacBook server. The server pushes
`command` messages (mode switches) over the same socket; the client streams
frames, 1 Hz system telemetry, and command results back.

Threading model per connection:
- main/worker thread: mode state machine, camera and model ownership
- receiver thread: sole socket reader, dispatches acks vs commands
- telemetry thread: samples and sends system telemetry at ~1 Hz

Backpressure: every outbound message consumes one in-flight slot released
when its ack arrives; frame production drops frames instead of queueing.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pi_client.camera import CameraFrame
from pi_client.camera_session import (
    CaptureSettings,
    FrameSourceError,
    make_capture_source,
    make_stream_source,
)
from pi_client.network import ClientConnectionError, PiClient
from pi_client.protocol import (
    make_camera_stream_frame_message,
    make_command_result_message,
    make_session_hello_message,
    make_system_telemetry_message,
)
from pi_client.telemetry import SystemTelemetrySampler


logger = logging.getLogger(__name__)

SESSION_MODES = ("idle", "stream", "depth", "yolo", "pipeline")
IN_FLIGHT_SLOTS = 4
CV_MODES = {"depth", "yolo", "pipeline"}


@dataclass
class SessionSettings:
    host: str
    port: int
    device_id: str
    source_kind: str = "camera"
    synthetic_image: Path = Path("data/cat.jpg")
    stream: CaptureSettings = field(default_factory=CaptureSettings)
    cv_capture: CaptureSettings = field(default_factory=lambda: CaptureSettings(fps=1.0))
    cv_fps: float = 1.0
    cv_jpeg_quality: int = 85
    telemetry_hz: float = 1.0
    initial_mode: str = "idle"
    connect_timeout_seconds: float = 5.0
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    yolo_model: str | None = None
    yolo_task: str = "detect"
    yolo_confidence: float = 0.5
    depth_backend: str = "depth-anything-v2"
    depth_model_path: str | None = None
    depth_encoder: str = "vits"
    depth_input_size: int = 392
    depth_is_metric: bool = False
    torch_threads: int = 3


class SessionRuntime:
    def __init__(self, settings: SessionSettings) -> None:
        self.settings = settings
        self.session_id = str(uuid.uuid4())
        self._stop_event = threading.Event()
        self._disconnected = threading.Event()
        self._command_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._client: PiClient | None = None
        self._in_flight = threading.Semaphore(IN_FLIGHT_SLOTS)
        self._send_timestamps: deque[float] = deque(maxlen=120)
        self._mode = "idle"
        self._requested_mode = settings.initial_mode
        self._source: Any = None
        self._warm_yolo: Any = None
        self._warm_depth: Any = None
        self._frame_index = 0

    def stop(self) -> None:
        self._stop_event.set()
        self._disconnected.set()

    # ---------------------------------------------------------------- run

    def run(self) -> int:
        backoff = self.settings.reconnect_min_seconds
        while not self._stop_event.is_set():
            client = PiClient(
                host=self.settings.host,
                port=self.settings.port,
                timeout_seconds=self.settings.connect_timeout_seconds,
            )
            try:
                client.connect()
                client.set_socket_timeout(None)
                self._client = client
                self._disconnected.clear()
                self._in_flight = threading.Semaphore(IN_FLIGHT_SLOTS)
                self._send_hello()
            except ClientConnectionError as exc:
                logger.warning("Connect failed: %s (retry in %.1fs)", exc, backoff)
                self._sleep(backoff)
                backoff = min(backoff * 2, self.settings.reconnect_max_seconds)
                continue

            backoff = self.settings.reconnect_min_seconds
            receiver = threading.Thread(target=self._receiver_loop, daemon=True, name="receiver")
            telemetry = threading.Thread(target=self._telemetry_loop, daemon=True, name="telemetry")
            receiver.start()
            telemetry.start()

            try:
                self._worker_loop()
            finally:
                self._disconnected.set()
                self._teardown_source()
                # The source is gone; drop to idle so the requested mode is
                # fully re-applied (models stay warm) after reconnecting.
                self._mode = "idle"
                client.close()
                receiver.join(timeout=2)
                telemetry.join(timeout=2)
                self._client = None

            if not self._stop_event.is_set():
                logger.info("Connection lost, reconnecting in %.1fs", backoff)
                self._sleep(backoff)
        return 0

    def _sleep(self, seconds: float) -> None:
        self._stop_event.wait(timeout=seconds)

    # ------------------------------------------------------------- threads

    def _send_hello(self) -> None:
        assert self._client is not None
        models = {
            "yolo_model": self.settings.yolo_model,
            "yolo_task": self.settings.yolo_task,
            "depth_backend": self.settings.depth_backend,
            "depth_model_path": self.settings.depth_model_path,
            "depth_encoder": self.settings.depth_encoder,
            "depth_input_size": self.settings.depth_input_size,
        }
        message = make_session_hello_message(
            device_id=self.settings.device_id,
            session_id=self.session_id,
            mode=self._requested_mode,
            capabilities=list(SESSION_MODES),
            models=models,
        )
        self._in_flight.acquire()
        self._client.send_message(message)
        logger.info("Session hello sent (session_id=%s)", self.session_id)

    def _receiver_loop(self) -> None:
        client = self._client
        assert client is not None
        while not self._disconnected.is_set():
            try:
                message, _ = client.receive_response()
            except ClientConnectionError:
                self._disconnected.set()
                return

            if message.type == "command":
                self._command_queue.put(dict(message.payload))
            elif message.type in {"ack", "error"}:
                if message.type == "error":
                    logger.warning("Server error response: %s", message.payload.get("error"))
                self._in_flight.release()
            else:
                logger.debug("Ignoring unexpected message type=%s", message.type)

    def _telemetry_loop(self) -> None:
        sampler = SystemTelemetrySampler()
        interval = 1.0 / max(self.settings.telemetry_hz, 0.1)
        while not self._disconnected.is_set():
            self._disconnected.wait(timeout=interval)
            if self._disconnected.is_set():
                return

            sample = sampler.sample()
            message = make_system_telemetry_message(
                device_id=self.settings.device_id,
                session_id=self.session_id,
                telemetry_payload=sample.to_payload(),
                mode=self._mode,
                fps_actual=self._fps_actual(),
            )
            self._send_with_slot(message, b"", slot_timeout=0.2, drop_label="telemetry")

    def _fps_actual(self) -> float | None:
        now = time.monotonic()
        recent = [ts for ts in self._send_timestamps if now - ts <= 5.0]
        if len(recent) < 2:
            return None
        span = recent[-1] - recent[0]
        if span <= 0:
            return None
        return (len(recent) - 1) / span

    # -------------------------------------------------------------- worker

    def _worker_loop(self) -> None:
        # Apply the initial mode as if it were commanded (without a result).
        if self._requested_mode != "idle":
            self._switch_mode(self._requested_mode, command_id=None)

        next_cv_tick = time.monotonic()
        while not self._stop_event.is_set() and not self._disconnected.is_set():
            self._drain_commands()

            try:
                if self._mode == "stream":
                    self._produce_stream_frame()
                elif self._mode in CV_MODES:
                    now = time.monotonic()
                    if now < next_cv_tick:
                        self._disconnected.wait(timeout=min(0.1, next_cv_tick - now))
                        continue
                    next_cv_tick = max(next_cv_tick + 1.0 / max(self.settings.cv_fps, 0.05), now)
                    self._produce_cv_frame()
                else:
                    self._disconnected.wait(timeout=0.2)
            except ClientConnectionError:
                self._disconnected.set()
                return
            except Exception as exc:
                # A single bad frame (camera glitch, model postprocessing
                # error) must not take down the whole session process.
                logger.error("Frame production failed in mode=%s: %s; falling back to idle", self._mode, exc)
                self._switch_mode("idle", command_id=None)

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._command_queue.get_nowait()
            except queue.Empty:
                return
            self._handle_command(command)

    def _handle_command(self, command: dict[str, Any]) -> None:
        command_id = str(command.get("command_id", ""))
        action = str(command.get("action", ""))
        if action == "set_mode":
            mode = str(command.get("mode", "")).strip().lower()
            if mode not in SESSION_MODES:
                self._send_command_result(command_id, ok=False, previous_mode=self._mode, error=f"unknown mode: {mode}", elapsed_ms=0.0)
                return
            self._switch_mode(mode, command_id=command_id)
        elif action == "ping":
            self._send_command_result(command_id, ok=True, previous_mode=self._mode, error=None, elapsed_ms=0.0)
        else:
            self._send_command_result(command_id, ok=False, previous_mode=self._mode, error=f"unknown action: {action}", elapsed_ms=0.0)

    def _switch_mode(self, new_mode: str, command_id: str | None) -> None:
        previous_mode = self._mode
        started_at = time.perf_counter()

        if new_mode == previous_mode:
            if command_id is not None:
                self._send_command_result(command_id, ok=True, previous_mode=previous_mode, error=None, elapsed_ms=0.0)
            return

        logger.info("Switching mode: %s -> %s", previous_mode, new_mode)
        self._teardown_source()

        try:
            if new_mode in {"yolo", "pipeline"}:
                self._ensure_warm_yolo()
            if new_mode in {"depth", "pipeline"}:
                self._ensure_warm_depth()
            self._setup_source(new_mode)
            self._mode = new_mode
            self._requested_mode = new_mode
            error = None
            ok = True
        except Exception as exc:  # model/camera failures must not kill the session
            logger.error("Mode switch to %s failed: %s", new_mode, exc)
            self._teardown_source()
            self._mode = "idle"
            error = str(exc)
            ok = False

        if command_id is not None:
            self._send_command_result(
                command_id,
                ok=ok,
                previous_mode=previous_mode,
                error=error,
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
            )

    def _send_command_result(
        self,
        command_id: str,
        ok: bool,
        previous_mode: str,
        error: str | None,
        elapsed_ms: float,
    ) -> None:
        message = make_command_result_message(
            device_id=self.settings.device_id,
            session_id=self.session_id,
            command_id=command_id,
            ok=ok,
            mode=self._mode,
            previous_mode=previous_mode,
            error=error,
            elapsed_ms=elapsed_ms,
        )
        self._send_with_slot(message, b"", slot_timeout=2.0, drop_label="command_result")

    def _setup_source(self, mode: str) -> None:
        if mode == "stream":
            self._source = make_stream_source(
                self.settings.source_kind, self.settings.stream, self.settings.synthetic_image
            )
            self._source.start()
        elif mode in CV_MODES:
            self._source = make_capture_source(
                self.settings.source_kind, self.settings.cv_capture, self.settings.synthetic_image
            )
            self._source.start()
        else:
            self._source = None

    def _teardown_source(self) -> None:
        if self._source is not None:
            try:
                self._source.stop()
            except Exception:
                pass
            self._source = None

    def _ensure_warm_yolo(self) -> None:
        if self._warm_yolo is not None:
            return
        if not self.settings.yolo_model:
            raise RuntimeError("yolo_model is not configured (set session.yolo_model or --yolo-model)")
        from pi_client.cv_models import WarmYolo

        logger.info("Loading YOLO model %s (task=%s)...", self.settings.yolo_model, self.settings.yolo_task)
        self._warm_yolo = WarmYolo(model_path=self.settings.yolo_model, task=self.settings.yolo_task)
        logger.info("YOLO model ready")

    def _ensure_warm_depth(self) -> None:
        if self._warm_depth is not None:
            return
        from pi_client.cv_models import make_warm_depth_model

        logger.info("Loading depth model backend=%s...", self.settings.depth_backend)
        self._warm_depth = make_warm_depth_model(
            backend=self.settings.depth_backend,
            model_path=self.settings.depth_model_path,
            encoder=self.settings.depth_encoder,
            input_size=self.settings.depth_input_size,
            is_metric=self.settings.depth_is_metric,
            torch_threads=self.settings.torch_threads,
        )
        logger.info("Depth model ready")

    # -------------------------------------------------------------- frames

    def _produce_stream_frame(self) -> None:
        data = self._source.next_stream_jpeg(timeout=5.0)
        self._send_frame(
            data,
            width=self.settings.stream.width,
            height=self.settings.stream.height,
            fps=self.settings.stream.fps,
            jpeg_quality=self.settings.stream.jpeg_quality,
            view="camera",
            inference_ms=None,
        )

    def _produce_cv_frame(self) -> None:
        image_bgr = self._source.capture_bgr()
        height, width = image_bgr.shape[:2]
        inference_ms = 0.0

        if self._mode == "depth":
            result = self._warm_depth.infer_bgr(image_bgr)
            data = result.heatmap_image
            view = "depth"
            inference_ms = result.inference_ms
        elif self._mode == "yolo":
            result = self._warm_yolo.infer_bgr(image_bgr, self.settings.yolo_confidence)
            data = result.annotated_image
            view = "yolo"
            inference_ms = result.inference_ms
        elif self._mode == "pipeline":
            from pi_client.cv_models import attach_depth_to_detections, render_combined_image

            yolo_result = self._warm_yolo.infer_bgr(image_bgr, self.settings.yolo_confidence)
            depth_result = self._warm_depth.infer_bgr(image_bgr)
            detections = attach_depth_to_detections(
                detections=yolo_result.detections,
                depth_map=depth_result.depth_map,
                image_width=width,
                image_height=height,
                is_metric=self.settings.depth_is_metric,
            )
            data = render_combined_image(_encode_jpeg(image_bgr, self.settings.cv_jpeg_quality), detections)
            view = "combined"
            inference_ms = yolo_result.inference_ms + depth_result.inference_ms
        else:
            return

        self._send_frame(
            data,
            width=width,
            height=height,
            fps=self.settings.cv_fps,
            jpeg_quality=self.settings.cv_jpeg_quality,
            view=view,
            inference_ms=inference_ms,
        )

    def _send_frame(
        self,
        data: bytes,
        width: int,
        height: int,
        fps: float,
        jpeg_quality: int,
        view: str,
        inference_ms: float | None,
    ) -> None:
        self._frame_index += 1
        frame = CameraFrame(
            frame_id=str(uuid.uuid4()),
            width=width,
            height=height,
            image_format="jpeg",
            content_type="image/jpeg",
            data=data,
        )
        message = make_camera_stream_frame_message(
            device_id=self.settings.device_id,
            frame=frame,
            session_id=self.session_id,
            frame_index=self._frame_index,
            fps=fps,
            jpeg_quality=jpeg_quality,
            save_frame=False,
            view=view,
            mode=self._mode,
            inference_ms=inference_ms,
        )
        sent = self._send_with_slot(message, data, slot_timeout=0.5, drop_label="frame")
        if sent:
            self._send_timestamps.append(time.monotonic())

    def _send_with_slot(self, message: Any, binary_payload: bytes, slot_timeout: float, drop_label: str) -> bool:
        client = self._client
        if client is None or self._disconnected.is_set():
            return False
        if not self._in_flight.acquire(timeout=slot_timeout):
            logger.debug("Dropping %s: in-flight window full", drop_label)
            return False
        try:
            client.send_message(message, binary_payload)
            return True
        except ClientConnectionError:
            self._in_flight.release()
            self._disconnected.set()
            return False


def _encode_jpeg(image_bgr: Any, quality: int) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise FrameSourceError("Failed to encode JPEG")
    return encoded.tobytes()
