"""Frame sources for the persistent session client.

Each source owns its camera resources between start() and stop(); only the
session worker thread ever touches a source, so mode switches are race-free.
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pi_client.camera import CameraFocusOptions, CameraUnavailableError, _QueuedJpegOutput


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureSettings:
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    jpeg_quality: int = 85
    focus_options: CameraFocusOptions = CameraFocusOptions()


class FrameSourceError(RuntimeError):
    """Raised when a frame source cannot start or produce frames."""


class PicameraStreamSource:
    """Full-rate MJPEG stream: hardware-paced JPEG frames via JpegEncoder."""

    def __init__(self, settings: CaptureSettings) -> None:
        self.settings = settings
        self._picam2: Any = None
        self._frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=2)

    def start(self) -> None:
        try:
            from picamera2 import Picamera2
            from picamera2.encoders import JpegEncoder
            from picamera2.outputs import FileOutput
            from libcamera import controls
        except ImportError as exc:
            raise FrameSourceError(
                "Picamera2 streaming dependencies are not available. Install with: "
                "sudo apt install python3-picamera2 --no-install-recommends"
            ) from exc

        from pi_client.camera import _apply_focus_controls

        picam2 = Picamera2()
        try:
            video_config = picam2.create_video_configuration(
                main={"size": (self.settings.width, self.settings.height)},
                controls={"FrameRate": self.settings.fps},
            )
            picam2.configure(video_config)
            _apply_focus_controls(picam2, controls, self.settings.focus_options)
            picam2.start_recording(
                JpegEncoder(q=self.settings.jpeg_quality),
                FileOutput(_QueuedJpegOutput(self._frame_queue)),
            )
        except Exception as exc:
            try:
                picam2.close()
            except Exception:
                pass
            raise FrameSourceError(f"Failed to start camera stream: {exc}") from exc
        self._picam2 = picam2

    def stop(self) -> None:
        if self._picam2 is None:
            return
        try:
            self._picam2.stop_recording()
        except Exception:
            pass
        try:
            self._picam2.close()
        except Exception:
            pass
        self._picam2 = None

    def next_stream_jpeg(self, timeout: float = 5.0) -> bytes:
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise FrameSourceError("Timed out waiting for camera stream frame") from exc

    def capture_bgr(self) -> Any:
        raise FrameSourceError("PicameraStreamSource does not support array capture")


class PicameraCaptureSource:
    """Low-rate BGR array capture for CV modes.

    Uses a video configuration (no per-frame still warmup) with an RGB888 main
    stream, which Picamera2 delivers in BGR memory order — matching cv2.
    """

    def __init__(self, settings: CaptureSettings) -> None:
        self.settings = settings
        self._picam2: Any = None

    def start(self) -> None:
        try:
            from picamera2 import Picamera2
            from libcamera import controls
        except ImportError as exc:
            raise FrameSourceError(
                "Picamera2 is not available. Install with: "
                "sudo apt install python3-picamera2 --no-install-recommends"
            ) from exc

        from pi_client.camera import _apply_focus_controls

        picam2 = Picamera2()
        try:
            video_config = picam2.create_video_configuration(
                main={"size": (self.settings.width, self.settings.height), "format": "RGB888"},
            )
            picam2.configure(video_config)
            _apply_focus_controls(picam2, controls, self.settings.focus_options)
            picam2.start()
            time.sleep(0.5)
        except Exception as exc:
            try:
                picam2.close()
            except Exception:
                pass
            raise FrameSourceError(f"Failed to start camera capture: {exc}") from exc
        self._picam2 = picam2

    def stop(self) -> None:
        if self._picam2 is None:
            return
        try:
            self._picam2.stop()
        except Exception:
            pass
        try:
            self._picam2.close()
        except Exception:
            pass
        self._picam2 = None

    def next_stream_jpeg(self, timeout: float = 5.0) -> bytes:
        raise FrameSourceError("PicameraCaptureSource does not support JPEG streaming")

    def capture_bgr(self) -> Any:
        if self._picam2 is None:
            raise FrameSourceError("Camera capture source is not started")
        try:
            import numpy as np

            array = self._picam2.capture_array("main")
            # Picamera2 raw captures are frequently a strided view (row
            # padding) and may carry a stray alpha channel; downstream
            # NCNN inference needs a plain contiguous HxWx3 uint8 buffer,
            # same as what cv2.imdecode always produced on the old path.
            if array.ndim == 3 and array.shape[2] > 3:
                array = array[:, :, :3]
            return np.ascontiguousarray(array)
        except FrameSourceError:
            raise
        except Exception as exc:
            raise FrameSourceError(f"Failed to capture camera frame: {exc}") from exc


class SyntheticSource:
    """Loops a static image; used for loopback testing without a camera."""

    def __init__(self, settings: CaptureSettings, image_path: Path) -> None:
        self.settings = settings
        self.image_path = image_path
        self._jpeg_bytes: bytes | None = None

    def start(self) -> None:
        if not self.image_path.exists():
            raise FrameSourceError(f"Synthetic source image not found: {self.image_path}")
        self._jpeg_bytes = self.image_path.read_bytes()

    def stop(self) -> None:
        self._jpeg_bytes = None

    def next_stream_jpeg(self, timeout: float = 5.0) -> bytes:
        if self._jpeg_bytes is None:
            raise FrameSourceError("Synthetic source is not started")
        time.sleep(1.0 / max(self.settings.fps, 0.1))
        return self._jpeg_bytes

    def capture_bgr(self) -> Any:
        if self._jpeg_bytes is None:
            raise FrameSourceError("Synthetic source is not started")
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise FrameSourceError("NumPy and OpenCV are required for synthetic capture") from exc

        data = np.frombuffer(self._jpeg_bytes, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise FrameSourceError("Failed to decode synthetic source image")
        return image


def make_stream_source(source_kind: str, settings: CaptureSettings, image_path: Path) -> Any:
    if source_kind == "camera":
        return PicameraStreamSource(settings)
    if source_kind == "synthetic":
        return SyntheticSource(settings, image_path)
    raise FrameSourceError(f"Unsupported source: {source_kind}")


def make_capture_source(source_kind: str, settings: CaptureSettings, image_path: Path) -> Any:
    if source_kind == "camera":
        return PicameraCaptureSource(settings)
    if source_kind == "synthetic":
        return SyntheticSource(settings, image_path)
    raise FrameSourceError(f"Unsupported source: {source_kind}")
