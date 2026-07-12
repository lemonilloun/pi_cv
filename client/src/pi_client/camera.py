"""Raspberry Pi camera capture helpers."""

from __future__ import annotations

from dataclasses import dataclass
import io
import queue
import time
import uuid
from typing import Iterator


class CameraUnavailableError(RuntimeError):
    """Raised when Picamera2 is not available or capture fails."""


@dataclass(frozen=True)
class CameraFrame:
    frame_id: str
    width: int
    height: int
    image_format: str
    content_type: str
    data: bytes


@dataclass(frozen=True)
class CameraFocusOptions:
    autofocus_mode: str = "continuous"
    autofocus_range: str = "normal"
    autofocus_speed: str = "fast"
    lens_position: float | None = None


def capture_camera_frame(
    width: int,
    height: int,
    image_format: str = "jpeg",
    focus_options: CameraFocusOptions | None = None,
) -> CameraFrame:
    if image_format != "jpeg":
        raise CameraUnavailableError("Only jpeg camera capture is supported right now")

    try:
        from picamera2 import Picamera2
        from libcamera import controls
    except ImportError as exc:
        raise CameraUnavailableError(
            "Picamera2 is not available. On Raspberry Pi OS install it with: "
            "sudo apt install python3-picamera2 --no-install-recommends. "
            "If you use venv, create it with: python3 -m venv --system-site-packages .venv"
        ) from exc

    picam2 = Picamera2()
    try:
        config = picam2.create_still_configuration(main={"size": (width, height)})
        picam2.configure(config)
        _apply_focus_controls(
            picam2,
            controls,
            focus_options or CameraFocusOptions(),
        )
        picam2.start()
        time.sleep(1)

        output = io.BytesIO()
        picam2.capture_file(output, format=image_format)
        data = output.getvalue()
    except Exception as exc:
        raise CameraUnavailableError(f"Failed to capture camera frame: {exc}") from exc
    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            picam2.close()
        except Exception:
            pass

    return CameraFrame(
        frame_id=str(uuid.uuid4()),
        width=width,
        height=height,
        image_format=image_format,
        content_type="image/jpeg",
        data=data,
    )


def stream_camera_frames(
    width: int,
    height: int,
    fps: float,
    jpeg_quality: int,
    duration_seconds: float | None,
    frame_limit: int | None,
    focus_options: CameraFocusOptions,
) -> Iterator[CameraFrame]:
    try:
        from picamera2 import Picamera2
        from picamera2.encoders import JpegEncoder
        from picamera2.outputs import FileOutput
        from libcamera import controls
    except ImportError as exc:
        raise CameraUnavailableError(
            "Picamera2 streaming dependencies are not available. Install with: "
            "sudo apt install python3-picamera2 --no-install-recommends. "
            "If you use venv, create it with: python3 -m venv --system-site-packages .venv"
        ) from exc

    if fps <= 0:
        raise CameraUnavailableError("Stream FPS must be greater than zero")
    if not 1 <= jpeg_quality <= 95:
        raise CameraUnavailableError("JPEG quality must be between 1 and 95")
    if frame_limit is not None and frame_limit <= 0:
        raise CameraUnavailableError("Frame limit must be greater than zero")
    if duration_seconds is not None and duration_seconds <= 0:
        raise CameraUnavailableError("Stream duration must be greater than zero")

    frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=2)
    output = _QueuedJpegOutput(frame_queue)
    picam2 = Picamera2()

    try:
        video_config = picam2.create_video_configuration(
            main={"size": (width, height)},
            controls={"FrameRate": fps},
        )
        picam2.configure(video_config)
        _apply_focus_controls(picam2, controls, focus_options)
        picam2.start_recording(JpegEncoder(q=jpeg_quality), FileOutput(output))

        start_time = time.monotonic()
        frame_index = 0

        while True:
            if duration_seconds is not None and time.monotonic() - start_time >= duration_seconds:
                break
            if frame_limit is not None and frame_index >= frame_limit:
                break

            try:
                data = frame_queue.get(timeout=5)
            except queue.Empty as exc:
                raise CameraUnavailableError("Timed out waiting for camera stream frame") from exc

            frame_index += 1
            yield CameraFrame(
                frame_id=str(uuid.uuid4()),
                width=width,
                height=height,
                image_format="jpeg",
                content_type="image/jpeg",
                data=data,
            )
    except Exception as exc:
        if isinstance(exc, CameraUnavailableError):
            raise
        raise CameraUnavailableError(f"Failed to stream camera frames: {exc}") from exc
    finally:
        try:
            picam2.stop_recording()
        except Exception:
            pass
        try:
            picam2.close()
        except Exception:
            pass


class _QueuedJpegOutput(io.BufferedIOBase):
    def __init__(self, frame_queue: queue.Queue[bytes]) -> None:
        self._frame_queue = frame_queue

    def writable(self) -> bool:
        return True

    def write(self, buffer: bytes) -> int:
        frame = bytes(buffer)
        while True:
            try:
                self._frame_queue.put_nowait(frame)
                break
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
        return len(buffer)


def _apply_focus_controls(picam2: object, controls: object, focus_options: CameraFocusOptions) -> None:
    control_values = {}

    af_mode = _enum_value(
        getattr(controls, "AfModeEnum", None),
        {
            "manual": "Manual",
            "auto": "Auto",
            "continuous": "Continuous",
        }.get(focus_options.autofocus_mode),
    )
    if af_mode is not None:
        control_values["AfMode"] = af_mode

    af_range = _enum_value(
        getattr(controls, "AfRangeEnum", None),
        {
            "normal": "Normal",
            "macro": "Macro",
            "full": "Full",
        }.get(focus_options.autofocus_range),
    )
    if af_range is not None:
        control_values["AfRange"] = af_range

    af_speed = _enum_value(
        getattr(controls, "AfSpeedEnum", None),
        {
            "normal": "Normal",
            "fast": "Fast",
        }.get(focus_options.autofocus_speed),
    )
    if af_speed is not None:
        control_values["AfSpeed"] = af_speed

    if focus_options.lens_position is not None:
        control_values["LensPosition"] = focus_options.lens_position

    if control_values:
        picam2.set_controls(control_values)


def _enum_value(enum_class: object | None, name: str | None) -> object | None:
    if enum_class is None or name is None:
        return None
    return getattr(enum_class, name, None)
