"""Raspberry Pi camera capture helpers."""

from __future__ import annotations

from dataclasses import dataclass
import io
import time
import uuid


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


def capture_camera_frame(width: int, height: int, image_format: str = "jpeg") -> CameraFrame:
    if image_format != "jpeg":
        raise CameraUnavailableError("Only jpeg camera capture is supported right now")

    try:
        from picamera2 import Picamera2
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
