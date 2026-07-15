"""Client-side helpers for the shared protocol."""

from __future__ import annotations

import platform
from pathlib import Path
import socket
import sys
import time

from pi_client.camera import CameraFrame
from shared.messages import Message, make_message


def make_hello_message(device_id: str) -> Message:
    return make_message(
        device_id=device_id,
        message_type="status",
        payload={"message": "hello"},
    )


def make_image_message(device_id: str, image_path: Path, byte_count: int) -> Message:
    return make_message(
        device_id=device_id,
        message_type="image",
        payload={
            "filename": image_path.name,
            "content_type": "image/jpeg",
            "byte_count": byte_count,
            "source": "test_file",
        },
    )


def make_camera_frame_message(device_id: str, frame: CameraFrame) -> Message:
    return make_message(
        device_id=device_id,
        message_type="camera_frame",
        payload={
            "frame_id": frame.frame_id,
            "width": frame.width,
            "height": frame.height,
            "format": frame.image_format,
            "content_type": frame.content_type,
            "byte_count": len(frame.data),
            "source": "picamera2",
        },
    )


def make_camera_stream_frame_message(
    device_id: str,
    frame: CameraFrame,
    session_id: str,
    frame_index: int,
    fps: float,
    jpeg_quality: int,
    save_frame: bool,
    view: str = "camera",
    mode: str | None = None,
    inference_ms: float | None = None,
    objects: list[dict[str, object]] | None = None,
) -> Message:
    payload = {
        "session_id": session_id,
        "frame_id": frame.frame_id,
        "frame_index": frame_index,
        "width": frame.width,
        "height": frame.height,
        "format": frame.image_format,
        "content_type": frame.content_type,
        "byte_count": len(frame.data),
        "source": "picamera2",
        "fps": fps,
        "jpeg_quality": jpeg_quality,
        "save_frame": save_frame,
        "view": view,
    }
    if mode is not None:
        payload["mode"] = mode
    if inference_ms is not None:
        payload["inference_ms"] = round(inference_ms, 1)
    if objects is not None:
        payload["objects"] = objects
    return make_message(
        device_id=device_id,
        message_type="camera_stream_frame",
        payload=payload,
    )


def make_cv_result_message(
    device_id: str,
    run_id: str,
    pipeline_type: str,
    byte_count: int,
    summary: dict[str, object],
) -> Message:
    return make_message(
        device_id=device_id,
        message_type="cv_result",
        payload={
            "run_id": run_id,
            "pipeline_type": pipeline_type,
            "content_type": "application/zip",
            "byte_count": byte_count,
            "summary": summary,
        },
    )


def make_session_hello_message(
    device_id: str,
    session_id: str,
    mode: str,
    capabilities: list[str],
    models: dict[str, object],
) -> Message:
    return make_message(
        device_id=device_id,
        message_type="session_hello",
        payload={
            "session_id": session_id,
            "role": "persistent",
            "mode": mode,
            "capabilities": capabilities,
            "models": models,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
    )


def make_system_telemetry_message(
    device_id: str,
    session_id: str,
    telemetry_payload: dict[str, object],
    mode: str,
    fps_actual: float | None,
) -> Message:
    payload = dict(telemetry_payload)
    payload["session_id"] = session_id
    payload["mode"] = mode
    payload["fps_actual"] = round(fps_actual, 2) if fps_actual is not None else None
    return make_message(
        device_id=device_id,
        message_type="system_telemetry",
        payload=payload,
    )


def make_command_result_message(
    device_id: str,
    session_id: str,
    command_id: str,
    ok: bool,
    mode: str,
    previous_mode: str,
    error: str | None,
    elapsed_ms: float,
) -> Message:
    return make_message(
        device_id=device_id,
        message_type="command_result",
        payload={
            "session_id": session_id,
            "command_id": command_id,
            "ok": ok,
            "mode": mode,
            "previous_mode": previous_mode,
            "error": error,
            "elapsed_ms": round(elapsed_ms, 1),
        },
    )


def make_telemetry_message(device_id: str) -> Message:
    return make_message(
        device_id=device_id,
        message_type="telemetry",
        payload={
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "monotonic_seconds": round(time.monotonic(), 3),
        },
    )
