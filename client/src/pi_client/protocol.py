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


# ------------------------------------------------------------- scene3d
# Online transfer of scene-recording keyframes (docs/scene3d.md): the
# recorder streams each keyframe to the Mac as it is captured, so nothing
# accumulates on the Pi's microSD. The binary payload of a scene_keyframe
# is rgb.jpg and masks.png concatenated; the header carries both lengths.


def make_scene_session_start_message(
    device_id: str,
    scene_session: str,
    intrinsics: dict[str, object] | None,
    settings: dict[str, object],
) -> Message:
    return make_message(
        device_id=device_id,
        message_type="scene_session_start",
        payload={
            "scene_session": scene_session,
            "intrinsics": intrinsics,
            "settings": settings,
        },
    )


def make_scene_keyframe_message(
    device_id: str,
    scene_session: str,
    frame_idx: int,
    meta: dict[str, object],
    rgb_bytes: int,
    masks_bytes: int,
) -> Message:
    return make_message(
        device_id=device_id,
        message_type="scene_keyframe",
        payload={
            "scene_session": scene_session,
            "frame_idx": frame_idx,
            "meta": meta,
            "rgb_bytes": rgb_bytes,
            "masks_bytes": masks_bytes,
        },
    )


def make_scene_session_end_message(
    device_id: str,
    scene_session: str,
    session_meta: dict[str, object],
) -> Message:
    return make_message(
        device_id=device_id,
        message_type="scene_session_end",
        payload={
            "scene_session": scene_session,
            "session_meta": session_meta,
        },
    )


def make_imu_cal_state_message(device_id: str, state: dict[str, object]) -> Message:
    """Live state of the guided IMU calibration wizard (imu_calibration.py)
    so the panel can tell the operator what to do right now. Pure display
    telemetry — the measurement itself is complete on the Pi whether or not
    anyone is watching, so the server just keeps the latest one."""
    return make_message(
        device_id=device_id,
        message_type="imu_cal_state",
        payload=state,
    )


def make_time_sync_message(device_id: str, t0_pi_ns: int, seq: int = 0) -> Message:
    """Ask the server for its monotonic clock so the two can be joined.

    Carries the Pi's send-time so the server can echo it back; the Pi pairs
    that with its own receive-time to get an RTT without needing the server
    to track anything per-client. See shared/clock_sync.py for why the
    minimum-RTT sample of a burst is the one to keep.
    """
    return make_message(
        device_id=device_id,
        message_type="time_sync",
        payload={"t0_pi_ns": int(t0_pi_ns), "seq": int(seq)},
    )


# ----------------------------------------------------------------- VLA data
# Behaviour-cloning episodes. Unlike scene_keyframe, these are sampled at a
# FIXED rate: a policy learns a mapping from observation to action at a
# regular control interval, so parallax-gated selection (which is right for
# reconstruction) would hand it a time axis that stretches and compresses
# with how fast the robot happened to be moving.


def make_vla_session_start_message(
    device_id: str,
    episode_id: str,
    task: str,
    settings: dict[str, object],
    clock: dict[str, object] | None = None,
) -> Message:
    """`clock`: the measured Pi<->Mac offset (shared/clock_sync). Sent up
    front so the server can stamp frames into a single timeline even if the
    episode is cut short."""
    payload: dict[str, object] = {
        "episode_id": episode_id,
        "task": task,
        "settings": settings,
    }
    if clock is not None:
        payload["clock"] = clock
    return make_message(
        device_id=device_id, message_type="vla_session_start", payload=payload,
    )


def make_vla_frame_message(
    device_id: str,
    episode_id: str,
    frame_idx: int,
    t_pi_mono_ns: int,
    jpeg_bytes: int,
    imu: dict[str, object] | None = None,
) -> Message:
    return make_message(
        device_id=device_id,
        message_type="vla_frame",
        payload={
            "episode_id": episode_id,
            "frame_idx": frame_idx,
            "t_pi_mono_ns": int(t_pi_mono_ns),
            "jpeg_bytes": jpeg_bytes,
            "imu": imu,
        },
    )


def make_vla_session_end_message(
    device_id: str, episode_id: str, episode_meta: dict[str, object]
) -> Message:
    return make_message(
        device_id=device_id,
        message_type="vla_session_end",
        payload={"episode_id": episode_id, "episode_meta": episode_meta},
    )
