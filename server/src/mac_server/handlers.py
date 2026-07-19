"""Message handlers for the MacBook server."""

from __future__ import annotations

import logging
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
import time
import zipfile
from io import BytesIO

from mac_server.protocol import make_ack_response, make_cv_result_ack_response, make_image_ack_response
from mac_server.preview import LatestFrameStore
from mac_server.registry import ClientHandle, ClientRegistry, TelemetryStore
from shared.messages import Message


logger = logging.getLogger(__name__)

QUIET_MESSAGE_TYPES = {"camera_stream_frame", "system_telemetry"}


@dataclass
class SessionContext:
    """Per-connection context so handlers can register session clients."""

    registry: ClientRegistry
    telemetry_store: TelemetryStore
    client_socket: socket.socket
    client_address: tuple[str, int]
    send_lock: threading.Lock


def handle_message(
    message: Message,
    binary_payload: bytes = b"",
    storage_dir: Path | None = None,
    preview_store: LatestFrameStore | None = None,
    session_context: SessionContext | None = None,
) -> Message:
    log = logger.debug if message.type in QUIET_MESSAGE_TYPES else logger.info
    log(
        "Handling message from device_id=%s type=%s payload=%s binary_payload_bytes=%s",
        message.device_id,
        message.type,
        message.payload,
        len(binary_payload),
    )

    if message.type == "image":
        if storage_dir is None:
            raise ValueError("storage_dir is required for image messages")
        return _handle_image_message(message, binary_payload, storage_dir)

    if message.type == "camera_frame":
        if storage_dir is None:
            raise ValueError("storage_dir is required for camera frame messages")
        return _handle_camera_frame_message(message, binary_payload, storage_dir)

    if message.type == "camera_stream_frame":
        return _handle_camera_stream_frame_message(message, binary_payload, storage_dir, preview_store)

    if message.type == "cv_result":
        if storage_dir is None:
            raise ValueError("storage_dir is required for CV result messages")
        return _handle_cv_result_message(message, binary_payload, storage_dir)

    if message.type == "session_hello" and session_context is not None:
        return _handle_session_hello_message(message, session_context)

    if message.type == "system_telemetry" and session_context is not None:
        return _handle_system_telemetry_message(message, session_context)

    if message.type == "command_result" and session_context is not None:
        return _handle_command_result_message(message, session_context)

    if message.type in {"scene_session_start", "scene_keyframe", "scene_session_end"}:
        if storage_dir is None:
            raise ValueError("storage_dir is required for scene messages")
        return _handle_scene_message(message, binary_payload, storage_dir)

    return make_ack_response(message)


# ------------------------------------------------------------- scene3d
# The scene recorder streams keyframes online (docs/scene3d.md) so nothing
# accumulates on the Pi's microSD; the server materializes the exact same
# session directory layout the offline/rsync path would produce.


def _scene_session_dir(storage_dir: Path, scene_session: str) -> Path:
    safe = "".join(c for c in scene_session if c.isalnum() or c == "_")
    if not safe:
        raise ValueError(f"Invalid scene session name: {scene_session!r}")
    # storage_dir is data/received; sessions live next to it in data/.
    return storage_dir.parent / "scene_sessions" / safe


def _handle_scene_message(message: Message, binary_payload: bytes, storage_dir: Path) -> Message:
    import json

    payload = message.payload
    session_dir = _scene_session_dir(storage_dir, str(payload.get("scene_session", "")))

    if message.type == "scene_session_start":
        (session_dir / "keyframes").mkdir(parents=True, exist_ok=True)
        intrinsics = payload.get("intrinsics")
        if intrinsics:
            (session_dir / "intrinsics.json").write_text(
                json.dumps(intrinsics, indent=2), encoding="utf-8"
            )
        logger.info("Scene session started: %s (intrinsics: %s)",
                    session_dir.name, "yes" if intrinsics else "no")
        return make_ack_response(message)

    if message.type == "scene_keyframe":
        rgb_bytes = int(payload.get("rgb_bytes", 0))
        masks_bytes = int(payload.get("masks_bytes", 0))
        if rgb_bytes <= 0 or rgb_bytes + masks_bytes != len(binary_payload):
            raise ValueError(
                f"scene_keyframe payload mismatch: rgb={rgb_bytes} masks={masks_bytes} "
                f"actual={len(binary_payload)}"
            )
        frame_idx = int(payload.get("frame_idx", 0))
        kf_dir = session_dir / "keyframes" / f"{frame_idx:06d}"
        kf_dir.mkdir(parents=True, exist_ok=True)
        (kf_dir / "rgb.jpg").write_bytes(binary_payload[:rgb_bytes])
        if masks_bytes > 0:
            (kf_dir / "masks.png").write_bytes(binary_payload[rgb_bytes:])
        (kf_dir / "meta.json").write_text(
            json.dumps(payload.get("meta", {}), indent=1), encoding="utf-8"
        )
        return make_ack_response(message)

    # scene_session_end
    (session_dir).mkdir(parents=True, exist_ok=True)
    (session_dir / "session_meta.json").write_text(
        json.dumps(payload.get("session_meta", {}), indent=2), encoding="utf-8"
    )
    logger.info("Scene session finished: %s", session_dir.name)
    return make_ack_response(message)


def _handle_session_hello_message(message: Message, context: SessionContext) -> Message:
    handle = ClientHandle(
        device_id=message.device_id,
        session_id=str(message.payload.get("session_id", "")),
        sock=context.client_socket,
        send_lock=context.send_lock,
        address=context.client_address,
        hello_payload=dict(message.payload),
        mode=str(message.payload.get("mode", "idle")),
    )
    context.registry.register(handle)
    return make_ack_response(message)


def _handle_system_telemetry_message(message: Message, context: SessionContext) -> Message:
    context.telemetry_store.add(message.device_id, message.payload)
    handle = context.registry.find_by_socket(context.client_socket)
    if handle is not None and message.payload.get("mode"):
        handle.mode = str(message.payload["mode"])
    return make_ack_response(message)


def _handle_command_result_message(message: Message, context: SessionContext) -> Message:
    handle = context.registry.find_by_socket(context.client_socket)
    if handle is not None:
        handle.last_command_result = dict(message.payload)
        if message.payload.get("ok") and message.payload.get("mode"):
            handle.mode = str(message.payload["mode"])
    logger.info(
        "Command result from device_id=%s command_id=%s ok=%s mode=%s error=%s",
        message.device_id,
        message.payload.get("command_id"),
        message.payload.get("ok"),
        message.payload.get("mode"),
        message.payload.get("error"),
    )
    return make_ack_response(message)


def _handle_image_message(message: Message, binary_payload: bytes, storage_dir: Path) -> Message:
    expected_byte_count = int(message.payload.get("byte_count", -1))
    actual_byte_count = len(binary_payload)

    if expected_byte_count != actual_byte_count:
        raise ValueError(
            f"Image byte count mismatch: expected {expected_byte_count}, got {actual_byte_count}"
        )

    original_filename = Path(str(message.payload.get("filename", "image.jpg"))).name
    timestamp_ms = int(time.time() * 1000)
    saved_path = storage_dir / f"{timestamp_ms}_{original_filename}"

    storage_dir.mkdir(parents=True, exist_ok=True)
    saved_path.write_bytes(binary_payload)

    logger.info("Saved image to %s", saved_path)
    return make_image_ack_response(message, str(saved_path), actual_byte_count)


def _handle_camera_frame_message(message: Message, binary_payload: bytes, storage_dir: Path) -> Message:
    expected_byte_count = int(message.payload.get("byte_count", -1))
    actual_byte_count = len(binary_payload)

    if expected_byte_count != actual_byte_count:
        raise ValueError(
            f"Camera frame byte count mismatch: expected {expected_byte_count}, got {actual_byte_count}"
        )

    saved_path = _save_camera_frame(message, binary_payload, storage_dir / "camera")
    logger.info("Saved camera frame to %s", saved_path)
    return make_image_ack_response(message, str(saved_path), actual_byte_count)


def _handle_camera_stream_frame_message(
    message: Message,
    binary_payload: bytes,
    storage_dir: Path | None,
    preview_store: LatestFrameStore | None,
) -> Message:
    expected_byte_count = int(message.payload.get("byte_count", -1))
    actual_byte_count = len(binary_payload)

    if expected_byte_count != actual_byte_count:
        raise ValueError(
            f"Camera stream frame byte count mismatch: expected {expected_byte_count}, got {actual_byte_count}"
        )

    if preview_store is not None:
        preview_store.update(binary_payload, message.payload)

    saved_path = ""
    if bool(message.payload.get("save_frame")):
        if storage_dir is None:
            raise ValueError("storage_dir is required to save stream frames")
        saved_path = str(_save_camera_frame(message, binary_payload, storage_dir / "stream"))

    frame_index = int(message.payload.get("frame_index", 0))
    log = logger.info if frame_index == 1 or frame_index % 30 == 0 else logger.debug
    log(
        "Received stream frame session=%s index=%s bytes=%s saved=%s",
        message.payload.get("session_id"),
        frame_index,
        actual_byte_count,
        bool(saved_path),
    )
    return make_image_ack_response(message, saved_path, actual_byte_count)


def _save_camera_frame(message: Message, binary_payload: bytes, target_dir: Path) -> Path:
    frame_id = _safe_filename_part(str(message.payload.get("frame_id", "frame")))
    device_id = _safe_filename_part(message.device_id)
    image_format = str(message.payload.get("format", "jpeg"))
    extension = "jpg" if image_format == "jpeg" else image_format
    timestamp_ms = int(time.time() * 1000)

    saved_path = target_dir / f"{timestamp_ms}_{device_id}_{frame_id}.{extension}"
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_path.write_bytes(binary_payload)
    return saved_path


def _handle_cv_result_message(message: Message, binary_payload: bytes, storage_dir: Path) -> Message:
    expected_byte_count = int(message.payload.get("byte_count", -1))
    actual_byte_count = len(binary_payload)

    if expected_byte_count != actual_byte_count:
        raise ValueError(f"CV result byte count mismatch: expected {expected_byte_count}, got {actual_byte_count}")

    run_id = _safe_filename_part(str(message.payload.get("run_id", "cv_run")))
    target_dir = storage_dir / "cv" / run_id
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_path = target_dir / "result.zip"
    zip_path.write_bytes(binary_payload)

    try:
        with zipfile.ZipFile(BytesIO(binary_payload)) as archive:
            _safe_extract_zip(archive, target_dir)
    except zipfile.BadZipFile as exc:
        raise ValueError("CV result payload is not a valid zip file") from exc

    logger.info(
        "Saved CV result run_id=%s pipeline=%s bytes=%s dir=%s",
        run_id,
        message.payload.get("pipeline_type"),
        actual_byte_count,
        target_dir,
    )
    return make_cv_result_ack_response(
        request=message,
        saved_path=str(zip_path),
        extracted_dir=str(target_dir),
        byte_count=actual_byte_count,
    )


def _safe_extract_zip(archive: zipfile.ZipFile, target_dir: Path) -> None:
    resolved_target = target_dir.resolve()
    for member in archive.infolist():
        member_path = target_dir / member.filename
        resolved_member = member_path.resolve()
        if resolved_target != resolved_member and resolved_target not in resolved_member.parents:
            raise ValueError(f"Unsafe zip member path: {member.filename}")
    archive.extractall(target_dir)


def _safe_filename_part(value: str) -> str:
    safe_chars = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe_value = "".join(safe_chars).strip("_")
    return safe_value or "unknown"
