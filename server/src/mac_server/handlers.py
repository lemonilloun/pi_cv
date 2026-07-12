"""Message handlers for the MacBook server."""

from __future__ import annotations

import logging
from pathlib import Path
import time
import zipfile
from io import BytesIO

from mac_server.protocol import make_ack_response, make_cv_result_ack_response, make_image_ack_response
from mac_server.preview import LatestFrameStore
from shared.messages import Message


logger = logging.getLogger(__name__)


def handle_message(
    message: Message,
    binary_payload: bytes = b"",
    storage_dir: Path | None = None,
    preview_store: LatestFrameStore | None = None,
) -> Message:
    log = logger.debug if message.type == "camera_stream_frame" else logger.info
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
