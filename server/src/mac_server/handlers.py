"""Message handlers for the MacBook server."""

from __future__ import annotations

import logging
from pathlib import Path
import time

from mac_server.protocol import make_ack_response, make_image_ack_response
from shared.messages import Message


logger = logging.getLogger(__name__)


def handle_message(message: Message, binary_payload: bytes = b"", storage_dir: Path | None = None) -> Message:
    logger.info(
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
