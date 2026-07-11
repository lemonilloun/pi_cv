"""Server-side helpers for the shared protocol."""

from __future__ import annotations

from shared.messages import Message, make_message


SERVER_DEVICE_ID = "macbook_server"


def make_ack_response(request: Message) -> Message:
    return make_message(
        device_id=SERVER_DEVICE_ID,
        message_type="ack",
        payload={
            "received_type": request.type,
            "received_payload": request.payload,
        },
    )


def make_image_ack_response(request: Message, saved_path: str, byte_count: int) -> Message:
    return make_message(
        device_id=SERVER_DEVICE_ID,
        message_type="ack",
        payload={
            "received_type": request.type,
            "filename": request.payload.get("filename"),
            "bytes_received": byte_count,
            "saved_path": saved_path,
        },
    )


def make_error_response(error_message: str) -> Message:
    return make_message(
        device_id=SERVER_DEVICE_ID,
        message_type="error",
        payload={"error": error_message},
    )
