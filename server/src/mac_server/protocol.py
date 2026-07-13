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
            "frame_id": request.payload.get("frame_id"),
            "bytes_received": byte_count,
            "saved_path": saved_path,
        },
    )


def make_cv_result_ack_response(
    request: Message,
    saved_path: str,
    extracted_dir: str,
    byte_count: int,
) -> Message:
    return make_message(
        device_id=SERVER_DEVICE_ID,
        message_type="ack",
        payload={
            "received_type": request.type,
            "run_id": request.payload.get("run_id"),
            "pipeline_type": request.payload.get("pipeline_type"),
            "bytes_received": byte_count,
            "saved_path": saved_path,
            "extracted_dir": extracted_dir,
        },
    )


def make_command_message(
    command_id: str,
    action: str,
    mode: str | None = None,
    params: dict[str, object] | None = None,
) -> Message:
    payload: dict[str, object] = {
        "command_id": command_id,
        "action": action,
        "params": params or {},
    }
    if mode is not None:
        payload["mode"] = mode
    return make_message(
        device_id=SERVER_DEVICE_ID,
        message_type="command",
        payload=payload,
    )


def make_error_response(error_message: str) -> Message:
    return make_message(
        device_id=SERVER_DEVICE_ID,
        message_type="error",
        payload={"error": error_message},
    )
