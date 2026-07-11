"""Lightweight validation for protocol message dictionaries."""

from __future__ import annotations

from typing import Any


REQUIRED_MESSAGE_FIELDS = {
    "device_id": str,
    "type": str,
    "timestamp": (int, float),
    "payload": dict,
}


def validate_message_dict(data: dict[str, Any]) -> None:
    for field_name, expected_type in REQUIRED_MESSAGE_FIELDS.items():
        if field_name not in data:
            raise ValueError(f"Missing required message field: {field_name}")
        if not isinstance(data[field_name], expected_type):
            raise ValueError(f"Invalid type for message field: {field_name}")

    if not data["device_id"]:
        raise ValueError("device_id must not be empty")

    if not data["type"]:
        raise ValueError("type must not be empty")
