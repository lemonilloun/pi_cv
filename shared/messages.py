"""Common JSON message format for Raspberry Pi and MacBook communication."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any

from shared.schemas import validate_message_dict


@dataclass(frozen=True)
class Message:
    """Protocol message sent over TCP as newline-delimited JSON."""

    device_id: str
    type: str
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "type": self.type,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }

    def to_json_line(self) -> bytes:
        return (json.dumps(self.to_dict(), separators=(",", ":")) + "\n").encode("utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        validate_message_dict(data)
        return cls(
            device_id=data["device_id"],
            type=data["type"],
            timestamp=float(data["timestamp"]),
            payload=data["payload"],
        )

    @classmethod
    def from_json_line(cls, raw_line: bytes) -> "Message":
        try:
            data = json.loads(raw_line.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError("Message is not valid UTF-8") from exc
        except json.JSONDecodeError as exc:
            raise ValueError("Message is not valid JSON") from exc

        if not isinstance(data, dict):
            raise ValueError("Message JSON must be an object")

        return cls.from_dict(data)


def make_message(device_id: str, message_type: str, payload: dict[str, Any] | None = None) -> Message:
    return Message(device_id=device_id, type=message_type, payload=payload or {})
