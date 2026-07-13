"""Registry of connected persistent session clients and their telemetry."""

from __future__ import annotations

import logging
import socket
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from mac_server.protocol import make_command_message
from shared.framing import send_packet


logger = logging.getLogger(__name__)

TELEMETRY_HISTORY_LIMIT = 600


class CommandDispatchError(RuntimeError):
    """Raised when a command cannot be delivered to a session client."""


@dataclass
class ClientHandle:
    device_id: str
    session_id: str
    sock: socket.socket
    send_lock: threading.Lock
    address: tuple[str, int]
    connected_at: float = field(default_factory=time.time)
    hello_payload: dict[str, Any] = field(default_factory=dict)
    mode: str = "idle"
    last_command: dict[str, Any] | None = None
    last_command_result: dict[str, Any] | None = None

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "session_id": self.session_id,
            "address": f"{self.address[0]}:{self.address[1]}",
            "connected_at": self.connected_at,
            "mode": self.mode,
            "models": self.hello_payload.get("models", {}),
            "capabilities": self.hello_payload.get("capabilities", []),
            "hostname": self.hello_payload.get("hostname"),
            "last_command": self.last_command,
            "last_command_result": self.last_command_result,
        }


class ClientRegistry:
    """Tracks session clients (registered via session_hello) keyed by device_id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: dict[str, ClientHandle] = {}

    def register(self, handle: ClientHandle) -> None:
        with self._lock:
            previous = self._clients.get(handle.device_id)
            self._clients[handle.device_id] = handle
        if previous is not None:
            logger.info(
                "Replacing session registration for device_id=%s (old session %s)",
                handle.device_id,
                previous.session_id,
            )
        logger.info(
            "Registered session client device_id=%s session_id=%s from %s:%s",
            handle.device_id,
            handle.session_id,
            handle.address[0],
            handle.address[1],
        )

    def unregister(self, sock: socket.socket) -> None:
        with self._lock:
            for device_id, handle in list(self._clients.items()):
                if handle.sock is sock:
                    del self._clients[device_id]
                    logger.info("Unregistered session client device_id=%s", device_id)
                    return

    def find_by_socket(self, sock: socket.socket) -> ClientHandle | None:
        with self._lock:
            for handle in self._clients.values():
                if handle.sock is sock:
                    return handle
        return None

    def get(self, device_id: str) -> ClientHandle | None:
        with self._lock:
            return self._clients.get(device_id)

    def list_clients(self) -> list[ClientHandle]:
        with self._lock:
            return list(self._clients.values())

    def send_command(
        self,
        device_id: str | None,
        action: str,
        mode: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Push a command to a session client. When device_id is None the
        command goes to the only connected session (error if 0 or 2+)."""
        with self._lock:
            if device_id is None:
                if len(self._clients) == 1:
                    handle = next(iter(self._clients.values()))
                elif not self._clients:
                    raise CommandDispatchError("No session client connected")
                else:
                    raise CommandDispatchError(
                        "Multiple session clients connected; specify device_id"
                    )
            else:
                handle = self._clients.get(device_id)  # type: ignore[assignment]
                if handle is None:
                    raise CommandDispatchError(f"No session client for device_id={device_id}")

        command_id = str(uuid.uuid4())
        message = make_command_message(command_id=command_id, action=action, mode=mode, params=params)

        try:
            with handle.send_lock:
                send_packet(handle.sock, message)
        except OSError as exc:
            self.unregister(handle.sock)
            raise CommandDispatchError(f"Failed to send command: {exc}") from exc

        command_info = {
            "command_id": command_id,
            "action": action,
            "mode": mode,
            "sent_at": time.time(),
        }
        handle.last_command = command_info
        logger.info(
            "Sent command to device_id=%s action=%s mode=%s command_id=%s",
            handle.device_id,
            action,
            mode,
            command_id,
        )
        return {"device_id": handle.device_id, **command_info}


class TelemetryStore:
    """Rolling per-device telemetry history."""

    def __init__(self, history_limit: int = TELEMETRY_HISTORY_LIMIT) -> None:
        self._lock = threading.Lock()
        self._history: dict[str, deque[dict[str, Any]]] = {}
        self._history_limit = history_limit

    def add(self, device_id: str, payload: dict[str, Any]) -> None:
        entry = dict(payload)
        entry.setdefault("received_at", time.time())
        with self._lock:
            history = self._history.setdefault(device_id, deque(maxlen=self._history_limit))
            history.append(entry)

    def latest(self, device_id: str) -> dict[str, Any] | None:
        with self._lock:
            history = self._history.get(device_id)
            return dict(history[-1]) if history else None

    def recent(self, device_id: str, seconds: float) -> list[dict[str, Any]]:
        cutoff = time.time() - seconds
        with self._lock:
            history = self._history.get(device_id)
            if not history:
                return []
            return [dict(entry) for entry in history if entry.get("received_at", 0) >= cutoff]

    def device_ids(self) -> list[str]:
        with self._lock:
            return list(self._history.keys())
