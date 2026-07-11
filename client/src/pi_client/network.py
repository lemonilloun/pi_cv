"""TCP client for the Raspberry Pi node."""

from __future__ import annotations

import logging
import socket
from types import TracebackType

from shared.messages import Message


logger = logging.getLogger(__name__)


class ClientConnectionError(ConnectionError):
    """Raised when the TCP client cannot communicate with the server."""


class PiClient:
    def __init__(self, host: str, port: int, timeout_seconds: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self._socket: socket.socket | None = None
        self._reader = None

    def connect(self) -> None:
        if self._socket is not None:
            return

        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout_seconds)
        except OSError as exc:
            raise ClientConnectionError(f"Failed to connect to {self.host}:{self.port}") from exc

        self._socket = sock
        self._reader = sock.makefile("rb")
        logger.info("Connected to server %s:%s", self.host, self.port)

    def send_message(self, message: Message) -> None:
        if self._socket is None:
            raise ClientConnectionError("Client is not connected")

        try:
            self._socket.sendall(message.to_json_line())
        except OSError as exc:
            self.close()
            raise ClientConnectionError("Failed to send message") from exc

        logger.info("Sent message type=%s payload=%s", message.type, message.payload)

    def receive_response(self) -> Message:
        if self._reader is None:
            raise ClientConnectionError("Client is not connected")

        try:
            raw_line = self._reader.readline()
        except socket.timeout as exc:
            self.close()
            raise ClientConnectionError("Timed out waiting for server response") from exc
        except OSError as exc:
            self.close()
            raise ClientConnectionError("Failed to receive response") from exc

        if not raw_line:
            self.close()
            raise ClientConnectionError("Server closed the connection")

        response = Message.from_json_line(raw_line)
        logger.info("Received response type=%s payload=%s", response.type, response.payload)
        return response

    def request(self, message: Message) -> Message:
        self.connect()
        self.send_message(message)
        return self.receive_response()

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

        if self._socket is not None:
            self._socket.close()
            self._socket = None
            logger.info("Client connection closed")

    def __enter__(self) -> "PiClient":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
