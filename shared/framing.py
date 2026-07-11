"""Length-prefixed TCP framing for JSON headers and optional binary payloads."""

from __future__ import annotations

import socket
import struct

from shared.messages import Message


FRAME_HEADER_FORMAT = "!IQ"
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FORMAT)
MAX_JSON_HEADER_BYTES = 1024 * 1024


def send_packet(sock: socket.socket, message: Message, binary_payload: bytes = b"") -> None:
    json_header = message.to_json_bytes()
    if len(json_header) > MAX_JSON_HEADER_BYTES:
        raise ValueError("JSON header is too large")

    frame_header = struct.pack(FRAME_HEADER_FORMAT, len(json_header), len(binary_payload))
    sock.sendall(frame_header + json_header + binary_payload)


def receive_packet(sock: socket.socket) -> tuple[Message, bytes]:
    frame_header = _recv_exact(sock, FRAME_HEADER_SIZE)
    json_header_length, binary_payload_length = struct.unpack(FRAME_HEADER_FORMAT, frame_header)

    if json_header_length <= 0:
        raise ValueError("JSON header length must be positive")
    if json_header_length > MAX_JSON_HEADER_BYTES:
        raise ValueError("JSON header is too large")

    json_header = _recv_exact(sock, json_header_length)
    binary_payload = _recv_exact(sock, binary_payload_length) if binary_payload_length else b""
    return Message.from_json_bytes(json_header), binary_payload


def _recv_exact(sock: socket.socket, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    bytes_received = 0

    while bytes_received < byte_count:
        chunk = sock.recv(byte_count - bytes_received)
        if not chunk:
            raise EOFError("Connection closed while receiving packet")
        chunks.append(chunk)
        bytes_received += len(chunk)

    return b"".join(chunks)
