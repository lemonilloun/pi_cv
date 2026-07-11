# Architecture

## Goal

The first project stage implements a minimal communication layer between:

- Raspberry Pi 5 as an edge sensor/client node.
- MacBook as a processing/server node.

The protocol is intentionally small and based only on Python standard libraries.

## Components

### `shared/`

Contains the common message protocol used by both sides.

- `messages.py` defines the `Message` dataclass and JSON serialization.
- `schemas.py` performs lightweight runtime validation.

Messages are sent over TCP as newline-delimited JSON. Each message is one JSON object followed by `\n`.

Example:

```json
{
  "device_id": "raspberry_pi_01",
  "type": "status",
  "timestamp": 123456789,
  "payload": {
    "message": "hello"
  }
}
```

### `client/`

Contains Raspberry Pi client code.

- `network.py` owns socket connection, send, receive, and disconnect behavior.
- `protocol.py` creates client messages.
- `client.py` is the command-line entry point.

### `server/`

Contains MacBook server code.

- `server.py` starts the TCP server and manages client sockets.
- `handlers.py` processes valid messages and returns responses.
- `protocol.py` creates server responses.

## Current Flow

1. Server opens a TCP socket and listens on the configured host and port.
2. Client connects to the server.
3. Client sends a JSON `status` message with payload `{"message": "hello"}`.
4. Server validates and handles the message.
5. Server sends an `ack` response.
6. Client logs the response and closes the connection.

## Next Steps

- Add structured message types for camera frames, IMU samples, status, and errors.
- Add integration tests that start a server on a random local port.
- Add reconnection/backoff logic on the client.
- Add binary payload strategy for images instead of embedding large data in JSON.
