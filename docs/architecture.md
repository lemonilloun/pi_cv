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

Messages are sent over TCP as length-prefixed packets. Each packet has:

```text
[4 bytes JSON header length][8 bytes binary payload length][JSON header][binary payload]
```

This keeps JSON metadata readable while allowing binary payloads such as JPEG files or future camera frames.

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
- `camera.py` captures one JPEG frame with `Picamera2` when running on Raspberry Pi.
- `protocol.py` creates client messages for `hello`, telemetry, test image transfer, and camera frames.
- `client.py` is the command-line entry point.

### `server/`

Contains MacBook server code.

- `server.py` starts the TCP server and manages client sockets.
- `handlers.py` processes valid messages, saves image and camera frame payloads, and returns responses.
- `protocol.py` creates server responses.

## Current Flow

1. Server opens a TCP socket and listens on the configured host and port.
2. Client connects to the server.
3. Client sends a framed `status` packet with payload `{"message": "hello"}` and no binary payload.
4. Server validates and handles the message.
5. Server sends an `ack` response.
6. Client logs the response and closes the connection.

## Image Transfer Flow

1. Client reads `data/cat.jpg`.
2. Client creates an `image` message with filename, content type, byte count, and source.
3. Client sends the message header plus raw JPEG bytes in one framed packet.
4. Server validates the byte count.
5. Server saves the file into `data/received/`.
6. Server returns an `ack` response containing the saved path and byte count.

## Camera Frame Flow

1. Raspberry Pi captures one JPEG frame with `Picamera2` into memory.
2. Client creates a `camera_frame` message with frame ID, resolution, format, content type, byte count, and source.
3. Client sends the JSON header plus raw JPEG bytes in one framed packet.
4. Server validates the byte count.
5. Server saves the frame into `data/received/camera/`.
6. Server returns an `ack` response containing the saved path and byte count.

## Next Steps

- Add structured message types for camera frames, IMU samples, status, and errors.
- Add integration tests that start a server on a random local port.
- Add reconnection/backoff logic on the client.
- Add chunking or streaming mode for continuous camera frames.
