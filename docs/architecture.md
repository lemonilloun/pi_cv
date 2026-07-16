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

## Camera Stream Flow

1. Raspberry Pi starts `Picamera2` video capture and JPEG encoding.
2. Client sends each frame as a `camera_stream_frame` packet.
3. Server validates the byte count and updates the latest in-memory preview frame.
4. Server exposes the latest frames as MJPEG at `http://127.0.0.1:8080/`.
5. Stream frames are not saved by default; use `--stream-save-frames true` only for debugging.

## CV Experiment Flow

1. Raspberry Pi captures one camera image or reads a local image file.
2. `run_cv_client.sh` runs `depth`, `yolo`, or `pipeline`.
3. Results are packaged as a zip containing images and `metadata.json`.
4. Client sends the package as a `cv_result` message.
5. Server stores and extracts it in `data/received/cv/<run_id>/`.

## Persistent Session Flow

`scripts/run_pi_session.sh` starts one long-lived client (`pi_client/session.py`)
controlled from the web panel served at `http://127.0.0.1:8080/`.

Threads on the client per connection:

- worker (main thread): mode state machine (idle/stream/depth/yolo/pipeline),
  owns the camera and warm models, produces frames.
- receiver: sole socket reader; acks release in-flight send slots, `command`
  messages go to the worker's queue.
- telemetry: 1 Hz `system_telemetry` from `/proc/stat`, `/proc/meminfo`, and
  `/sys/class/thermal` (no psutil).

On the server, `session_hello` registers the connection in `ClientRegistry`
(`mac_server/registry.py`); the HTTP control panel (`mac_server/preview.py`)
pushes `command` messages over the same TCP socket under a per-connection send
lock. Telemetry history is kept in a rolling `TelemetryStore` deque and served
via `GET /api/telemetry`.

Backpressure: each outbound client message consumes one of 4 in-flight slots
released by its ack; frames are dropped (never queued) when the window is full,
so telemetry and command results stay responsive during streaming.

The client reconnects with exponential backoff (1→30 s), re-sends
`session_hello`, and re-applies the last requested mode. Models load lazily on
first use and stay warm across mode switches and reconnects.

## Server-Side CV + Room Mapping Flow

`mac_server/cv_worker.py` runs Depth Anything V2 metric (indoor) on the Mac
(MPS) against the incoming 30 fps camera stream: heatmap JPEGs go to the
`depth` view store, raw metric depth arrays (`DepthFrame`) go to a single-slot
`LatestItemStore` consumed by the mapper. `FrameStoreHub` (preview.py) holds
named frame stores (`pi`/`depth`/`map`) served via `/stream.mjpg?view=...`.

Room mapping (`mac_server/mapping/`):

1. `geometry.py` — pinhole intrinsics from the Camera Module 3 Wide FOV
   (102°x67°), depth→3D points, height-band filtering (floor/ceiling removal),
   per-frame wall-distance anchoring, camera→room transforms for the four
   axis-aligned scan directions.
2. `grid.py` — hit-count + max-height occupancy grid (0.05 m cells) and PNG
   rendering (walls bright, obstacles colored by height, 1 m gridlines).
3. `rooms.py` — room configs in `data/rooms/<id>.json`, scan state and
   estimated dimensions in `data/rooms/<id>/state.json`.
4. `scanner.py` — `ScanController` thread per active scan: consume depth
   frames, anchor by facing-wall distance (EMA + jump rejection), accumulate
   into the direction grid, re-render the fused map every 0.5 s, persist on
   stop.

Scan convention without IMU: the camera faces one wall and moves only along
the viewing axis; each frame is independently anchored by its measured wall
distance, so a stationary camera is a valid degenerate case and drift cannot
accumulate. Four directional scans fuse into one grid by axis-aligned
transforms and auto-estimate the room rectangle.

## Smart Monitoring Flow

For a fixed camera placement, `mac_server/monitoring/` turns the enriched
25 fps detection stream (Hailo YOLO on the Pi + metric depth on the Mac) into
a persistent semantic log:

1. `cv_worker` publishes every enriched objects batch — including empty
   ones — into `objects_store` (a `LatestItemStore`).
2. `controller.py` (`MonitorController`, one thread per active session)
   consumes it: `tracker.py` (greedy IoU/centroid association, no Kalman)
   maintains short-term tracks; `events.py` derives hysteresis-gated events
   (`entered`, `exited`, `stationary`/`moving`, `on_furniture` with posture
   from bbox aspect ratio); `signatures.py` resolves persistent entities
   (`Person#1`) via HSV-histogram matching, split-biased so uncertain matches
   create new entities rather than merging strangers.
3. `store.py` persists everything to SQLite (WAL, controller thread is the
   only writer) with bbox-crop snapshots and a retention sweep; `scenes.py`
   holds per-placement configs with explicitly frozen furniture anchors.
4. `agent.py` (optional) talks to apfel — Apple's on-device model behind an
   OpenAI-compatible endpoint — producing 5-minute event digests and
   on-demand hierarchical summaries within the model's 4096-token budget.

Timestamps are Mac wall clock at enrichment time; Pi frame indices are used
only for deduplication, sidestepping Pi↔Mac clock skew.

## Next Steps

- Hailo osnet re-id embeddings behind the `SignatureProvider` seam (robust
  person identity across clothing changes).
- TF-Luna ToF sensor as metric-depth ground truth (`depth_scale_correction`
  calibration), then revisit room mapping.
- Add structured message types for IMU samples; fuse IMU odometry into scans.
- Add optional H.264 path with `rpicam-vid` for higher quality/lower bandwidth streaming.
