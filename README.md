# Raspberry Pi CV Communication Layer

Minimal Python communication layer for a distributed computer vision setup:

- `client/` runs on Raspberry Pi 5.
- `server/` runs on MacBook.
- `shared/` contains the common JSON message protocol.

The current first-stage implementation starts a TCP server, connects a TCP client, sends a test `hello` message, and receives an acknowledgement.
It can also send a JPEG file as a binary payload with JSON metadata.
On Raspberry Pi, it can capture one camera frame with `Picamera2` and send it to the MacBook.

## Requirements

- Python 3.10+
- No third-party Python packages
- On Raspberry Pi camera capture uses the system package `python3-picamera2`

## Project Structure

```text
pi_cv/
├── client/
│   ├── src/pi_client/
│   │   ├── client.py
│   │   ├── network.py
│   │   └── protocol.py
│   ├── tests/
│   └── requirements.txt
├── server/
│   ├── src/mac_server/
│   │   ├── handlers.py
│   │   ├── protocol.py
│   │   └── server.py
│   ├── tests/
│   └── requirements.txt
├── shared/
│   ├── messages.py
│   └── schemas.py
├── config/
│   └── default.json
├── data/
│   ├── cat.jpg
│   └── received/
├── docs/
│   └── architecture.md
├── README.md
└── .gitignore
```

## Configuration

Default settings are stored in `config/default.json`.
Local network overrides can be stored in `.env`. A template is provided in `.env.example`.

For local testing on one machine, keep:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8765
  },
  "client": {
    "device_id": "raspberry_pi_01",
    "server_host": "127.0.0.1",
    "connect_timeout_seconds": 5
  }
}
```

For Raspberry Pi to MacBook Wi-Fi communication:

- On MacBook, keep server bind host as `0.0.0.0`.
- On Raspberry Pi, set `PI_CV_SERVER_HOST` in `.env` to the MacBook IP address on the same Wi-Fi network.
- Do not use `127.0.0.1` from Raspberry Pi when connecting to MacBook. On Raspberry Pi, `127.0.0.1` means the Raspberry Pi itself.

## Run

Open two terminals from the repository root:

```bash
cd /Users/lehacho/Desktop/works/cv_pojects/pi_cv
```

Terminal 1, start the server:

```bash
./scripts/run_server.sh
```

Terminal 2, run the client:

```bash
./scripts/run_client.sh
```

For Raspberry Pi connecting to MacBook, pass the MacBook IP explicitly:

```bash
./scripts/run_client.sh --host MACBOOK_IP_ADDRESS
```

Expected result:

- Client sends a JSON message with payload `{"message": "hello"}`.
- Server logs the received message.
- Server responds with a JSON `ack`.
- Client logs the server response.

## Send Test Image

Start the server:

```bash
cd /Users/lehacho/Desktop/works/cv_pojects/pi_cv
./scripts/run_server.sh
```

In another terminal, send `data/cat.jpg`:

```bash
cd /Users/lehacho/Desktop/works/cv_pojects/pi_cv
./scripts/run_client.sh --image data/cat.jpg
```

The client sends:

- JSON metadata: filename, content type, byte count, source.
- Binary payload: raw JPEG bytes.

The server saves the received image into:

```text
data/received/
```

For Raspberry Pi connecting to MacBook:

```bash
./scripts/run_client.sh --host MACBOOK_IP_ADDRESS --image data/cat.jpg
```

## Send Telemetry

Send a lightweight JSON-only device information packet:

```bash
./scripts/run_client.sh --telemetry
```

Send telemetry and image in one client run over the same TCP connection:

```bash
./scripts/run_client.sh --telemetry --image data/cat.jpg
```

## Raspberry Pi Camera Check

Before using the project camera mode, check the camera directly on Raspberry Pi with a connected monitor:

```bash
rpicam-hello --list-cameras
```

Start a live preview:

```bash
rpicam-hello --timeout 0
```

Capture a test still image outside the project:

```bash
rpicam-still -o ~/camera_test.jpg
```

If you only need a JPEG capture check without a long preview:

```bash
rpicam-jpeg -o ~/camera_test.jpg --timeout 2000
```

## Raspberry Pi venv

Install the minimal system packages:

```bash
sudo apt update
sudo apt install python3-venv python3-picamera2 --no-install-recommends
```

Create a virtual environment in the project. Use `--system-site-packages` so the venv can see `python3-picamera2` installed by `apt`:

```bash
cd ~/pi_cv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

Check `Picamera2` inside the venv:

```bash
python3 -c "from picamera2 import Picamera2; print('picamera2 ok')"
```

`client/requirements.txt` intentionally stays minimal because `Picamera2` should be installed through Raspberry Pi OS packages, not `pip`.

## Send Camera Shot

Start the server on MacBook:

```bash
cd /Users/lehacho/Desktop/works/cv_pojects/pi_cv
./scripts/run_server.sh
```

On Raspberry Pi over SSH:

```bash
ssh pi@192.168.1.137
cd ~/pi_cv
source .venv/bin/activate
./scripts/run_client.sh --telemetry --camera-shot
```

Optional camera settings:

```bash
./scripts/run_client.sh --camera-shot --camera-width 1280 --camera-height 720 --camera-format jpeg
```

By default, the captured camera frame is kept in memory and sent to the MacBook without writing a JPEG to the Raspberry Pi SD card.
If you explicitly want a local debug copy on Raspberry Pi:

```bash
./scripts/run_client.sh --camera-shot --camera-save-local true
```

The server saves camera frames into:

```text
data/received/camera/
```

## Stream Camera Video

Start the MacBook TCP server. It also starts a local browser preview server:

```bash
cd /Users/lehacho/Desktop/works/cv_pojects/pi_cv
./scripts/run_server.sh
```

Open this URL on the MacBook:

```text
http://127.0.0.1:8080/
```

From Raspberry Pi over SSH, start streaming:

```bash
ssh pi@192.168.1.137
cd ~/pi_cv
source .venv/bin/activate
./scripts/run_client.sh --telemetry --camera-stream
```

Default stream settings are conservative for Wi-Fi and SD-card lifetime:

```text
1280x720, 12 FPS, JPEG quality 85, no per-frame saving on Raspberry Pi or MacBook
```

Higher quality local Wi-Fi test:

```bash
./scripts/run_client.sh --camera-stream \
  --camera-width 1920 \
  --camera-height 1080 \
  --stream-fps 15 \
  --stream-quality 90
```

Stop streaming with `Ctrl+C` in the Raspberry Pi SSH terminal.

If you explicitly want the MacBook server to save every streamed frame, use:

```bash
./scripts/run_client.sh --camera-stream --stream-save-frames true
```

This can fill disk quickly, so keep it off for normal preview.

### Camera Module 3 Autofocus

Continuous autofocus, normal range:

```bash
./scripts/run_client.sh --camera-stream \
  --camera-autofocus-mode continuous \
  --camera-autofocus-range normal \
  --camera-autofocus-speed fast
```

Macro / close-object streaming:

```bash
./scripts/run_client.sh --camera-stream \
  --camera-autofocus-mode continuous \
  --camera-autofocus-range macro \
  --camera-autofocus-speed fast
```

Manual focus example. `lens-position` is in dioptres: `0.0` is infinity, `2.0` is about 0.5 m:

```bash
./scripts/run_client.sh --camera-stream \
  --camera-autofocus-mode manual \
  --camera-lens-position 2.0
```

The same autofocus flags also work with one-shot capture:

```bash
./scripts/run_client.sh --telemetry --camera-shot --camera-autofocus-range macro
```

### Optional H.264 Direction

The built-in project stream is MJPEG over the existing Python TCP protocol, which is simple and good for debugging and future CV frame handling.
For the highest quality and lower bandwidth, the next streaming option should use `rpicam-vid` H.264. Raspberry Pi documents network video streaming with:

```bash
rpicam-vid -t 0 -n --inline -o udp://MACBOOK_IP:PORT
```

or TCP listener mode:

```bash
rpicam-vid -t 0 -n --inline --listen -o tcp://0.0.0.0:PORT
```

## One-shot Test

For a quick local check, start the server in one-shot mode:

```bash
./scripts/run_server.sh --once
```

Then run the client once:

```bash
./scripts/run_client.sh
```

The server exits after handling one client connection.

One-shot image test:

```bash
./scripts/run_server.sh --once
```

Then:

```bash
./scripts/run_client.sh --image data/cat.jpg
```

## Manual Commands

If you do not want to use scripts, run from the repository root:

```bash
PYTHONPATH=.:server/src python3 -m mac_server.server
PYTHONPATH=.:client/src python3 -m pi_client.client
```

If you are already inside `server/src`, run:

```bash
PYTHONPATH=../..:. python3 -m mac_server.server
```

If you are already inside `client/src`, run:

```bash
PYTHONPATH=../..:. python3 -m pi_client.client
```

## Check Port Conflicts

The project uses TCP port `8765` by default. To check whether something else is using it on macOS:

```bash
lsof -nP -iTCP:8765 -sTCP:LISTEN
```

## Protocol

TCP is a byte stream, so the project uses length-prefixed packets:

```text
[4 bytes JSON header length][8 bytes binary payload length][JSON header][binary payload]
```

For `hello`, binary payload length is `0`.
For `--image data/cat.jpg`, the JSON header contains image metadata and the binary payload contains the JPEG bytes.
For `--camera-shot`, the JSON header has type `camera_frame` and the binary payload contains JPEG bytes captured from `Picamera2`.

## CV Experiments

The `research/` directory is ignored by Git and can keep local notes, model experiments, and downloaded references.
The project code exposes a separate CV CLI:

```bash
./scripts/run_cv_client.sh --help
```

Install only the CV dependencies you need on Raspberry Pi.
For real Depth Anything V2 testing, use the setup script below instead of only installing the minimal file:

```bash
cd ~/pi_cv
source .venv/bin/activate
```

For YOLO on Raspberry Pi:

```bash
pip install -r client/requirements-yolo.txt
yolo detect predict model=yolo11n.pt
yolo export model=yolo11n.pt format=ncnn
```

Start the MacBook server:

```bash
cd /Users/lehacho/Desktop/works/cv_pojects/pi_cv
./scripts/run_server.sh
```

### Depth Test

Install Depth Anything V2 Small and download its real checkpoint:

```bash
./scripts/setup_depth_anything_v2.sh small
```

Check the real depth environment:

```bash
./scripts/run_cv_client.sh depth-check
```

Run a real camera depth test:

```bash
./scripts/run_cv_client.sh depth \
  --source camera \
  --depth-backend depth-anything-v2 \
  --depth-model-path models/depth_anything_v2_vits.pth \
  --depth-encoder vits \
  --depth-input-size 392 \
  --telemetry
```

The server saves the result in:

```text
data/received/cv/<run_id>/
```

Open these files on the MacBook:

- `original.jpg` - source image from Raspberry Pi camera.
- `depth_heatmap.jpg` - visual depth map.
- `depth_raw.npz` - raw float32 depth map.
- `metadata.json` - model name, inference time, min/max depth, image source.

To test the model on an existing image file instead of the camera:

```bash
./scripts/run_cv_client.sh depth \
  --source image \
  --image data/cat.jpg \
  --depth-backend depth-anything-v2 \
  --depth-model-path models/depth_anything_v2_vits.pth \
  --depth-encoder vits \
  --depth-input-size 392 \
  --telemetry
```

Synthetic depth still exists only as a transport smoke test:

```bash
./scripts/run_cv_client.sh depth --source image --image data/cat.jpg --depth-backend synthetic
```

Depth Anything V2 model variants:

```bash
./scripts/setup_depth_anything_v2.sh small  # vits, first choice for Raspberry Pi 5 CPU
./scripts/setup_depth_anything_v2.sh base   # vitb, heavier
./scripts/setup_depth_anything_v2.sh large  # vitl, likely too slow/heavy on Pi CPU
```

### YOLO Test

For detection with exported NCNN model:

```bash
./scripts/run_cv_client.sh yolo \
  --source camera \
  --yolo-model models/yolo11n_ncnn_model \
  --yolo-task detect \
  --yolo-confidence 0.5 \
  --telemetry
```

For segmentation, use a segmentation model such as `yolo11n-seg.pt` exported to NCNN:

```bash
./scripts/run_cv_client.sh yolo \
  --source camera \
  --yolo-model models/yolo11n-seg_ncnn_model \
  --yolo-task segment \
  --yolo-confidence 0.5
```

### Combined Pipeline

Capture one image, run YOLO, run depth, attach median depth values to each object box, and send all artifacts to the MacBook:

```bash
./scripts/run_cv_client.sh pipeline \
  --source camera \
  --yolo-model models/yolo11n_ncnn_model \
  --yolo-task detect \
  --yolo-confidence 0.5 \
  --depth-backend depth-anything-v2 \
  --depth-model-path models/depth_anything_v2_vits.pth \
  --depth-encoder vits \
  --depth-input-size 392 \
  --telemetry
```

The server saves each result package here:

```text
data/received/cv/<run_id>/
```

Each run contains:

- `metadata.json` with model settings, detections, and depth statistics.
- `original.jpg`
- `yolo_annotated.jpg` for YOLO and pipeline runs.
- `depth_heatmap.jpg` for depth and pipeline runs.
- `depth_raw.npz` for depth and pipeline runs.
- `combined.jpg` for pipeline runs.

Depth Anything V2 relative checkpoints produce relative depth, not true distance in meters. Use `--depth-is-metric` only when the selected model output is metric.
