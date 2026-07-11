# Raspberry Pi CV Communication Layer

Minimal Python communication layer for a distributed computer vision setup:

- `client/` runs on Raspberry Pi 5.
- `server/` runs on MacBook.
- `shared/` contains the common JSON message protocol.

The current first-stage implementation starts a TCP server, connects a TCP client, sends a test `hello` message, and receives an acknowledgement.
It can also send a JPEG file as a binary payload with JSON metadata.

## Requirements

- Python 3.10+
- No third-party Python packages

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
