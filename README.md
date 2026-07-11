# Raspberry Pi CV Communication Layer

Minimal Python communication layer for a distributed computer vision setup:

- `client/` runs on Raspberry Pi 5.
- `server/` runs on MacBook.
- `shared/` contains the common JSON message protocol.

The current first-stage implementation starts a TCP server, connects a TCP client, sends a test `hello` message, and receives an acknowledgement.

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
├── docs/
│   └── architecture.md
├── README.md
└── .gitignore
```

## Configuration

Default settings are stored in `config/default.json`.

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
- On Raspberry Pi, set `client.server_host` to the MacBook IP address on the same Wi-Fi network.
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
