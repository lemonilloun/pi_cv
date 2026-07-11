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
    "host": "127.0.0.1",
    "port": 5000
  },
  "client": {
    "device_id": "raspberry_pi_01",
    "connect_timeout_seconds": 5
  }
}
```

For Raspberry Pi to MacBook Wi-Fi communication, set `server.host` on the Raspberry Pi side to the MacBook IP address on the same network.

## Run

Open two terminals from the repository root.

Terminal 1, start the server:

```bash
PYTHONPATH=.:server/src python3 -m mac_server.server
```

Terminal 2, run the client:

```bash
PYTHONPATH=.:client/src python3 -m pi_client.client
```

Expected result:

- Client sends a JSON message with payload `{"message": "hello"}`.
- Server logs the received message.
- Server responds with a JSON `ack`.
- Client logs the server response.

## One-shot Test

For a quick local check, start the server in one-shot mode:

```bash
PYTHONPATH=.:server/src python3 -m mac_server.server --once
```

Then run the client once:

```bash
PYTHONPATH=.:client/src python3 -m pi_client.client
```

The server exits after handling one client connection.
