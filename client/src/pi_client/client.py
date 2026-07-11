"""Command-line entry point for the Raspberry Pi TCP client."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pi_client.network import ClientConnectionError, PiClient
from pi_client.protocol import make_hello_message, make_image_message, make_telemetry_message
from shared.config import load_config
from shared.messages import Message


DEFAULT_CONFIG_PATH = REPO_ROOT / "config/default.json"
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Raspberry Pi TCP client")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--host", help="Override server host from config")
    parser.add_argument("--port", type=int, help="Override server port from config")
    parser.add_argument("--image", type=Path, help="Send an image file as binary payload")
    parser.add_argument("--telemetry", action="store_true", help="Send a device telemetry JSON packet")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_ENV_PATH)
    configure_logging(config.get("logging", {}).get("level", "INFO"))

    server_config = config["server"]
    client_config = config["client"]

    host = args.host or client_config.get("server_host") or server_config["host"]
    port = args.port or int(server_config["port"])
    timeout_seconds = float(client_config.get("connect_timeout_seconds", 5))
    device_id = client_config["device_id"]

    packets: list[tuple[Message, bytes]] = []

    if args.telemetry:
        packets.append((make_telemetry_message(device_id), b""))

    if args.image:
        image_path = args.image if args.image.is_absolute() else REPO_ROOT / args.image
        image_bytes = image_path.read_bytes()
        packets.append((make_image_message(device_id, image_path, len(image_bytes)), image_bytes))

    if not packets:
        packets.append((make_hello_message(device_id), b""))

    try:
        with PiClient(host=host, port=port, timeout_seconds=timeout_seconds) as client:
            for message, binary_payload in packets:
                response, _ = client.request(message, binary_payload)
                logging.getLogger(__name__).info("Server response: %s", response.to_dict())
    except (OSError, ValueError, ClientConnectionError) as exc:
        logging.getLogger(__name__).error("Client failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
