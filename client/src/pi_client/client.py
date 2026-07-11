"""Command-line entry point for the Raspberry Pi TCP client."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from pi_client.network import ClientConnectionError, PiClient
from pi_client.protocol import make_hello_message


DEFAULT_CONFIG_PATH = Path("config/default.json")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config.get("logging", {}).get("level", "INFO"))

    server_config = config["server"]
    client_config = config["client"]

    host = args.host or server_config["host"]
    port = args.port or int(server_config["port"])
    timeout_seconds = float(client_config.get("connect_timeout_seconds", 5))
    device_id = client_config["device_id"]

    message = make_hello_message(device_id)

    try:
        with PiClient(host=host, port=port, timeout_seconds=timeout_seconds) as client:
            response = client.request(message)
    except (OSError, ValueError, ClientConnectionError) as exc:
        logging.getLogger(__name__).error("Client failed: %s", exc)
        return 1

    logging.getLogger(__name__).info("Server response: %s", response.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
