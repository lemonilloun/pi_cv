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

from pi_client.camera import CameraUnavailableError, capture_camera_frame
from pi_client.network import ClientConnectionError, PiClient
from pi_client.protocol import (
    make_camera_frame_message,
    make_hello_message,
    make_image_message,
    make_telemetry_message,
)
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
    parser.add_argument("--camera-shot", action="store_true", help="Capture one JPEG frame from Picamera2")
    parser.add_argument("--camera-width", type=int, default=1280, help="Camera capture width")
    parser.add_argument("--camera-height", type=int, default=720, help="Camera capture height")
    parser.add_argument("--camera-format", choices=["jpeg"], default="jpeg", help="Camera capture format")
    parser.add_argument(
        "--camera-save-local",
        nargs="?",
        const="true",
        default="false",
        type=parse_bool,
        help="Save captured camera frame locally on Raspberry Pi before sending",
    )
    return parser.parse_args()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


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

    try:
        packets: list[tuple[Message, bytes]] = []

        if args.telemetry:
            packets.append((make_telemetry_message(device_id), b""))

        if args.image:
            image_path = args.image if args.image.is_absolute() else REPO_ROOT / args.image
            image_bytes = image_path.read_bytes()
            packets.append((make_image_message(device_id, image_path, len(image_bytes)), image_bytes))

        if args.camera_shot:
            frame = capture_camera_frame(
                width=args.camera_width,
                height=args.camera_height,
                image_format=args.camera_format,
            )
            if args.camera_save_local:
                local_path = REPO_ROOT / "data/camera_local" / f"{frame.frame_id}.jpg"
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(frame.data)
                logging.getLogger(__name__).info("Saved local camera frame to %s", local_path)

            packets.append((make_camera_frame_message(device_id, frame), frame.data))

        if not packets:
            packets.append((make_hello_message(device_id), b""))

        with PiClient(host=host, port=port, timeout_seconds=timeout_seconds) as client:
            for message, binary_payload in packets:
                response, _ = client.request(message, binary_payload)
                logging.getLogger(__name__).info("Server response: %s", response.to_dict())
    except (OSError, ValueError, ClientConnectionError, CameraUnavailableError) as exc:
        logging.getLogger(__name__).error("Client failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
