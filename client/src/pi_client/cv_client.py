"""Command-line entry point for Raspberry Pi CV experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pi_client.cv_models import CvModelError
from pi_client.cv_pipeline import build_cv_package, load_source_image
from pi_client.network import ClientConnectionError, PiClient
from pi_client.protocol import make_cv_result_message, make_telemetry_message
from shared.config import load_config


DEFAULT_CONFIG_PATH = REPO_ROOT / "config/default.json"
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Raspberry Pi CV experiments")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    for mode in ("depth", "yolo", "pipeline"):
        subparser = subparsers.add_parser(mode)
        _add_common_args(subparser)

    return parser.parse_args()


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--host", help="Override server host from config")
    parser.add_argument("--port", type=int, help="Override server port from config")
    parser.add_argument("--source", choices=["camera", "image"], default="camera")
    parser.add_argument("--image", type=Path, help="Input image path when --source image")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--telemetry", action="store_true")

    parser.add_argument("--yolo-model", help="Path to YOLO .pt or exported NCNN model directory")
    parser.add_argument("--yolo-task", choices=["detect", "segment"], default="detect")
    parser.add_argument("--yolo-confidence", type=float, default=0.5)

    parser.add_argument(
        "--depth-backend",
        choices=["synthetic", "depth-anything-v2", "depth-anything-v3"],
        default="synthetic",
    )
    parser.add_argument("--depth-model-path", help="Depth model checkpoint path or Hugging Face model id")
    parser.add_argument("--depth-encoder", choices=["vits", "vitb", "vitl"], default="vits")
    parser.add_argument("--depth-input-size", type=int, default=392)
    parser.add_argument("--depth-is-metric", action="store_true")


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_ENV_PATH)
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)

    server_config = config["server"]
    client_config = config["client"]
    host = args.host or client_config.get("server_host") or server_config["host"]
    port = args.port or int(server_config["port"])
    timeout_seconds = float(client_config.get("connect_timeout_seconds", 5))
    device_id = client_config["device_id"]

    try:
        image_path = None
        if args.image is not None:
            image_path = args.image if args.image.is_absolute() else REPO_ROOT / args.image

        image_bytes, source_metadata = load_source_image(
            source=args.source,
            image_path=image_path,
            width=args.camera_width,
            height=args.camera_height,
        )
        run_id, package_bytes, summary = build_cv_package(
            mode=args.mode,
            device_id=device_id,
            image_bytes=image_bytes,
            source_metadata=source_metadata,
            yolo_model=args.yolo_model,
            yolo_task=args.yolo_task,
            yolo_confidence=args.yolo_confidence,
            depth_backend=args.depth_backend,
            depth_model_path=args.depth_model_path,
            depth_encoder=args.depth_encoder,
            depth_input_size=args.depth_input_size,
            depth_is_metric=args.depth_is_metric,
        )

        with PiClient(host=host, port=port, timeout_seconds=timeout_seconds) as client:
            if args.telemetry:
                response, _ = client.request(make_telemetry_message(device_id))
                logger.info("Telemetry response: %s", response.to_dict())

            message = make_cv_result_message(
                device_id=device_id,
                run_id=run_id,
                pipeline_type=args.mode,
                byte_count=len(package_bytes),
                summary=summary,
            )
            response, _ = client.request(message, package_bytes)
            logger.info("CV result response: %s", response.to_dict())

    except (OSError, ValueError, ClientConnectionError, CvModelError) as exc:
        logger.error("CV client failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
