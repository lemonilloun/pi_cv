"""Command-line entry point for the persistent Raspberry Pi session client."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pi_client.camera import CameraFocusOptions
from pi_client.camera_session import CaptureSettings
from pi_client.session import SESSION_MODES, SessionRuntime, SessionSettings
from shared.config import load_config


DEFAULT_CONFIG_PATH = REPO_ROOT / "config/default.json"
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Per-frame send logging would flood the console at stream FPS.
    logging.getLogger("pi_client.network").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run persistent Raspberry Pi session client")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--host", help="Override server host from config")
    parser.add_argument("--port", type=int, help="Override server port from config")
    parser.add_argument("--source", choices=["camera", "synthetic"], help="Frame source")
    parser.add_argument("--synthetic-image", type=Path, help="Image for --source synthetic")
    parser.add_argument("--initial-mode", choices=list(SESSION_MODES), help="Mode to enter on start")

    parser.add_argument("--stream-fps", type=float, help="Full-rate stream FPS")
    parser.add_argument("--cv-fps", type=float, help="CV modes FPS")
    parser.add_argument("--camera-width", type=int, help="Capture width")
    parser.add_argument("--camera-height", type=int, help="Capture height")
    parser.add_argument("--stream-quality", type=int, help="Stream JPEG quality 1-95")

    parser.add_argument("--camera-autofocus-mode", choices=["manual", "auto", "continuous"])
    parser.add_argument("--camera-autofocus-range", choices=["normal", "macro", "full"])
    parser.add_argument("--camera-autofocus-speed", choices=["normal", "fast"])
    parser.add_argument("--camera-lens-position", type=float)

    parser.add_argument("--yolo-model", help="Path to YOLO .pt or exported NCNN model directory")
    parser.add_argument("--yolo-task", choices=["detect", "segment"])
    parser.add_argument("--yolo-confidence", type=float)

    parser.add_argument("--depth-backend", choices=["synthetic", "depth-anything-v2"])
    parser.add_argument("--depth-model-path", help="Depth model checkpoint path")
    parser.add_argument("--depth-encoder", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--depth-input-size", type=int)
    parser.add_argument("--depth-is-metric", action="store_true")
    parser.add_argument("--torch-threads", type=int)

    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> SessionSettings:
    config = load_config(args.config, DEFAULT_ENV_PATH)
    configure_logging(config.get("logging", {}).get("level", "INFO"))

    server_config = config["server"]
    client_config = config["client"]
    session_config = config.get("session", {})

    def pick(arg_value, key, default):
        if arg_value is not None:
            return arg_value
        return session_config.get(key, default)

    host = args.host or client_config.get("server_host") or server_config["host"]
    port = args.port or int(server_config["port"])

    focus_options = CameraFocusOptions(
        autofocus_mode=args.camera_autofocus_mode or "continuous",
        autofocus_range=args.camera_autofocus_range or "normal",
        autofocus_speed=args.camera_autofocus_speed or "fast",
        lens_position=args.camera_lens_position,
    )

    width = int(pick(args.camera_width, "camera_width", 1280))
    height = int(pick(args.camera_height, "camera_height", 720))
    stream_fps = float(pick(args.stream_fps, "stream_fps", 30.0))
    stream_quality = int(pick(args.stream_quality, "stream_quality", 85))
    cv_fps = float(pick(args.cv_fps, "cv_fps", 1.0))

    synthetic_image = args.synthetic_image or Path(session_config.get("synthetic_image", "data/cat.jpg"))
    if not synthetic_image.is_absolute():
        synthetic_image = REPO_ROOT / synthetic_image

    return SessionSettings(
        host=host,
        port=port,
        device_id=client_config["device_id"],
        source_kind=str(pick(args.source, "source", "camera")),
        synthetic_image=synthetic_image,
        stream=CaptureSettings(
            width=width,
            height=height,
            fps=stream_fps,
            jpeg_quality=stream_quality,
            focus_options=focus_options,
        ),
        cv_capture=CaptureSettings(
            width=width,
            height=height,
            fps=cv_fps,
            jpeg_quality=stream_quality,
            focus_options=focus_options,
        ),
        cv_fps=cv_fps,
        cv_jpeg_quality=stream_quality,
        telemetry_hz=float(session_config.get("telemetry_hz", 1.0)),
        initial_mode=str(pick(args.initial_mode, "initial_mode", "idle")),
        connect_timeout_seconds=float(client_config.get("connect_timeout_seconds", 5)),
        reconnect_min_seconds=float(session_config.get("reconnect_min_seconds", 1.0)),
        reconnect_max_seconds=float(session_config.get("reconnect_max_seconds", 30.0)),
        yolo_model=pick(args.yolo_model, "yolo_model", None),
        yolo_task=str(pick(args.yolo_task, "yolo_task", "detect")),
        yolo_confidence=float(pick(args.yolo_confidence, "yolo_confidence", 0.5)),
        depth_backend=str(pick(args.depth_backend, "depth_backend", "depth-anything-v2")),
        depth_model_path=pick(args.depth_model_path, "depth_model_path", None),
        depth_encoder=str(pick(args.depth_encoder, "depth_encoder", "vits")),
        depth_input_size=int(pick(args.depth_input_size, "depth_input_size", 392)),
        depth_is_metric=bool(args.depth_is_metric or session_config.get("depth_is_metric", False)),
        torch_threads=int(pick(args.torch_threads, "torch_threads", 3)),
    )


def main() -> int:
    args = parse_args()
    settings = build_settings(args)
    logger = logging.getLogger(__name__)
    logger.info(
        "Starting session client device_id=%s server=%s:%s source=%s initial_mode=%s",
        settings.device_id,
        settings.host,
        settings.port,
        settings.source_kind,
        settings.initial_mode,
    )

    runtime = SessionRuntime(settings)

    def handle_signal(signum: int, _frame) -> None:
        logger.info("Received signal %s, shutting down", signum)
        runtime.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    return runtime.run()


if __name__ == "__main__":
    raise SystemExit(main())
