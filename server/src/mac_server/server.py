"""TCP server for the MacBook processing node."""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
from pathlib import Path
from types import TracebackType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mac_server.handlers import SessionContext, handle_message
from mac_server.preview import FrameStoreHub, MjpegPreviewServer
from mac_server.protocol import make_error_response
from mac_server.registry import ClientRegistry, TelemetryStore
from shared.config import load_config
from shared.framing import receive_packet, send_packet
from shared.messages import Message


logger = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = REPO_ROOT / "config/default.json"
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


class MacServer:
    def __init__(
        self,
        host: str,
        port: int,
        backlog: int = 5,
        storage_dir: Path | None = None,
        preview_host: str = "127.0.0.1",
        preview_port: int = 8080,
        preview_enabled: bool = True,
        server_cv_config: dict[str, Any] | None = None,
        mapping_config: dict[str, Any] | None = None,
        monitoring_config: dict[str, Any] | None = None,
        scene3d_config: dict[str, Any] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.backlog = backlog
        self.storage_dir = storage_dir or REPO_ROOT / "data/received"
        self.frame_hub = FrameStoreHub()
        self.preview_store = self.frame_hub.get("pi")
        self.registry = ClientRegistry()
        self.telemetry_store = TelemetryStore()
        # Separate store: imu_telemetry arrives at ~10 Hz (vs 1 Hz for
        # system_telemetry) so the panel's IMU tab can poll it on its own
        # cadence without growing the payload every other view already polls.
        self.imu_telemetry_store = TelemetryStore(history_limit=1200)
        self.cv_worker = self._build_cv_worker(server_cv_config or {}, mapping_config or {})
        self.room_store, self.scan_controller = self._build_mapping(mapping_config or {})
        self.monitor_controller = self._build_monitoring(monitoring_config or {})
        self.scene_pipeline = self._build_scene3d(scene3d_config or {})
        self.preview_server = (
            MjpegPreviewServer(
                preview_host,
                preview_port,
                self.frame_hub,
                registry=self.registry,
                telemetry_store=self.telemetry_store,
                imu_telemetry_store=self.imu_telemetry_store,
                cv_status_provider=(self.cv_worker.status if self.cv_worker is not None else None),
                scan_controller=self.scan_controller,
                room_store=self.room_store,
                monitor_controller=self.monitor_controller,
                scene_pipeline=self.scene_pipeline,
            )
            if preview_enabled
            else None
        )
        self._socket: socket.socket | None = None
        self._stop_event = threading.Event()

    def _build_cv_worker(self, cv_config: dict[str, Any], mapping_config: dict[str, Any]):
        if not cv_config.get("enabled", False):
            return None
        try:
            from mac_server.cv_worker import ServerCvWorker
        except ImportError as exc:
            logger.warning("Server CV disabled (import failed): %s", exc)
            return None
        return ServerCvWorker(
            pi_store=self.preview_store,
            frame_hub=self.frame_hub,
            config=cv_config,
            repo_root=REPO_ROOT,
            mapping_config=mapping_config,
        )

    def _build_mapping(self, mapping_config: dict[str, Any]):
        if self.cv_worker is None or not mapping_config:
            return None, None
        try:
            from mac_server.mapping.rooms import RoomStore
            from mac_server.mapping.scanner import ScanController
        except ImportError as exc:
            logger.warning("Mapping disabled (import failed): %s", exc)
            return None, None
        rooms_dir = Path(mapping_config.get("rooms_dir", "data/rooms"))
        if not rooms_dir.is_absolute():
            rooms_dir = REPO_ROOT / rooms_dir
        room_store = RoomStore(rooms_dir, mapping_config)
        scan_controller = ScanController(
            room_store=room_store,
            cv_worker=self.cv_worker,
            frame_hub=self.frame_hub,
            registry=self.registry,
            config=mapping_config,
        )
        return room_store, scan_controller

    def _build_monitoring(self, monitoring_config: dict[str, Any]):
        if self.cv_worker is None or not monitoring_config.get("enabled", False):
            return None
        try:
            from mac_server.monitoring.controller import MonitorController
            from mac_server.monitoring.scenes import SceneStore
            from mac_server.monitoring.store import MonitoringStore
        except ImportError as exc:
            logger.warning("Monitoring disabled (import failed): %s", exc)
            return None

        data_dir = Path(monitoring_config.get("data_dir", "data/monitoring"))
        if not data_dir.is_absolute():
            data_dir = REPO_ROOT / data_dir
        self._monitoring_data_dir = data_dir

        agent = None
        agent_config = monitoring_config.get("agent", {})
        if agent_config.get("enabled", False):
            try:
                from mac_server.monitoring.agent import ApfelClient

                agent = ApfelClient(
                    base_url=str(agent_config.get("base_url", "http://127.0.0.1:11434")),
                    model=str(agent_config.get("model", "apple-foundationmodel")),
                    timeout_s=float(agent_config.get("timeout_s", 20.0)),
                    max_input_tokens=int(agent_config.get("max_input_tokens", 2500)),
                    max_output_tokens=int(agent_config.get("max_output_tokens", 400)),
                )
            except Exception as exc:
                logger.warning("Monitoring agent disabled: %s", exc)

        vision = None
        vision_config = monitoring_config.get("vision", {})
        if vision_config.get("enabled", False):
            try:
                from mac_server.monitoring.vision import OllamaVisionClient

                vision = OllamaVisionClient(
                    base_url=str(vision_config.get("base_url", "http://127.0.0.1:11434")),
                    model=str(vision_config.get("model", "gemma4:e4b-it-qat")),
                    timeout_s=float(vision_config.get("timeout_s", 30.0)),
                    keep_alive=str(vision_config.get("keep_alive", "0s")),
                )
            except Exception as exc:
                logger.warning("Monitoring vision captioning disabled: %s", exc)

        return MonitorController(
            cv_worker=self.cv_worker,
            frame_hub=self.frame_hub,
            registry=self.registry,
            scene_store=SceneStore(data_dir / "scenes"),
            store=MonitoringStore(data_dir),
            config=monitoring_config,
            vision=vision,
            scan_controller=self.scan_controller,
            agent=agent,
        )

    def _build_scene3d(self, scene3d_config: dict[str, Any]):
        if not scene3d_config:
            return None
        try:
            from mac_server.scene3d.pipeline import ScenePipeline
        except ImportError as exc:
            logger.warning("Scene3D disabled (import failed): %s", exc)
            return None
        sessions_dir = Path(scene3d_config.get("sessions_dir", "data/scene_sessions"))
        if not sessions_dir.is_absolute():
            sessions_dir = REPO_ROOT / sessions_dir
        return ScenePipeline(sessions_dir, scene3d_config, REPO_ROOT)

    def start(self) -> None:
        if self._socket is not None:
            return

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(self.backlog)

        self._socket = server_socket
        logger.info("Server listening on %s:%s", self.host, self.port)
        if self.preview_server is not None:
            self.preview_server.start()
        if self.cv_worker is not None:
            self.cv_worker.start()
        # apfel/Ollama lifecycle is tied to monitoring sessions, not the
        # server: MonitorController starts them on monitor start and stops
        # them (only the copies it spawned) on monitor stop.

    def serve_forever(self, once: bool = False) -> None:
        self.start()
        assert self._socket is not None

        while not self._stop_event.is_set():
            try:
                client_socket, address = self._socket.accept()
            except OSError:
                if self._stop_event.is_set():
                    break
                raise

            logger.info("Accepted connection from %s:%s", address[0], address[1])

            if once:
                self._handle_client(client_socket, address)
                break

            thread = threading.Thread(
                target=self._handle_client,
                args=(client_socket, address),
                daemon=True,
            )
            thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self.monitor_controller is not None:
            self.monitor_controller.shutdown()
        if self.scan_controller is not None:
            self.scan_controller.shutdown()
        if self.cv_worker is not None:
            self.cv_worker.stop()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
            logger.info("Server stopped")
        if self.preview_server is not None:
            self.preview_server.stop()

    def _handle_client(self, client_socket: socket.socket, address: tuple[str, int]) -> None:
        send_lock = threading.Lock()
        session_context = SessionContext(
            registry=self.registry,
            telemetry_store=self.telemetry_store,
            client_socket=client_socket,
            client_address=address,
            send_lock=send_lock,
            imu_telemetry_store=self.imu_telemetry_store,
        )
        try:
            with client_socket:
                while not self._stop_event.is_set():
                    try:
                        message, binary_payload = receive_packet(client_socket)
                    except EOFError:
                        logger.info("Client disconnected %s:%s", address[0], address[1])
                        return
                    except ValueError as exc:
                        logger.warning("Invalid packet from %s:%s: %s", address[0], address[1], exc)
                        try:
                            with send_lock:
                                send_packet(client_socket, make_error_response(str(exc)))
                        except OSError:
                            return
                        return
                    except OSError as exc:
                        logger.warning("Failed reading from %s:%s: %s", address[0], address[1], exc)
                        return

                    response = self._process_message(message, binary_payload, session_context)

                    try:
                        with send_lock:
                            send_packet(client_socket, response)
                    except OSError as exc:
                        logger.warning("Failed sending to %s:%s: %s", address[0], address[1], exc)
                        return
        finally:
            self.registry.unregister(client_socket)

    def _process_message(
        self,
        message: Message,
        binary_payload: bytes,
        session_context: SessionContext,
    ) -> Message:
        try:
            return handle_message(
                message,
                binary_payload,
                self.storage_dir,
                self.preview_store,
                session_context=session_context,
            )
        except ValueError as exc:
            logger.warning("Invalid message: %s", exc)
            return make_error_response(str(exc))

    def __enter__(self) -> "MacServer":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MacBook TCP server")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--host", help="Override server host from config")
    parser.add_argument("--port", type=int, help="Override server port from config")
    parser.add_argument("--once", action="store_true", help="Handle one connection and exit")
    parser.add_argument("--preview-host", help="HTTP MJPEG preview bind host")
    parser.add_argument("--preview-port", type=int, help="HTTP MJPEG preview port")
    parser.add_argument("--no-preview", action="store_true", help="Disable HTTP MJPEG preview server")
    parser.add_argument("--no-cv", action="store_true", help="Disable server-side CV worker and mapping")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_ENV_PATH)
    configure_logging(config.get("logging", {}).get("level", "INFO"))

    server_config = config["server"]
    host = args.host or server_config["host"]
    port = args.port or int(server_config["port"])
    backlog = int(server_config.get("backlog", 5))
    storage_dir = Path(server_config.get("storage_dir", "data/received"))
    if not storage_dir.is_absolute():
        storage_dir = REPO_ROOT / storage_dir
    preview_config = config.get("preview", {})
    preview_host = args.preview_host or preview_config.get("host", "127.0.0.1")
    preview_port = args.preview_port or int(preview_config.get("port", 8080))
    preview_enabled = bool(preview_config.get("enabled", True)) and not args.no_preview

    server_cv_config = dict(config.get("server_cv", {}))
    mapping_config = dict(config.get("mapping", {}))
    monitoring_config = dict(config.get("monitoring", {}))
    scene3d_config = dict(config.get("scene3d", {}))
    if args.no_cv:
        server_cv_config["enabled"] = False
        mapping_config = {}
        monitoring_config = {}

    server = MacServer(
        host=host,
        port=port,
        backlog=backlog,
        storage_dir=storage_dir,
        preview_host=preview_host,
        preview_port=preview_port,
        preview_enabled=preview_enabled,
        server_cv_config=server_cv_config,
        mapping_config=mapping_config,
        monitoring_config=monitoring_config,
        scene3d_config=scene3d_config,
    )
    try:
        server.serve_forever(once=args.once)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except OSError as exc:
        logger.error("Server failed: %s", exc)
        return 1
    finally:
        server.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
