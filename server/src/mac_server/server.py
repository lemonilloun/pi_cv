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

from mac_server.handlers import handle_message
from mac_server.preview import LatestFrameStore, MjpegPreviewServer
from mac_server.protocol import make_error_response
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
    ) -> None:
        self.host = host
        self.port = port
        self.backlog = backlog
        self.storage_dir = storage_dir or REPO_ROOT / "data/received"
        self.preview_store = LatestFrameStore()
        self.preview_server = (
            MjpegPreviewServer(preview_host, preview_port, self.preview_store)
            if preview_enabled
            else None
        )
        self._socket: socket.socket | None = None
        self._stop_event = threading.Event()

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
        if self._socket is not None:
            self._socket.close()
            self._socket = None
            logger.info("Server stopped")
        if self.preview_server is not None:
            self.preview_server.stop()

    def _handle_client(self, client_socket: socket.socket, address: tuple[str, int]) -> None:
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
                        send_packet(client_socket, make_error_response(str(exc)))
                    except OSError:
                        return
                    return
                except OSError as exc:
                    logger.warning("Failed reading from %s:%s: %s", address[0], address[1], exc)
                    return

                response = self._process_message(message, binary_payload)

                try:
                    send_packet(client_socket, response)
                except OSError as exc:
                    logger.warning("Failed sending to %s:%s: %s", address[0], address[1], exc)
                    return

    def _process_message(self, message: Message, binary_payload: bytes) -> Message:
        try:
            return handle_message(message, binary_payload, self.storage_dir, self.preview_store)
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

    server = MacServer(
        host=host,
        port=port,
        backlog=backlog,
        storage_dir=storage_dir,
        preview_host=preview_host,
        preview_port=preview_port,
        preview_enabled=preview_enabled,
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
