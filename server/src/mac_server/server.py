"""TCP server for the MacBook processing node."""

from __future__ import annotations

import argparse
import json
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
from mac_server.protocol import make_error_response
from shared.messages import Message


logger = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = REPO_ROOT / "config/default.json"


class MacServer:
    def __init__(self, host: str, port: int, backlog: int = 5) -> None:
        self.host = host
        self.port = port
        self.backlog = backlog
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

    def _handle_client(self, client_socket: socket.socket, address: tuple[str, int]) -> None:
        with client_socket:
            reader = client_socket.makefile("rb")
            with reader:
                while not self._stop_event.is_set():
                    try:
                        raw_line = reader.readline()
                    except OSError as exc:
                        logger.warning("Failed reading from %s:%s: %s", address[0], address[1], exc)
                        return

                    if not raw_line:
                        logger.info("Client disconnected %s:%s", address[0], address[1])
                        return

                    response = self._process_raw_message(raw_line)

                    try:
                        client_socket.sendall(response.to_json_line())
                    except OSError as exc:
                        logger.warning("Failed sending to %s:%s: %s", address[0], address[1], exc)
                        return

    def _process_raw_message(self, raw_line: bytes) -> Message:
        try:
            message = Message.from_json_line(raw_line)
            return handle_message(message)
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


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config.get("logging", {}).get("level", "INFO"))

    server_config = config["server"]
    host = args.host or server_config["host"]
    port = args.port or int(server_config["port"])
    backlog = int(server_config.get("backlog", 5))

    server = MacServer(host=host, port=port, backlog=backlog)
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
