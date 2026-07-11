"""Configuration loading with optional .env overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_config(config_path: Path, env_path: Path | None = None) -> dict[str, Any]:
    if env_path is not None:
        load_env_file(env_path)

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    apply_env_overrides(config)
    return config


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")

        if key:
            os.environ.setdefault(key, value)


def apply_env_overrides(config: dict[str, Any]) -> None:
    server_config = config.setdefault("server", {})
    client_config = config.setdefault("client", {})
    logging_config = config.setdefault("logging", {})

    if value := os.environ.get("PI_CV_SERVER_BIND_HOST"):
        server_config["host"] = value
    if value := os.environ.get("PI_CV_SERVER_PORT"):
        server_config["port"] = int(value)
    if value := os.environ.get("PI_CV_STORAGE_DIR"):
        server_config["storage_dir"] = value

    if value := os.environ.get("PI_CV_SERVER_HOST"):
        client_config["server_host"] = value
    if value := os.environ.get("PI_CV_DEVICE_ID"):
        client_config["device_id"] = value
    if value := os.environ.get("PI_CV_CONNECT_TIMEOUT_SECONDS"):
        client_config["connect_timeout_seconds"] = float(value)

    if value := os.environ.get("PI_CV_LOG_LEVEL"):
        logging_config["level"] = value
