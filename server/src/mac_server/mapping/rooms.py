"""Room configuration storage for mapping scans.

Each room lives in <rooms_dir>/<room_id>.json (user-editable config) with
scan state (estimated dims, per-direction max wall distances) persisted in
<rooms_dir>/<room_id>/state.json.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class RoomConfig:
    room_id: str
    name: str
    width_m: float | None = None
    length_m: float | None = None
    camera_height_m: float = 0.3
    hfov_deg: float = 102.0
    vfov_deg: float = 67.0
    created_at: float = field(default_factory=time.time)


def _safe_room_id(value: str) -> str:
    safe_chars = []
    for char in value.strip().lower():
        if char.isalnum():
            safe_chars.append(char)
        elif char in {" ", "-", "_"}:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("_")
    return safe or "room"


class RoomStore:
    def __init__(self, rooms_dir: Path, mapping_config: dict[str, Any]) -> None:
        self.rooms_dir = rooms_dir
        self._mapping_config = mapping_config
        self._lock = threading.Lock()
        self.rooms_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------- config

    def list_rooms(self) -> list[dict[str, Any]]:
        rooms = []
        with self._lock:
            for path in sorted(self.rooms_dir.glob("*.json")):
                try:
                    config = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    logger.warning("Skipping unreadable room config %s: %s", path, exc)
                    continue
                room_id = str(config.get("room_id", path.stem))
                state = self._load_state_locked(room_id)
                config["scans_done"] = sorted(state.get("scans", {}).keys())
                config["estimated_dims"] = state.get("estimated_dims")
                rooms.append(config)
        return rooms

    def create_room(self, body: dict[str, Any]) -> dict[str, Any]:
        name = str(body.get("name", "")).strip()
        if not name:
            raise ValueError("Room name must not be empty")
        room_id = _safe_room_id(name)
        path = self.rooms_dir / f"{room_id}.json"
        if path.exists():
            raise ValueError(f"Room already exists: {room_id}")

        defaults = self._mapping_config
        config = RoomConfig(
            room_id=room_id,
            name=name,
            width_m=_float_or_none(body.get("width_m")),
            length_m=_float_or_none(body.get("length_m")),
            camera_height_m=float(
                body.get("camera_height_m", defaults.get("camera_height_m", 0.3))
            ),
            hfov_deg=float(body.get("hfov_deg", defaults.get("hfov_deg", 102.0))),
            vfov_deg=float(body.get("vfov_deg", defaults.get("vfov_deg", 67.0))),
        )
        with self._lock:
            path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
        logger.info("Created room %s (%s)", room_id, name)
        return asdict(config)

    def load_room(self, room_id: str) -> RoomConfig | None:
        path = self.rooms_dir / f"{_safe_room_id(room_id)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable room config %s: %s", path, exc)
            return None
        known = {f for f in RoomConfig.__dataclass_fields__}
        return RoomConfig(**{k: v for k, v in data.items() if k in known})

    # ------------------------------------------------------------ state

    def room_dir(self, room_id: str) -> Path:
        directory = self.rooms_dir / _safe_room_id(room_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def load_state(self, room_id: str) -> dict[str, Any]:
        with self._lock:
            return self._load_state_locked(room_id)

    def _load_state_locked(self, room_id: str) -> dict[str, Any]:
        path = self.rooms_dir / _safe_room_id(room_id) / "state.json"
        if not path.exists():
            return {"scans": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"scans": {}}

    def save_state(self, room_id: str, state: dict[str, Any]) -> None:
        with self._lock:
            path = self.room_dir(room_id) / "state.json"
            path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def update_scan_state(
        self,
        room_id: str,
        direction: str,
        scan_info: dict[str, Any],
        body_offset_m: float,
    ) -> dict[str, Any]:
        """Record a finished directional scan and refresh estimated dims."""
        state = self.load_state(room_id)
        scans = state.setdefault("scans", {})
        scans[direction] = scan_info

        def axis_estimate(dir_a: str, dir_b: str) -> float | None:
            d_values = [
                scans[d].get("d_max_m")
                for d in (dir_a, dir_b)
                if d in scans and scans[d].get("d_max_m")
            ]
            if not d_values:
                return None
            return round(max(d_values) + body_offset_m, 2)

        state["estimated_dims"] = {
            "length_m": axis_estimate("north", "south"),
            "width_m": axis_estimate("east", "west"),
        }
        self.save_state(room_id, state)
        return state


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
