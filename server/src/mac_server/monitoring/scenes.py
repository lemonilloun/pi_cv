"""Scene configuration storage: named camera placements with frozen anchors.

A scene is a JSON file data/monitoring/scenes/<scene_id>.json. Anchors are
furniture detections explicitly frozen by the user while the room is in a
known-good state (deterministic, unlike live furniture detections that
flicker under occlusion — e.g. a person sitting on the couch).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any
from pathlib import Path


logger = logging.getLogger(__name__)


def _safe_scene_id(value: str) -> str:
    safe_chars = []
    for char in value.strip().lower():
        if char.isalnum():
            safe_chars.append(char)
        elif char in {" ", "-", "_"}:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_") or "scene"


class SceneStore:
    def __init__(self, scenes_dir: Path) -> None:
        self.scenes_dir = scenes_dir
        self._lock = threading.Lock()
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

    def list_scenes(self) -> list[dict[str, Any]]:
        scenes = []
        with self._lock:
            for path in sorted(self.scenes_dir.glob("*.json")):
                try:
                    scenes.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, ValueError) as exc:
                    logger.warning("Skipping unreadable scene %s: %s", path, exc)
        return scenes

    def create_scene(self, body: dict[str, Any]) -> dict[str, Any]:
        name = str(body.get("name", "")).strip()
        if not name:
            raise ValueError("Scene name must not be empty")
        scene_id = _safe_scene_id(name)
        path = self.scenes_dir / f"{scene_id}.json"
        if path.exists():
            raise ValueError(f"Scene already exists: {scene_id}")
        scene = {
            "scene_id": scene_id,
            "name": name,
            "notes": str(body.get("notes", "")),
            "anchors": [],
            "created_at": time.time(),
        }
        with self._lock:
            path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
        logger.info("Created scene %s (%s)", scene_id, name)
        return scene

    def load_scene(self, scene_id: str) -> dict[str, Any] | None:
        path = self.scenes_dir / f"{_safe_scene_id(scene_id)}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable scene %s: %s", path, exc)
            return None

    def set_anchors(self, scene_id: str, objects: list[dict[str, Any]], anchor_classes: set[str]) -> list[dict[str, Any]]:
        """Freeze current enriched objects of anchor classes into the scene."""
        scene = self.load_scene(scene_id)
        if scene is None:
            raise ValueError(f"Unknown scene: {scene_id}")

        anchors: list[dict[str, Any]] = []
        counters: dict[str, int] = {}
        for obj in objects:
            class_name = str(obj.get("class", ""))
            if class_name not in anchor_classes:
                continue
            counters[class_name] = counters.get(class_name, 0) + 1
            anchors.append(
                {
                    "anchor_id": f"{class_name.replace(' ', '_')}_{counters[class_name]}",
                    "class": class_name,
                    "bbox_xyxy": obj.get("bbox_xyxy"),
                    "depth_m": obj.get("depth_median_m"),
                    "lateral_m": obj.get("lateral_m"),
                    "forward_m": obj.get("forward_m"),
                    "frozen_at": time.time(),
                }
            )

        scene["anchors"] = anchors
        path = self.scenes_dir / f"{_safe_scene_id(scene_id)}.json"
        with self._lock:
            path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
        logger.info("Froze %d anchors for scene %s", len(anchors), scene_id)
        return anchors
