"""Reading Pi scene sessions and locating derived artifacts (the data
contract between the Pi recorder and the Mac pipeline — docs/scene3d.md)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Keyframe:
    index: int
    dir: Path

    @property
    def rgb_path(self) -> Path:
        return self.dir / "rgb.jpg"

    @property
    def masks_path(self) -> Path:
        return self.dir / "masks.png"

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    def meta(self) -> dict[str, Any]:
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def rgb_bgr(self):
        import cv2

        image = cv2.imread(str(self.rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Unreadable keyframe image: {self.rgb_path}")
        return image

    def masks(self):
        import cv2

        masks = cv2.imread(str(self.masks_path), cv2.IMREAD_UNCHANGED)
        if masks is None:
            raise IOError(f"Unreadable masks: {self.masks_path}")
        return masks


class SceneSession:
    """One recorded session directory + its derived/ artifact tree."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.session_id = self.root.name
        self.keyframes_dir = self.root / "keyframes"
        self.derived = self.root / "derived"

    # ------------------------------------------------------------- input

    def exists(self) -> bool:
        return self.keyframes_dir.is_dir()

    def keyframes(self) -> list[Keyframe]:
        frames = []
        for entry in sorted(self.keyframes_dir.iterdir()):
            if entry.is_dir() and (entry / "rgb.jpg").exists():
                try:
                    frames.append(Keyframe(index=int(entry.name), dir=entry))
                except ValueError:
                    continue
        return frames

    def session_meta(self) -> dict[str, Any]:
        path = self.root / "session_meta.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def intrinsics(self, fallback_hfov_deg: float = 102.0) -> dict[str, Any]:
        """Calibrated intrinsics, or a FOV-based estimate when the session
        was recorded before calibration (flagged `estimated: true` — good
        enough to exercise the pipeline, not for metric accuracy)."""
        path = self.root / "intrinsics.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        sample = self.keyframes()
        if not sample:
            raise IOError(f"Session has no keyframes: {self.root}")
        import cv2

        image = cv2.imread(str(sample[0].rgb_path))
        h, w = image.shape[:2]
        fx = (w / 2.0) / math.tan(math.radians(fallback_hfov_deg / 2.0))
        return {
            "width": w,
            "height": h,
            "fx": fx,
            "fy": fx,
            "cx": w / 2.0,
            "cy": h / 2.0,
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "estimated": True,
            "fallback_hfov_deg": fallback_hfov_deg,
        }

    # ----------------------------------------------------------- derived

    def depth_dir(self) -> Path:
        return self.derived / "depth"

    def depth_path(self, index: int) -> Path:
        return self.depth_dir() / f"{index:06d}.npy"

    def colmap_dir(self) -> Path:
        return self.derived / "colmap"

    def poses_path(self) -> Path:
        return self.derived / "poses.json"

    def scale_report_path(self) -> Path:
        return self.derived / "scale_report.json"

    def mesh_path(self) -> Path:
        return self.derived / "room_mesh.ply"

    def plan_path(self) -> Path:
        return self.derived / "floor_plan.png"

    def objects_path(self) -> Path:
        return self.derived / "objects.json"

    def objects_pcd_dir(self) -> Path:
        return self.derived / "objects_pcd"

    def graph_path(self) -> Path:
        return self.derived / "scene_graph.json"

    def plan_labeled_path(self) -> Path:
        return self.derived / "floor_plan_labeled.png"

    def state_path(self) -> Path:
        return self.derived / "pipeline_state.json"

    def load_poses(self) -> dict[int, Any]:
        """poses.json -> {keyframe_index: 4x4 world_from_cam (metric)}."""
        import numpy as np

        data = json.loads(self.poses_path().read_text(encoding="utf-8"))
        return {
            int(k): np.asarray(v, dtype=np.float64)
            for k, v in data["world_from_cam"].items()
        }


def list_sessions(sessions_dir: Path) -> list[dict[str, Any]]:
    """Session summaries for the panel, newest first."""
    out = []
    if not sessions_dir.is_dir():
        return out
    for entry in sorted(sessions_dir.iterdir(), reverse=True):
        if not entry.is_dir() or not (entry / "keyframes").is_dir():
            continue
        session = SceneSession(entry)
        state = {}
        if session.state_path().exists():
            try:
                state = json.loads(session.state_path().read_text(encoding="utf-8"))
            except (OSError, ValueError):
                state = {}
        out.append(
            {
                "session_id": session.session_id,
                "keyframes": len(session.keyframes()),
                "meta": session.session_meta(),
                "calibrated": (entry / "intrinsics.json").exists(),
                "steps": state.get("steps", {}),
            }
        )
    return out
