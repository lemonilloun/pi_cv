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

    def intrinsics(self, fallback_hfov_deg: float = 75.0) -> dict[str, Any]:
        """As-recorded intrinsics: the Pi's calibration, or a FOV-based
        estimate when the session predates calibration (flagged
        `estimated: true`). This is the *input* to SfM — everything
        downstream of the poses step wants `active_intrinsics()` instead."""
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

    def refined_intrinsics(self) -> dict[str, Any] | None:
        """The camera COLMAP's bundle adjustment converged on, if the poses
        step has run. None before that."""
        path = self.refined_intrinsics_path()
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def active_intrinsics(self, fallback_hfov_deg: float = 75.0) -> dict[str, Any]:
        """The camera every step *after* `poses` must use.

        The poses step hands COLMAP a starting camera and lets bundle
        adjustment refine the focal length whenever the calibration was only
        estimated. The resulting poses are expressed in terms of the
        *refined* camera, so unprojecting depth with the original estimate
        reconstructs a different camera than the one the poses describe —
        the geometry is then internally inconsistent no matter how good the
        depth model is. (Measured on session_20260724_144728: BA converged on
        fx=989 while the 102-degree fallback assumed fx=622, a 59% mismatch,
        and the mesh spanned 12x15 m for a room the camera crossed in 2x4 m.)

        So: refined if we have it, as-recorded otherwise.
        """
        refined = self.refined_intrinsics()
        if refined:
            return refined
        return self.intrinsics(fallback_hfov_deg)

    # ----------------------------------------------------------- derived

    def depth_dir(self) -> Path:
        return self.derived / "depth"

    def depth_path(self, index: int) -> Path:
        return self.depth_dir() / f"{index:06d}.npy"

    def depth_filtered_dir(self) -> Path:
        return self.derived / "depth_filtered"

    def depth_filtered_path(self, index: int) -> Path:
        return self.depth_filtered_dir() / f"{index:06d}.npy"

    def refined_intrinsics_path(self) -> Path:
        return self.derived / "intrinsics_refined.json"

    def occupancy_path(self) -> Path:
        return self.derived / "occupancy.npz"

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

    def imu_up_vector(self, poses: dict[int, Any] | None = None):
        """World-space up from the recorder's IMU, or None when absent.

        The seam for S6. Each keyframe's meta.json may carry
        ``"gravity": [x, y, z]`` — the accelerometer vector in the *camera*
        frame, pointing down. Rotating each one into world space with that
        keyframe's pose and averaging gives a gravity direction that does
        not care how the camera was tilted, which is exactly what the naive
        camera-average estimate cannot do.

        Sessions recorded before the IMU exists simply have no such field
        and this returns None, so the contract is backwards compatible and
        `occupancy.estimate_gravity` falls through to the floor fit.
        """
        import numpy as np

        if poses is None:
            if not self.poses_path().exists():
                return None
            poses = self.load_poses()

        downs = []
        for kf in self.keyframes():
            if kf.index not in poses:
                continue
            try:
                gravity = kf.meta().get("gravity")
            except (OSError, ValueError):
                continue
            if not gravity or len(gravity) != 3:
                continue
            vec = np.asarray(gravity, dtype=np.float64)
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                continue
            downs.append(np.asarray(poses[kf.index])[:3, :3] @ (vec / norm))

        if not downs:
            return None
        mean = np.mean(downs, axis=0)
        norm = np.linalg.norm(mean)
        if norm < 1e-6:
            return None
        return -(mean / norm)

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
