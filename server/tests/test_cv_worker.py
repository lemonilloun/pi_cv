"""Unit tests for cv_worker._wall_points: the live wall/obstacle scatter
that backs the IMU debug tab's map (docs/scene3d.md IMU section). Reuses
test_mapping's synthetic room generator since this is the same geometry
(depth_to_points + filter_height_band), just wired up per-frame instead of
accumulated across a scan.

Run: python3 -m unittest discover server/tests
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from test_mapping import make_synthetic_room_depth  # noqa: E402


@unittest.skipUnless(HAS_NUMPY, "numpy is required for cv_worker tests")
class WallPointsTest(unittest.TestCase):
    def test_empty_without_mapping_config(self) -> None:
        from mac_server.cv_worker import _wall_points
        from mac_server.mapping.geometry import CameraIntrinsics

        intr = CameraIntrinsics.from_fov(320, 240, hfov_deg=102.0, vfov_deg=67.0)
        depth = np.full((240, 320), 3.0, dtype=np.float32)
        self.assertEqual(_wall_points(depth, intr, np, {}), [])

    def test_isolates_the_wall_band_from_floor_and_ceiling(self) -> None:
        from mac_server.cv_worker import _wall_points
        from mac_server.mapping.geometry import CameraIntrinsics

        intr = CameraIntrinsics.from_fov(320, 240, hfov_deg=102.0, vfov_deg=67.0)
        wall_distance = 3.0
        depth = make_synthetic_room_depth(
            intr, wall_distance=wall_distance, camera_height=0.3, ceiling_height=2.7
        )
        mapping_config = {
            "camera_height_m": 0.3,
            "height_band_m": [0.1, 2.0],
            "wall_points_stride": 8,
            "edge_crop_frac": 0.1,
            "max_depth_use_m": 8.0,
        }
        points = _wall_points(depth, intr, np, mapping_config)
        self.assertGreater(len(points), 0)
        for lateral_m, forward_m in points:
            # Every kept point should be the wall, not the floor/ceiling
            # planes the height-band filter is supposed to exclude.
            self.assertAlmostEqual(forward_m, wall_distance, delta=0.05)

    def test_stride_controls_point_count(self) -> None:
        from mac_server.cv_worker import _wall_points
        from mac_server.mapping.geometry import CameraIntrinsics

        intr = CameraIntrinsics.from_fov(320, 240, hfov_deg=102.0, vfov_deg=67.0)
        depth = make_synthetic_room_depth(intr, wall_distance=3.0, camera_height=0.3)
        base_config = {
            "camera_height_m": 0.3, "height_band_m": [0.1, 2.0],
            "edge_crop_frac": 0.1, "max_depth_use_m": 8.0,
        }
        coarse = _wall_points(depth, intr, np, {**base_config, "wall_points_stride": 32})
        fine = _wall_points(depth, intr, np, {**base_config, "wall_points_stride": 4})
        self.assertGreater(len(fine), len(coarse))


def _world_to_local(world_points, pose_x: float, pose_y: float, yaw_rad: float):
    """Inverse of cv_worker's own local->world stamping: what the camera at
    (pose_x, pose_y, yaw_rad) would see of these world-fixed points, in its
    own [lateral, forward] axes. Used only to build synthetic test scans —
    the ground truth _update_live_map's pose accumulation is checked against."""
    shifted = world_points - np.array([pose_x, pose_y])
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    lateral = cos_y * shifted[:, 0] + sin_y * shifted[:, 1]
    forward = -sin_y * shifted[:, 0] + cos_y * shifted[:, 1]
    return np.stack([lateral, forward], axis=1)


def _make_worker():
    from mac_server.cv_worker import ServerCvWorker
    from mac_server.preview import FrameStoreHub, LatestFrameStore

    return ServerCvWorker(
        pi_store=LatestFrameStore(),
        frame_hub=FrameStoreHub(),
        config={},
        repo_root=REPO_ROOT,
        mapping_config={"grid_resolution_m": 0.05, "live_map_min_points": 15},
    )


@unittest.skipUnless(HAS_NUMPY and HAS_SCIPY, "numpy and scipy are required")
class LiveMapPoseTest(unittest.TestCase):
    """_update_live_map composes each tick's ICP result into a running
    world pose — this is the part test_scan_match.py can't cover on its
    own (it only checks icp_2d in isolation), and the part most at risk of
    a sign-convention mistake (rotate-then-add vs add-then-rotate, which
    axis is "forward"). Ground truth here is a synthetic world with a
    known true trajectory; each tick's scan is built by projecting that
    world into the camera's local frame at the true pose, mirroring
    test_scan_match's apply_true_motion but through the full worker."""

    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        # A wide, dense "room" so every tested pose still sees a wall.
        lateral = rng.uniform(-3.0, 3.0, 800)
        forward = rng.uniform(-3.0, 8.0, 800)
        self.world_points = np.stack([lateral, forward], axis=1)

    def test_straight_line_accumulates_forward_distance(self) -> None:
        worker = _make_worker()
        step = 0.25
        for i in range(6):  # tick 0 has no previous scan; 5 real steps follow
            pose_y = i * step
            scan = _world_to_local(self.world_points, 0.0, pose_y, 0.0)
            live_map = worker._update_live_map(scan, np)

        self.assertAlmostEqual(live_map["pose"]["x"], 0.0, delta=0.03)
        self.assertAlmostEqual(live_map["pose"]["y"], 5 * step, delta=0.05)
        self.assertAlmostEqual(live_map["pose"]["yaw_rad"], 0.0, delta=0.02)
        self.assertEqual(len(live_map["trail"]), 6)
        self.assertGreater(len(live_map["explored"]), 0)

    def test_turning_in_place_accumulates_yaw(self) -> None:
        worker = _make_worker()
        step_deg = 5.0
        for i in range(5):  # tick 0, then 4 turns of step_deg each
            yaw = math.radians(i * step_deg)
            scan = _world_to_local(self.world_points, 0.0, 0.0, yaw)
            live_map = worker._update_live_map(scan, np)

        self.assertAlmostEqual(math.degrees(live_map["pose"]["yaw_rad"]), 4 * step_deg, delta=1.0)
        self.assertAlmostEqual(live_map["pose"]["x"], 0.0, delta=0.03)
        self.assertAlmostEqual(live_map["pose"]["y"], 0.0, delta=0.03)

    def test_reset_clears_accumulated_state(self) -> None:
        worker = _make_worker()
        for i in range(3):
            scan = _world_to_local(self.world_points, 0.0, i * 0.2, 0.0)
            worker._update_live_map(scan, np)
        self.assertGreater(len(worker._trail), 0)

        worker.reset_live_map()
        self.assertTrue(worker._reset_live_map.is_set())
        # Mirrors what _run()'s loop does on seeing the flag set.
        worker._pose = {"x": 0.0, "y": 0.0, "yaw_rad": 0.0}
        worker._prev_scan_points = None
        worker._trail.clear()
        worker._explored_cells.clear()
        worker._reset_live_map.clear()

        first_scan = _world_to_local(self.world_points, 0.0, 0.0, 0.0)
        live_map = worker._update_live_map(first_scan, np)
        self.assertEqual(live_map["pose"], {"x": 0.0, "y": 0.0, "yaw_rad": 0.0})
        self.assertEqual(len(live_map["trail"]), 1)


if __name__ == "__main__":
    unittest.main()
