"""Unit tests for cv_worker._wall_points: the live wall/obstacle scatter
that backs the IMU debug tab's map (docs/scene3d.md IMU section). Reuses
test_mapping's synthetic room generator since this is the same geometry
(depth_to_points + filter_height_band), just wired up per-frame instead of
accumulated across a scan.

Run: python3 -m unittest discover server/tests
"""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
