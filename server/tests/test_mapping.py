"""Unit tests for the mapping math core (geometry + grid).

Run: python3 -m unittest discover server/tests
"""

from __future__ import annotations

import math
import sys
import tempfile
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


def make_synthetic_room_depth(
    intrinsics,
    wall_distance: float,
    camera_height: float,
    ceiling_height: float = 2.7,
) -> "np.ndarray":
    """Analytic depth map of an infinitely wide room slice: a flat wall at
    `wall_distance` ahead, a floor plane below the camera, and a ceiling
    plane above. Depth here is z-depth (distance along the optical axis),
    matching what DA2 metric models output.
    """
    height, width = intrinsics.height, intrinsics.width
    us = np.arange(width, dtype=np.float32)
    vs = np.arange(height, dtype=np.float32)
    _, vv = np.meshgrid(us, vs)

    # Ray direction per pixel row: y_cam = (v - cy) / fy * z.
    # Floor plane: y_cam = camera_height  -> z_floor = camera_height * fy / (v - cy) for v > cy
    # Ceiling:     y_cam = -(ceiling_height - camera_height)
    depth = np.full((height, width), wall_distance, dtype=np.float32)

    dy = vv - intrinsics.cy
    with np.errstate(divide="ignore", invalid="ignore"):
        z_floor = camera_height * intrinsics.fy / dy
        z_ceiling = -(ceiling_height - camera_height) * intrinsics.fy / dy

    below = dy > 0
    depth[below] = np.minimum(depth[below], z_floor[below])
    above = dy < 0
    ceiling_hit = above & (z_ceiling < depth)
    depth[ceiling_hit] = z_ceiling[ceiling_hit]
    return depth


@unittest.skipUnless(HAS_NUMPY, "numpy is required for mapping tests")
class IntrinsicsTest(unittest.TestCase):
    def test_from_fov_camera_module_3_wide(self) -> None:
        from mac_server.mapping.geometry import CameraIntrinsics

        intr = CameraIntrinsics.from_fov(1280, 720, hfov_deg=102.0, vfov_deg=67.0)
        self.assertAlmostEqual(intr.fx, 518.0, delta=2.0)
        self.assertAlmostEqual(intr.fy, 543.9, delta=2.0)
        self.assertEqual(intr.cx, 640.0)
        self.assertEqual(intr.cy, 360.0)

    def test_center_pixel_projects_on_axis(self) -> None:
        from mac_server.mapping.geometry import CameraIntrinsics, depth_to_points

        intr = CameraIntrinsics.from_fov(1280, 720, 102.0, 67.0)
        depth = np.full((720, 1280), 2.0, dtype=np.float32)
        points = depth_to_points(depth, intr, stride=1, edge_crop_frac=0.0)
        center = points[np.argmin(np.abs(points[:, 0]) + np.abs(points[:, 1]))]
        self.assertAlmostEqual(center[0], 0.0, delta=0.01)
        self.assertAlmostEqual(center[1], 0.0, delta=0.01)
        self.assertAlmostEqual(center[2], 2.0, delta=0.001)

    def test_scaled_intrinsics(self) -> None:
        from mac_server.mapping.geometry import CameraIntrinsics

        intr = CameraIntrinsics.from_fov(1280, 720, 102.0, 67.0)
        scaled = intr.scaled(640, 360)
        self.assertAlmostEqual(scaled.fx, intr.fx / 2)
        self.assertAlmostEqual(scaled.cy, intr.cy / 2)


@unittest.skipUnless(HAS_NUMPY, "numpy is required for mapping tests")
class BboxBearingTest(unittest.TestCase):
    def setUp(self) -> None:
        from mac_server.mapping.geometry import CameraIntrinsics

        self.intr = CameraIntrinsics.from_fov(1280, 720, hfov_deg=102.0, vfov_deg=67.0)

    def test_centered_bbox_has_zero_bearing(self) -> None:
        from mac_server.mapping.geometry import bbox_bearing_deg

        bbox = [590.0, 300.0, 690.0, 500.0]  # center x = 640 = cx
        self.assertAlmostEqual(bbox_bearing_deg(bbox, self.intr), 0.0, delta=0.01)

    def test_edge_bbox_approaches_half_hfov(self) -> None:
        from mac_server.mapping.geometry import bbox_bearing_deg

        left_edge = [0.0, 300.0, 0.0, 500.0]
        right_edge = [1280.0, 300.0, 1280.0, 500.0]
        self.assertAlmostEqual(bbox_bearing_deg(left_edge, self.intr), -51.0, delta=0.5)
        self.assertAlmostEqual(bbox_bearing_deg(right_edge, self.intr), 51.0, delta=0.5)

    def test_camera_xy_uses_tan_not_sin(self) -> None:
        """A polar (sin/cos) plot would diverge sharply from the correct
        tan-based lateral offset at wide angles — assert the real formula."""
        from mac_server.mapping.geometry import bbox_bearing_rad, bbox_camera_xy

        bbox = [1150.0, 300.0, 1280.0, 500.0]  # near the right edge
        forward_m = 3.0
        lateral, forward = bbox_camera_xy(bbox, forward_m, self.intr)

        bearing = bbox_bearing_rad(bbox, self.intr)
        expected_lateral = math.tan(bearing) * forward_m
        wrong_polar_lateral = math.sin(bearing) * forward_m

        self.assertAlmostEqual(lateral, expected_lateral, delta=1e-6)
        self.assertEqual(forward, forward_m)
        # The two formulas must disagree materially at this wide angle —
        # otherwise this test wouldn't actually catch the sin/cos regression.
        self.assertGreater(abs(lateral - wrong_polar_lateral), 0.3)

    def test_camera_xy_consistent_with_depth_to_points(self) -> None:
        """Same pixel column, same depth: bbox_camera_xy must agree with the
        already-tested full-frame projection depth_to_points."""
        from mac_server.mapping.geometry import bbox_camera_xy, depth_to_points

        wall_distance = 3.0
        depth = np.full((720, 1280), wall_distance, dtype=np.float32)
        points = depth_to_points(depth, self.intr, stride=1, edge_crop_frac=0.0)

        # Pick a point roughly 100px right of center at row = cy.
        target_u, target_v = 740, 360
        row = points[
            (np.abs(points[:, 2] - wall_distance) < 1e-3)
            & (np.abs(points[:, 0] - (target_u - self.intr.cx) / self.intr.fx * wall_distance) < 1e-2)
        ]
        self.assertTrue(len(row) > 0)
        expected_lateral = float(row[0, 0])

        bbox = [float(target_u), float(target_v), float(target_u), float(target_v)]
        lateral, _ = bbox_camera_xy(bbox, wall_distance, self.intr)
        self.assertAlmostEqual(lateral, expected_lateral, delta=1e-2)


@unittest.skipUnless(HAS_NUMPY, "numpy is required for mapping tests")
class SyntheticRoomTest(unittest.TestCase):
    """Camera at the south side of a 4 m wide x 5 m long room, facing north,
    3 m from the north wall, mounted 0.3 m above the floor."""

    def setUp(self) -> None:
        from mac_server.mapping.geometry import CameraIntrinsics

        self.intr = CameraIntrinsics.from_fov(640, 360, 102.0, 67.0)
        self.camera_height = 0.3
        self.wall_distance = 3.0
        self.depth = make_synthetic_room_depth(
            self.intr, self.wall_distance, self.camera_height
        )

    def test_wall_distance_estimate(self) -> None:
        from mac_server.mapping.geometry import estimate_wall_distance

        estimate = estimate_wall_distance(self.depth)
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate, 3.0, delta=0.05)

    def test_height_band_removes_floor_and_keeps_wall(self) -> None:
        from mac_server.mapping.geometry import depth_to_points, filter_height_band

        points = depth_to_points(self.depth, self.intr, stride=2, edge_crop_frac=0.1)
        xz, heights = filter_height_band(points, self.camera_height, band=(0.1, 2.0))

        self.assertGreater(len(xz), 0)
        # Every kept point is inside the band.
        self.assertTrue(np.all(heights >= 0.1))
        self.assertTrue(np.all(heights <= 2.0))
        # Points in the band belong to the wall (z == wall_distance), not the
        # floor: floor points have height ~0, ceiling points > 2 m.
        self.assertTrue(np.all(np.abs(xz[:, 1] - self.wall_distance) < 0.01))

    def test_wall_lands_at_room_north_edge(self) -> None:
        from mac_server.mapping.geometry import (
            camera_to_room,
            depth_to_points,
            filter_height_band,
        )
        from mac_server.mapping.grid import OccupancyGrid

        points = depth_to_points(self.depth, self.intr, stride=2, edge_crop_frac=0.1)
        xz, heights = filter_height_band(points, self.camera_height)
        room_xy = camera_to_room(
            xz,
            direction="north",
            wall_distance=self.wall_distance,
            lateral_offset=2.0,
            room_w=4.0,
            room_l=5.0,
        )

        grid = OccupancyGrid(width_m=4.0, length_m=5.0, resolution_m=0.05)
        landed = grid.accumulate(room_xy, heights)
        self.assertGreater(landed, 0)

        occupied_rows = np.where(grid.occupied_mask(min_hits=1).any(axis=1))[0]
        # Wall cells must lie within 2 cells (0.1 m) of y = 0 (north edge).
        self.assertLessEqual(occupied_rows.max(), 2)

    def test_direction_consistency(self) -> None:
        """The same physical obstacle observed from north- and east-facing
        scans must land in the same room cell."""
        from mac_server.mapping.geometry import camera_to_room

        # Obstacle at room (x=1.0, y=2.0) in a 4x5 room.
        # North scan: camera on the midline x=2.0 at wall distance d=3.0
        #   (camera at y=3.0): X_lat = 1.0 - 2.0 = -1.0, Z_fwd = 3.0 - 2.0 = 1.0
        north_xy = camera_to_room(
            np.array([[-1.0, 1.0]], dtype=np.float32),
            direction="north",
            wall_distance=3.0,
            lateral_offset=2.0,
            room_w=4.0,
            room_l=5.0,
        )
        # East scan: camera on midline y=2.5 at wall distance d=2.5
        #   (camera at x = W - d = 1.5): Z_fwd = 1.0 - 1.5 -> negative...
        #   pick d=3.0 (camera at x=1.0): Z_fwd = 1.0 - 1.0 = 0 -> on camera.
        #   Use d=3.5 (camera at x=0.5): Z_fwd = 0.5, X_lat = 2.0 - 2.5 = -0.5.
        east_xy = camera_to_room(
            np.array([[-0.5, 0.5]], dtype=np.float32),
            direction="east",
            wall_distance=3.5,
            lateral_offset=2.5,
            room_w=4.0,
            room_l=5.0,
        )
        np.testing.assert_allclose(north_xy[0], [1.0, 2.0], atol=1e-5)
        np.testing.assert_allclose(east_xy[0], [1.0, 2.0], atol=1e-5)


@unittest.skipUnless(HAS_NUMPY, "numpy is required for mapping tests")
class GridTest(unittest.TestCase):
    def test_accumulate_and_masks(self) -> None:
        from mac_server.mapping.grid import OccupancyGrid

        grid = OccupancyGrid(width_m=2.0, length_m=2.0, resolution_m=0.1)
        xy = np.array([[0.55, 0.55]] * 5, dtype=np.float32)
        heights = np.array([1.8] * 5, dtype=np.float32)
        landed = grid.accumulate(xy, heights)

        self.assertEqual(landed, 5)
        self.assertEqual(grid.hits[5, 5], 5)
        self.assertTrue(grid.occupied_mask(min_hits=3)[5, 5])
        self.assertTrue(grid.wall_mask(min_wall_height=1.6)[5, 5])
        self.assertFalse(grid.occupied_mask(min_hits=3)[0, 0])

    def test_out_of_bounds_points_ignored(self) -> None:
        from mac_server.mapping.grid import OccupancyGrid

        grid = OccupancyGrid(width_m=1.0, length_m=1.0, resolution_m=0.1)
        xy = np.array([[-0.5, 0.5], [0.5, 5.0], [0.5, 0.5]], dtype=np.float32)
        heights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        landed = grid.accumulate(xy, heights)
        self.assertEqual(landed, 1)

    def test_merge_and_npz_roundtrip(self) -> None:
        from mac_server.mapping.grid import OccupancyGrid

        a = OccupancyGrid(width_m=2.0, length_m=2.0, resolution_m=0.1)
        b = OccupancyGrid(width_m=2.0, length_m=2.0, resolution_m=0.1)
        xy = np.array([[1.0, 1.0]], dtype=np.float32)
        a.accumulate(xy, np.array([0.5], dtype=np.float32))
        b.accumulate(xy, np.array([1.9], dtype=np.float32))
        a.merge(b)

        self.assertEqual(a.hits[10, 10], 2)
        self.assertAlmostEqual(float(a.height_max[10, 10]), 1.9, places=5)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.npz"
            a.save_npz(path)
            loaded = OccupancyGrid.load_npz(path)
            np.testing.assert_array_equal(loaded.hits, a.hits)
            np.testing.assert_array_equal(loaded.height_max, a.height_max)
            self.assertEqual(loaded.frames, a.frames)

    def test_render_map_png(self) -> None:
        from mac_server.mapping.grid import OccupancyGrid, render_map_png

        grid = OccupancyGrid(width_m=2.0, length_m=2.0, resolution_m=0.1)
        xy = np.array([[0.5, 0.05]] * 5, dtype=np.float32)
        grid.accumulate(xy, np.array([1.8] * 5, dtype=np.float32))
        png = render_map_png(grid, label="test 2.0x2.0m")
        self.assertGreater(len(png), 100)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
