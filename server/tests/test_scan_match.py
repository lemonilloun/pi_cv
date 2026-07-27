"""Unit tests for mac_server.mapping.scan_match (2D ICP, the live map's
frame-to-frame ego-motion source — see docs/scene3d.md's IMU section for
why the IMU alone can't do this job yet).

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


def make_wall_scan(rng, n: int = 200, lateral_range=(-2.0, 2.0), forward_range=(0.5, 4.0)):
    lateral = rng.uniform(*lateral_range, n)
    forward = rng.uniform(*forward_range, n)
    return np.stack([lateral, forward], axis=1)


def rotation_2d(angle_rad: float):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def apply_true_motion(prev_points, rotation, translation):
    """Build the `curr` scan such that (by construction) rotation @ curr_i +
    translation == prev_i for every point — i.e. exactly the relationship
    icp_2d is meant to recover, so this is the ground truth its output is
    checked against."""
    return (prev_points - translation) @ rotation  # rotation is orthogonal: R^-1 == R^T


@unittest.skipUnless(HAS_NUMPY and HAS_SCIPY, "numpy and scipy are required")
class Icp2dTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def test_recovers_pure_translation(self) -> None:
        from mac_server.mapping.scan_match import icp_2d

        prev = make_wall_scan(self.rng)
        translation = np.array([0.3, 0.5])
        curr = apply_true_motion(prev, np.eye(2), translation)

        result = icp_2d(prev, curr)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result["dx"], 0.3, delta=0.01)
        self.assertAlmostEqual(result["dy"], 0.5, delta=0.01)
        self.assertAlmostEqual(result["dyaw_rad"], 0.0, delta=0.01)
        self.assertGreater(result["fitness"], 0.9)

    def test_recovers_pure_rotation(self) -> None:
        from mac_server.mapping.scan_match import icp_2d

        prev = make_wall_scan(self.rng)
        angle = math.radians(10.0)
        curr = apply_true_motion(prev, rotation_2d(angle), np.zeros(2))

        result = icp_2d(prev, curr, initial_yaw_rad=angle)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(math.degrees(result["dyaw_rad"]), 10.0, delta=0.5)
        self.assertAlmostEqual(result["dx"], 0.0, delta=0.02)
        self.assertAlmostEqual(result["dy"], 0.0, delta=0.02)

    def test_recovers_combined_motion(self) -> None:
        from mac_server.mapping.scan_match import icp_2d

        prev = make_wall_scan(self.rng, n=400)
        angle = math.radians(-6.0)
        translation = np.array([-0.15, 0.4])
        curr = apply_true_motion(prev, rotation_2d(angle), translation)

        result = icp_2d(prev, curr, initial_yaw_rad=angle)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result["dx"], -0.15, delta=0.02)
        self.assertAlmostEqual(result["dy"], 0.4, delta=0.02)
        self.assertAlmostEqual(math.degrees(result["dyaw_rad"]), -6.0, delta=0.5)

    def test_tolerates_partial_overlap_as_the_robot_explores_new_area(self) -> None:
        from mac_server.mapping.scan_match import icp_2d

        prev = make_wall_scan(self.rng, n=300)
        translation = np.array([0.0, 0.3])
        curr_matched = apply_true_motion(prev, np.eye(2), translation)
        # A quarter of curr is genuinely new geometry the previous scan
        # never saw (newly-revealed wall as the robot moves forward) — the
        # correspondence search must reject these, not be dragged by them.
        # (Consecutive real ticks overlap far more than this; this checks
        # the correspondence rejection itself, not a realistic worst case.)
        new_geometry = make_wall_scan(self.rng, n=75, forward_range=(4.0, 6.0))
        curr = np.concatenate([curr_matched[:225], new_geometry], axis=0)

        # A cold identity guess isn't enough with a quarter of the scan
        # being outliers (nearest-neighbor correspondence has no way to
        # know which quarter) — this is exactly why cv_worker passes the
        # previous tick's own estimate as a warm start every frame.
        result = icp_2d(prev, curr, initial_translation=(0.0, 0.25), min_fitness=0.3)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result["dy"], 0.3, delta=0.03)

    def test_none_when_scans_barely_overlap(self) -> None:
        from mac_server.mapping.scan_match import icp_2d

        prev = make_wall_scan(self.rng, forward_range=(0.5, 2.0))
        curr = make_wall_scan(self.rng, forward_range=(8.0, 10.0))  # disjoint region
        self.assertIsNone(icp_2d(prev, curr))

    def test_none_with_too_few_points(self) -> None:
        from mac_server.mapping.scan_match import icp_2d

        prev = make_wall_scan(self.rng, n=5)
        curr = make_wall_scan(self.rng, n=5)
        self.assertIsNone(icp_2d(prev, curr))


if __name__ == "__main__":
    unittest.main()
