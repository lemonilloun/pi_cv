"""Unit tests for the calibration sanity gates in pi_client.scene_calibrate.

These exist because `config/scene_intrinsics.json` shipped a structurally wrong
camera for weeks — fx 534.5 at 1536x864 (a 110 deg field) with cy=500 in an
864-tall frame — and nothing caught it. Every solvePnP and every reconstruction
downstream silently inherited the error.

Run from the repo root with client/src on PYTHONPATH:
    python3 -m unittest client.tests.test_scene_calibrate -v
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "client/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import numpy as np  # noqa: E402

from pi_client.scene_calibrate import (  # noqa: E402
    board_skew,
    board_tilt_spread_deg,
    check_calibration_sanity,
)


W, H = 1536, 864
GOOD_FX = W / 2.0 / math.tan(math.radians(75.0 / 2.0))   # ~1001


class SanityGateTests(unittest.TestCase):
    def test_accepts_a_well_conditioned_camera(self):
        self.assertEqual(
            check_calibration_sanity(GOOD_FX, GOOD_FX, W / 2, H / 2, W, H), []
        )

    def test_accepts_the_measured_vggt_figure(self):
        """VGGT refined 75.6 deg on session_20260730_161007, fx->991 at 1536."""
        self.assertEqual(check_calibration_sanity(991.0, 993.0, 768.0, 432.0, W, H), [])

    def test_rejects_the_real_broken_file(self):
        """The actual contents of config/scene_intrinsics.json. Both failures
        must be reported — the field of view AND the principal point — because
        together they identify the cause (wrong capture resolution)."""
        problems = check_calibration_sanity(534.457, 536.296, 732.15, 499.96, W, H)
        self.assertTrue(any("horizontal field" in p for p in problems))
        self.assertTrue(any("cy=" in p for p in problems))
        self.assertGreaterEqual(len(problems), 2)

    def test_rejects_an_off_centre_principal_point_alone(self):
        problems = check_calibration_sanity(GOOD_FX, GOOD_FX, 768.0, 300.0, W, H)
        self.assertEqual(len(problems), 1)
        self.assertIn("cy=", problems[0])

    def test_rejects_asymmetric_focal_lengths(self):
        """Square pixels; a large fx/fy split means the solve is degenerate."""
        problems = check_calibration_sanity(GOOD_FX, GOOD_FX * 1.10, W / 2, H / 2, W, H)
        self.assertTrue(any("differ by" in p for p in problems))

    def test_tolerates_small_realistic_deviations(self):
        """Must not be so tight that a genuinely good calibration fails: a few
        percent off centre and 1% fx/fy asymmetry are normal."""
        self.assertEqual(
            check_calibration_sanity(GOOD_FX, GOOD_FX * 1.008, W / 2 + 30, H / 2 - 20, W, H),
            [],
        )

    def test_non_positive_focal_short_circuits(self):
        problems = check_calibration_sanity(0.0, 0.0, W / 2, H / 2, W, H)
        self.assertEqual(len(problems), 1)
        self.assertIn("did not converge", problems[0])

    def test_expected_hfov_is_a_parameter_not_a_constant(self):
        """A different lens must be checkable without editing the module."""
        wide_fx = W / 2.0 / math.tan(math.radians(110.0 / 2.0))
        self.assertEqual(
            check_calibration_sanity(wide_fx, wide_fx, W / 2, H / 2, W, H,
                                     expected_hfov_deg=110.0),
            [],
        )


class BoardTiltSpreadTests(unittest.TestCase):
    """Focal length is constrained by how much the board TURNS, not by how far
    away it is. This is the gate that would have caught the 2026-08-03 run:
    fx=1098 (69.9 deg) with RMS 0.484, full frame coverage and three distance
    buckets, while a tape measure put the truth at 968 (76.9 deg)."""

    @staticmethod
    def _rvec_tilted(deg: float):
        """A board rotated `deg` about the camera's X axis."""
        return np.array([math.radians(deg), 0.0, 0.0], dtype=np.float64)

    def test_fronto_parallel_capture_has_no_spread(self):
        """A board taped to a wall, shot by a robot that only drives and yaws."""
        rvecs = [self._rvec_tilted(0.0) for _ in range(30)]
        self.assertLess(board_tilt_spread_deg(rvecs), 1.0)

    def test_measures_a_known_spread(self):
        rvecs = [self._rvec_tilted(-20.0), self._rvec_tilted(0.0), self._rvec_tilted(25.0)]
        self.assertAlmostEqual(board_tilt_spread_deg(rvecs), 45.0, delta=0.5)

    def test_empty_is_zero_not_a_crash(self):
        self.assertEqual(board_tilt_spread_deg([]), 0.0)

    def test_sanity_gate_rejects_a_head_on_capture(self):
        problems = check_calibration_sanity(
            GOOD_FX, GOOD_FX, W / 2, H / 2, W, H, tilt_spread_deg=3.0
        )
        self.assertTrue(any("almost head-on" in p for p in problems))
        # The fix must be one the operator can actually perform: the board is
        # glued to the wall by design (the IMU gravity calibration needs it
        # plumb), so the advice has to be about moving the ROBOT.
        self.assertTrue(any("DRIVE THE ROBOT" in p for p in problems))
        self.assertFalse(any("HOLD THE BOARD" in p for p in problems))

    def test_sanity_gate_accepts_a_varied_capture(self):
        self.assertEqual(
            check_calibration_sanity(GOOD_FX, GOOD_FX, W / 2, H / 2, W, H,
                                     tilt_spread_deg=55.0),
            [],
        )

    def test_tilt_is_optional_so_old_callers_still_work(self):
        self.assertEqual(check_calibration_sanity(GOOD_FX, GOOD_FX, W / 2, H / 2, W, H), [])

    def test_the_real_failed_run_is_caught_by_tilt_even_though_fov_squeaks_through(self):
        """fx=1098 is 69.9 deg — only 5 deg off 75, inside the 12 deg FOV gate.
        Nothing else flagged it either. The tilt spread is what catches it."""
        fov_only = check_calibration_sanity(1098.18, 1095.32, 763.53, 403.78, W, H)
        self.assertEqual(fov_only, [])
        with_tilt = check_calibration_sanity(1098.18, 1095.32, 763.53, 403.78, W, H,
                                             tilt_spread_deg=4.0)
        self.assertEqual(len(with_tilt), 1)
        self.assertIn("almost head-on", with_tilt[0])


class BoardSkewTests(unittest.TestCase):
    """Intrinsics-free obliquity, measured from the projected trapezoid.

    This is what the live gauge shows during capture, and it exists because
    obliquity cannot be measured with solvePnP here — that would need the
    very intrinsics being solved for.
    """

    PATTERN = (9, 6)

    @staticmethod
    def _grid(left_len: float, right_len: float, top_len: float = 100.0,
              bottom_len: float = 100.0):
        """A trapezoid with the given edge lengths, as a corner array."""
        cols, rows = 9, 6
        pts = np.zeros((rows, cols, 2), dtype=np.float64)
        for r in range(rows):
            t = r / (rows - 1)
            ly, ry = t * left_len, t * right_len
            for c in range(cols):
                u = c / (cols - 1)
                pts[r, c, 0] = u * (top_len + u * 0 + (bottom_len - top_len) * t)
                pts[r, c, 1] = ly + u * (ry - ly)
        return pts.reshape(-1, 1, 2)

    def test_head_on_board_reads_zero(self):
        h, _v = board_skew(self._grid(100.0, 100.0), self.PATTERN)
        self.assertAlmostEqual(h, 0.0, places=6)

    def test_sign_distinguishes_left_from_right(self):
        from_left, _ = board_skew(self._grid(80.0, 120.0), self.PATTERN)
        from_right, _ = board_skew(self._grid(120.0, 80.0), self.PATTERN)
        self.assertGreater(from_left, 0.0)
        self.assertLess(from_right, 0.0)
        self.assertAlmostEqual(from_left, -from_right, places=6)

    def test_magnitude_grows_with_obliquity(self):
        mild, _ = board_skew(self._grid(95.0, 105.0), self.PATTERN)
        strong, _ = board_skew(self._grid(70.0, 130.0), self.PATTERN)
        self.assertGreater(abs(strong), abs(mild))

    def test_a_wall_board_and_a_driving_robot_can_reach_the_span(self):
        """The whole point: parking off to one side and turning to face the
        board produces plenty of obliquity without touching the board."""
        from pi_client.scene_calibrate import MIN_SKEW_SPAN

        left, _ = board_skew(self._grid(75.0, 125.0), self.PATTERN)
        right, _ = board_skew(self._grid(125.0, 75.0), self.PATTERN)
        self.assertGreaterEqual(left - right, MIN_SKEW_SPAN)


class PinnedFocalResolveTests(unittest.TestCase):
    """End-to-end proof that a head-on capture is RECOVERABLE offline.

    This is the promise that the capture never has to be repeated: project a
    known camera through nearly head-on board poses, confirm the free solve
    drifts badly on focal length (with an essentially perfect RMS — which is
    why nothing caught it), then confirm that pinning fx to an externally
    measured value recovers the principal point and distortion exactly.
    """

    TRUE_FX = 968.0
    TRUE_DIST = [-0.05, 0.12, 0.0, 0.0, -0.03]

    def _synthesise(self):
        import cv2

        camera = np.array([[self.TRUE_FX, 0, W / 2], [0, self.TRUE_FX, H / 2], [0, 0, 1]])
        dist = np.array(self.TRUE_DIST)
        cols, rows, square = 9, 6, 0.024
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square

        rng = np.random.default_rng(0)
        obj_points, img_points = [], []
        for _ in range(30):
            # Tiny angles: a board on a wall, seen by a robot on the floor.
            rvec = 0.02 * rng.standard_normal(3)
            tvec = np.array([0.03 * rng.standard_normal(), 0.02 * rng.standard_normal(),
                             0.5 + 0.25 * rng.random()])
            pts, _ = cv2.projectPoints(objp, rvec, tvec, camera, dist)
            obj_points.append(objp.copy())
            img_points.append(pts.astype(np.float32))
        return obj_points, img_points

    def test_free_solve_drifts_despite_a_perfect_rms(self):
        import cv2

        obj_points, img_points = self._synthesise()
        rms, K, _dist, _r, _t = cv2.calibrateCamera(obj_points, img_points, (W, H), None, None)
        self.assertLess(rms, 0.1)                       # looks flawless
        self.assertGreater(abs(K[0, 0] - self.TRUE_FX) / self.TRUE_FX, 0.15)  # and is not

    def test_pinning_the_focal_length_recovers_everything_else(self):
        import cv2

        obj_points, img_points = self._synthesise()
        guess = np.array([[self.TRUE_FX, 0, W / 2], [0, self.TRUE_FX, H / 2], [0, 0, 1]],
                         dtype=np.float64)
        flags = cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_USE_INTRINSIC_GUESS
        _rms, K, dist, _r, _t = cv2.calibrateCamera(
            obj_points, img_points, (W, H), guess, np.zeros(5), flags=flags
        )
        self.assertAlmostEqual(K[0, 0], self.TRUE_FX, places=3)
        self.assertAlmostEqual(K[0, 2], W / 2, delta=1.0)
        self.assertAlmostEqual(K[1, 2], H / 2, delta=1.0)
        for got, want in zip(dist.reshape(-1)[:5], self.TRUE_DIST):
            self.assertAlmostEqual(float(got), want, places=3)
