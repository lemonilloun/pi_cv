"""Unit tests for the yaw-validation math in pi_client.imu_calibrate.

The physical claim under test: rotating the rig about gravity changes the
camera's PnP yaw and the IMU's fused yaw by the SAME amount, so a fit of one
against the other has gain +/-1. Everything here is synthetic geometry — the
live phase needs a board on a wall.

Run from the repo root with client/src on PYTHONPATH:
    python3 -m unittest client.tests.test_imu_yaw_check -v
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(REPO_ROOT / "client/src"),):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import numpy as np  # noqa: E402

from pi_client.imu_calibrate import (  # noqa: E402
    camera_yaw_about_up_deg,
    check_intrinsics_sanity,
    corner_pitch_px,
    fit_yaw_relation,
    unwrap_deg,
)


def board_rvec_for_yaw(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """rvec of a plumb wall board seen by a camera yawed by `yaw_deg`.

    Built the way the real thing arises: start from a board squarely facing
    the camera, rotate the CAMERA about the world vertical, and express the
    board in the rotated camera's frame. `pitch_deg` tilts the camera up, as
    the real rig is (measured 6.4 deg), to prove the yaw readout is not
    contaminated by pitch.
    """
    import cv2

    # A board hanging on a wall FACES the camera, so its outward normal
    # points back along -Z, not along +Z. The first version of this fixture
    # had the board facing away — unphysical, and it let a real bug through:
    # the un-negated normal put the measured yaw at +/-180, on the atan2
    # discontinuity, where unwrapping turned noise into fake rotation.
    board = np.diag([1.0, -1.0, -1.0])
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    # Camera yaw about the vertical (-Y), then pitch about the camera's X.
    r_yaw = np.array([
        [math.cos(yaw), 0.0, math.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-math.sin(yaw), 0.0, math.cos(yaw)],
    ])
    r_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(pitch), -math.sin(pitch)],
        [0.0, math.sin(pitch), math.cos(pitch)],
    ])
    board_in_camera = r_pitch @ r_yaw @ board
    rvec, _ = cv2.Rodrigues(board_in_camera)
    return rvec


class UnwrapTests(unittest.TestCase):
    def test_crosses_the_180_boundary_continuously(self):
        wrapped = [170.0, 178.0, -175.0, -168.0]
        self.assertEqual(
            [round(v, 3) for v in unwrap_deg(wrapped)],
            [170.0, 178.0, 185.0, 192.0],
        )

    def test_empty_and_single(self):
        self.assertEqual(unwrap_deg([]), [])
        self.assertEqual(unwrap_deg([5.0]), [5.0])


class CameraYawTests(unittest.TestCase):
    UP = np.array([0.0, -1.0, 0.0])

    def test_measures_near_zero_when_facing_the_board(self):
        """The regression guard. A wall board faces the camera, so a naive
        normal puts this at +/-180 — the atan2 discontinuity, where the
        unwrapped series banks noise flips as real rotation (that produced
        gain -2.19 on the first real run)."""
        yaw = camera_yaw_about_up_deg(board_rvec_for_yaw(0.0), self.UP)
        assert yaw is not None
        self.assertLess(abs(yaw), 5.0)

    def test_stays_off_the_wrap_boundary_across_the_sweep(self):
        for turn in (-40.0, -20.0, 0.0, 20.0, 40.0):
            yaw = camera_yaw_about_up_deg(board_rvec_for_yaw(turn), self.UP)
            assert yaw is not None
            self.assertLess(abs(yaw), 150.0, f"turn {turn} landed near the wrap")

    def test_tracks_a_known_rotation_one_to_one(self):
        base = camera_yaw_about_up_deg(board_rvec_for_yaw(0.0), self.UP)
        assert base is not None
        for turn in (-30.0, -10.0, 10.0, 25.0):
            measured = camera_yaw_about_up_deg(board_rvec_for_yaw(turn), self.UP)
            assert measured is not None
            self.assertAlmostEqual(measured - base, turn, delta=0.2)

    def test_camera_pitch_does_not_leak_into_yaw(self):
        """The rig's camera is pitched up 6.4 deg. Measuring yaw about the
        camera's own axis instead of about gravity would smear that in."""
        flat = camera_yaw_about_up_deg(board_rvec_for_yaw(20.0), self.UP)
        pitched = camera_yaw_about_up_deg(board_rvec_for_yaw(20.0, pitch_deg=6.4), self.UP)
        assert flat is not None and pitched is not None
        self.assertAlmostEqual(flat, pitched, delta=0.5)


class FitYawRelationTests(unittest.TestCase):
    def test_recovers_unit_gain_from_an_agreeing_pair(self):
        camera = [float(v) for v in np.linspace(-30.0, 30.0, 60)]
        imu = [c + 137.0 for c in camera]  # arbitrary datum offset
        report = fit_yaw_relation(camera, imu)
        assert report is not None
        self.assertAlmostEqual(report["gain"], 1.0, places=3)
        self.assertAlmostEqual(report["offset_deg"], 137.0, places=1)
        self.assertLess(report["residual_rms_deg"], 0.01)
        self.assertEqual(report["sign"], "agrees")
        self.assertTrue(report["ok"])

    def test_detects_an_inverted_yaw_convention(self):
        """The failure this exists to catch: the robot turns left and the
        filter believes it turned right. Gravity calibration cannot see it."""
        camera = [float(v) for v in np.linspace(-30.0, 30.0, 60)]
        imu = [-c + 10.0 for c in camera]
        report = fit_yaw_relation(camera, imu)
        assert report is not None
        self.assertAlmostEqual(report["gain"], -1.0, places=3)
        self.assertEqual(report["sign"], "inverted")
        self.assertTrue(report["ok"])  # inverted but self-consistent

    def test_flags_a_non_vertical_yaw_axis(self):
        camera = [float(v) for v in np.linspace(-30.0, 30.0, 60)]
        imu = [0.70 * c for c in camera]
        report = fit_yaw_relation(camera, imu)
        assert report is not None
        self.assertFalse(report["ok"])
        self.assertTrue(any("not +/-1" in p for p in report["problems"]))

    def test_measures_drift_against_elapsed_time(self):
        """A slow ramp on top of a correct gain is exactly IMU drift, and is
        the number Ekf2DHeading's bias_walk_per_s was only guessed at."""
        n = 120
        t = list(np.linspace(0.0, 60.0, n))
        camera = [float(30.0 * math.sin(i / 8.0)) for i in range(n)]
        imu = [c + 0.5 * (ti / 60.0) for c, ti in zip(camera, t)]  # 0.5 deg/min
        report = fit_yaw_relation(camera, imu, t)
        assert report is not None
        self.assertAlmostEqual(report["drift_deg_per_min"], 0.5, delta=0.05)

    def test_refuses_a_rotation_too_small_to_fit(self):
        """Reporting a gain fitted to sensor noise would be worse than
        reporting nothing."""
        camera = [0.1 * math.sin(i) for i in range(40)]
        imu = [c + 3.0 for c in camera]
        self.assertIsNone(fit_yaw_relation(camera, imu))

    def test_refuses_too_few_samples(self):
        self.assertIsNone(fit_yaw_relation([0.0, 20.0], [0.0, 20.0]))


class YawCheckGuardTests(unittest.TestCase):
    """The two ways the first real run went wrong, as tests."""

    def test_refuses_a_series_sitting_on_the_wrap_boundary(self):
        """Real report that motivated this: offset_deg -174.4, gain -2.19,
        residual only 1.6 deg — smooth, confident and meaningless."""
        camera = [179.0 if i % 2 else -179.0 for i in range(60)]
        imu = [float(i) for i in range(60)]
        self.assertIsNone(fit_yaw_relation(camera, imu))

    def test_refuses_a_sweep_narrower_than_the_floor(self):
        """The first run swept 15.8 deg; the fitted gain's own uncertainty
        was larger than the deviation from 1 it was meant to detect."""
        camera = [float(v) for v in np.linspace(-8.0, 8.0, 60)]
        imu = [c + 3.0 for c in camera]
        self.assertIsNone(fit_yaw_relation(camera, imu))

    def test_rejects_pnp_pose_flips_without_dragging_the_gain(self):
        """Planar PnP is two-fold ambiguous near head-on, so occasional
        pose flips are expected rather than exceptional."""
        camera = [float(v) for v in np.linspace(-30.0, 30.0, 80)]
        imu = [c + 20.0 for c in camera]
        for i in (7, 23, 51):
            imu[i] += 45.0
        report = fit_yaw_relation(camera, imu)
        assert report is not None
        self.assertAlmostEqual(report["gain"], 1.0, places=2)
        self.assertGreaterEqual(report["samples_rejected"], 3)

    def test_an_imprecise_gain_reads_as_inconclusive_not_as_a_fault(self):
        """Calling a noisy run a hardware fault sends you chasing wiring
        that is fine."""
        rng = np.random.default_rng(0)
        camera = [float(v) for v in np.linspace(-15.0, 15.0, 60)]
        imu = [1.4 * c + rng.normal(0.0, 12.0) for c in camera]
        report = fit_yaw_relation(camera, imu)
        if report is not None:
            self.assertFalse(report["ok"])
            self.assertTrue(any("inconclusive" in p for p in report["problems"]))


class IntrinsicsSanityTests(unittest.TestCase):
    """The rig's own scene_intrinsics.json fails these, which is why they
    exist: fx 534 at 1536x864 is a 110 deg field, while VGGT-refined
    (75.6 deg) and COLMAP/DA3 (75.7/74.4 deg) agree on ~75."""

    @staticmethod
    def _matrix(fx: float, cx: float, cy: float) -> np.ndarray:
        return np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])

    def test_accepts_intrinsics_matching_this_camera(self):
        problems = check_intrinsics_sanity(self._matrix(991.0, 768.0, 432.0), 1536, 864)
        self.assertEqual(problems, [])

    def test_flags_the_rigs_actual_bad_file(self):
        problems = check_intrinsics_sanity(self._matrix(534.457, 732.15, 499.96), 1536, 864)
        self.assertTrue(any("horizontal field" in p for p in problems))
        self.assertTrue(any("cy=" in p for p in problems))

    def test_flags_an_off_centre_principal_point(self):
        problems = check_intrinsics_sanity(self._matrix(991.0, 768.0, 300.0), 1536, 864)
        self.assertTrue(any("cy=" in p for p in problems))


class CornerPitchTests(unittest.TestCase):
    """Intrinsics-free 'is the board big enough in frame'. Deliberately not
    a distance threshold: distance comes out of solvePnP, which is exactly
    what a wrong focal length corrupts."""

    def test_measures_a_known_grid_spacing(self):
        cols, rows, step = 9, 6, 24.0
        grid = np.zeros((rows, cols, 2), dtype=np.float64)
        for r in range(rows):
            for c in range(cols):
                grid[r, c] = (100.0 + c * step, 50.0 + r * step)
        self.assertAlmostEqual(
            corner_pitch_px(grid.reshape(-1, 1, 2), (cols, rows)), step, places=6
        )

    def test_shrinks_as_the_board_gets_further_away(self):
        cols, rows = 9, 6
        pitches = []
        for step in (40.0, 20.0, 8.0):
            grid = np.zeros((rows, cols, 2), dtype=np.float64)
            for r in range(rows):
                for c in range(cols):
                    grid[r, c] = (c * step, r * step)
            pitches.append(corner_pitch_px(grid.reshape(-1, 1, 2), (cols, rows)))
        self.assertEqual(pitches, sorted(pitches, reverse=True))
