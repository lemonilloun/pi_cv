"""Unit tests for pi_client.imu_rvc: frame decoding and the tilt-model fix
for the metric-scale bug (docs/scene3d.md, poses_step.imu_scale_samples).

Run directly (no discover wiring exists for client/tests yet):
    python3 -m unittest client.tests.test_imu_rvc -v
from the repo root, with client/src on PYTHONPATH.
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

from pi_client.imu_rvc import (  # noqa: E402
    ImuIntegrator,
    ImuSample,
    checksum,
    fit_tilt_model,
    iter_frames,
    parse_frame,
)


def make_frame(index: int, yaw_c: int, pitch_c: int, roll_c: int,
               ax_mg: int, ay_mg: int, az_mg: int) -> bytes:
    def i16(value: int) -> bytes:
        return int(value & 0xFFFF).to_bytes(2, "little")

    body = bytes([index]) + i16(yaw_c) + i16(pitch_c) + i16(roll_c) + \
        i16(ax_mg) + i16(ay_mg) + i16(az_mg) + bytes([0, 0, 0])  # MI, MR, reserved
    frame = b"\xaa\xaa" + body
    return frame + bytes([checksum(frame)])


class ParseFrameTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        frame = make_frame(7, 100, -50, 25, -10, 22, 1002)
        sample = parse_frame(frame, monotonic=1.0)
        assert sample is not None
        self.assertEqual(sample.index, 7)
        self.assertAlmostEqual(sample.yaw_deg, 1.0)
        self.assertAlmostEqual(sample.pitch_deg, -0.5)
        self.assertAlmostEqual(sample.roll_deg, 0.25)
        self.assertEqual(sample.accel_mg, (-10.0, 22.0, 1002.0))

    def test_bad_checksum_rejected(self) -> None:
        frame = make_frame(1, 0, 0, 0, 0, 0, 1000)
        corrupted = frame[:-1] + bytes([(frame[-1] + 1) % 256])
        self.assertIsNone(parse_frame(corrupted))

    def test_iter_frames_resyncs_after_one_bad_byte(self) -> None:
        good = make_frame(1, 0, 0, 0, 0, 0, 1000)
        buffer = b"\x00" + good + good  # one junk byte before two good frames
        samples, remainder = iter_frames(buffer)
        self.assertEqual(len(samples), 2)
        self.assertEqual(remainder, b"")


class FitTiltModelTests(unittest.TestCase):
    """The core of the fix: recovering a (Δpitch, Δroll) -> Δup mapping from
    the same multi-attitude data imu_calibrate.py's `full` mode collects."""

    def test_recovers_a_known_small_angle_model(self) -> None:
        # A synthetic ground truth: pitch tilts up_x, roll tilts up_y, small
        # angle regime (a few degrees, matching what the rig actually does).
        ref_pitch, ref_roll = 1.4, 0.6
        ref_up = (0.02, 0.01, 0.9997)
        attitudes = [(0.0, 0.0), (5.0, 0.0), (0.0, 4.0), (-3.0, 2.0), (3.0, -3.0)]

        def true_up(d_pitch_deg: float, d_roll_deg: float) -> tuple[float, float, float]:
            dp, dr = math.radians(d_pitch_deg), math.radians(d_roll_deg)
            x, y, z = ref_up[0] + dp, ref_up[1] + dr, ref_up[2]
            norm = math.sqrt(x * x + y * y + z * z)
            return (x / norm, y / norm, z / norm)

        pitch_roll = [(ref_pitch + dp, ref_roll + dr) for dp, dr in attitudes]
        up_sensor = [true_up(dp, dr) for dp, dr in attitudes]

        model = fit_tilt_model(pitch_roll, up_sensor, ref_pitch, ref_roll, ref_up)
        self.assertIsNotNone(model)
        assert model is not None
        self.assertLess(model["fit_residual_deg_mean"], 0.5)
        self.assertEqual(model["attitudes_used"], len(attitudes))

    def test_rejects_single_axis_attitude_spread(self) -> None:
        # Propping the rig up the same way every time (pitch varies, roll
        # never does) cannot constrain a 2D sensitivity — must return None
        # rather than silently extrapolating along the untested axis.
        ref_pitch, ref_roll = 0.0, 0.0
        ref_up = (0.0, 0.0, 1.0)
        pitch_roll = [(0.0, 0.0), (3.0, 0.0), (6.0, 0.0), (-3.0, 0.0)]
        up_sensor = [(math.sin(math.radians(p)), 0.0, math.cos(math.radians(p)))
                     for p, _ in pitch_roll]
        model = fit_tilt_model(pitch_roll, up_sensor, ref_pitch, ref_roll, ref_up)
        self.assertIsNone(model)

    def test_none_with_too_few_attitudes(self) -> None:
        model = fit_tilt_model([(0, 0), (1, 1)], [(0, 0, 1), (0.01, 0.01, 1)], 0, 0, (0, 0, 1))
        self.assertIsNone(model)


class ImuIntegratorDynamicGravityTests(unittest.TestCase):
    def _sample(self, pitch_deg: float, roll_deg: float,
                accel_mg: tuple[float, float, float]) -> ImuSample:
        return ImuSample(
            index=0, yaw_deg=0.0, pitch_deg=pitch_deg, roll_deg=roll_deg,
            accel_mg=accel_mg, monotonic=0.0,
        )

    def test_static_reference_leaks_gravity_when_chassis_tilts(self) -> None:
        # This reproduces the bug the last commit measured: the rig has
        # actually tilted 3 deg (fused pitch), but the accelerometer still
        # reads straight down its own Z because it cannot tell tilt from
        # acceleration on its own — a constant-reference subtraction then
        # leaves the full 9.81*sin(3 deg) leak in "linear" acceleration.
        integrator = ImuIntegrator(up_sensor=(0.0, 0.0, 1.0))
        tilted = self._sample(3.0, 0.0, (0.0, 0.0, 1000.0))
        linear = integrator.linear_accel(tilted)
        assert linear is not None
        self.assertAlmostEqual(linear[0], 0.0, places=3)  # nothing to correct with

    def test_tilt_model_moves_the_dynamic_up_toward_the_true_tilt(self) -> None:
        # A tilt model saying "1 rad of pitch shifts up_x by 1 unit, 1 rad
        # of roll shifts up_y by 1 unit" (a clean synthetic small-angle
        # relationship — the point is the mechanism, not this number).
        # Fed a sample 3 deg pitched from the reference, the dynamic up
        # estimate must move off the static reference toward that tilt,
        # instead of staying frozen at (0, 0, 1) the way the old
        # constant-reference code always did.
        tilt_model = {
            "pitch_ref_deg": 0.0,
            "roll_ref_deg": 0.0,
            "up_ref": [0.0, 0.0, 1.0],
            "sensitivity": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
        integrator = ImuIntegrator(up_sensor=(0.0, 0.0, 1.0), tilt_model=tilt_model)
        tilted = self._sample(3.0, 0.0, (0.0, 0.0, 1000.0))
        up = integrator.current_up_sensor(tilted)
        assert up is not None
        expected_x = math.sin(math.radians(3.0))  # normalized, small-angle
        self.assertAlmostEqual(up[0], expected_x, places=3)
        self.assertGreater(up[0], 0.01)  # moved off the static (0, 0, 1) reference

    def test_no_tilt_model_keeps_static_reference(self) -> None:
        integrator = ImuIntegrator(up_sensor=(0.0, 0.0, 1.0))
        tilted = self._sample(3.0, 0.0, (0.0, 0.0, 1000.0))
        up = integrator.current_up_sensor(tilted)
        self.assertEqual(up, (0.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
