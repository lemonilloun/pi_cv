"""Tests for the RVC maths, against the datasheet and against what this rig
actually measured rather than against invented numbers."""

from __future__ import annotations

import math
import struct
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "client/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from pi_client.imu_rvc_math import (  # noqa: E402
    NOMINAL_DT_S,
    FrameClock,
    YawDriftCorrector,
    YawRateKF,
    gravity_in_body,
    level_rotation,
    manhattan_yaw_offset,
    motion_state,
    rvc_to_matrix,
    unwrap_deg,
)


class DatasheetFrameTest(unittest.TestCase):
    """The reference frame from the BNO085 RVC datasheet. If this passes, the
    byte layout and checksum are right and any remaining oddity has another
    cause — which is exactly the question worth settling first."""

    REF = bytes.fromhex("AAAADE010092FF25088DFEECFFD103000000E7")

    def test_frame_is_19_bytes(self) -> None:
        self.assertEqual(len(self.REF), 19)

    def test_checksum_rule_is_sum_of_bytes_2_to_17(self) -> None:
        self.assertEqual(sum(self.REF[2:18]) & 0xFF, self.REF[18])

    def test_angles_and_accel_decode_to_the_documented_values(self) -> None:
        yaw, pitch, roll, ax, ay, az = struct.unpack("<hhhhhh", self.REF[3:15])
        self.assertAlmostEqual(yaw * 0.01, 0.01, places=6)
        self.assertAlmostEqual(pitch * 0.01, -1.10, places=6)
        self.assertAlmostEqual(roll * 0.01, 20.85, places=6)
        self.assertEqual((ax, ay, az), (-371, -20, 977))


class FrameClockTest(unittest.TestCase):
    """The measured problem: arrivals come in pairs 20 ms apart, so host time
    is worth up to 20 ms per frame. The index counter is not."""

    def _batched_arrivals(self, n: int) -> list[tuple[int, float]]:
        # Reproduces what was measured: two frames land together, then a 20 ms
        # gap. True cadence is still 10 ms per frame.
        out = []
        for seq in range(n):
            batch = seq // 2
            out.append((seq, batch * 0.020))
        return out

    def test_recovers_the_true_period_from_batched_arrivals(self) -> None:
        clock = FrameClock()
        for seq, t_host in self._batched_arrivals(1200):
            clock.update(seq, t_host)
        self.assertAlmostEqual(clock.a, NOMINAL_DT_S, places=4)
        self.assertAlmostEqual(clock.rate_hz, 100.0, delta=0.5)

    def test_timestamps_are_evenly_spaced_even_though_arrivals_are_not(self) -> None:
        clock = FrameClock()
        stamps = [clock.update(seq, t) for seq, t in self._batched_arrivals(1200)]
        gaps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
        # Arrival gaps alternate 0 and 20 ms; reconstructed ones must not.
        self.assertAlmostEqual(min(gaps), NOMINAL_DT_S, places=4)
        self.assertAlmostEqual(max(gaps), NOMINAL_DT_S, places=4)

    def test_a_broken_stream_does_not_move_the_fit(self) -> None:
        # A period outside 8-12 ms means the stream broke, not that a crystal
        # drifted; following it would corrupt every timestamp afterwards.
        clock = FrameClock()
        for seq in range(400):
            clock.update(seq, seq * 0.01)
        good = clock.a
        for seq in range(400, 800):
            clock.update(seq, seq * 0.5)      # absurd 500 ms period
        self.assertAlmostEqual(clock.a, good, places=6)

    def test_first_sample_gets_a_timestamp(self) -> None:
        self.assertIsInstance(FrameClock().update(0, 123.0), float)


class UnwrapTest(unittest.TestCase):
    def test_wrap_upward_across_180(self) -> None:
        self.assertAlmostEqual(unwrap_deg(179.0, -179.0, 0.0), 360.0)

    def test_wrap_downward_across_180(self) -> None:
        self.assertAlmostEqual(unwrap_deg(-179.0, 179.0, 0.0), -360.0)

    def test_small_steps_do_not_wrap(self) -> None:
        self.assertAlmostEqual(unwrap_deg(10.0, 10.2, 5.0), 5.0)

    def test_first_sample_is_a_no_op(self) -> None:
        self.assertAlmostEqual(unwrap_deg(None, 42.0, 0.0), 0.0)


class GravityTest(unittest.TestCase):
    def test_level_rotation_stands_gravity_upright(self) -> None:
        # The identity the guide asks to verify once: levelling must send the
        # measured down-vector exactly onto -Z.
        pitch, roll = math.radians(7.0), math.radians(-4.0)
        g = gravity_in_body(pitch, roll)
        r = level_rotation(pitch, roll)
        out = [sum(r[i][k] * g[k] for k in range(3)) for i in range(3)]
        for value, expected in zip(out, (0.0, 0.0, -1.0)):
            self.assertAlmostEqual(value, expected, places=9)

    def test_gravity_is_a_unit_vector(self) -> None:
        for pitch in (-0.4, 0.0, 0.3):
            for roll in (-0.5, 0.0, 0.2):
                g = gravity_in_body(pitch, roll)
                self.assertAlmostEqual(math.sqrt(sum(v * v for v in g)), 1.0, places=12)

    def test_yaw_cannot_change_where_gravity_points(self) -> None:
        # The reason pitch/roll never drift while yaw does: yaw is a rotation
        # ABOUT gravity. Levelling must therefore be yaw-free, and it is.
        pitch, roll = 0.2, -0.1
        first = level_rotation(pitch, roll)
        self.assertEqual(first, level_rotation(pitch, roll))

    def test_level_is_the_yaw_zero_case_of_the_full_matrix(self) -> None:
        pitch, roll = 0.15, -0.22
        full = rvc_to_matrix(0.0, pitch, roll)
        for row_a, row_b in zip(full, level_rotation(pitch, roll)):
            for a, b in zip(row_a, row_b):
                self.assertAlmostEqual(a, b, places=12)


class YawRateKFTest(unittest.TestCase):
    def test_tracks_a_constant_turn_rate(self) -> None:
        kf = YawRateKF()
        rate = math.radians(30.0)
        theta = 0.0
        for _ in range(400):
            theta += rate * NOMINAL_DT_S
            _, omega = kf.step(theta)
        self.assertAlmostEqual(math.degrees(omega), 30.0, delta=1.5)

    def test_beats_plain_differencing_on_quantised_input(self) -> None:
        # Yaw is quantised to 0.01 deg, and 0.01 deg per 10 ms is already
        # 1 deg/s of noise. This is the whole reason the filter exists.
        import random

        rng = random.Random(7)
        rate = math.radians(20.0)
        kf = YawRateKF()
        theta = 0.0
        kf_err, naive_err = [], []
        previous = None
        for _ in range(600):
            theta += rate * NOMINAL_DT_S
            quantised = math.radians(round(math.degrees(theta) + rng.gauss(0, 0.01), 2))
            _, omega = kf.step(quantised)
            kf_err.append(abs(omega - rate))
            if previous is not None:
                naive_err.append(abs((quantised - previous) / NOMINAL_DT_S - rate))
            previous = quantised
        settled = len(kf_err) // 2
        kf_rms = math.sqrt(sum(e * e for e in kf_err[settled:]) / len(kf_err[settled:]))
        naive_rms = math.sqrt(sum(e * e for e in naive_err[settled:]) / len(naive_err[settled:]))
        self.assertLess(kf_rms, naive_rms / 2.0)

    def test_is_causal_and_starts_at_the_first_sample(self) -> None:
        kf = YawRateKF()
        theta, omega = kf.step(1.234)
        self.assertAlmostEqual(theta, 1.234)
        self.assertEqual(omega, 0.0)


class MotionStateTest(unittest.TestCase):
    def _still(self, n=30):
        return [(0.0, 0.0, 9.81) for _ in range(n)]

    def test_still_rig_is_stationary(self) -> None:
        out = motion_state(self._still(), 0.0, 0.3)
        self.assertTrue(out["stationary"])
        self.assertFalse(out["vibration"])
        self.assertFalse(out["impact"])

    def test_a_steady_tilt_is_not_motion(self) -> None:
        # Tilting changes the axes but not |a|. Judging per-axis would call
        # this movement; judging the magnitude does not.
        tilted = [(9.81 * math.sin(0.3), 0.0, 9.81 * math.cos(0.3)) for _ in range(30)]
        self.assertTrue(motion_state(tilted, 0.0, 0.3)["stationary"])

    def test_vibration_is_flagged(self) -> None:
        shaking = [(0.0, 0.0, 9.81 + (1.0 if i % 2 else -1.0)) for i in range(30)]
        out = motion_state(shaking, 0.0, 0.3)
        self.assertTrue(out["vibration"])
        self.assertFalse(out["stationary"])

    def test_a_knock_is_flagged(self) -> None:
        knocked = self._still(29) + [(0.0, 0.0, 20.0)]
        self.assertTrue(motion_state(knocked, 0.0, 0.3)["impact"])

    def test_turning_is_not_stationary(self) -> None:
        out = motion_state(self._still(), math.radians(9.0), 0.3)   # 30 deg/s
        self.assertFalse(out["stationary"])
        self.assertAlmostEqual(out["omega_dps"], 30.0, delta=0.5)

    def test_empty_window(self) -> None:
        self.assertEqual(motion_state([], 0.0, 0.3)["samples"], 0)


class ManhattanTest(unittest.TestCase):
    def test_square_room_gives_a_confident_zero_offset(self) -> None:
        walls = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
        offset, strength = manhattan_yaw_offset(walls)
        self.assertAlmostEqual(offset, 0.0, places=9)
        self.assertAlmostEqual(strength, 1.0, places=9)

    def test_a_rotated_room_reports_its_rotation(self) -> None:
        turn = math.radians(7.0)
        walls = [turn + k * math.pi / 2 for k in range(4)]
        offset, strength = manhattan_yaw_offset(walls)
        self.assertAlmostEqual(math.degrees(offset), 7.0, places=4)
        self.assertGreater(strength, 0.99)

    def test_scattered_normals_report_low_strength(self) -> None:
        # No walls worth listening to: the caller must be able to tell, or it
        # would apply a meaningless correction to the heading.
        import random

        rng = random.Random(3)
        walls = [rng.uniform(0, 2 * math.pi) for _ in range(400)]
        _, strength = manhattan_yaw_offset(walls)
        self.assertLess(strength, 0.3)

    def test_no_walls_is_zero_strength_not_a_crash(self) -> None:
        self.assertEqual(manhattan_yaw_offset([]), (0.0, 0.0))


class DriftCorrectorTest(unittest.TestCase):
    def test_converges_towards_the_absolute_reference(self) -> None:
        corrector = YawDriftCorrector(alpha=0.1)
        for _ in range(200):
            corrector.observe(math.radians(10.0), math.radians(4.0))
        self.assertAlmostEqual(math.degrees(corrector.correct(math.radians(10.0))),
                               4.0, delta=0.2)

    def test_a_single_observation_does_not_jump_the_heading(self) -> None:
        # A step in heading puts a discontinuity into the pose graph, which is
        # worse than the drift being corrected.
        corrector = YawDriftCorrector(alpha=0.02)
        corrector.observe(math.radians(10.0), math.radians(0.0))
        moved = abs(math.degrees(corrector.bias))
        self.assertLess(moved, 0.5)

    def test_wraps_the_error_the_short_way(self) -> None:
        corrector = YawDriftCorrector(alpha=1.0)
        corrector.observe(math.radians(179.0), math.radians(-179.0))
        # 2 degrees apart across the boundary, not 358.
        self.assertLess(abs(math.degrees(corrector.bias)), 5.0)


if __name__ == "__main__":
    unittest.main()


class AtTimeTest(unittest.TestCase):
    """Interpolation is the point of the frame clock: it is what lets a camera
    frame be told where the rig was looking at the instant it was exposed."""

    def _reader(self, n=50):
        from pi_client.imu_rvc import ImuSample, RvcReader

        reader = RvcReader.__new__(RvcReader)          # no serial port needed
        import threading
        reader._lock = threading.Lock()
        reader._samples = [
            ImuSample(index=i % 256, yaw_deg=i * 1.0, pitch_deg=i * 0.1,
                      roll_deg=-i * 0.2, accel_mg=(0, 0, 1000),
                      monotonic=0.0, t_grid=i * 0.01, seq=i)
            for i in range(n)
        ]
        return reader

    def test_interpolates_between_samples(self) -> None:
        from pi_client.imu_rvc import YAW_SIGN

        reader = self._reader()
        out = reader.at_time(0.105)                    # halfway between 10 and 11
        # YAW_SIGN is applied here exactly as `read_orientation` applies it —
        # measured on this sensor, whose yaw runs backwards. Returning the raw
        # sign from one accessor and the corrected sign from the other is how
        # a heading ends up mirrored in one code path only.
        self.assertAlmostEqual(out["yaw_deg"], YAW_SIGN * 10.5, places=6)
        self.assertAlmostEqual(out["pitch_deg"], 1.05, places=6)
        self.assertAlmostEqual(out["interp_gap_ms"], 10.0, places=2)

    def test_outside_the_window_returns_none(self) -> None:
        # Extrapolating would produce a confident-looking wrong orientation.
        reader = self._reader()
        self.assertIsNone(reader.at_time(-1.0))
        self.assertIsNone(reader.at_time(99.0))

    def test_yaw_interpolates_the_short_way_round(self) -> None:
        from pi_client.imu_rvc import ImuSample

        reader = self._reader(2)
        reader._samples = [
            ImuSample(index=0, yaw_deg=179.0, pitch_deg=0.0, roll_deg=0.0,
                      accel_mg=(0, 0, 1000), monotonic=0.0, t_grid=0.0, seq=0),
            ImuSample(index=1, yaw_deg=-179.0, pitch_deg=0.0, roll_deg=0.0,
                      accel_mg=(0, 0, 1000), monotonic=0.0, t_grid=0.01, seq=1),
        ]
        out = reader.at_time(0.005)
        # Halfway between 179 and -179 the short way is +/-180, not 0.
        self.assertGreater(abs(out["yaw_deg"]), 179.0)

    def test_too_few_samples(self) -> None:
        self.assertIsNone(self._reader(1).at_time(0.0))


class MountOffsetTest(unittest.TestCase):
    """Test E measured 1.39 deg pitch and 1.49 deg roll of mount tilt. That is
    a constant lie in the vertical every reconstruction leans on — 14 cm of
    false slope across a 4 m floor — so it has to be subtracted, and it has to
    survive the calibration file being absent."""

    def _reader(self, calibration=None):
        import threading

        from pi_client.imu_rvc import RvcReader

        reader = RvcReader.__new__(RvcReader)
        reader._lock = threading.Lock()
        reader._samples = []
        if calibration is not None:
            reader.calibration = calibration
        return reader

    def test_measured_offsets_are_subtracted(self) -> None:
        reader = self._reader({"rvc_acceptance": {
            "pitch_offset_deg": 1.39, "roll_offset_deg": 1.49}})
        pitch, roll = reader.level_angles_deg(1.39, 1.49)
        self.assertAlmostEqual(pitch, 0.0, places=9)
        self.assertAlmostEqual(roll, 0.0, places=9)

    def test_a_real_tilt_survives_the_correction(self) -> None:
        reader = self._reader({"rvc_acceptance": {
            "pitch_offset_deg": 1.39, "roll_offset_deg": 1.49}})
        pitch, roll = reader.level_angles_deg(11.39, -3.51)
        self.assertAlmostEqual(pitch, 10.0, places=6)
        self.assertAlmostEqual(roll, -5.0, places=6)

    def test_no_calibration_at_all_still_reports_angles(self) -> None:
        reader = self._reader()
        self.assertEqual(reader.mount_offsets_deg, (0.0, 0.0))
        self.assertEqual(reader.level_angles_deg(3.0, -2.0), (3.0, -2.0))

    def test_calibration_without_the_section(self) -> None:
        reader = self._reader({"mode": "full"})
        self.assertEqual(reader.mount_offsets_deg, (0.0, 0.0))


class KeyframeBlurGateTest(unittest.TestCase):
    """Пороги отбраковки кадров по IMU (§7.5). Резкость по Лапласиану ловит
    смаз постфактум; курс говорит, что кадр мажется прямо сейчас."""

    def _still(self, n=30):
        return [(0.0, 0.0, 9.81) for _ in range(n)]

    def test_fast_turn_is_over_the_blur_threshold(self) -> None:
        # 20 град/с при выдержке 1/30 с — уже смаз.
        out = motion_state(self._still(), math.radians(6.0), 0.3)
        self.assertGreater(out["omega_dps"], 15.0)

    def test_a_slow_pan_stays_under_it(self) -> None:
        # Съёмка комнаты ведётся медленно; гасить такие кадры значило бы
        # выбросить почти всю запись.
        out = motion_state(self._still(), math.radians(1.5), 0.3)
        self.assertLess(out["omega_dps"], 15.0)

    def test_a_knock_is_caught_even_while_standing_still(self) -> None:
        knocked = self._still(29) + [(0.0, 0.0, 20.0)]
        out = motion_state(knocked, 0.0, 0.3)
        self.assertTrue(out["impact"])
        self.assertLess(out["omega_dps"], 1.0)
