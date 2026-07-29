"""Unit tests for pi_client.imu_shtp_uart: pure report-batch parsing.

These cover only the I/O-free logic (walk_reports/decode_vec3/plausibility
bounds) - the framing/resync/reconnect behavior needs a real or fake serial
port and is instead verified empirically (see docs/imu_shtp_uart.md for the
measured results from the actual sensor).

Run directly (no discover wiring exists for client/tests yet):
    python3 -m unittest client.tests.test_imu_shtp_uart -v
from the repo root, with client/src on PYTHONPATH.
"""

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(REPO_ROOT / "client/src"),):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from pi_client.imu_shtp_uart import (  # noqa: E402
    _ACCEL_SCALAR,
    _GYRO_SCALAR,
    decode_vec3,
    is_plausible_accel,
    is_plausible_gyro,
    walk_reports,
)


def make_report(report_id: int, x: int, y: int, z: int, extra: bytes = b"") -> bytes:
    """4-byte common header (report_id + 3 don't-care bytes) + 3x int16 LE."""
    return bytes([report_id, 0, 0, 0]) + struct.pack("<hhh", x, y, z) + extra


class WalkReportsTests(unittest.TestCase):
    def test_single_accel_report(self):
        payload = make_report(0x01, 100, 200, -2521)  # -2521*1/256 ~ -9.85 m/s^2
        reports, unknown = walk_reports(payload)
        self.assertEqual(unknown, 0)
        self.assertEqual(len(reports), 1)
        report_id, body = reports[0]
        self.assertEqual(report_id, 0x01)
        self.assertEqual(len(body), 10)

    def test_accel_then_gyro_back_to_back(self):
        payload = make_report(0x01, 100, 200, -2521) + make_report(0x02, 0, 0, 0)
        reports, unknown = walk_reports(payload)
        self.assertEqual(unknown, 0)
        self.assertEqual([r[0] for r in reports], [0x01, 0x02])

    def test_unknown_report_uses_fallback_length_and_keeps_going(self):
        # Tap Detector (0x10) isn't in _KNOWN_REPORT_LENGTHS - the walker
        # should fall back to the 6-byte guess and still recover the
        # accelerometer report that follows it.
        tap = bytes([0x10, 0, 0, 0, 0, 0])  # 6 bytes, matches the fallback length
        payload = tap + make_report(0x01, 100, 200, -2521)
        reports, unknown = walk_reports(payload)
        self.assertEqual(unknown, 1)
        self.assertEqual([r[0] for r in reports], [0x10, 0x01])

    def test_truncated_trailing_report_is_dropped_not_crashed(self):
        # A report claiming more bytes than are left in the payload (e.g.
        # the sensor's batch got cut by a transport-level continuation) -
        # walk_reports must stop cleanly, keeping whatever came before.
        payload = make_report(0x01, 100, 200, -2521) + bytes([0x02, 0, 0])  # truncated gyro
        reports, unknown = walk_reports(payload)
        self.assertEqual([r[0] for r in reports], [0x01])

    def test_control_style_id_stops_the_walk(self):
        # IDs >= 0xF0 are control-channel-style (e.g. BASE_TIMESTAMP is the
        # one exception already in _KNOWN_REPORT_LENGTHS) and shouldn't be
        # guessed at with the short-report fallback.
        payload = make_report(0x01, 100, 200, -2521) + bytes([0xF1, 1, 2, 3])
        reports, unknown = walk_reports(payload)
        self.assertEqual([r[0] for r in reports], [0x01])
        self.assertEqual(unknown, 0)


class DecodeVec3Tests(unittest.TestCase):
    def test_gravity_at_rest(self):
        body = make_report(0x01, 0, 0, -2521)
        x, y, z = decode_vec3(body, _ACCEL_SCALAR)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(z, -2521 / 256, places=5)

    def test_gyro_scalar(self):
        body = make_report(0x02, 512, -512, 0)
        x, y, z = decode_vec3(body, _GYRO_SCALAR)
        self.assertAlmostEqual(x, 1.0, places=5)
        self.assertAlmostEqual(y, -1.0, places=5)
        self.assertAlmostEqual(z, 0.0)

    def test_all_axes_negative(self):
        # Every axis negative at once (e.g. gravity fully inverted relative
        # to a right-side-up mount, or a rotation the sign convention below
        # calls negative on all three gyro axes at once) - confirms
        # decode_vec3 isn't accidentally relying on at-most-one-negative
        # test fixtures elsewhere in this file.
        body = make_report(0x01, -100, -200, -300)
        x, y, z = decode_vec3(body, _ACCEL_SCALAR)
        self.assertAlmostEqual(x, -100 / 256, places=5)
        self.assertAlmostEqual(y, -200 / 256, places=5)
        self.assertAlmostEqual(z, -300 / 256, places=5)

    def test_negative_int16_extremes_round_trip(self):
        # struct.unpack_from("<hhh", ...) must treat each field as a SIGNED
        # short. -32768 is the most negative representable int16 and the
        # one value where a signed/unsigned mixup is most obvious (an
        # unsigned read would come back as +32768, i.e. wrong sign AND
        # wrong magnitude after scaling). -1 is the other classic trap
        # (0xFFFF unsigned vs -1 signed).
        body = make_report(0x02, -32768, -1, 100)
        x, y, z = decode_vec3(body, _GYRO_SCALAR)
        self.assertAlmostEqual(x, -32768 * _GYRO_SCALAR, places=5)
        self.assertAlmostEqual(y, -1 * _GYRO_SCALAR, places=5)
        self.assertAlmostEqual(z, 100 * _GYRO_SCALAR, places=5)
        self.assertLess(x, 0)
        self.assertLess(y, 0)
        self.assertGreater(z, 0)


class PlausibilityTests(unittest.TestCase):
    def test_gravity_is_plausible(self):
        self.assertTrue(is_plausible_accel((0.5, 0.5, -9.85)))

    def test_full_scale_int16_garbage_is_not_plausible(self):
        # This is exactly the corruption pattern seen on real hardware: a
        # decoded value near the int16 full-scale edge (~127 m/s^2 at this
        # scalar), which no real accelerometer report on this sensor can
        # produce.
        self.assertFalse(is_plausible_accel((0.5, 0.5, 126.5)))

    def test_gyro_over_sensor_full_scale_is_not_plausible(self):
        # BNO08x tops out ~35 rad/s at full scale (~2000 dps) - 63 rad/s
        # (seen repeatedly in testing) cannot be a real report.
        self.assertFalse(is_plausible_gyro((0.0, 0.0, 63.246)))

    def test_gyro_at_rest_is_plausible(self):
        self.assertTrue(is_plausible_gyro((0.0, 0.002, -0.002)))

    def test_negative_accel_symmetric_with_positive(self):
        # Plausibility is a magnitude bound (max(abs(v) for v in accel)), so
        # a negative reading must be accepted/rejected exactly like its
        # positive mirror - a sign mishandled as "always non-negative"
        # (e.g. a stray abs() dropped, or a bound check written as
        # `v <= MAX` instead of `abs(v) <= MAX`) would make one side of this
        # pair disagree.
        self.assertTrue(is_plausible_accel((-0.5, -0.5, -9.85)))
        self.assertFalse(is_plausible_accel((-0.5, -0.5, -126.5)))

    def test_negative_gyro_symmetric_with_positive(self):
        self.assertTrue(is_plausible_gyro((-0.002, 0.0, -0.002)))
        self.assertFalse(is_plausible_gyro((0.0, 0.0, -63.246)))

    def test_negative_just_inside_bound_is_plausible(self):
        # Boundary case on the negative side specifically (not just "a big
        # negative number is rejected", but "a negative number just under
        # the limit is still accepted").
        self.assertTrue(is_plausible_accel((0.0, 0.0, -24.9)))
        self.assertTrue(is_plausible_gyro((0.0, 0.0, -34.9)))


class AxisConventionTests(unittest.TestCase):
    """Documents (as executable assertions, not just prose) the physical
    mounting convention the user confirmed for this board, 2026-07-28:

        X forward, Y right, Z up; rotation about Z is positive CLOCKWISE
        as viewed from above.

    That Z-rotation sign is LEFT-handed - it is the opposite of the
    standard right-hand rule (positive Z rotation = counter-clockwise from
    above) used elsewhere in this repo's world/camera-frame math (e.g.
    imu_rvc.py's yaw handling, mapping/geometry.py's bearing math). This
    module (imu_shtp_uart.py) is a raw SHTP decoder with no world-frame
    opinion - it does not negate anything - so `gyro_rads[2]` as returned by
    decode_vec3/ShtpUartReader.latest() is a raw sensor-frame value where
    POSITIVE = CLOCKWISE-FROM-ABOVE. A future integration layer that wants
    "positive = counter-clockwise / turning left" (the usual robotics
    convention, and what a right-handed consumer would assume) must negate
    gyro_rads[2] itself; nothing here does it automatically.

    This class has no logic to test (there is no "convention-aware" code
    path here yet, by design - see docs/imu_shtp_uart.md "Next steps" #1)
    - it exists so this fact lives in the codebase next to the decoder
    it concerns, not only in a doc/commit message.
    """

    def test_clockwise_positive_convention_is_left_handed_not_right_handed(self):
        # A standard right-handed CCW-positive yaw and this sensor's
        # documented CW-positive convention give opposite signs for the
        # same physical rotation - i.e. they are never both correct for
        # the same raw value. Encode that disagreement explicitly so it
        # can't be silently "fixed" into a right-hand assumption later
        # without someone noticing this test describes why it's flipped.
        physical_turn_is_clockwise_from_above = True
        sensor_reports_positive_gyro_z = physical_turn_is_clockwise_from_above
        right_hand_rule_would_report_positive_gyro_z = not physical_turn_is_clockwise_from_above
        self.assertNotEqual(
            sensor_reports_positive_gyro_z,
            right_hand_rule_would_report_positive_gyro_z,
            "sensor's CW-positive Z convention must be the opposite sign of "
            "the repo's usual right-hand-rule convention, not equal to it",
        )


if __name__ == "__main__":
    unittest.main()
