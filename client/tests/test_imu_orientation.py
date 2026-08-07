"""Unit tests for pi_client.imu_orientation_test.

The yaw SIGN and GAIN are the only things about this sensor that have never
been measured on the rig, and two earlier attempts failed for tooling
reasons. So the analysis is tested against synthetic signals whose truth is
known exactly, before it is pointed at hardware.

    python3 -m unittest client.tests.test_imu_orientation -v
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

from pi_client.imu_orientation_test import (  # noqa: E402
    analyse_push,
    analyse_still,
    analyse_turn,
    unwrap_deg,
    verdict,
)


def wrapped(values):
    return [(v + 180.0) % 360.0 - 180.0 for v in values]


def still_samples(n=400, drift_deg_per_min=0.0, accel_mg=(0.0, 0.0, 1000.0), hz=100.0):
    out = []
    for i in range(n):
        t = i / hz
        yaw = drift_deg_per_min * t / 60.0
        out.append((t, (yaw + 180.0) % 360.0 - 180.0, -0.5, 0.5, accel_mg))
    return out


class UnwrapTests(unittest.TestCase):
    def test_a_full_turn_through_the_boundary_is_continuous(self):
        true = [i * 3.0 for i in range(121)]          # 0 .. 360
        self.assertAlmostEqual(unwrap_deg(wrapped(true))[-1] - unwrap_deg(wrapped(true))[0],
                               360.0, places=6)


class TurnTests(unittest.TestCase):
    def test_a_clean_left_turn_gives_gain_one(self):
        yaw = wrapped([i * 3.0 for i in range(121)])
        result = analyse_turn(yaw, +360.0)
        self.assertTrue(result["ok"], result.get("problems"))
        self.assertAlmostEqual(result["gain"], 1.0, places=3)
        self.assertEqual(result["sign"], "agrees")

    def test_an_inverted_sensor_is_detected_not_hidden(self):
        """The failure the gravity calibration physically cannot see: the
        robot turns left and the filter believes it turned right."""
        yaw = wrapped([-i * 3.0 for i in range(121)])
        result = analyse_turn(yaw, +360.0)
        self.assertAlmostEqual(result["gain"], -1.0, places=3)
        self.assertEqual(result["sign"], "inverted")
        self.assertTrue(result["ok"])     # inverted but self-consistent

    def test_a_half_scale_sensor_fails_the_gain_check(self):
        yaw = wrapped([i * 1.5 for i in range(121)])
        result = analyse_turn(yaw, +360.0)
        self.assertFalse(result["ok"])
        self.assertTrue(any("not +/-1" in p for p in result["problems"]))

    def test_a_wobbly_turn_is_refused_rather_than_averaged(self):
        """Back-and-forth motion makes the endpoint difference meaningless
        even when it happens to land near 360."""
        yaw = []
        angle = 0.0
        for i in range(200):
            angle += 6.0 if (i // 10) % 2 == 0 else -2.4
            yaw.append(angle)
        result = analyse_turn(wrapped(yaw), +360.0)
        self.assertFalse(result["ok"])
        self.assertTrue(any("backwards" in p for p in result["problems"]))

    def test_too_few_samples_is_reported(self):
        self.assertFalse(analyse_turn([0.0, 1.0], 360.0)["ok"])


class StillTests(unittest.TestCase):
    def test_a_quiet_sensor_passes(self):
        result = analyse_still(still_samples())
        self.assertTrue(result["ok"], result.get("problems"))
        self.assertAlmostEqual(result["accel_mg_median"], 1000.0, places=1)

    def test_flags_real_drift(self):
        result = analyse_still(still_samples(drift_deg_per_min=15.0))
        self.assertFalse(result["ok"])
        self.assertTrue(any("drifts" in p for p in result["problems"]))

    def test_tolerates_the_measured_0_03_deg_per_min(self):
        """The figure actually measured on this device over 90 s."""
        self.assertTrue(analyse_still(still_samples(drift_deg_per_min=0.03))["ok"])

    def test_flags_a_wrong_accelerometer_scale(self):
        result = analyse_still(still_samples(accel_mg=(0.0, 0.0, 500.0)))
        self.assertFalse(result["ok"])
        self.assertTrue(any("scale is wrong" in p for p in result["problems"]))


class PushTests(unittest.TestCase):
    @staticmethod
    def _push(peak_mg=120.0, yaw_drift_deg=0.0, n=1000, push_from=300, push_to=600):
        """Realistic shape: still, a push in the middle, still again — which
        is what a 10 s phase with a ~3 s push actually looks like."""
        out = []
        for i in range(n):
            t = i / 100.0
            if push_from <= i < push_to:
                bump = peak_mg * math.sin(math.pi * (i - push_from) / (push_to - push_from))
            else:
                bump = 0.0
            out.append((t, yaw_drift_deg * i / n, -0.5, 0.5, (bump, 0.0, 1000.0)))
        return out

    def test_a_real_push_registers(self):
        result = analyse_push(self._push())
        self.assertTrue(result["ok"], result.get("problems"))
        self.assertGreater(result["peak_deviation_mg"], 15.0)

    def test_a_robot_that_never_moved_is_caught(self):
        result = analyse_push(self._push(peak_mg=1.0))
        self.assertFalse(result["ok"])
        self.assertTrue(any("barely moved" in p for p in result["problems"]))

    def test_a_curved_push_is_caught(self):
        result = analyse_push(self._push(yaw_drift_deg=60.0))
        self.assertFalse(result["ok"])
        self.assertTrue(any("straight push" in p for p in result["problems"]))


class VerdictTests(unittest.TestCase):
    @staticmethod
    def _report(left_gain=1.0, right_gain=1.0, still_ok=True):
        ok_still = {"ok": still_ok}
        return {
            "turn_left": {"ok": True, "gain": left_gain},
            "turn_right": {"ok": True, "gain": right_gain},
            "still_start": ok_still, "still_mid": ok_still, "still_end": ok_still,
        }

    def test_aligned_sensor_is_usable(self):
        self.assertIn("aligned", verdict(self._report()))

    def test_consistently_inverted_is_still_usable(self):
        """An inverted convention is fine as long as it is known — the filter
        just needs the sign. It is only fatal when nobody measured it."""
        v = verdict(self._report(left_gain=-1.0, right_gain=-1.0))
        self.assertIn("INVERTED", v)
        self.assertIn("USABLE", v)

    def test_turns_disagreeing_about_sign_is_broken(self):
        self.assertIn("BROKEN", verdict(self._report(left_gain=1.0, right_gain=-1.0)))

    def test_a_bad_stationary_check_outranks_a_good_turn(self):
        self.assertIn("SENSOR PROBLEM", verdict(self._report(still_ok=False)))

    def test_no_clean_turn_is_inconclusive_not_a_pass(self):
        self.assertIn("INCONCLUSIVE", verdict({"turn_left": {"ok": False}}))


class PushMeasuresTheVectorNotTheMagnitude(unittest.TestCase):
    """Regression: horizontal acceleration adds to gravity in quadrature, so
    a magnitude-based test is nearly blind to a real push. 120 mg on top of
    1000 mg moves |a| by 7 mg. This project made that exact mistake once
    before, in imu_distance_analysis, where a magnitude detector called 100%
    of a genuine 3 m push 'stationary'."""

    def test_a_purely_horizontal_push_is_seen_at_full_size(self):
        samples = [
            (i / 100.0, 0.0, -0.5, 0.5,
             (120.0 * math.sin(math.pi * (i - 300) / 300) if 300 <= i < 600 else 0.0,
              0.0, 1000.0))
            for i in range(1000)
        ]
        result = analyse_push(samples)
        self.assertTrue(result["ok"], result.get("problems"))
        # The vector deviation must recover ~120 mg, not the ~7 mg a
        # magnitude test would report.
        self.assertGreater(result["peak_deviation_mg"], 100.0)

    def test_magnitude_would_have_missed_it(self):
        """Pin the physics that motivates the design, so nobody 'simplifies'
        it back to a magnitude comparison later."""
        rest = math.sqrt(0.0 ** 2 + 1000.0 ** 2)
        pushed = math.sqrt(120.0 ** 2 + 1000.0 ** 2)
        self.assertLess(pushed - rest, 8.0)


class PushRestReferenceTests(unittest.TestCase):
    """An operator who pushes for the whole window makes the in-window median
    a mid-push value, and the excursion collapses. The preceding stationary
    phase is the honest reference."""

    @staticmethod
    def _continuous_push(peak_mg=120.0, n=300):
        return [
            (i / 100.0, 0.0, -0.5, 0.5, (peak_mg * math.sin(math.pi * i / n), 0.0, 1000.0))
            for i in range(n)
        ]

    def test_in_window_median_understates_a_window_filling_push(self):
        result = analyse_push(self._continuous_push())
        self.assertLess(result["peak_deviation_mg"], 100.0)

    def test_an_explicit_rest_reference_recovers_the_full_size(self):
        result = analyse_push(self._continuous_push(), rest_vector=(0.0, 0.0, 1000.0))
        self.assertGreater(result["peak_deviation_mg"], 115.0)
        self.assertTrue(result["ok"], result.get("problems"))
