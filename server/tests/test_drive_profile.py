"""Unit tests for mac_server.drive_profile.

A DC motor that cannot turn draws its STALL current — the largest it will
ever draw — and all of it becomes heat. Commanding just below the moving
threshold therefore costs more battery than driving does. With three 300 mAh
packs dying in 1-2 minutes, that band is worth engineering around.

    python3 -m unittest server.tests.test_drive_profile -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from mac_server.drive_profile import (  # noqa: E402
    DriveProfile,
    energy_note,
    motor_pwm_to_request,
    request_to_motor_pwm,
    shape,
)


class EspMappingTests(unittest.TestCase):
    """Mirror of scaleCmd. Without it this module and the wire disagree about
    what a number means — which is exactly how the localization spin came to
    request 110 and deliver 86."""

    def test_reproduces_the_measured_spin_numbers(self):
        self.assertAlmostEqual(request_to_motor_pwm(110), 86, delta=1)
        self.assertAlmostEqual(request_to_motor_pwm(160), 108, delta=1)

    def test_the_esp_deadband_is_a_hard_zero(self):
        self.assertEqual(request_to_motor_pwm(4), 0)
        self.assertEqual(request_to_motor_pwm(-4), 0)

    def test_endpoints(self):
        self.assertEqual(request_to_motor_pwm(255), 150)
        self.assertEqual(request_to_motor_pwm(-255), -150)

    def test_sign_is_preserved(self):
        self.assertEqual(request_to_motor_pwm(-160), -request_to_motor_pwm(160))

    def test_inverse_rounds_up_never_down(self):
        """Rounding down would land the command back inside the stall band
        the caller just asked to escape."""
        for motor in range(41, 150):
            recovered = request_to_motor_pwm(motor_pwm_to_request(motor))
            self.assertGreaterEqual(recovered, motor, f"motor={motor}")


class ShapeTests(unittest.TestCase):
    def setUp(self):
        self.profile = DriveProfile(straight_motor_pwm_min=70, turn_motor_pwm_min=105)

    def test_zero_stays_zero(self):
        plan = shape(0, 0, self.profile)
        self.assertEqual((plan.left, plan.right), (0, 0))
        self.assertFalse(plan.pulsed)

    def test_a_strong_command_passes_through_untouched(self):
        plan = shape(200, 200, self.profile)
        self.assertEqual((plan.left, plan.right), (200, 200))
        self.assertEqual(plan.duty, 1.0)

    def test_a_stalling_command_is_raised_above_the_threshold(self):
        """The command that would sit in the stall band gets raised out of
        it rather than held where it only makes heat."""
        plan = shape(40, 40, self.profile)     # ~46 at the motor, below 70
        self.assertGreaterEqual(request_to_motor_pwm(plan.left), 70)
        self.assertIn("stall", plan.reason)

    def test_recording_never_pulses_by_default(self):
        """Pulsing is an alternation between a command and a stop, but the
        action log records one command per decision — so a pulsed label is
        one the wheels never received. Losing the creep is cheap; a
        systematically wrong action label is not."""
        plan = shape(40, 40, self.profile)
        self.assertFalse(plan.pulsed)
        self.assertEqual(plan.duty, 1.0)

    def test_pulsing_is_available_to_callers_that_want_it(self):
        plan = shape(40, 40, self.profile, allow_pulsing=True)
        self.assertTrue(plan.pulsed)
        self.assertLess(plan.duty, 1.0)

    def test_turning_uses_the_higher_threshold(self):
        """Both wheels scrub sideways in a turn, so it needs more than
        driving straight — measured: 86 did not turn the chassis, 108 did."""
        straight = shape(90, 90, self.profile, allow_pulsing=True)
        turning = shape(90, -90, self.profile, allow_pulsing=True)
        self.assertFalse(straight.pulsed)      # ~76 clears the straight 70
        self.assertTrue(turning.pulsed)        # ~76 does not clear the turn 105

    def test_boosting_preserves_the_turn_the_operator_asked_for(self):
        """Both wheels scale together, so the radius survives the boost.

        Checked in MOTOR units, not request units. The ESP's map is affine
        with a large MIN_PWM offset, so it compresses ratios hard: requests
        40 and 20 arrive as 55 and 46 — a ratio of 1.2, not 2.0. Wheel speed
        follows the motor value, so any reasoning about turn radius has to
        happen there; comparing request ratios would be comparing the wrong
        quantity."""
        plan = shape(40, 20, self.profile)
        before = request_to_motor_pwm(40) / request_to_motor_pwm(20)
        after = (request_to_motor_pwm(plan.left) / request_to_motor_pwm(plan.right))
        self.assertAlmostEqual(after, before, delta=0.08)

    def test_the_esp_map_compresses_ratios_which_is_why_motor_units_matter(self):
        """Pin the surprise itself, so the reasoning above is not lost."""
        self.assertAlmostEqual(request_to_motor_pwm(40) / request_to_motor_pwm(20),
                               1.2, delta=0.1)

    def test_duty_never_drops_to_a_lurch(self):
        plan = shape(6, 6, self.profile, allow_pulsing=True)
        self.assertGreaterEqual(plan.duty, 0.25)

    def test_below_the_esp_deadband_is_an_honest_stop(self):
        """Commanding 3 makes the ESP emit exactly nothing, so pretending
        otherwise would put a fictional action in the dataset."""
        plan = shape(3, 3, self.profile)
        self.assertEqual((plan.left, plan.right), (0, 0))
        self.assertIn("deadband", plan.reason)

    def test_reverse_is_handled_symmetrically(self):
        forward = shape(40, 40, self.profile, allow_pulsing=True)
        reverse = shape(-40, -40, self.profile, allow_pulsing=True)
        self.assertEqual(forward.left, -reverse.left)
        self.assertAlmostEqual(forward.duty, reverse.duty)


class EnergyNoteTests(unittest.TestCase):
    def setUp(self):
        self.profile = DriveProfile(straight_motor_pwm_min=70, turn_motor_pwm_min=105)

    def test_flags_a_stalling_command(self):
        note = energy_note(shape(40, 40, self.profile).__class__(40, 40), self.profile)
        self.assertTrue(note["stalling"])

    def test_a_shaped_command_never_stalls(self):
        note = energy_note(shape(40, 40, self.profile), self.profile)
        self.assertFalse(note["stalling"])

    def test_reports_whether_the_profile_was_ever_measured(self):
        """An unmeasured profile must be visible as such — placeholder
        thresholds that look authoritative are how a guess becomes a fact."""
        self.assertFalse(energy_note(shape(200, 200, self.profile), self.profile)
                         ["profile_measured"])


class ProfileIoTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "drive_profile.json"
            DriveProfile(straight_motor_pwm_min=66, turn_motor_pwm_min=99,
                         measured=True, notes="measured 2026-08-03").save(path)
            loaded = DriveProfile.load(path)
            self.assertEqual(loaded.straight_motor_pwm_min, 66)
            self.assertTrue(loaded.measured)

    def test_missing_file_gives_an_unmeasured_placeholder(self):
        profile = DriveProfile.load(Path("/nonexistent/drive_profile.json"))
        self.assertFalse(profile.measured)

    def test_corrupt_file_does_not_crash_the_robot_link(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "drive_profile.json"
            path.write_text("{not json")
            self.assertFalse(DriveProfile.load(path).measured)
