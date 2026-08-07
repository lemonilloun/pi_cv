"""Unit tests for mac_server.vla.replay.

The replay video is the acceptance check for the whole dataset pipeline, so
the thing it ASSERTS has to be right. `turn_direction` is that assertion in
code form: it reads a wheel pair and says which way the robot turned, using
this chassis' swapped-wheel convention.

    python3 -m unittest server.tests.test_replay -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from mac_server.robocar import mix_drive  # noqa: E402
from mac_server.vla.replay import turn_direction  # noqa: E402


def sample(left, right):
    return {"raw_action": (left, right), "state": [0.0, 1.0, 0, 0, 0, 0, 0]}


class TurnDirectionTests(unittest.TestCase):
    def test_agrees_with_mix_drive_for_a_left_turn(self):
        """The decisive consistency check: whatever mix_drive produces when
        the operator steers LEFT must be read back as LEFT. These two live
        in different modules and could drift apart silently — this is what
        stops them."""
        left, right = mix_drive(throttle=0.0, steer=-1.0, speed=140)
        self.assertEqual(turn_direction(sample(left, right)), "LEFT")

    def test_agrees_with_mix_drive_for_a_right_turn(self):
        left, right = mix_drive(throttle=0.0, steer=+1.0, speed=140)
        self.assertEqual(turn_direction(sample(left, right)), "RIGHT")

    def test_forward_and_reverse(self):
        self.assertEqual(turn_direction(sample(*mix_drive(1.0, 0.0, 140))), "forward")
        self.assertEqual(turn_direction(sample(*mix_drive(-1.0, 0.0, 140))), "reverse")

    def test_stop(self):
        self.assertEqual(turn_direction(sample(0, 0)), "stop")

    def test_a_gentle_curve_still_reads_as_forward(self):
        """A small wheel difference is a drift, not a turn — labelling every
        1-PWM asymmetry as a turn would make the overlay unreadable."""
        self.assertEqual(turn_direction(sample(140, 135)), "forward")

    def test_a_real_curve_reads_as_a_turn(self):
        self.assertEqual(turn_direction(sample(140, 40)), "LEFT")

    def test_reverse_turns_are_still_named_by_wheel_order(self):
        """Reversing while turning: the naming stays wheel-based rather than
        trying to guess the operator's intent, so the overlay never
        contradicts the numbers printed beside it."""
        self.assertEqual(turn_direction(sample(-40, -140)), "LEFT")
