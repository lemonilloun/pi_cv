"""Непрерывность локализации: взгляд в дверь не должен переносить робота."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.fix_gate import FixGate


class FixGateTest(unittest.TestCase):
    def test_first_fix_is_always_taken(self):
        self.assertTrue(FixGate().accept("room", (0.0, 0.0), 0.0)["accepted"])

    def test_glance_through_doorway_is_refused(self):
        # Ровно наблюдавшийся отказ: один кадр из комнаты в коридор.
        gate = FixGate()
        gate.accept("room", (0.0, 0.0), 0.0)
        verdict = gate.accept("hall", (8.0, 0.0), 0.7, similarity=0.71)
        self.assertFalse(verdict["accepted"])

    def test_real_transit_is_accepted(self):
        gate = FixGate()
        gate.accept("room", (0.0, 0.0), 0.0)
        for i in range(FixGate().confirmations):
            verdict = gate.accept("hall", (2.0, 0.0), 1.0 + i, similarity=0.75)
        self.assertTrue(verdict["accepted"])
        self.assertTrue(verdict["switched_session"])

    def test_a_single_stray_frame_does_not_count_toward_a_switch(self):
        # Подтверждения должны идти ПОДРЯД, иначе редкие ложные совпадения
        # накапливаются и рано или поздно всё равно уводят робота.
        gate = FixGate()
        gate.accept("room", (0.0, 0.0), 0.0)
        gate.accept("hall", (8.0, 0.0), 1.0, similarity=0.7)
        gate.accept("room", (0.1, 0.0), 2.0)          # сброс счётчика
        verdict = gate.accept("hall", (8.0, 0.0), 3.0, similarity=0.7)
        self.assertFalse(verdict["accepted"])

    def test_impossible_speed_within_one_room_is_refused(self):
        gate = FixGate()
        gate.accept("room", (0.0, 0.0), 0.0)
        verdict = gate.accept("room", (5.0, 0.0), 0.5)
        self.assertFalse(verdict["accepted"])
        self.assertEqual(gate.rejected, 1)

    def test_walking_pace_within_one_room_is_accepted(self):
        gate = FixGate()
        gate.accept("room", (0.0, 0.0), 0.0)
        self.assertTrue(gate.accept("room", (0.3, 0.0), 1.0)["accepted"])

    def test_overwhelming_match_switches_immediately(self):
        # Иначе робот, перенесённый руками в другую комнату, залипает в старой.
        gate = FixGate()
        gate.accept("room", (0.0, 0.0), 0.0)
        verdict = gate.accept("hall", (8.0, 0.0), 1.0, similarity=0.97)
        self.assertTrue(verdict["accepted"])


if __name__ == "__main__":
    unittest.main()
