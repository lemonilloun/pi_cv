"""Приём путевых точек: проверяется, как он ПЕРЕСТАЁТ ехать.

Решение принимает робот, но останавливают его отсюда — и оператор, и
сторож по таймауту. Робот, который не умеет надёжно останавливаться,
опаснее робота, который не умеет ехать.
"""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mac_server.nav2 import relay as relay_mod
from mac_server.nav2.relay import Nav2Relay


class _Robocar:
    def __init__(self):
        self.commands = []
        self.hooks = []

    def add_takeover_hook(self, hook):
        self.hooks.append(hook)

    def drive(self, left, right, **kwargs):
        self.commands.append((left, right))
        return True


AHEAD = {"waypoint": [0.05, 0.0], "node": 3, "nodes": 20,
         "distance": 4.0, "speed": 120}


class RelayTest(unittest.TestCase):
    def setUp(self):
        self.robocar = _Robocar()
        self.relay = Nav2Relay(self.robocar)

    def tearDown(self):
        self.relay.shutdown()

    def test_a_waypoint_reaches_the_wheels(self):
        out = self.relay.handle(AHEAD)
        self.assertTrue(out["ok"])
        self.assertEqual(self.robocar.commands[-1], (out["left"], out["right"]))
        self.assertGreater(out["throttle"], 0.9)

    def test_a_point_to_the_left_steers_left(self):
        # Знак — тот же класс ошибки, что уже ловился на этом шасси:
        # вперёд-назад выглядит верно, и промах виден лишь на повороте.
        out = self.relay.handle({**AHEAD, "waypoint": [0.05, 0.02]})
        self.assertLess(out["steer"], 0.0)

    def test_an_explicit_stop_halts(self):
        self.relay.handle(AHEAD)
        self.relay.handle({"stop": True})
        self.assertEqual(self.robocar.commands[-1], (0, 0))
        self.assertFalse(self.relay.status()["active"])

    def test_the_operator_wins_immediately(self):
        """Ждать, пока робот узнает о перехвате, нельзя: между перехватом и
        доставкой сообщения он продолжал бы ехать."""
        self.relay.handle(AHEAD)
        self.robocar.hooks[0]("operator took over")
        self.assertEqual(self.robocar.commands[-1], (0, 0))
        out = self.relay.handle(AHEAD)
        self.assertFalse(out["ok"])
        self.assertEqual(self.robocar.commands[-1], (0, 0))

    def test_resume_is_explicit_after_a_takeover(self):
        self.relay.handle(AHEAD)
        self.robocar.hooks[0]("operator took over")
        self.assertFalse(self.relay.handle(AHEAD)["ok"])
        self.relay.resume()
        self.assertTrue(self.relay.handle(AHEAD)["ok"])

    def test_the_watchdog_stops_when_points_dry_up(self):
        """Прошивка глушит моторы только при обрыве СЕТИ. Замерший Pi
        держит соединение, и без своего сторожа последняя команда жила бы."""
        original = relay_mod.WAYPOINT_TTL_S
        relay_mod.WAYPOINT_TTL_S = 0.2
        try:
            self.relay.handle(AHEAD)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and self.relay.status()["active"]:
                time.sleep(0.05)
            self.assertFalse(self.relay.status()["active"])
            self.assertEqual(self.robocar.commands[-1], (0, 0))
            self.assertIn("перестали приходить", self.relay.status()["reason"])
        finally:
            relay_mod.WAYPOINT_TTL_S = original

    def test_a_malformed_waypoint_is_refused(self):
        with self.assertRaises(ValueError):
            self.relay.handle({"waypoint": [1, 2, 3]})


if __name__ == "__main__":
    unittest.main()
