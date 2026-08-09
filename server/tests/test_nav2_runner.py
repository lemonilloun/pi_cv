"""Автономный цикл: проверяется не езда, а то, как он ПЕРЕСТАЁТ ехать.

Модели здесь нет — только поведение потока. Робот, который не умеет
надёжно останавливаться, опаснее робота, который не умеет ехать.
"""

import io
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mac_server.nav2.runner import Nav2Runner


class _Frame:
    def __init__(self, data):
        self.data = data


class _Store:
    """Отдаёт новый кадр на каждый опрос, пока `frozen` не выставлен."""

    def __init__(self):
        self.n = 0
        self.frozen = False

    def latest(self):
        if not self.frozen:
            self.n += 1
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (self.n % 255, 10, 20)).save(buf, "JPEG")
        return self.n, _Frame(buf.getvalue())


class _Policy:
    class config:
        context_size = 2


class _Robocar:
    def __init__(self):
        self.commands = []
        self.hooks = []
        self.lock = threading.Lock()

    def add_takeover_hook(self, hook):
        self.hooks.append(hook)

    def drive(self, left, right, **kwargs):
        with self.lock:
            self.commands.append((left, right))
        return True


class _Localizer:
    def __init__(self, reached=False):
        self.closest_node = 0
        self.reached = reached

    def step(self, context, goal_node=None):
        return {"node": 1, "distance": 5.0,
                "waypoint": np.array([[0.05, 0.0]] * 5),
                "window": [0, 2], "reached_goal": self.reached}


def _runner(store=None, robocar=None, localizer=None, **kwargs):
    runner = Nav2Runner(_Policy(), [0, 1, 2, 3], robocar or _Robocar(),
                        store or _Store(), **kwargs)
    runner.localizer = localizer or _Localizer()
    return runner


class SafetyTest(unittest.TestCase):
    def _wait_idle(self, runner, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not runner.status()["active"]:
                return True
            time.sleep(0.05)
        return False

    def test_operator_taking_over_stops_it(self):
        robocar = _Robocar()
        runner = _runner(robocar=robocar)
        runner.start()
        time.sleep(0.6)
        robocar.hooks[0]("operator took over")
        self.assertTrue(self._wait_idle(runner))
        self.assertIn("перехват", runner.status()["reason"])

    def test_it_always_ends_with_a_stop_command(self):
        # Что бы ни случилось, последнее, что получает привод, — ноль.
        robocar = _Robocar()
        runner = _runner(robocar=robocar)
        runner.start()
        time.sleep(0.6)
        runner.stop()
        self.assertTrue(self._wait_idle(runner))
        self.assertEqual(robocar.commands[-1], (0, 0))

    def test_frames_drying_up_stops_it(self):
        # Ехать по устаревшему кадру хуже, чем стоять: политика уверенно
        # ведёт туда, где робота уже нет.
        store = _Store()
        runner = _runner(store=store, stale_frame_s=0.3)
        runner.start()
        time.sleep(0.5)
        store.frozen = True
        self.assertTrue(self._wait_idle(runner))
        self.assertEqual(runner.status()["reason"], "кадры не приходят")

    def test_reaching_the_goal_stops_it(self):
        runner = _runner(localizer=_Localizer(reached=True))
        runner.start()
        self.assertTrue(self._wait_idle(runner))
        self.assertEqual(runner.status()["reason"], "цель достигнута")

    def test_a_crash_in_the_loop_still_stops_the_robot(self):
        class _Exploding(_Localizer):
            def step(self, context, goal_node=None):
                raise RuntimeError("бум")

        robocar = _Robocar()
        runner = _runner(robocar=robocar, localizer=_Exploding())
        runner.start()
        self.assertTrue(self._wait_idle(runner))
        self.assertIn("ошибка", runner.status()["reason"])
        self.assertEqual(robocar.commands[-1], (0, 0))

    def test_it_refuses_to_start_twice(self):
        runner = _runner()
        self.assertTrue(runner.start()[0])
        self.assertFalse(runner.start()[0])
        runner.stop()

    def test_it_refuses_without_a_robot(self):
        runner = Nav2Runner(_Policy(), [0, 1], None, _Store())
        ok, detail = runner.start()
        self.assertFalse(ok)
        self.assertIn("привод", detail)


if __name__ == "__main__":
    unittest.main()
