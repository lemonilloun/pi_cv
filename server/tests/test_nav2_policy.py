"""Арифметика окна локализатора — без модели и без весов.

Окно и есть вся защита от «телепорта», из-за которого прошлая навигация
закрыта. Проверяется оно логикой, а не чекпоинтом: замер на реальных кадрах
(scripts/eval_topomap.py) отвечает на другой вопрос — годится ли модель для
нашей камеры.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mac_server.nav2.policy import TopologicalLocalizer


class _FakePolicy:
    """Возвращает заданные дистанции. Модель здесь ни при чём."""

    def __init__(self, distance_of):
        self.distance_of = distance_of
        self.asked = []

    def predict(self, context, goals):
        self.asked.append(list(goals))
        dists = np.asarray([self.distance_of(g) for g in goals], dtype=float)
        waypoints = np.zeros((len(goals), 5, 2))
        return dists, waypoints


class WindowTest(unittest.TestCase):
    def setUp(self):
        self.topomap = list(range(20))

    def test_search_never_leaves_the_window(self):
        # Узел 19 «идеален», но лежит за окном — и не должен быть найден.
        policy = _FakePolicy(lambda g: 0.0 if g == 19 else 10.0)
        loc = TopologicalLocalizer(policy, self.topomap, radius=2, start_node=0)
        out = loc.step(list(range(6)))
        self.assertLessEqual(out["node"], 3)
        self.assertNotIn(19, policy.asked[0])

    def test_node_moves_at_most_radius_per_tick(self):
        # Прошлая навигация могла перескочить в любую точку карты за один
        # кадр. Здесь верхняя граница смещения — структурная.
        policy = _FakePolicy(lambda g: float(abs(g - 12)) if g != 12 else 20.0)
        loc = TopologicalLocalizer(policy, self.topomap, radius=3, start_node=0)
        first = loc.step(list(range(6)))["node"]
        self.assertLessEqual(first, 3 + 1)

    def test_window_advances_as_the_robot_moves(self):
        policy = _FakePolicy(lambda g: 10.0)
        loc = TopologicalLocalizer(policy, self.topomap, radius=2, start_node=0)
        loc.step(list(range(6)))
        loc.closest_node = 8
        loc.step(list(range(6)))
        # Окно асимметрично: назад radius, вперёд radius+1. Так в апстриме, и
        # это осмысленно — робот едет вперёд, поэтому следующий узел стоит
        # держать в поле зрения раньше, чем предыдущий из него уйдёт.
        self.assertEqual(min(policy.asked[-1]), 8 - 2)
        self.assertEqual(max(policy.asked[-1]), 8 + 2 + 1)

    def test_arriving_at_a_node_moves_the_target_on(self):
        # Иначе робот припаркуется на промежуточном узле и встанет.
        policy = _FakePolicy(lambda g: 0.5 if g == 4 else 9.0)
        loc = TopologicalLocalizer(policy, self.topomap, radius=4,
                                   close_threshold=3.0, start_node=4)
        out = loc.step(list(range(6)))
        self.assertEqual(out["node"], 5)

    def test_window_is_clamped_at_both_ends(self):
        policy = _FakePolicy(lambda g: 10.0)
        loc = TopologicalLocalizer(policy, self.topomap, radius=5, start_node=0)
        loc.step(list(range(6)))
        self.assertEqual(min(policy.asked[0]), 0)
        loc.closest_node = 19
        loc.step(list(range(6)))
        self.assertEqual(max(policy.asked[-1]), 19)


if __name__ == "__main__":
    unittest.main()
