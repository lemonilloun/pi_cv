"""Локализация в топологическом графе кадров — общая для Pi и сервера.

Здесь нет ни модели, ни фреймворка: только арифметика окна поиска. Ровно
поэтому модуль общий — на Pi политика исполняется через onnxruntime, на
ноутбуке через torch, а правило «где я» обязано быть одним и тем же. Две
копии этой арифметики разошлись бы, и расхождение выглядело бы как «на
роботе почему-то иначе».

Политика передаётся любым объектом с методом `predict(context, goals)`,
возвращающим (дистанции, путевые точки).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class TopologicalLocalizer:
    """Текущий узел графа, обновляемый только локальным поиском.

    `radius` — вся защита от «телепорта»: узел может сместиться не более чем
    на radius за такт, поэтому переход между удалёнными частями карты
    физически не выражается в этой структуре. Глобальный аргмаксимум, на
    котором сломалась прошлая навигация, здесь не вычисляется никогда.
    """

    def __init__(self, policy: VintPolicy, topomap: list, radius: int = 4,
                 close_threshold: float = 3.0, start_node: int = 0) -> None:
        self.policy = policy
        self.topomap = topomap
        self.radius = radius
        self.close_threshold = close_threshold
        self.closest_node = start_node

    def step(self, context: list, goal_node: int | None = None) -> dict[str, Any]:
        """Один такт: обновить текущий узел и вернуть путевую точку."""
        goal = len(self.topomap) - 1 if goal_node is None else goal_node
        start = max(self.closest_node - self.radius, 0)
        end = min(self.closest_node + self.radius + 1, max(goal, start))
        goals = self.topomap[start:end + 1]
        dists, waypoints = self.policy.predict(context, goals)
        best = int(np.argmin(dists))

        # Дошли до узла — цель переносится на следующий, иначе робот
        # припаркуется на промежуточной точке и никуда не поедет.
        if dists[best] > self.close_threshold:
            node = start + best
            waypoint = waypoints[best]
        else:
            node = min(start + best + 1, goal)
            waypoint = waypoints[min(best + 1, len(waypoints) - 1)]
        self.closest_node = node
        return {
            "node": node,
            "distance": float(dists[best]),
            "waypoint": waypoint,
            "window": [start, end],
            "reached_goal": node >= goal and dists[best] <= self.close_threshold,
        }
