"""Путевая точка -> намерение. Главное здесь — знак поворота.

На этом шасси уже ловилась перевёрнутая рулёжка (нажимаешь влево, едет
вправо): при перепутанном знаке движение вперёд-назад выглядит правильно,
и ошибка не видна до первого поворота. Поэтому знак пинается тестом, а не
описывается комментарием.
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mac_server.nav2.control import waypoint_to_drive
from mac_server.robocar import mix_drive


class SignTest(unittest.TestCase):
    def test_straight_ahead_does_not_steer(self):
        throttle, steer = waypoint_to_drive([0.05, 0.0])
        self.assertGreater(throttle, 0.9)
        self.assertAlmostEqual(steer, 0.0, places=6)

    def test_point_to_the_left_steers_left(self):
        # dy > 0 — влево по соглашению ROS; у нас влево это ОТРИЦАТЕЛЬНЫЙ
        # steer (панель шлёт D -> +1, то есть + это вправо).
        _, steer = waypoint_to_drive([0.05, 0.02])
        self.assertLess(steer, 0.0)

    def test_point_to_the_right_steers_right(self):
        _, steer = waypoint_to_drive([0.05, -0.02])
        self.assertGreater(steer, 0.0)

    def test_the_sign_survives_all_the_way_to_the_wheels(self):
        """Сквозная проверка до колёс.

        Знак может быть верным здесь и перевернуться в микшере — там стоит
        компенсация зеркальной распайки. Ошибиться дважды и получить
        правильный ответ тоже можно, поэтому проверяется цепочка целиком.
        """
        left_turn = mix_drive(*waypoint_to_drive([0.0, 0.05]), 120)
        right_turn = mix_drive(*waypoint_to_drive([0.0, -0.05]), 120)
        self.assertNotEqual(left_turn, right_turn)
        self.assertEqual(left_turn, tuple(-v for v in right_turn))


class ProportionalityTest(unittest.TestCase):
    def test_steer_grows_with_heading_error(self):
        # Закон апстрима здесь насыщался уже на 6° и робота раскачивало.
        angles = [5.0, 15.0, 30.0]
        steers = []
        for deg in angles:
            rad = math.radians(deg)
            _, steer = waypoint_to_drive([0.05 * math.cos(rad), 0.05 * math.sin(rad)])
            steers.append(abs(steer))
        self.assertLess(steers[0], steers[1])
        self.assertLess(steers[1], steers[2])
        self.assertLess(steers[0], 0.5)

    def test_throttle_falls_as_the_point_moves_off_axis(self):
        ahead, _ = waypoint_to_drive([0.05, 0.0])
        oblique, _ = waypoint_to_drive([0.035, 0.035])
        self.assertGreater(ahead, oblique)

    def test_a_near_point_asks_for_less_throttle(self):
        far, _ = waypoint_to_drive([0.05, 0.0])
        near, _ = waypoint_to_drive([0.01, 0.0])
        self.assertGreater(far, near)


class BehindTest(unittest.TestCase):
    def test_a_point_behind_turns_instead_of_reversing(self):
        throttle, steer = waypoint_to_drive([-0.04, 0.02])
        self.assertEqual(throttle, 0.0)
        self.assertLess(steer, 0.0)   # позади-слева -> доворот влево

    def test_a_point_behind_turns_the_short_way(self):
        """Апстримовский atan(dy/dx) теряет квадрант и доворачивает длинной
        стороной: точка позади-СЛЕВА получала бы поворот ВПРАВО."""
        _, left_behind = waypoint_to_drive([-0.04, 0.02])
        _, right_behind = waypoint_to_drive([-0.04, -0.02])
        self.assertLess(left_behind, 0.0)
        self.assertGreater(right_behind, 0.0)


class DegenerateTest(unittest.TestCase):
    def test_zero_waypoint_is_a_stop(self):
        self.assertEqual(waypoint_to_drive([0.0, 0.0]), (0.0, 0.0))

    def test_zero_offset_with_a_heading_turns_on_the_spot(self):
        # 4-мерная точка: смещения нет, курс есть — доворот без хода.
        throttle, steer = waypoint_to_drive([0.0, 0.0, 0.0, 1.0])
        self.assertEqual(throttle, 0.0)
        self.assertLess(steer, 0.0)

    def test_a_wrong_shaped_waypoint_is_refused(self):
        with self.assertRaises(ValueError):
            waypoint_to_drive([0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
