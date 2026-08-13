"""Математика управления шасси. Ни железа, ни сети — только числа."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.drive_control import (
    ESP_MAX_PWM, ESP_MIN_PWM, INNER_WHEEL_RATIO, HeadingHold, KickStart, arc_turn, motor_pwm,
    request_for_motor_pwm, wrap_deg,
)


class MotorMappingTest(unittest.TestCase):
    def test_full_request_reaches_only_the_configured_ceiling(self):
        """Это и есть корень «моторчикам тяжело»: полный газ доходит до
        мотора как MAX_PWM из 255, а не как 255."""
        self.assertEqual(motor_pwm(255), ESP_MAX_PWM)
        self.assertLess(ESP_MAX_PWM / 255, 0.6)

    def test_the_spin_speed_that_did_not_move_the_chassis(self):
        # Замеренный факт: запрос 110 приходил на мотор как ~86 и не
        # проворачивал шасси, 160 приходит как ~108 и проворачивает.
        self.assertLess(motor_pwm(110), 90)
        self.assertGreater(motor_pwm(160), 100)

    def test_the_mapping_round_trips(self):
        for pwm in range(ESP_MIN_PWM, ESP_MAX_PWM + 1, 10):
            self.assertLessEqual(abs(motor_pwm(request_for_motor_pwm(pwm)) - pwm), 1)

    def test_raising_the_ceiling_raises_the_torque_available(self):
        self.assertGreater(motor_pwm(255, max_pwm=220), motor_pwm(255, max_pwm=150))

    def test_sign_is_preserved(self):
        self.assertEqual(motor_pwm(-255), -ESP_MAX_PWM)


class ArcTurnTest(unittest.TestCase):
    def test_a_turn_from_standstill_becomes_an_arc(self):
        """Разворот на месте на этом шасси недостижим: оба колеса скребут
        вбок. Дуга переводит скольжение в качение."""
        throttle, steer = arc_turn(0.0, 1.0)
        self.assertGreater(throttle, 0.0)
        self.assertLess(abs(steer), throttle)      # внутреннее колесо едет

    def test_both_wheels_keep_driving_forward(self):
        """Обесточенный редукторный мотор не катится, а сопротивляется —
        то есть становится тормозом, который второе колесо таскает."""
        throttle, steer = arc_turn(0.0, 1.0)
        inner = throttle - abs(steer)
        outer = throttle + abs(steer)
        self.assertGreater(inner, 0.0)
        self.assertAlmostEqual(inner / outer, INNER_WHEEL_RATIO, places=6)

    def test_the_pair_never_saturates_the_mixer(self):
        """Иначе приоритет поворота при насыщении сдвинет пару и выкинет
        ровно ту продольную составляющую, ради которой дуга и строится."""
        for magnitude in (0.1, 0.5, 1.0):
            throttle, steer = arc_turn(0.0, magnitude)
            self.assertLessEqual(throttle + abs(steer), 1.0 + 1e-9)

    def test_a_sharper_request_gives_a_tighter_arc(self):
        _, gentle = arc_turn(0.0, 0.2)
        _, sharp = arc_turn(0.0, 1.0)
        self.assertGreater(abs(sharp), abs(gentle))

    def test_a_gentle_nudge_is_nearly_straight(self):
        throttle, steer = arc_turn(0.0, 0.05)
        self.assertGreater(throttle, 0.9)
        self.assertLess(abs(steer), 0.05)

    def test_the_turn_direction_is_preserved(self):
        self.assertGreater(arc_turn(0.0, 0.6)[1], 0.0)
        self.assertLess(arc_turn(0.0, -0.6)[1], 0.0)

    def test_a_robot_already_moving_is_left_alone(self):
        # Этот случай работает и без нас: два мотора вперёд едут хорошо.
        self.assertEqual(arc_turn(1.0, 0.5), (1.0, 0.5))

    def test_straight_line_is_untouched(self):
        self.assertEqual(arc_turn(0.7, 0.0), (0.7, 0.0))

    def test_a_true_pivot_stays_available(self):
        # Обзорному спину нужно как раз не ехать.
        self.assertEqual(arc_turn(0.0, 1.0, assist=0.0), (0.0, 1.0))


class KickStartTest(unittest.TestCase):
    def test_moving_off_from_rest_gets_a_kick(self):
        kick = KickStart()
        out = kick.request((0, 0), (100, 100), now=0.0)
        self.assertIsNotNone(out)
        self.assertEqual(out, (255, 255))

    def test_the_kick_keeps_the_turn(self):
        """Бить обоими колёсами поровну значило бы поехать прямо ровно в тот
        момент, когда начинается поворот."""
        kick = KickStart()
        out = kick.request((0, 0), (100, -50), now=0.0)
        self.assertEqual(out[0], 255)
        self.assertLess(out[1], 0)
        self.assertLess(abs(out[1]), 255)

    def test_it_lasts_only_as_long_as_configured(self):
        kick = KickStart(duration_s=0.1)
        kick.request((0, 0), (100, 100), now=0.0)
        self.assertIsNotNone(kick.request((0, 0), (100, 100), now=0.05))
        self.assertIsNone(kick.request((50, 50), (100, 100), now=0.2))

    def test_a_moving_robot_is_not_kicked(self):
        kick = KickStart()
        self.assertIsNone(kick.request((100, 100), (120, 120), now=10.0))

    def test_a_reversal_is_kicked_too(self):
        # Переход через ноль — это проход зоны застревания дважды.
        kick = KickStart()
        self.assertIsNotNone(kick.request((100, 100), (-100, -100), now=10.0))

    def test_stopping_is_never_kicked(self):
        kick = KickStart()
        self.assertIsNone(kick.request((100, 100), (0, 0), now=10.0))

    def test_repeated_commands_do_not_hammer_the_motors(self):
        kick = KickStart(duration_s=0.05, cooldown_s=1.0)
        kick.request((0, 0), (100, 100), now=0.0)
        kick.request((0, 0), (100, 100), now=0.06)      # удар кончился
        self.assertIsNone(kick.request((0, 0), (100, 100), now=0.5))


class HeadingHoldTest(unittest.TestCase):
    def test_wrap_survives_the_zero_crossing(self):
        # Без этого ошибка в 359 градусов выкручивает руль не в ту сторону.
        self.assertAlmostEqual(wrap_deg(359.0), -1.0)
        self.assertAlmostEqual(wrap_deg(-181.0), 179.0)

    def test_drifting_left_is_corrected_to_the_right(self):
        # Курс растёт = уводит влево; вправо у нас положительный руль.
        hold = HeadingHold()
        hold.update(0.0, 0.0, now=0.0)
        corrected = hold.update(10.0, 0.0, now=0.1)
        self.assertGreater(corrected, 0.0)

    def test_drifting_right_is_corrected_to_the_left(self):
        hold = HeadingHold()
        hold.update(0.0, 0.0, now=0.0)
        self.assertLess(hold.update(-10.0, 0.0, now=0.1), 0.0)

    def test_a_constant_bias_is_eventually_cancelled(self):
        """Ради этого и нужна интегральная часть: чисто пропорциональный
        регулятор против постоянного увода всегда оставляет остаточную
        ошибку, потому что выдаёт поправку только пока ошибка есть."""
        hold = HeadingHold()
        hold.update(0.0, 0.0, now=0.0)
        t, heading = 0.0, 0.0
        for _ in range(200):
            t += 0.05
            correction = hold.update(heading, 0.0, now=t)
            # Модель: увод 6 град/с, руль правит 40 град/с на единицу.
            heading += (6.0 - 40.0 * correction) * 0.05
        self.assertLess(abs(heading), 3.0)
        self.assertGreater(abs(hold.integral), 0.0)

    def test_the_integral_cannot_wind_up_while_saturated(self):
        hold = HeadingHold(max_integral=5.0)
        hold.update(0.0, 0.0, now=0.0)
        t = 0.0
        for _ in range(100):
            t += 0.1
            hold.update(35.0, 0.0, now=t)          # большая стойкая ошибка
        self.assertLessEqual(abs(hold.integral), 5.0)

    def test_operator_steering_is_never_fought(self):
        hold = HeadingHold()
        hold.update(0.0, 0.0, now=0.0)
        hold.update(20.0, 0.0, now=0.5)
        self.assertEqual(hold.update(20.0, 0.8, now=1.0), 0.8)
        self.assertEqual(hold.integral, 0.0)

    def test_the_setpoint_relatches_after_a_turn(self):
        hold = HeadingHold()
        hold.update(0.0, 0.0, now=0.0)
        hold.update(90.0, 1.0, now=1.0)            # оператор развернул
        self.assertAlmostEqual(hold.setpoint, 90.0)
        self.assertAlmostEqual(hold.update(90.0, 0.0, now=1.1), 0.0)

    def test_a_silent_imu_disables_the_hold(self):
        """Замерший курс неотличим от ровного хода — так однажды уже
        пряталась поломка связи с датчиком."""
        hold = HeadingHold()
        hold.update(0.0, 0.0, now=0.0)
        self.assertEqual(hold.update(None, 0.3, now=1.0), 0.3)

    def test_standing_still_does_not_accumulate(self):
        hold = HeadingHold()
        hold.update(0.0, 0.0, now=0.0, moving=True)
        for i in range(20):
            hold.update(15.0, 0.0, now=1.0 + i * 0.1, moving=False)
        self.assertEqual(hold.integral, 0.0)

    def test_an_inverted_yaw_sign_is_caught_instead_of_spinning_the_robot(self):
        """Знак курса на этом стенде не проверен, а перевёрнутый превращает
        обратную связь в положительную: регулятор усиливает увод."""
        hold = HeadingHold()
        hold.update(0.0, 0.0, now=0.0)
        t, heading = 0.0, 0.0
        for _ in range(100):
            t += 0.05
            correction = hold.update(heading, 0.0, now=t)
            heading += (6.0 + 40.0 * correction) * 0.05   # знак перевёрнут
            if hold.diverged:
                break
        self.assertTrue(hold.diverged)
        self.assertLess(abs(heading), 90.0)

    def test_the_configured_sign_flips_the_correction(self):
        normal = HeadingHold()
        flipped = HeadingHold(yaw_sign=-1.0)
        for hold in (normal, flipped):
            hold.update(0.0, 0.0, now=0.0)
        self.assertGreater(normal.update(10.0, 0.0, now=0.1), 0.0)
        self.assertLess(flipped.update(10.0, 0.0, now=0.1), 0.0)


if __name__ == "__main__":
    unittest.main()
