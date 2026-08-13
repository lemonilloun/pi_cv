"""Unit tests for mac_server.robocar: the pure logic behind driving.

The socket plumbing is exercised by the live run; what is tested here is
the arithmetic and the refusals — the parts that decide how fast a wheel
actually turns and which commands a browser is allowed to send.

    python3 -m unittest server.tests.test_robocar -v
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from mac_server import robocar  # noqa: E402
from mac_server.robocar import (  # noqa: E402
    DEFAULT_MAX_PWM,
    SPIN_SPEED,
    PWM_LIMIT,
    RobocarService,
    clamp_pwm,
    mix_drive,
    parse_drive_payload,
)


class MixDriveTests(unittest.TestCase):
    def test_straight_ahead_drives_both_wheels_equally(self):
        self.assertEqual(mix_drive(1.0, 0.0, 160), (160, 160))

    def test_reverse_is_symmetric(self):
        self.assertEqual(mix_drive(-1.0, 0.0, 160), (-160, -160))

    def test_a_true_pivot_counter_rotates(self):
        # Симметрия — инвариант пивота; величина несёт TURN_BOOST, потому что
        # вращение скребёт покрышками вбок, а езда нет. Пивот теперь надо
        # запрашивать явно: обычный поворот стал дугой, иначе на потяжелевшем
        # шасси он не выполняется вовсе.
        left, right = mix_drive(0.0, 1.0, 200, turn_assist=0.0)
        self.assertEqual(left, -right)
        self.assertGreater(abs(left), 200)

    def test_a_plain_turn_request_is_an_arc_with_both_wheels_driving(self):
        left, right = mix_drive(0.0, 1.0, 200)
        self.assertGreater(left, 0)
        self.assertGreater(right, 0)
        self.assertNotEqual(left, right)

    def test_a_spin_asks_for_more_effort_than_the_same_speed_straight(self):
        # The measured reason for TURN_BOOST: a request of 110 would not
        # rotate this chassis at all while manual driving at the same number
        # was fine, so a turn needs materially more than a straight line.
        straight = max(abs(v) for v in mix_drive(1.0, 0.0, 150))
        spin = max(abs(v) for v in mix_drive(0.0, 1.0, 150))
        self.assertGreater(spin, straight)

    def test_idle_is_exactly_zero(self):
        self.assertEqual(mix_drive(0.0, 0.0, 255), (0, 0))

    def test_saturation_costs_speed_not_the_turn(self):
        """Full throttle plus full steer used to come out as (120, 0): one
        wheel driving and the other DEAD. A geared motor with no power on it
        resists rather than freewheeling, so that wheel became a brake the
        other had to drag round while the caster scrubbed — the measured
        cause of "forward+left barely moves the car".

        Clamping now slides both wheels into range by the same amount, which
        preserves `left - right` exactly, so both motors push the rotation."""
        left, right = mix_drive(1.0, 1.0, 120)
        self.assertLess(min(left, right), 0, "inner wheel must drive, not idle")
        self.assertEqual(left, -right)

    def test_a_gentler_steer_gives_an_arc_with_both_wheels_forward(self):
        # Driving round a corner, as opposed to spinning: the outer wheel
        # leads, the inner one still pulls in the same direction.
        left, right = mix_drive(1.0, 0.3, 150)
        self.assertGreater(left, 0)
        self.assertGreater(right, 0)
        self.assertNotEqual(left, right)

    def test_steering_is_mirrored_for_this_chassis_wiring(self):
        """The motor leads are wired swapped relative to `M <l> <r>`: press
        A and the robot used to turn right. Forward/back hid it, because a
        swapped pair cancels when both wheels get the same sign."""
        self.assertEqual(mix_drive(1.0, 0.0, 100), (100, 100))   # unaffected
        # turn_assist=0: проверяется распайка, а не форма дуги.
        steer_right = mix_drive(0.0, 1.0, 100, turn_assist=0.0)
        steer_left = mix_drive(0.0, -1.0, 100, turn_assist=0.0)
        # Directions are what this test is about; magnitudes carry TURN_BOOST.
        self.assertLess(steer_right[0], 0)
        self.assertGreater(steer_right[1], 0)
        self.assertEqual(steer_right, tuple(-v for v in steer_left))

    def test_speed_is_clamped_to_the_pwm_range(self):
        self.assertEqual(mix_drive(1.0, 0.0, 9999), (PWM_LIMIT, PWM_LIMIT))
        self.assertEqual(mix_drive(1.0, 0.0, -50), (0, 0))


class ClampTests(unittest.TestCase):
    def test_clamps_and_survives_junk(self):
        self.assertEqual(clamp_pwm(300), PWM_LIMIT)
        self.assertEqual(clamp_pwm(-300), -PWM_LIMIT)
        self.assertEqual(clamp_pwm("80"), 80)
        self.assertEqual(clamp_pwm(None), 0)
        self.assertEqual(clamp_pwm("fast"), 0)


class ParsePayloadTests(unittest.TestCase):
    def test_accepts_explicit_wheel_values(self):
        body = json.dumps({"left": 100, "right": -100}).encode()
        cmd = parse_drive_payload(body)
        self.assertEqual((cmd.left, cmd.right), (100, -100))
        self.assertEqual(cmd.form, "wheels")
        self.assertIsNone(cmd.throttle)   # nothing to infer; do not invent one

    def test_accepts_and_mixes_a_throttle_steer_trio(self):
        body = json.dumps({"throttle": 1.0, "steer": 0.0, "speed": 90}).encode()
        cmd = parse_drive_payload(body)
        self.assertEqual((cmd.left, cmd.right), (90, 90))
        self.assertEqual(cmd.form, "mixed")

    def test_keeps_the_pre_mix_intent_alongside_the_wheels(self):
        """mix_drive swaps the wheels for this chassis' mirrored wiring. If
        that wiring is ever fixed, a wheels-only recording becomes silently
        wrong while the intent stays true — so both are kept."""
        body = json.dumps({"throttle": 0.0, "steer": -1.0, "speed": 100}).encode()
        cmd = parse_drive_payload(body)
        self.assertEqual((cmd.throttle, cmd.steer, cmd.speed), (0.0, -1.0, 100))
        # Оба колеса ведут: поворот с места стал дугой. Обесточенное
        # колесо редукторного мотора не катится, а тормозит.
        self.assertGreater(cmd.left, 0)
        self.assertGreater(cmd.right, 0)
        self.assertNotEqual(cmd.left, cmd.right)

    def test_clamps_out_of_range_throttle(self):
        body = json.dumps({"throttle": 5.0, "steer": 0.0, "speed": 100}).encode()
        cmd = parse_drive_payload(body)
        self.assertEqual((cmd.left, cmd.right), (100, 100))

    def test_rejects_malformed_json(self):
        self.assertIsNone(parse_drive_payload(b"{not json"))
        self.assertIsNone(parse_drive_payload(b"[1,2,3]"))

    def test_empty_body_is_a_valid_stop(self):
        cmd = parse_drive_payload(b"")
        self.assertEqual((cmd.left, cmd.right), (0, 0))


class CommandAllowlistTests(unittest.TestCase):
    """FORGET wipes the ESP's stored Wi-Fi credentials and REBOOT drops the
    link. Neither belongs one stray fetch away in a browser."""

    def setUp(self):
        self.service = RobocarService()

    def test_refuses_destructive_commands(self):
        for command in ("FORGET", "REBOOT", "M 255 255", "rm -rf /"):
            ok, detail = self.service.send_command(command)
            self.assertFalse(ok, command)
            self.assertIn("not allowed", detail)

    def test_allowed_command_gets_past_the_allowlist(self):
        """No ESP connected, so it fails at the link — but on 'no robot',
        not on the allowlist. That distinction is the test."""
        ok, detail = self.service.send_command("PING")
        self.assertFalse(ok)
        self.assertEqual(detail, "no robot connected")

    def test_max_is_parsed_and_clamped(self):
        ok, detail = self.service.send_command("MAX 9999")
        self.assertEqual(detail, "no robot connected")   # passed validation
        ok, detail = self.service.send_command("MAX fast")
        self.assertFalse(ok)
        self.assertIn("needs a number", detail)

    def test_drive_without_a_robot_reports_failure(self):
        self.assertFalse(self.service.drive(100, 100))

    def test_status_is_serializable_with_no_robot(self):
        status = self.service.status()
        self.assertFalse(status["connected"])
        json.dumps(status)   # the panel has to be able to render this


class SpinTests(unittest.TestCase):
    """An in-place survey spin. Timed, not measured: this chassis has no
    wheel encoders, and the IMU heading that used to close the loop came
    from the metric localization stack, which no longer exists."""

    def setUp(self):
        self.service = RobocarService()

    def test_refuses_without_a_robot(self):
        ok, detail = self.service.start_spin()
        self.assertFalse(ok)
        self.assertEqual(detail, "no robot connected")

    def test_spins_on_a_step_count(self):
        """The button must do something. "The button does nothing" was the
        worst failure mode of the earlier heading-gated version."""
        class _FakeLink:
            stop = threading.Event()
            addr = ("10.0.0.9", 5000)
            repeat_cmd = None
            repeat_until = 0.0
            last_hb = ""
            lines: list = []
            connected_at = 0.0

            def send(self, command):
                return True

        self.service._link = _FakeLink()
        ok, detail = self.service.start_spin()
        self.assertTrue(ok)
        self.assertIn("steps", detail)
        self.service.cancel_spin()
        if self.service._spin_thread:
            self.service._spin_thread.join(timeout=3)
        self.assertEqual(self.service.status()["spin"]["mode"], "timed")

    def test_status_exposes_an_idle_spin(self):
        spin = self.service.status()["spin"]
        self.assertFalse(spin["active"])
        self.assertEqual(spin["steps"], 0)

    def test_manual_drive_cancels_a_spin(self):
        """The operator grabbing the keys must win immediately, with no
        separate cancel step."""
        self.service._spin = {"active": True, "turned_deg": 90.0, "reason": None}

        class _FakeThread:
            def is_alive(self):
                return True

        self.service._spin_thread = _FakeThread()
        self.service.drive(100, 100)          # no robot, but the cancel still fires
        self.assertTrue(self.service._spin_cancel.is_set())
        self.assertEqual(self.service._spin["reason"], "operator took over")

    def test_internal_drive_does_not_cancel_the_spin(self):
        self.service._spin = {"active": True, "turned_deg": 90.0, "reason": None}

        class _FakeThread:
            def is_alive(self):
                return True

        self.service._spin_thread = _FakeThread()
        self.service.drive(100, 100, _internal=True)
        self.assertFalse(self.service._spin_cancel.is_set())


class MaxPwmTests(unittest.TestCase):
    """The sketch boots at MAX_PWM 80, tuned on a bare chassis; loaded with
    the Pi 5 and battery it barely moves."""

    def test_default_is_raised_above_the_sketch_default(self):
        self.assertGreaterEqual(DEFAULT_MAX_PWM, 150)
        self.assertEqual(RobocarService().max_pwm, DEFAULT_MAX_PWM)

    def test_clamped_to_the_sketch_accepted_range(self):
        self.assertEqual(RobocarService(max_pwm=5).max_pwm, 20)
        self.assertEqual(RobocarService(max_pwm=9999).max_pwm, PWM_LIMIT)


class SpinSpeedTests(unittest.TestCase):
    """Turning in place fights far more friction than driving straight, and
    the sketch maps a request onto MIN_PWM..MAX_PWM — so the first attempt
    at 110 against MAX_PWM 150 landed near 86 and did not move the chassis
    at all."""

    def test_spin_speed_clears_the_static_friction_floor(self):
        self.assertGreaterEqual(SPIN_SPEED, 150)

    def test_spin_command_is_a_pure_counter_rotation(self):
        # The spin passes turn_boost=0.0 because SPIN_SPEED was measured
        # against this exact scrub already — boosting it again would double
        # the compensation and put ~232 on the motors, past what the pulsed
        # spin was tuned for.
        left, right = mix_drive(0.0, 1.0, SPIN_SPEED, turn_boost=0.0, turn_assist=0.0)
        self.assertEqual(left, -right)
        self.assertEqual(abs(left), SPIN_SPEED)


class PulsedSpinTests(unittest.TestCase):
    """The spin pulses and stops rather than rotating continuously: the
    place index is matched on a CLIP embedding of ONE frame, and a frame
    grabbed mid-rotation is motion-blurred into something that matches
    nothing in an index built from a slow, sharp walk."""

    def test_settle_outlasts_a_nav_tick(self):
        """The nav loop runs at ~1.5 Hz. A settle shorter than that period
        could let a step go unphotographed, which defeats the purpose."""
        from mac_server.robocar import SPIN_SETTLE_S

        self.assertGreater(SPIN_SETTLE_S, 1.0 / 1.5)

    def test_a_single_pulse_cannot_approach_the_unwrap_ambiguity(self):
        """Accumulating with a shortest-arc unwrap is only valid while one
        step stays well under 180 deg."""
        from mac_server.robocar import SPIN_PULSE_S

        # Even at an implausibly fast 360 deg/s the chassis turns < 180 deg.
        self.assertLess(SPIN_PULSE_S * 360.0, 180.0)

    def test_blind_mode_is_bounded_by_steps(self):
        from mac_server.robocar import SPIN_BLIND_STEPS

        self.assertGreaterEqual(SPIN_BLIND_STEPS, 8)

    def test_status_reports_steps(self):
        self.assertIn("steps", RobocarService().status()["spin"])


class PanTests(unittest.TestCase):
    """The tripod's servo, used for 2-3 fixed look positions.

    Panning costs milliamps; rotating the chassis scrubs both wheels sideways
    and is the most expensive manoeuvre this robot has. Three views before a
    single wheel turns is worth real runtime on a robot whose battery dies in
    minutes.
    """

    class _FakeLink:
        addr = ("10.0.0.9", 5000)
        repeat_until = 0.0
        last_hb = ""
        connected_at = 0.0

        def __init__(self, repeat_cmd=None):
            self.stop = threading.Event()
            self.repeat_cmd = repeat_cmd
            self.lines = []
            self.sent = []

        def send(self, command):
            self.sent.append(command)
            return True

    def setUp(self):
        self.service = RobocarService()

    def test_refuses_without_a_robot(self):
        ok, detail = self.service.set_pan(30)
        self.assertFalse(ok)
        self.assertEqual(detail, "no robot connected")

    def test_refuses_while_the_wheels_are_driving(self):
        """The ESP shares timer1 between the servo and the motor PWM, so the
        sketch detaches the servo while driving. Refusing here gives a reason
        instead of a silently ignored command."""
        self.service._link = self._FakeLink(repeat_cmd="M 140 140")
        ok, detail = self.service.set_pan(30)
        self.assertFalse(ok)
        self.assertIn("while the wheels are driving", detail)

    def test_sends_the_command_when_parked(self):
        link = self._FakeLink()
        self.service._link = link
        ok, _ = self.service.set_pan(-30)
        self.assertTrue(ok)
        self.assertEqual(link.sent, ["PAN -30"])
        self.assertEqual(self.service.status()["pan_deg"], -30.0)

    def test_angle_is_clamped_to_the_mechanical_limit(self):
        """Clamped to the same number the sketch clamps to. If the server
        asked for more, the firmware would trim it silently and the server
        would believe the camera is somewhere it is not."""
        link = self._FakeLink()
        self.service._link = link
        self.service.set_pan(999)
        self.assertEqual(link.sent, [f"PAN {int(robocar.PAN_LIMIT_DEG)}"])
        self.assertEqual(self.service.status()["pan_deg"], robocar.PAN_LIMIT_DEG)

    def test_the_survey_never_asks_beyond_the_limit(self):
        for angle in robocar.PAN_ANGLES_DEG:
            self.assertLessEqual(abs(angle), robocar.PAN_LIMIT_DEG)

    def test_unknown_pan_is_never_reported_as_centred(self):
        """`None` means never commanded, which is not the same claim as
        'pointing straight ahead' — a survey must not skip a robot whose
        camera is at an unknown angle."""
        self.assertIsNone(self.service.status()["pan_deg"])

    def test_old_firmware_is_learned_from_the_reply_and_not_retried(self):
        self.service._link = self._FakeLink()
        self.service.note_pan_supported(False)
        ok, detail = self.service.set_pan(30)
        self.assertFalse(ok)
        self.assertIn("reflash", detail)

    def test_survey_visits_every_angle_and_recentres(self):
        link = self._FakeLink()
        self.service._link = link
        reached = self.service.pan_survey(angles=(-40.0, 0.0, 40.0), settle_s=0.001)
        self.assertEqual(reached, [-40.0, 0.0, 40.0])
        self.assertEqual(link.sent[-1], "PAN 0")     # ends centred

    def test_survey_stops_cleanly_on_old_firmware(self):
        self.service._link = self._FakeLink()
        self.service.note_pan_supported(False)
        self.assertEqual(self.service.pan_survey(settle_s=0.001), [])


class DeadbandTests(unittest.TestCase):
    """A wheel commanded between zero and the moving threshold is the worst
    of both worlds: a stalled DC motor draws its STALL current, the largest
    it ever draws, and turns nothing. That is the "energy going nowhere" the
    weak turns were spending."""

    def test_no_wheel_is_left_inside_the_stall_band(self):
        from mac_server.robocar import WHEEL_DEADBAND

        for steer in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            for speed in (60, 120, 200, 255):
                for throttle in (-1.0, -0.5, 0.0, 0.5, 1.0):
                    for wheel in mix_drive(throttle, steer, speed):
                        self.assertFalse(
                            0 < abs(wheel) < WHEEL_DEADBAND,
                            f"throttle={throttle} steer={steer} speed={speed} "
                            f"put {wheel} in the stall band")

    def test_a_full_stop_is_still_exactly_zero(self):
        # The snap must not invent motion out of an idle command.
        self.assertEqual(mix_drive(0.0, 0.0, 200), (0, 0))

    def test_the_turn_survives_the_snap(self):
        # Snapping the inner wheel up must not accidentally equalise the pair
        # and cancel the turn.
        left, right = mix_drive(1.0, 0.25, 200)
        self.assertNotEqual(left, right)


class WheelTrimTests(unittest.TestCase):
    """Моторы N20 при одинаковом PWM крутятся по-разному — замер одометрии
    показал это косвенно: команда была симметричной во всех шести проездах, а
    робота уводило настолько, что длинные проезды занижали скорость на 24%."""

    def test_trim_is_off_by_default(self):
        self.assertEqual(mix_drive(1.0, 0.0, 200), (200, 200))

    def test_trim_lands_on_the_wheel_it_names(self):
        # Слот 0 команды `M <l> <r>` — левое колесо. Ослабив ЛЕВОЕ, ожидаем
        # изменения именно в слоте 0. Микшер переставляет колёса внутри, и
        # проверка ловит, если подстройка уедет не туда.
        left, right = mix_drive(1.0, 0.0, 200, trim_left=0.9)
        self.assertLess(left, 200)
        self.assertEqual(right, 200)

    def test_trimming_the_right_wheel_touches_only_it(self):
        left, right = mix_drive(1.0, 0.0, 200, trim_right=0.9)
        self.assertEqual(left, 200)
        self.assertLess(right, 200)

    def test_trim_does_not_break_a_turn(self):
        left, right = mix_drive(1.0, -0.55, 200, trim_left=0.92)
        self.assertNotEqual(left, right)

    def test_trim_never_parks_a_wheel_in_the_stall_band(self):
        from mac_server.robocar import WHEEL_DEADBAND

        for trim in (0.5, 0.7, 0.9, 1.0):
            for speed in (60, 120, 200):
                for wheel in mix_drive(1.0, 0.0, speed, trim_left=trim):
                    self.assertFalse(0 < abs(wheel) < WHEEL_DEADBAND)
