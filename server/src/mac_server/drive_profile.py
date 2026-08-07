"""Turn a desired motion into wheel commands that actually move the robot.

Between "stop" and "moving" there is a band of commands that make the motors
draw current without turning the wheels. That band is not merely wasteful —
**it is the worst possible state for the battery.** A DC motor that cannot
turn draws its stall current, the largest it will ever draw, and every watt
of it becomes heat in the windings. Commanding just below the threshold costs
more than driving.

Measured on this chassis: an in-place turn requested at PWM 110 (which the
sketch maps to ~86 after `scaleCmd`) did not move it at all, while 160 (~108)
did. With three 300 mAh packs dying in 1-2 minutes, avoiding the stall band is
not a micro-optimisation.

Two rules follow, and they are the whole module:

1. **Never emit a command inside the dead band.** Round it up to the moving
   threshold, or down to a clean zero. There is no useful command in between.
2. **To go slower than the threshold, pulse.** Alternate above-threshold
   bursts with coasting, rather than holding a command that stalls. This is
   the standard way to creep with a stiction-limited drivetrain, and the
   `robocar` localization spin already works this way for an unrelated reason
   (motion blur), which is a useful precedent.

A third, smaller rule: **coast rather than brake.** The sketch's `BRAKE`
shorts the motor terminals; `STOP` lets them ramp down and freewheel. Braking
dissipates the robot's kinetic energy on purpose, so it is for emergencies,
not for ordinary stopping.

Thresholds here are placeholders until `scripts/measure_stiction.py` has run
on the real chassis with a healthy battery — they are deliberately named and
defaulted so an unmeasured profile is obvious rather than plausible.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PWM_LIMIT = 255

# The ESP's own mapping, mirrored so this module can reason in the units the
# motors actually see rather than in request units. See robocar_esp.ino:
#   if (abs(v) < 5) return 0;
#   m = map(abs(v), 5, 255, MIN_PWM, MAX_PWM)
ESP_DEADBAND_REQUEST = 5
ESP_MIN_PWM = 40
ESP_MAX_PWM = 150


def request_to_motor_pwm(request: int, min_pwm: int = ESP_MIN_PWM,
                         max_pwm: int = ESP_MAX_PWM) -> int:
    """What the motor actually receives for a given `M <v>` request.

    Mirrors `scaleCmd`. Without this the numbers in this module and the
    numbers on the wire mean different things, which is exactly how the spin
    ended up requesting 110 and delivering 86.
    """
    magnitude = abs(int(request))
    if magnitude < ESP_DEADBAND_REQUEST:
        return 0
    span = max_pwm - min_pwm
    scaled = min_pwm + (magnitude - ESP_DEADBAND_REQUEST) * span // (255 - ESP_DEADBAND_REQUEST)
    scaled = max(min_pwm, min(max_pwm, scaled))
    return scaled if request > 0 else -scaled


def motor_pwm_to_request(motor_pwm: int, min_pwm: int = ESP_MIN_PWM,
                         max_pwm: int = ESP_MAX_PWM) -> int:
    """Inverse of `request_to_motor_pwm`, rounded UP.

    Rounding up matters: rounding down would land the command back inside
    the stall band that the caller asked to escape.
    """
    magnitude = abs(int(motor_pwm))
    if magnitude <= 0:
        return 0
    span = max(max_pwm - min_pwm, 1)
    request = ESP_DEADBAND_REQUEST + math.ceil(
        (magnitude - min_pwm) * (255 - ESP_DEADBAND_REQUEST) / span
    )
    request = max(ESP_DEADBAND_REQUEST, min(255, request))
    return request if motor_pwm > 0 else -request


@dataclass
class DriveProfile:
    """Measured behaviour of this particular chassis and battery.

    `measured` stays False until `scripts/measure_stiction.py` has written
    real numbers, so an unmeasured profile is never mistaken for a
    characterized one.
    """

    straight_motor_pwm_min: int = 70
    turn_motor_pwm_min: int = 105
    max_pwm: int = ESP_MAX_PWM
    min_pwm: int = ESP_MIN_PWM
    measured: bool = False
    notes: str = "placeholder — run scripts/measure_stiction.py"

    @classmethod
    def load(cls, path: Path) -> "DriveProfile":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    def threshold_for(self, turning: bool) -> int:
        """Turning needs more than driving straight: both wheels scrub
        sideways against the floor instead of rolling."""
        return self.turn_motor_pwm_min if turning else self.straight_motor_pwm_min


@dataclass(frozen=True)
class DrivePlan:
    """What to send, and for how long.

    `duty` below 1.0 means the caller should alternate `wheels` with a stop:
    the only way to creep slower than the stall threshold without stalling.
    """

    left: int
    right: int
    duty: float = 1.0
    reason: str = "direct"

    @property
    def pulsed(self) -> bool:
        return self.duty < 0.999


def shape(
    left: int,
    right: int,
    profile: DriveProfile,
    allow_pulsing: bool = False,
    min_duty: float = 0.25,
) -> DrivePlan:
    """Move a wheel pair out of the stall band, pulsing if it has to.

    The turning/straight distinction comes from the wheels themselves — a
    pair with opposite signs, or very different magnitudes, is a turn — so
    callers do not have to remember to say which they meant.

    **`allow_pulsing` is off by default, and that is a deliberate trade.**
    Pulsing lets the robot creep slower than the stall threshold, which is
    genuinely useful for autonomous manoeuvres. But it costs the one thing
    the dataset cannot afford: the action log records a single command per
    decision, while a pulsed command is really an alternation between that
    command and a stop. The policy would then be trained on a label that was
    never what the wheels received. Losing the ability to creep is a small
    price; a systematically wrong action label is not.

    So for teleop and recording, a below-threshold request is simply rounded
    UP to the threshold. The robot moves faster than asked, the log says so,
    and observation and action stay consistent. Pulsing is available to
    callers that own their own action bookkeeping.

    `min_duty` floors how sparse the pulsing gets. Below roughly a quarter
    the robot lurches rather than creeps, and a lurching demonstration is
    poor training data as well as poor driving.
    """
    if left == 0 and right == 0:
        return DrivePlan(0, 0, reason="stop")

    turning = (left * right < 0) or (abs(left - right) > 0.3 * PWM_LIMIT)
    threshold = profile.threshold_for(turning)

    motor_left = request_to_motor_pwm(left, profile.min_pwm, profile.max_pwm)
    motor_right = request_to_motor_pwm(right, profile.min_pwm, profile.max_pwm)
    strongest = max(abs(motor_left), abs(motor_right))
    if strongest >= threshold:
        return DrivePlan(left, right, reason="above threshold")

    # Below the threshold. Scale the pair up until the stronger wheel clears
    # it, then pulse to give back the speed that scaling added. Both wheels
    # scale together so the turn radius the operator asked for is preserved.
    if strongest <= 0:
        return DrivePlan(0, 0, reason="below the ESP's own 5-unit deadband")
    boost = threshold / strongest
    boosted_left = motor_pwm_to_request(int(round(motor_left * boost)),
                                        profile.min_pwm, profile.max_pwm)
    boosted_right = motor_pwm_to_request(int(round(motor_right * boost)),
                                         profile.min_pwm, profile.max_pwm)
    if not allow_pulsing:
        return DrivePlan(
            boosted_left, boosted_right, duty=1.0,
            reason=f"raised: {strongest} would stall below {threshold}",
        )
    duty = max(min_duty, min(1.0, 1.0 / boost))
    return DrivePlan(
        boosted_left, boosted_right, duty=duty,
        reason=f"pulsed: {strongest} would stall below {threshold}",
    )


def energy_note(plan: DrivePlan, profile: DriveProfile) -> dict[str, Any]:
    """Diagnostics for the panel: what this command does to the battery.

    Not a wattage estimate — there is no current sensing on this robot. It
    reports the one thing that is knowable from the numbers alone: whether
    the command is in the stall band, which is the peak-drain state.
    """
    motor = max(abs(request_to_motor_pwm(plan.left, profile.min_pwm, profile.max_pwm)),
                abs(request_to_motor_pwm(plan.right, profile.min_pwm, profile.max_pwm)))
    turning = (plan.left * plan.right < 0) or (abs(plan.left - plan.right) > 0.3 * PWM_LIMIT)
    threshold = profile.threshold_for(turning)
    return {
        "motor_pwm": motor,
        "threshold": threshold,
        "stalling": 0 < motor < threshold,
        "duty": round(plan.duty, 3),
        "pulsed": plan.pulsed,
        "profile_measured": profile.measured,
    }
