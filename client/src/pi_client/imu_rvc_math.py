"""Pure maths for the BNO085 UART-RVC link: time base, rates, motion state,
gravity, and heading correction.

Split out of `imu_rvc.py` on purpose — everything here is a function of numbers
and can be tested on a laptop with no sensor, no robot and no room. The driver
keeps the serial port; this keeps the reasoning.

The organising fact about RVC, from the datasheet and confirmed by measurement
on this rig, is that the link gives three things of very different quality:

* **pitch/roll** are tied to gravity and NEVER drift. They are the most
  valuable output for this project: a reconstruction from VGGT or a depth model
  arrives in an arbitrarily tilted frame, and without a true vertical there is
  no floor to find, no honest floor plan and no object height.
* **yaw** drifts (spec ~0.5 deg/min) and has no absolute zero. Excellent as a
  relative heading over seconds to tens of seconds; not a session-long compass.
* **accelerometer** is body-frame WITH gravity in it. Useful for detecting
  rest, vibration and knocks. Useless for position: a 0.5 deg tilt error is
  0.086 m/s^2 of phantom acceleration, which double-integrates to 0.43 m in
  10 s. No filter repairs that — the error is in the input.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

G_MS2 = 9.80665
MG_TO_MS2 = G_MS2 / 1000.0

# RVC streams at exactly 100 Hz; the frame index is the only trustworthy clock.
NOMINAL_DT_S = 0.01
# A fitted period outside this band means the fit, not the sensor, is wrong.
DT_SANITY = (0.008, 0.012)


class FrameClock:
    """Reconstructs a per-sample timestamp from the RVC frame index.

    **Why arrival time cannot be used.** Measured on this rig (CH340 adapter,
    `scripts/imu_rvc_check.py`, 2026-08-08): 100.2 Hz with zero checksum errors
    and zero dropped frames, but arrival intervals had a median of 0.00 ms, a
    p95 of 20.07 ms, and 50.2% of gaps under 1 ms. Frames land in PAIRS — two
    back to back, then a 20 ms wait — because the adapter buffers. Unlike an
    FT232RL the CH340 exposes no `latency_timer` to turn that off, so this is
    not a setting that can be fixed; it has to be modelled.

    Twenty milliseconds is not a rounding error here: at 30 deg/s the heading
    attached to a camera frame would be wrong by 0.6 deg.

    The device emits one frame every 10 ms and counts them, so
    `t = a * seq + b` recovers the true cadence. `a` is fitted from the ends of
    a long window rather than by least squares over all of it: the batching
    puts a sawtooth on the arrival times, and comparing means far apart cancels
    it, while a plain regression is pulled around by it.
    """

    def __init__(self, window: int = 2000, refit_every: int = 50) -> None:
        self.a = NOMINAL_DT_S
        self.b: float | None = None
        self.window = window
        self.refit_every = refit_every
        self._points: deque[tuple[int, float]] = deque(maxlen=window)
        self.fits = 0

    def update(self, seq: int, t_host: float) -> float:
        """Register a sample and return its reconstructed timestamp."""
        self._points.append((seq, t_host))
        if self.b is None:
            self.b = t_host - self.a * seq
        if len(self._points) >= 200 and seq % self.refit_every == 0:
            self._refit(seq)
        return self.a * seq + self.b

    def _refit(self, seq: int) -> None:
        points = list(self._points)
        quarter = max(1, len(points) // 4)
        head = points[:quarter]
        tail = points[-quarter:]
        x1 = sum(p[0] for p in head) / len(head)
        y1 = sum(p[1] for p in head) / len(head)
        x2 = sum(p[0] for p in tail) / len(tail)
        y2 = sum(p[1] for p in tail) / len(tail)
        if x2 <= x1:
            return
        a = (y2 - y1) / (x2 - x1)
        # A period outside 8-12 ms means the stream broke, not that the crystal
        # drifted. Keeping the previous fit is safer than following the noise.
        if not (DT_SANITY[0] < a < DT_SANITY[1]):
            return
        # Adopt the new RATE but keep the timeline continuous: pin the offset
        # so THIS sample keeps the timestamp the old fit would have given it.
        #
        # Taking `b` straight from the window mean instead makes the clock jump
        # at every refit — caught by a test, which saw a 5 ms step in an
        # otherwise perfectly 10 ms sequence. That step is not cosmetic: these
        # timestamps are interpolated between to answer "where was the camera
        # looking at this instant", and a discontinuity there is a lie about
        # when a sample happened. A clock may change its rate; it may not
        # travel.
        previous = self.a * seq + self.b
        self.a = a
        self.b = previous - a * seq
        self.fits += 1

    @property
    def rate_hz(self) -> float:
        return 1.0 / self.a if self.a else 0.0


def unwrap_deg(previous_raw: float | None, raw: float, accumulated: float) -> float:
    """Running offset that keeps yaw continuous across the +/-180 boundary.

    Applied to RAW degrees before any conversion. A step of 359.9 -> 0.1 is a
    0.2 deg turn, but read literally it is -359.8, and anything downstream that
    differentiates yaw then sees a colossal spike. Every consumer of yaw here
    wants the continuous value.
    """
    if previous_raw is None:
        return accumulated
    delta = raw - previous_raw
    if delta > 180.0:
        return accumulated - 360.0
    if delta < -180.0:
        return accumulated + 360.0
    return accumulated


def gravity_in_body(pitch_rad: float, roll_rad: float) -> tuple[float, float, float]:
    """Unit vector pointing DOWN, in the sensor frame.

    Depends only on pitch and roll — yaw is a rotation ABOUT gravity and cannot
    change where gravity points. That is precisely why this vector never
    drifts while yaw does, and why it is the part of RVC worth building on.
    """
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    return (sp, -cp * sr, -cp * cr)


def level_rotation(pitch_rad: float, roll_rad: float) -> list[list[float]]:
    """Rotation that levels a cloud (Z up) WITHOUT touching its heading.

    `R = Ry(pitch) @ Rx(roll)`, i.e. `rvc_to_R` with yaw set to zero. Applying
    it to points from a depth model or VGGT makes the floor horizontal, which
    is the single transform that unlocks floor finding, an honest orthographic
    floor plan, and metric object height.
    """
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    ry = [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    rx = [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]
    return [[sum(ry[i][k] * rx[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def rvc_to_matrix(yaw_rad: float, pitch_rad: float, roll_rad: float) -> list[list[float]]:
    """Body->world rotation for RVC's yaw->pitch->roll order.

    `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`. The order is not a free choice — it
    is what the datasheet specifies, and applying the three angles in any other
    sequence gives a different orientation for the same numbers.
    """
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    rz = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
    level = level_rotation(pitch_rad, roll_rad)
    return [[sum(rz[i][k] * level[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


class YawRateKF:
    """Angular rate from yaw, without the lag that smoothing would add.

    Differencing consecutive samples gives about 3 deg/s of noise: yaw is
    quantised to 0.01 deg and 0.01 deg over 0.01 s is already 1 deg/s. A
    low-pass would fix the noise and add phase lag, which is fatal here because
    the rate is used to decide whether a camera frame is motion-blurred — a
    lagged answer marks the wrong frames.

    A constant-acceleration Kalman filter is causal: it uses no future samples
    and adds no delay. `q_alpha` is the assumed std-dev of angular
    ACCELERATION; 2.0 rad/s^2 is the documented default, cutting rate noise
    about fourfold against plain differencing.
    """

    def __init__(self, dt_s: float = NOMINAL_DT_S,
                 sigma_theta_deg: float = 0.03, q_alpha: float = 2.0) -> None:
        self.dt = dt_s
        self.theta = 0.0
        self.omega = 0.0
        self.p = [[1e-2, 0.0], [0.0, 1e-1]]
        var = q_alpha ** 2
        self.q = [[var * dt_s ** 4 / 4.0, var * dt_s ** 3 / 2.0],
                  [var * dt_s ** 3 / 2.0, var * dt_s ** 2]]
        self.r = math.radians(sigma_theta_deg) ** 2
        self._started = False

    def step(self, theta_rad: float, dt_s: float | None = None) -> tuple[float, float]:
        """Feed one unwrapped yaw sample; returns (smoothed theta, omega)."""
        if not self._started:
            self.theta = theta_rad
            self._started = True
            return (self.theta, 0.0)
        dt = self.dt if dt_s is None else max(1e-4, dt_s)

        # Predict.
        self.theta += self.omega * dt
        p = self.p
        p00 = p[0][0] + dt * (p[1][0] + p[0][1]) + dt * dt * p[1][1] + self.q[0][0]
        p01 = p[0][1] + dt * p[1][1] + self.q[0][1]
        p10 = p[1][0] + dt * p[1][1] + self.q[1][0]
        p11 = p[1][1] + self.q[1][1]

        # Update on the angle only.
        s = p00 + self.r
        k0, k1 = p00 / s, p10 / s
        innovation = theta_rad - self.theta
        self.theta += k0 * innovation
        self.omega += k1 * innovation
        self.p = [[(1 - k0) * p00, (1 - k0) * p01],
                  [p10 - k1 * p00, p11 - k1 * p01]]
        return (self.theta, self.omega)


def motion_state(accel_ms2: list[tuple[float, float, float]],
                 yaw_span_rad: float, dt_s: float,
                 stationary_std_accel: float = 0.08,
                 stationary_omega_dps: float = 1.5,
                 vibration_std: float = 0.4,
                 impact_ms2: float = 13.0) -> dict[str, Any]:
    """Is the rig still, shaking, or was it knocked?

    Judged on the MAGNITUDE of acceleration, whose mean is gravity and whose
    spread is everything else. Deliberately not on the individual axes: a
    steady tilt changes the axes while the magnitude stays at 1 g, and tilting
    is not motion for this purpose.

    Feeds two decisions: whether a keyframe is worth keeping (a blurred or
    knocked frame poisons DINOv2 features and depth alike), and whether a
    stretch of driving can be trusted.
    """
    if not accel_ms2:
        return {"stationary": False, "std_accel": 0.0, "omega_dps": 0.0,
                "vibration": False, "impact": False, "samples": 0}
    magnitudes = [math.sqrt(x * x + y * y + z * z) for x, y, z in accel_ms2]
    mean = sum(magnitudes) / len(magnitudes)
    variance = sum((m - mean) ** 2 for m in magnitudes) / len(magnitudes)
    std = math.sqrt(variance)
    omega_dps = abs(math.degrees(yaw_span_rad)) / max(dt_s, 1e-3)
    return {
        "stationary": std < stationary_std_accel and omega_dps < stationary_omega_dps,
        "std_accel": round(std, 4),
        "mean_accel": round(mean, 4),
        "omega_dps": round(omega_dps, 3),
        "vibration": std > vibration_std,
        "impact": max(magnitudes) > impact_ms2,
        "samples": len(magnitudes),
    }


def manhattan_yaw_offset(wall_azimuths_rad: list[float]) -> tuple[float, float]:
    """Heading correction from the room's own walls, and how much to trust it.

    Rooms are overwhelmingly rectangular, so wall normals cluster at multiples
    of 90 degrees. Folding every azimuth into [0, 90) and taking a circular
    mean on the QUADRUPLED angle (which makes the four wall directions
    coincide) recovers how far the heading has slipped against the building.

    This is the only correction here that does not accumulate error: it is an
    observation of geometry, not an integration. Returns (offset, strength);
    strength near 1 means the walls agreed, near 0 means there were no walls
    worth listening to and the offset must be ignored.
    """
    if not wall_azimuths_rad:
        return (0.0, 0.0)
    folded = [a % (math.pi / 2.0) for a in wall_azimuths_rad]
    cos_sum = sum(math.cos(4.0 * a) for a in folded) / len(folded)
    sin_sum = sum(math.sin(4.0 * a) for a in folded) / len(folded)
    offset = math.atan2(sin_sum, cos_sum) / 4.0
    strength = math.hypot(cos_sum, sin_sum)
    return (offset, strength)


class YawDriftCorrector:
    """Slowly pulls IMU yaw onto an absolute reference.

    `alpha` is small on purpose. Heading feeds a pose graph, and a step change
    in it puts a discontinuity into the reconstruction — worse than the drift
    being corrected. Absolute observations arrive rarely and are trusted a
    little at a time.
    """

    def __init__(self, alpha: float = 0.02) -> None:
        self.bias = 0.0
        self.alpha = alpha
        self.observations = 0

    def correct(self, yaw_rad: float) -> float:
        return yaw_rad - self.bias

    def observe(self, yaw_imu_rad: float, yaw_absolute_rad: float,
                weight: float = 1.0) -> None:
        error = (yaw_imu_rad - self.bias) - yaw_absolute_rad
        error = (error + math.pi) % (2.0 * math.pi) - math.pi
        self.bias += self.alpha * max(0.0, min(1.0, weight)) * error
        self.observations += 1
