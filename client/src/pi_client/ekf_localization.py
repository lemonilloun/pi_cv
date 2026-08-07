"""2D EKF for live robot localization in a room's floor-plan frame.

State [x, y, theta]: position in the plan frame (same convention
navindex.py/plan_frame.json use — meters, `theta` measured CCW from
`axis_a`, matching navindex._heading_deg), heading in radians.

Three measurement sources, each a standard textbook EKF update:

- `update_visual`: navindex.query()'s CLIP place-recognition fix — direct
  position+heading observation (H = identity). Loosest/tightest per-call
  covariance is the CALLER's job (from `similarity`/`heading_spread_deg`),
  not this module's — this stays pure filter math, no policy about how
  confident a given similarity score should be treated as.
- `update_landmark`: classic EKF-SLAM range-bearing observation against a
  known scene-graph object's world position (mapping/geometry.py's
  bbox_bearing_deg + a depth reading give the range/bearing pair).
- `update_range`: a single range measurement (e.g. forward depth vs. the
  occupancy grid's predicted wall distance) with a NUMERICALLY estimated
  Jacobian, since a grid map has no analytic "expected measurement"
  function. This is the shakiest of the three — occupancy-grid
  localization is more naturally a particle filter's job (that's what
  ROS's amcl does), forced into EKF form here because the position/
  heading state is already an EKF for the other two sources. Expect this
  term to be less reliable and worth revisiting if it misbehaves.

Predict step is an odometry motion model (Thrun/Burgard/Fox,
*Probabilistic Robotics*): given a body-frame displacement (forward,
lateral) and a heading change, rotate the displacement by the MIDPOINT
heading (theta + dtheta/2) — halves the heading-integration bias a naive
"rotate by the old heading" update would carry into position.

Important: the body-frame displacement comes from
imu_shtp_motion.ShtpMotionIntegrator's `delta_p_m`, which is a DOUBLE
INTEGRATION of accelerometer readings — reliable for the fraction of a
second between visual fixes, not for dead-reckoning over any longer
horizon (this project's own measurements: 78 m/s of accumulated drift
over 215s on session_20260726_193117). `predict`'s process noise grows
with both distance AND elapsed time for exactly this reason — the second
term is what keeps the filter from over-trusting a "the robot didn't
move much" segment that in fact only failed to accumulate visible drift
yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Ekf2D:
    x: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P: np.ndarray = field(default_factory=lambda: np.eye(3) * 1e-3)

    @classmethod
    def initialize(
        cls, x0: float, y0: float, theta0: float, pos_std: float = 0.3, theta_std_deg: float = 30.0
    ) -> "Ekf2D":
        """Start from a visual fix's own uncertainty rather than a fake
        zero-covariance origin — the initial state is exactly as trustworthy
        as whatever localized it (e.g. the spin-in-place warm-up fixes)."""
        p = np.diag([pos_std ** 2, pos_std ** 2, math.radians(theta_std_deg) ** 2])
        return cls(x=np.array([x0, y0, theta0], dtype=np.float64), P=p)

    # ------------------------------------------------------------ predict

    def predict(
        self,
        forward_m: float,
        lateral_m: float,
        dtheta_rad: float,
        dt_s: float,
        dist_noise_frac: float = 0.5,
        time_noise_per_s: float = 0.05,
        theta_noise_per_rad: float = 0.3,
        theta_noise_per_s: float = 0.02,
    ) -> None:
        """Advance the state by one IMU segment.

        `forward_m`/`lateral_m`: body-frame displacement since the last
        predict (imu_shtp_motion's delta_p_m[0]/[1] — X-forward/Y-right
        rig mounting). `dtheta_rad`: heading change since the last predict
        (from consecutive Madgwick orientation samples' yaw, or the
        gyro-integrated yaw directly).
        """
        theta = self.x[2]
        theta_mid = theta + dtheta_rad / 2.0
        c, s = math.cos(theta_mid), math.sin(theta_mid)
        dx = forward_m * c - lateral_m * s
        dy = forward_m * s + lateral_m * c

        self.x[0] += dx
        self.x[1] += dy
        self.x[2] = _wrap_angle(theta + dtheta_rad)

        # Jacobian of the motion model w.r.t. state, evaluated at theta_mid
        # (the same linearization point the update itself uses).
        F = np.array([
            [1.0, 0.0, -dy],
            [0.0, 1.0, dx],
            [0.0, 0.0, 1.0],
        ])

        dist = math.hypot(forward_m, lateral_m)
        # Both terms matter: a fast, large motion is uncertain because
        # double-integrated accel error grows with distance; a slow or
        # stationary segment is uncertain because bias drift grows with
        # time regardless of whether it produced visible displacement yet.
        pos_var = (dist_noise_frac * dist) ** 2 + (time_noise_per_s * dt_s) ** 2
        theta_var = (theta_noise_per_rad * abs(dtheta_rad)) ** 2 + (theta_noise_per_s * dt_s) ** 2
        Q = np.diag([pos_var, pos_var, theta_var])

        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------- visual fix

    def update_visual(self, x_meas: float, y_meas: float, theta_meas: float, R: np.ndarray) -> None:
        z = np.array([x_meas, y_meas, theta_meas])
        innovation = z - self.x
        innovation[2] = _wrap_angle(innovation[2])
        H = np.eye(3)
        self._apply(innovation, H, R)

    # ----------------------------------------------------------- landmark

    def update_landmark(
        self, landmark_x: float, landmark_y: float, bearing_meas: float, range_meas: float, R: np.ndarray
    ) -> None:
        """Standard EKF-SLAM range-bearing observation of a KNOWN landmark
        (the scene graph object's world position isn't itself a filter
        state — only the robot's pose is)."""
        x, y, theta = self.x
        dx, dy = landmark_x - x, landmark_y - y
        range_pred = math.hypot(dx, dy)
        if range_pred < 1e-6:
            return
        bearing_pred = _wrap_angle(math.atan2(dy, dx) - theta)

        innovation = np.array([
            range_meas - range_pred,
            _wrap_angle(bearing_meas - bearing_pred),
        ])
        q = range_pred ** 2
        H = np.array([
            [-dx / range_pred, -dy / range_pred, 0.0],
            [dy / q, -dx / q, -1.0],
        ])
        self._apply(innovation, H, R)

    # --------------------------------------------------------- range-only

    def update_range(self, predict_range_fn, range_meas: float, R: float, eps: float = 1e-3) -> None:
        """A single range measurement (e.g. forward depth vs. the occupancy
        grid's predicted wall distance along the current heading).
        `predict_range_fn(x, y, theta) -> float` does the ray-marching
        (grid-specific — deliberately not this module's concern); its
        Jacobian is estimated numerically since a grid has no closed form.
        `R` is a scalar variance, not a matrix, since this is a 1D
        observation."""
        x, y, theta = self.x
        range_pred = predict_range_fn(x, y, theta)
        if range_pred is None:
            return
        innovation = np.array([range_meas - range_pred])

        # A perturbed probe can legitimately fall off the grid/ray-march
        # (None) near a boundary even when the center evaluation didn't —
        # treat that axis's sensitivity as unknown (0) rather than crash,
        # since a real grid-based predict_range_fn will hit this in
        # practice, not just in a synthetic test.
        def _central_diff(plus, minus) -> float:
            if plus is None or minus is None:
                return 0.0
            return (plus - minus) / (2 * eps)

        h_x = _central_diff(predict_range_fn(x + eps, y, theta), predict_range_fn(x - eps, y, theta))
        h_y = _central_diff(predict_range_fn(x, y + eps, theta), predict_range_fn(x, y - eps, theta))
        h_theta = _central_diff(
            predict_range_fn(x, y, theta + eps), predict_range_fn(x, y, theta - eps)
        )
        H = np.array([[h_x, h_y, h_theta]])
        self._apply(innovation, H, np.array([[R]]))

    # ------------------------------------------------------------- shared

    def _apply(self, innovation: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[2] = _wrap_angle(self.x[2])
        I = np.eye(3)
        # Joseph form: numerically stabler than (I - KH) @ P alone (stays
        # symmetric/PSD under roundoff, matters over a long-running session).
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def heading_rad(self) -> float:
        return float(self.x[2])

    @property
    def position_std_m(self) -> tuple[float, float]:
        return float(math.sqrt(self.P[0, 0])), float(math.sqrt(self.P[1, 1]))


# ---------------------------------------------------------------------------
# Heading filter for the UART-RVC link
# ---------------------------------------------------------------------------


@dataclass
class Ekf2DHeading:
    """4-state filter for orientation-only navigation on the BNO08x UART-RVC link.

    State: [x, y, theta, b] — position and heading in the plan frame, plus
    `b`, the offset between the IMU's yaw datum and the map's heading zero.

    **Why a separate filter instead of reusing `Ekf2D`.** `Ekf2D.predict`
    consumes a heading *increment* (dtheta between two samples). That was
    the right shape for the SHTP link, where heading came from integrating
    raw gyro on the host and only differences were meaningful. UART-RVC
    reports a *fused, absolute* yaw computed on-chip at the sensor's full
    internal rate. Differencing it to recover an increment and then letting
    theta random-walk between fixes throws away the strongest signal this
    link has: measured on this device, yaw drifts **0.03 deg/min** with the
    rig stationary (90 s capture, 9011 samples). Absolute yaw held that
    steady is worth using as an absolute observation.

    **What `b` is for.** RVC yaw is relative — its zero is wherever the
    sensor happened to power up, and it carries no magnetometer, so it has
    no idea where the room's north is. The map frame's heading zero comes
    from `navindex`/`plan_frame.json`. The constant between them is
    unknown at startup and drifts slowly afterwards. Modelling it as a
    state makes it *observable*: each visual fix pins theta, and the
    difference against the concurrent IMU yaw identifies `b`. After a
    handful of fixes `b` is known to well under a degree, and from then on
    the IMU alone supplies absolute map heading between fixes rather than
    merely "you turned by this much".

    **Position is deliberately not dead-reckoned.** RVC reports no raw
    gyro and this rig has no wheel odometry, so there is no displacement
    source worth integrating — and the accelerometer route was measured at
    78x too large on this platform (see `Ekf2DVio`). `predict` therefore
    grows position uncertainty at a realistic platform speed and leaves
    x/y to the visual fix. That is the same validated behaviour the
    heading-only `Ekf2D` path had; what changes here is only that heading
    got much better.

    IMU yaw enters ONLY through `update_imu_yaw`, never through `predict`.
    Using it in both would count one measurement twice and make the filter
    overconfident.

    **The yaw handed in must already be in the plan frame's sense** (CCW
    positive). `b` is an additive offset and cannot absorb a SIGN, so a
    sensor whose convention runs the other way would make this filter steer
    the estimate backwards on every turn while looking perfectly healthy.
    The BNO08x's RVC output is a compass heading (CW positive) and is
    normalized at the driver boundary — see `imu_rvc.YAW_SIGN`, which was
    measured with two full turns against a floor mark rather than assumed.
    """

    x: np.ndarray = field(default_factory=lambda: np.zeros(4))
    P: np.ndarray = field(default_factory=lambda: np.eye(4) * 1e-3)

    @classmethod
    def initialize(
        cls,
        x0: float,
        y0: float,
        theta0: float,
        pos_std: float = 0.3,
        theta_std_deg: float = 30.0,
        bias_std_deg: float = 180.0,
    ) -> "Ekf2DHeading":
        """`bias_std_deg` defaults to 180 deg — a genuinely uninformative
        prior, because the RVC datum really is arbitrary at power-up. A
        tight prior here would fight the first few fixes instead of
        learning from them."""
        state = np.array([x0, y0, theta0, 0.0], dtype=np.float64)
        p = np.diag([
            pos_std ** 2,
            pos_std ** 2,
            math.radians(theta_std_deg) ** 2,
            math.radians(bias_std_deg) ** 2,
        ])
        return cls(x=state, P=p)

    # ------------------------------------------------------------ predict

    def predict(
        self,
        dt_s: float,
        speed_ms: float = 0.6,
        turned_rad: float | None = None,
        turn_noise_frac: float = 0.30,
        turn_rate_rad_s: float = 1.0,
        turn_rate_floor_rad_s: float = 0.02,
        bias_walk_per_s: float = 0.0002,
        forward_m: float | None = None,
        lateral_m: float = 0.0,
        motion_noise_frac: float = 0.6,
    ) -> None:
        """Advance time, and move the state if a displacement is supplied.

        **`forward_m`/`lateral_m` are body-frame displacement since the last
        predict**, normally `imu.segment.delta_p_m` from the RVC reader. Pass
        them and the marker moves when the robot drives; omit them and this
        degrades to the previous behaviour — pure uncertainty growth, the
        position frozen until the next visual fix.

        Feeding IMU displacement here is safe in a way that feeding it to the
        scene3d metric scale was NOT, and the difference is the correction
        interval rather than the sensor. Double-integrated acceleration drifts
        quadratically, so with this rig's ~3 deg of gravity error:

            uncorrected for 153 s (scene scale)  ->  kilometres, unusable
            corrected every 0.7 s (this filter)  ->  0.13 m
            a 3 s visual dropout                 ->  2.3 m

        The filter's job is to be right between fixes, not for three minutes,
        and 0.13 m of prediction error is far better than the alternative of
        not moving at all.

        `motion_noise_frac` (0.6) is the share of the reported displacement
        taken as its own 1-sigma error — deliberately large, because the
        magnitude of an accelerometer-derived step is the least trustworthy
        thing about it. Direction comes from the filter's heading `theta`,
        which the IMU yaw keeps tight, so a step lands in roughly the right
        direction with a generous doubt about how far. That asymmetry is the
        whole reason this is worth doing: a wrong distance along a right
        bearing is corrected by the next fix, and in the meantime the marker
        tracks the robot instead of sitting still.

        The caller must not pass displacement it cannot justify — in
        particular, a segment the driver marked `gravity_ok: false` means the
        rig left the pose its gravity was calibrated in, and the reported
        distance is then mostly un-subtracted gravity.

        `speed_ms` is the platform's realistic top speed: with no
        displacement source, "how far could it possibly have gone in
        dt_s" is the honest position uncertainty.

        **`turned_rad` is how much the IMU says the robot actually turned
        since the last predict, and passing it is what keeps the estimate
        steady.** Without it this falls back to a blanket
        `turn_rate_rad_s` envelope, which says "the heading could have
        changed by up to 57 deg/s no matter what" — at the ~1.5 Hz
        navigation rate that grows the heading variance by (38 deg)^2
        every single tick and effectively deletes the prior. The estimate
        then follows whichever measurement arrived last, and since the
        visual heading carries tens of degrees of spread, the fused marker
        visibly jumps. That was the observed behaviour on the first real
        run, and it was a parameter mistake, not a sensor one.
        Uncertainty should track what the robot DID, not what it could
        conceivably have done, so with `turned_rad` supplied the growth is
        a small fraction of the measured turn plus a floor for the time
        elapsed.

        Using the measured turn to size the NOISE is not double-counting
        the measurement: the state still moves only through
        `update_imu_yaw`. This is adaptive process noise, not a second
        update.

        `turn_noise_frac` is 0.30, and it is NOT the sensor's error (the
        IMU's turn gain measured 0.996-1.003, so 0.4%). It is the prior on
        how a measured yaw change should be apportioned. theta and b are
        only ever observed as `theta - b`, and the split follows their
        variance ratio — so during a turn theta's variance has to grow
        enough to dominate the bias variance that 40 visual fixes have
        already tightened to ~0.2 deg^2. Worked through for a 90 deg turn
        in ten 9 deg steps: 0.05 gives theta 90% of it (81 of 90 deg, and
        the filter comes out 14 deg short), 0.15 gives 98.8%, 0.30 gives
        99.7%, and beyond that the gain is pennies. Being generous here is
        the physically correct prior: a chassis can swing 90 deg in two
        seconds, a fusion datum cannot.

        The floor and `bias_walk_per_s` must stay far apart. theta and b
        enter `update_imu_yaw` only as the difference `theta - b`, so a
        change in IMU yaw is ambiguous between "the robot turned" and "the
        datum drifted", and the filter splits it in proportion to their
        variances. An early draft used 0.02 rad/s as the *whole* heading
        term, which says the robot turns no faster than 1 deg/s; a 90 deg
        turn during a visual dropout was then charged half to bias and the
        fused heading came out 10 deg short (caught by
        test_converged_bias_makes_imu_yaw_absolute_between_fixes). Here
        0.02 is only the floor, and a real turn raises it through
        `turned_rad`.

        `bias_walk_per_s` defaults to 0.0002 rad/s (~0.7 deg/min), a
        deliberately loose envelope around the 0.03 deg/min measured
        stationary — the rate under sustained motion was not measured, so
        the model allows more drift than the bench figure rather than
        pretending the bench figure holds while driving.
        """
        if dt_s <= 0:
            return

        if forward_m is None:
            # No displacement source: "how far could it possibly have gone"
            # is the honest uncertainty, and the state stays put.
            pos_var = (speed_ms * dt_s) ** 2
        else:
            # Rotate the body-frame step into the plan frame using the
            # filter's OWN heading, not the IMU's raw yaw. theta is the same
            # quantity the position lives in, and it already carries the
            # yaw-datum offset `b` that makes IMU yaw meaningful on this map;
            # using the raw yaw here would steer every step by that offset.
            theta = float(self.x[2])
            c, s_ = math.cos(theta), math.sin(theta)
            self.x[0] += forward_m * c - lateral_m * s_
            self.x[1] += forward_m * s_ + lateral_m * c
            step = math.hypot(forward_m, lateral_m)
            # Scale doubt with the step taken, with the same time-based floor
            # as before so a long tick is never treated as certain just
            # because the reported step was small.
            pos_var = (motion_noise_frac * step) ** 2 + (speed_ms * dt_s * 0.2) ** 2

        self.P[0, 0] += pos_var
        self.P[1, 1] += pos_var
        if turned_rad is None:
            theta_var = (turn_rate_rad_s * dt_s) ** 2
        else:
            theta_var = ((turn_noise_frac * abs(turned_rad)) ** 2
                         + (turn_rate_floor_rad_s * dt_s) ** 2)
        self.P[2, 2] += theta_var
        self.P[3, 3] += (bias_walk_per_s * dt_s) ** 2

    # -------------------------------------------------------- IMU heading

    def update_imu_yaw(self, yaw_imu_rad: float, sigma_deg: float = 1.0) -> None:
        """Observe the RVC fused yaw: z = theta - b.

        The measurement is informative about theta only once `b` has been
        identified by visual fixes, and informative about `b` only once
        theta has. The filter handles that automatically through the
        covariance — before the first fix this update mostly just ties the
        two together, which is exactly right.
        """
        innovation = np.array([_wrap_angle(yaw_imu_rad - (self.x[2] - self.x[3]))])
        H = np.array([[0.0, 0.0, 1.0, -1.0]])
        R = np.array([[math.radians(sigma_deg) ** 2]])
        self._apply(innovation, H, R)

    # ------------------------------------------------------- visual fix

    def update_visual(self, x_meas: float, y_meas: float, theta_meas: float, R: np.ndarray) -> None:
        innovation = np.array([
            x_meas - self.x[0],
            y_meas - self.x[1],
            _wrap_angle(theta_meas - self.x[2]),
        ])
        H = np.zeros((3, 4))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        self._apply(innovation, H, R)

    def update_position_only(self, x_meas: float, y_meas: float, R: np.ndarray) -> None:
        """Position fix with no usable heading (navindex returned a match
        whose neighbours disagree too much on facing direction). Keeping
        heading out of the update rather than feeding it with an inflated
        covariance avoids dragging a well-converged `b` toward noise."""
        innovation = np.array([x_meas - self.x[0], y_meas - self.x[1]])
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        self._apply(innovation, H, R)

    # ------------------------------------------------------------- shared

    def _apply(self, innovation: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[2] = _wrap_angle(self.x[2])
        self.x[3] = _wrap_angle(self.x[3])
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def heading_rad(self) -> float:
        return float(self.x[2])

    @property
    def yaw_bias_rad(self) -> float:
        return float(self.x[3])

    @property
    def yaw_bias_std_deg(self) -> float:
        """How well the IMU datum is pinned to the map frame. Watching this
        fall is how you tell the filter is actually fusing rather than just
        echoing the latest visual fix."""
        return math.degrees(math.sqrt(self.P[3, 3]))

    @property
    def heading_std_deg(self) -> float:
        return math.degrees(math.sqrt(self.P[2, 2]))

    @property
    def position_std_m(self) -> tuple[float, float]:
        return float(math.sqrt(self.P[0, 0])), float(math.sqrt(self.P[1, 1]))


# ---------------------------------------------------------------------------
# Bias-estimating VIO filter
# ---------------------------------------------------------------------------


@dataclass
class Ekf2DVio:
    """7-state filter that makes the accelerometer actually useful.

    State: [px, py, theta, vx, vy, bax, bay] — position and velocity in the
    plan frame, heading, and the accelerometer's BODY-frame bias.

    Why this exists (all measured on real data from this rig, not assumed):

    * Feeding `segment.delta_p_m` straight in is hopeless — over
      session_20260730_161007 it claimed 667 m for a walk VGGT measured at
      ~8.5 m (78x). Reproduced from the raw logs: a residual horizontal
      acceleration of **0.81 m/s²**, i.e. a ~4.7° error between the static
      gravity reference and the rig's true tilt while moving.
    * Tracking gravity live with the Madgwick quaternion instead did NOT
      help (leak got worse: 1.55 m/s² on the same log) — measured, so this
      filter does not do that.
    * An inertial ZUPT cannot rescue it either: on the `forward` log 82% of
      samples pass a textbook stationarity test, because at constant
      velocity an accelerometer genuinely reads the same as at rest. Zeroing
      velocity there would delete the very motion we want.

    What is left, and what this class does, is the standard VIO answer:
    treat the bias as an unknown to be ESTIMATED, and let an independent
    position observation (the CLIP visual fix) identify it. Velocity lives
    in the filter — not inside the integrator, whose own carried-over `_vel`
    is what diverged — so a visual fix corrects position, velocity and bias
    together through their covariance coupling.

    Consumes `segment.delta_v_ms` (the integral of horizontal acceleration
    over the segment), NOT `delta_p_m`: the latter already contains the
    integrator's divergent velocity carry-over, the former does not.

    Honest expectation: this makes the accelerometer contribute real
    short-horizon displacement between visual fixes. It is not a
    dead-reckoning system — with no visual fixes it still drifts, because
    an unaided MEMS IMU always does.
    """

    x: np.ndarray = field(default_factory=lambda: np.zeros(7))
    P: np.ndarray = field(default_factory=lambda: np.eye(7) * 1e-3)

    @classmethod
    def initialize(
        cls,
        x0: float,
        y0: float,
        theta0: float,
        pos_std: float = 0.3,
        theta_std_deg: float = 30.0,
        vel_std: float = 0.5,
        bias_std: float = 1.0,
    ) -> "Ekf2DVio":
        """`bias_std` defaults to 1.0 m/s² because the measured leak was
        0.81 m/s² — the prior has to admit a bias that large or the filter
        will refuse to learn the one actually present."""
        state = np.array([x0, y0, theta0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        p = np.diag([
            pos_std ** 2, pos_std ** 2, math.radians(theta_std_deg) ** 2,
            vel_std ** 2, vel_std ** 2, bias_std ** 2, bias_std ** 2,
        ])
        return cls(x=state, P=p)

    def predict(
        self,
        delta_v_body: tuple[float, float],
        dtheta_rad: float,
        dt_s: float,
        accel_noise: float = 0.3,
        gyro_noise_per_rad: float = 0.3,
        gyro_noise_per_s: float = 0.02,
        bias_walk_per_s: float = 0.01,
    ) -> None:
        """`delta_v_body`: (forward, lateral) integral of acceleration over
        this segment, in m/s — i.e. `segment["delta_v_ms"][0:2]`."""
        if dt_s <= 0:
            return
        px, py, theta, vx, vy, bax, bay = self.x

        # Average body-frame acceleration over the segment, bias removed.
        a0 = delta_v_body[0] / dt_s - bax
        a1 = delta_v_body[1] / dt_s - bay

        theta_mid = theta + dtheta_rad / 2.0
        c, s = math.cos(theta_mid), math.sin(theta_mid)
        aw0 = a0 * c - a1 * s
        aw1 = a0 * s + a1 * c

        self.x[0] = px + vx * dt_s + 0.5 * aw0 * dt_s ** 2
        self.x[1] = py + vy * dt_s + 0.5 * aw1 * dt_s ** 2
        self.x[2] = _wrap_angle(theta + dtheta_rad)
        self.x[3] = vx + aw0 * dt_s
        self.x[4] = vy + aw1 * dt_s
        # bias is modelled as constant + random walk (no deterministic update)

        d_aw0_dth = -a0 * s - a1 * c
        d_aw1_dth = a0 * c - a1 * s
        half = 0.5 * dt_s ** 2

        F = np.eye(7)
        F[0, 2] = half * d_aw0_dth
        F[0, 3] = dt_s
        F[0, 5] = half * (-c)
        F[0, 6] = half * s
        F[1, 2] = half * d_aw1_dth
        F[1, 4] = dt_s
        F[1, 5] = half * (-s)
        F[1, 6] = half * (-c)
        F[3, 2] = dt_s * d_aw0_dth
        F[3, 5] = dt_s * (-c)
        F[3, 6] = dt_s * s
        F[4, 2] = dt_s * d_aw1_dth
        F[4, 5] = dt_s * (-s)
        F[4, 6] = dt_s * (-c)

        av = (accel_noise * dt_s) ** 2
        ap = (0.5 * accel_noise * dt_s ** 2) ** 2
        th_var = (gyro_noise_per_rad * abs(dtheta_rad)) ** 2 + (gyro_noise_per_s * dt_s) ** 2
        bw = (bias_walk_per_s * dt_s) ** 2
        Q = np.diag([ap, ap, th_var, av, av, bw, bw])

        self.P = F @ self.P @ F.T + Q

    def update_visual(self, x_meas: float, y_meas: float, theta_meas: float, R: np.ndarray) -> None:
        """Observes position + heading only. Velocity and bias are corrected
        indirectly, through the covariance this predict step built between
        them — that coupling is the whole mechanism by which the bias
        becomes observable."""
        innovation = np.array([
            x_meas - self.x[0],
            y_meas - self.x[1],
            _wrap_angle(theta_meas - self.x[2]),
        ])
        H = np.zeros((3, 7))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        self._apply(innovation, H, R)

    def update_zero_velocity(self, R: float = 0.01 ** 2) -> None:
        """Observe v = 0. Only call when something OTHER than the
        accelerometer says the robot is stopped — e.g. the recorder's
        visual image-shift ZUPT. An inertial stillness test must not drive
        this: measured on the `forward` log, 82% of samples look stationary
        to it while the rig is actually rolling."""
        innovation = np.array([-self.x[3], -self.x[4]])
        H = np.zeros((2, 7))
        H[0, 3] = H[1, 4] = 1.0
        self._apply(innovation, H, np.eye(2) * R)

    def _apply(self, innovation: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[2] = _wrap_angle(self.x[2])
        I = np.eye(7)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def heading_rad(self) -> float:
        return float(self.x[2])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[3]), float(self.x[4])

    @property
    def bias(self) -> tuple[float, float]:
        return float(self.x[5]), float(self.x[6])

    @property
    def position_std_m(self) -> tuple[float, float]:
        return float(math.sqrt(self.P[0, 0])), float(math.sqrt(self.P[1, 1]))
