"""SHTP-over-UART motion/gravity reader for scene_recorder.py.

Same role as imu_rvc.RvcReader (gravity direction for the floor plan +
preintegrated per-keyframe motion for the metric-scale cross-check), but
backed by imu_shtp_uart.ShtpUartReader instead of UART-RVC. The difference
that matters: RVC never exposed a real gyroscope (only its own fused
yaw/pitch/roll), which is why imu_rvc.ImuIntegrator's gravity-tracking is a
static reference vector plus an optional linear "tilt_model" fit against the
fused Euler angles - a proxy for an independent tilt estimate, since RVC had
no other one. SHTP gives a real gyroscope, so this module tracks orientation
directly with a Madgwick filter (real gyro + real dt, no proxy needed) and
records that live quaternion on every keyframe - not just as an internal
detail, but as explicit, inspectable per-frame context in meta.json.

Gravity removal for the distance/speed preintegration still uses a STATIC
reference vector, not the live quaternion, deliberately: this project's rig
drives in the X-Y plane and never tilts (see CLAUDE.md), so gravity in the
camera frame is a constant and RVC's "reference" calibration mode already
covers this correctly and simply. The live quaternion is tracked anyway and
attached per-keyframe so an actual violation of that assumption (the rig
genuinely tilting) is visible and debuggable after the fact, rather than
silently mixed into the gravity-removal math.

**Calibration is NOT reused from config/imu_calibration.json as-is.** That
file was measured with imu_rvc.RvcReader against RVC's raw accelerometer
convention (its own docstring: ~+997 mg at rest, i.e. positive-up). This
project's SHTP-UART driver reads ~-9.7 m/s^2 on the same physical axis at
rest - convention mismatch confirmed empirically this session, not
guessed. Reusing the old up_imu_reference/cam_from_imu here without
re-measuring would silently apply the wrong sign convention, which is worse
than no calibration at all (this module returns None from `read()` without
one, same philosophy as RvcReader.read()'s own docstring). Re-measure with
this driver before trusting `gravity_camera` output.

The Madgwick step reuses the exact accel-sign convention discovered and
verified this session (see the imu_raw_log.py/madgwick offline analysis):
this driver's raw accelerometer reads NEGATIVE on the axis that is "up" at
rest, opposite of the sign the textbook Madgwick derivation assumes, so the
accel vector fed to the filter is negated before use.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

G_MS2 = 9.80665
MADGWICK_BETA = 0.05


def quat_mult(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def madgwick_step(
    q: tuple[float, float, float, float],
    gyro_rads: tuple[float, float, float],
    accel_ms2: tuple[float, float, float],
    dt: float,
    beta: float = MADGWICK_BETA,
) -> tuple[float, float, float, float]:
    """One Madgwick AHRS update (gyro + accel, no magnetometer).

    No magnetometer means yaw is NOT observable and drifts exactly like raw
    gyro integration - verified this session (offline test on forward.csv:
    Madgwick yaw matched naive gyro integration to within 1 degree). This is
    physics (accelerometer only measures gravity's direction, which rotation
    about gravity does not change), not a bug to chase. Roll/pitch, which
    accel DOES constrain, are what this buys over plain integration.
    """
    gx, gy, gz = gyro_rads
    ax, ay, az = accel_ms2
    # See module docstring: this driver's accel reads negative-up at rest,
    # opposite of the convention the gradient step below assumes.
    ax, ay, az = -ax, -ay, -az

    q0, q1, q2, q3 = q
    q_dot = tuple(0.5 * v for v in quat_mult(q, (0.0, gx, gy, gz)))

    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm > 1e-6:
        ax, ay, az = ax / norm, ay / norm, az / norm
        f1 = 2 * (q1 * q3 - q0 * q2) - ax
        f2 = 2 * (q0 * q1 + q2 * q3) - ay
        f3 = 2 * (0.5 - q1 * q1 - q2 * q2) - az
        j = (
            (-2 * q2, 2 * q3, -2 * q0, 2 * q1),
            (2 * q1, 2 * q0, 2 * q3, 2 * q2),
            (0.0, -4 * q1, -4 * q2, 0.0),
        )
        grad = [
            j[0][k] * f1 + j[1][k] * f2 + j[2][k] * f3
            for k in range(4)
        ]
        gnorm = math.sqrt(sum(v * v for v in grad))
        if gnorm > 1e-9:
            grad = [v / gnorm for v in grad]
        q_dot = tuple(qd - beta * g for qd, g in zip(q_dot, grad))

    q_new = tuple(qc + qd * dt for qc, qd in zip(q, q_dot))
    qnorm = math.sqrt(sum(v * v for v in q_new))
    if qnorm < 1e-9:
        return q
    return tuple(v / qnorm for v in q_new)


def quat_to_euler_deg(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """(roll, pitch, yaw) in degrees, standard aerospace convention."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def load_calibration(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable IMU calibration %s: %s", path, exc)
        return None


def angle_between_deg(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    denom = math.sqrt(sum(v * v for v in a)) * math.sqrt(sum(v * v for v in b))
    if denom < 1e-12:
        return float("nan")
    dot = sum(av * bv for av, bv in zip(a, b)) / denom
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


class ShtpMotionIntegrator:
    """Preintegrates motion between keyframes, same contract as
    imu_rvc.ImuIntegrator.cut() - see that class's docstring for why this
    matters (metric-scale cross-check against COLMAP's scale-free poses) and
    why velocity persists across cuts while everything else resets.

    Unlike imu_rvc.ImuIntegrator this has no `tilt_model`: this rig never
    tilts (CLAUDE.md), so gravity in the camera frame is a constant and a
    single reference vector is the correct model, not an approximation
    pending one. A real chassis tilt would show up as `gravity_mode` reads
    silently degrading to None (reference drifted past tolerance) - it would
    NOT be corrected on the fly. See the recorded live `orientation_quat` per
    keyframe if that ever needs investigating.
    """

    def __init__(self, up_sensor: tuple[float, float, float] | None = None) -> None:
        self.up_sensor = up_sensor
        self.reset()

    def reset(self) -> None:
        self._dt = 0.0
        self._samples = 0
        self._dv = [0.0, 0.0, 0.0]
        self._dp = [0.0, 0.0, 0.0]
        self._peak_linear = 0.0
        self._sum_linear = 0.0
        self._zupts = 0
        self._vel = [0.0, 0.0, 0.0]  # deliberately not reset - see ImuIntegrator.reset

    def set_reference(self, up_sensor: tuple[float, float, float] | None) -> None:
        self.up_sensor = up_sensor

    def linear_accel(self, accel_ms2: tuple[float, float, float]) -> tuple[float, float, float] | None:
        if self.up_sensor is None:
            return None
        ux, uy, uz = self.up_sensor
        ax, ay, az = accel_ms2
        return (ax - ux * G_MS2, ay - uy * G_MS2, az - uz * G_MS2)

    def add(self, accel_ms2: tuple[float, float, float], dt: float) -> None:
        if not (0.0 < dt < 0.5):
            return
        linear = self.linear_accel(accel_ms2)
        if linear is None:
            return
        # Shake/blur detection stays full 3D - vibration is not planar.
        magnitude = math.sqrt(sum(v * v for v in linear))
        self._peak_linear = max(self._peak_linear, magnitude)
        self._sum_linear += magnitude
        self._samples += 1
        self._dt += dt

        # Position/velocity integration is horizontal-plane-only by request:
        # this rig never leaves the floor, so any measured vertical
        # component is noise/gravity-removal residual, not real
        # displacement - project it OUT of the acceleration before
        # integrating (not just zeroed after) so it cannot leak into
        # distance_m/speed_ms at all, which are vector norms of dp/vel.
        ux, uy, uz = self.up_sensor  # linear_accel() already checked this isn't None
        vertical = linear[0] * ux + linear[1] * uy + linear[2] * uz
        horizontal = (
            linear[0] - vertical * ux,
            linear[1] - vertical * uy,
            linear[2] - vertical * uz,
        )
        for k in range(3):
            self._dp[k] += self._vel[k] * dt + 0.5 * horizontal[k] * dt * dt
            self._vel[k] += horizontal[k] * dt
            self._dv[k] += horizontal[k] * dt

    def zero_velocity(self) -> None:
        self._vel = [0.0, 0.0, 0.0]
        self._zupts += 1

    def cut(self) -> dict[str, Any]:
        segment = {
            "dt_s": round(self._dt, 4),
            "samples": self._samples,
            "delta_p_m": [round(v, 5) for v in self._dp],
            "delta_v_ms": [round(v, 5) for v in self._dv],
            "distance_m": round(math.sqrt(sum(v * v for v in self._dp)), 5),
            "speed_ms": round(math.sqrt(sum(v * v for v in self._vel)), 5),
            "peak_linear_accel_ms2": round(self._peak_linear, 4),
            "mean_linear_accel_ms2": round(self._sum_linear / self._samples if self._samples else 0.0, 4),
            "zupts": self._zupts,
        }
        velocity = list(self._vel)
        self.reset()
        self._vel = velocity
        return segment


class ShtpMotionReader:
    """Drop-in replacement for imu_rvc.RvcReader in scene_recorder.py.

    Interface used by scene_recorder.py / imu_calibrate.py: .port, .start(),
    .stop(), .window(), .averaged_up_sensor(), .integrator.zero_velocity(),
    .read_motion(), .stats().
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 3_000_000,
        calibration_path: Path | None = None,
        window_s: float = 0.5,
    ) -> None:
        from pi_client.imu_shtp_uart import ShtpUartReader

        self.port = port
        self.baud = baud
        self.window_s = window_s
        self.calibration = load_calibration(calibration_path) if calibration_path else None
        driver_tag = (self.calibration or {}).get("driver")
        if driver_tag not in (None, "shtp"):
            logger.error(
                "IMU calibration %s was measured with driver=%r, not 'shtp' — its "
                "accelerometer sign convention will not match this reader (see module "
                "docstring). Ignoring it; re-run imu_calibrate.py --driver shtp.",
                calibration_path, driver_tag,
            )
            self.calibration = None
        self._reader = ShtpUartReader(port=port, baudrate=baud)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._q: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        self._last_t: float | None = None
        self._window: list[tuple[float, tuple[float, float, float]]] = []
        self._suppressed = 0

        up_ref = (self.calibration or {}).get("up_imu_reference")
        self.integrator = ShtpMotionIntegrator(tuple(up_ref) if up_ref else None)

    # ------------------------------------------------------------ lifecycle

    def start(self) -> bool:
        if not self._reader.start():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="imu-shtp-motion", daemon=True)
        self._thread.start()
        logger.info("SHTP motion reader started on %s @ %d", self.port, self.baud)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._reader.stop()

    # --------------------------------------------------------------- thread

    def _run(self) -> None:
        # Dedup on reconstructed parse time, not sample.monotonic - the
        # exact bug euroc_capture.py's module docstring documents (polling
        # faster than data arrives makes .monotonic alone always look "new").
        while not self._stop.is_set():
            sample = self._reader.latest()
            if sample is not None:
                accel_t = sample.monotonic - sample.accel_age_s
                gyro_t = sample.monotonic - sample.gyro_age_s
                new_t = max(accel_t, gyro_t)
                if self._last_t is None:
                    self._last_t = new_t
                elif new_t > self._last_t:
                    dt = new_t - self._last_t
                    self._last_t = new_t
                    if 0.0 < dt < 0.5:
                        with self._lock:
                            self._q = madgwick_step(self._q, sample.gyro_rads, sample.accel_ms2, dt)
                            self.integrator.add(sample.accel_ms2, dt)
                            self._window.append((new_t, sample.accel_ms2))
                            cutoff = new_t - self.window_s
                            self._window = [w for w in self._window if w[0] >= cutoff]
            time.sleep(0.002)

    # ---------------------------------------------------------------- reads

    def window(self) -> list[tuple[float, tuple[float, float, float]]]:
        with self._lock:
            return list(self._window)

    def orientation_quat(self) -> tuple[float, float, float, float]:
        with self._lock:
            return self._q

    def averaged_up_sensor(self) -> tuple[float, float, float] | None:
        """Median unit 'up' (physical up, i.e. negative of the raw reading -
        see module docstring) over the current window. Median per axis, same
        rationale as imu_rvc.RvcReader.averaged_up_sensor - one jolt should
        not drag it."""
        samples = self.window()
        if not samples:
            return None
        vectors = []
        for _, (ax, ay, az) in samples:
            mag = math.sqrt(ax * ax + ay * ay + az * az)
            if mag > 1e-6:
                vectors.append((-ax / mag, -ay / mag, -az / mag))
        if not vectors:
            return None
        axes = [statistics.median(v[k] for v in vectors) for k in range(3)]
        norm = math.sqrt(sum(a * a for a in axes))
        if norm < 1e-6:
            return None
        return (axes[0] / norm, axes[1] / norm, axes[2] / norm)

    def averaged_pitch_roll_deg(self) -> tuple[float, float] | None:
        """Roll/pitch from the LIVE MADGWICK QUATERNION, not an independent
        on-chip fusion (SHTP mode here only has raw accel+gyro reports
        enabled, no onboard Rotation Vector). Note for imu_calibrate.py's
        `full` mode specifically: this is therefore not as independent a
        cross-check as RVC's on-chip Euler was - reasonable for `reference`
        mode calibration (the one this rig actually needs, since it never
        tilts), less rigorous if `full` mode is attempted with this driver."""
        roll, pitch, _ = quat_to_euler_deg(self.orientation_quat())
        return (pitch, roll)

    def stats(self) -> dict[str, Any]:
        base = self._reader.stats()
        with self._lock:
            base["window"] = len(self._window)
        base["calibrated"] = self.calibration is not None
        base["mode"] = (self.calibration or {}).get("mode")
        base["suppressed_off_reference"] = self._suppressed
        return base

    def read_motion(self) -> dict[str, Any] | None:
        with self._lock:
            q = self._q
            segment = self.integrator.cut()
        roll, pitch, yaw = quat_to_euler_deg(q)
        motion: dict[str, Any] = {
            "yaw_deg": round(yaw, 2),
            "pitch_deg": round(pitch, 2),
            "roll_deg": round(roll, 2),
            "orientation_quat": [round(v, 6) for v in q],
            "segment": segment,
            "gravity_mode": "static_reference" if self.integrator.up_sensor else "none",
        }
        gravity = self.read()
        if gravity is not None:
            motion["gravity_camera"] = [round(v, 5) for v in gravity]
        return motion

    def read(self) -> list[float] | None:
        """Gravity (pointing DOWN) in the CAMERA frame - same contract as
        imu_rvc.RvcReader.read(), see its docstring. Requires a calibration
        file measured WITH THIS DRIVER (see module docstring on why the old
        RVC one cannot be reused)."""
        if not self.calibration:
            return None
        up_sensor = self.averaged_up_sensor()
        if up_sensor is None:
            return None

        rotation = self.calibration.get("cam_from_imu")
        if self.calibration.get("mode") == "full" and rotation:
            up_camera = tuple(
                sum(rotation[r][c] * up_sensor[c] for c in range(3)) for r in range(3)
            )
        else:
            reference_imu = self.calibration.get("up_imu_reference")
            reference_cam = self.calibration.get("up_camera_reference")
            if not reference_imu or not reference_cam:
                return None
            drift = angle_between_deg(up_sensor, tuple(reference_imu))
            tolerance = float(self.calibration.get("tilt_tolerance_deg", 4.0))
            if not math.isfinite(drift) or drift > tolerance:
                self._suppressed += 1
                return None
            up_camera = tuple(reference_cam)

        norm = math.sqrt(sum(v * v for v in up_camera))
        if norm < 1e-6:
            return None
        return [-v / norm for v in up_camera]
