"""BNO08x UART-RVC driver: attitude for navigation, gravity for reconstruction.

The sensor is on a CH340 USB-serial bridge (`/dev/ttyUSB0`, 115200 8N1,
stable path `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0`) and streams
**UART-RVC**, not SHTP — a fixed 19-byte frame at 100 Hz:

    0  0xAA          header
    1  0xAA
    2  index         free-running counter, wraps at 256
    3.. yaw          int16 LE, 0.01 deg      (relative heading, no magnetometer)
    5.. pitch        int16 LE, 0.01 deg
    7.. roll         int16 LE, 0.01 deg
    9.. accel x      int16 LE, milli-g
   11.. accel y      int16 LE, milli-g
   13.. accel z      int16 LE, milli-g
   15  MI, 16 MR, 17 reserved
   18  checksum      sum(bytes 2..17) & 0xFF

Verified against the real device: 200/200 captured frames checksum-clean,
100 Hz, |accel| = 997 mg at rest, per-axis noise ~2 mg. Re-measured after
the 2026-08-01 rewiring (sensor pins straight to the CH340, straight to
USB): 501 frames in 5.0 s (100.2 Hz), **zero** checksum resyncs, **zero**
gaps in the frame index, |accel| median 1002 mg (998-1006). For contrast,
the SHTP-over-UART link this replaced was delivering ~25% well-formed
frames at the end — the numbers above are the reason this driver, not that
one, is now the default everywhere.

**What this link is good for: attitude.** Yaw/pitch/roll are fused on-chip
at the sensor's full internal rate, and measured on this device the yaw
drifts **0.03 deg/min** with the rig stationary (90 s, 9011 samples;
pitch/roll jitter +/-0.05 deg). `read_orientation()` is the interface for
that, and `ekf_localization.Ekf2DHeading` is what consumes it.

**What it is not good for: metres.** RVC reports no raw gyroscope, and
double-integrating this accelerometer for displacement was measured at 78x
too large on this rig (see `ekf_localization.Ekf2DVio`). `ImuIntegrator`
below still exists for the scene3d metric-scale experiment, but live
navigation deliberately does not dead-reckon position from it.

**Why gravity comes from the accelerometer and not from pitch/roll.** The
RVC report carries both, and they disagree by an axis permutation — measured
at rest, "down" from the fused Euler angles is (+0.007, +0.023, -0.9997)
while from the accelerometer it is (-0.024, -0.007, -0.9997). Same tilt
magnitude (~1.4 deg), different axis convention: the RVC Euler angles are not
expressed in the accelerometer's frame. Rather than guess which datasheet
convention applies, this module uses the accelerometer — a direct physical
measurement whose scale we verified — and *checks* it against the Euler tilt
angle, which is convention-independent.

The cost of that choice is that raw acceleration includes the robot's own
motion. Three things make it a non-issue here: the rig moves slowly, samples
are low-pass filtered over a window (`SAMPLE_WINDOW_S`), and the consumer
(`occupancy.estimate_gravity` on the Mac) averages the vector over every
keyframe of the session, so uncorrelated motion cancels.

The sensor frame is NOT assumed to match the camera frame. The rotation
between them is measured by `imu_calibrate.py` and stored in
`config/imu_calibration.json`; without that file this module still reports
raw samples, but refuses to claim a camera-frame gravity vector.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pi_client.imu_rvc_math import FrameClock

logger = logging.getLogger(__name__)

HEADER = b"\xaa\xaa"
FRAME_LEN = 19
DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 115200
SAMPLE_RATE_HZ = 100.0
SAMPLE_WINDOW_S = 0.5  # low-pass window for gravity; 50 samples at 100 Hz

# RVC reports yaw as a COMPASS heading: clockwise-positive seen from above.
# The plan frame is right-handed and counter-clockwise-positive (see
# navindex._heading_deg, which is an atan2 in the standard maths sense), so
# the two disagree in sign and the driver normalizes here rather than leaving
# every consumer to remember.
#
# MEASURED, not assumed (2026-08-03, imu_orientation_test.py): two full
# turns, one in each direction, against a floor mark.
#     turn left  +360 deg true -> -358.41 deg reported, gain -0.996
#     turn right -360 deg true -> +360.97 deg reported, gain -1.003
# The two agree with each other to 0.7% and both are within 0.4% of unit
# magnitude, so this is a clean sign convention and not a scale problem.
#
# Why it had to be measured: an additive datum offset (Ekf2DHeading's `b`)
# can absorb any constant, but it cannot absorb a SIGN. With the sign wrong
# the filter steers its heading estimate the wrong way on every turn, and
# nothing in the gravity calibration can detect that — rotation about
# gravity leaves the gravity vector unchanged.
YAW_SIGN = -1.0


@dataclass(frozen=True)
class ImuSample:
    """One decoded RVC frame. Accel is milli-g in the SENSOR frame."""

    index: int
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    accel_mg: tuple[float, float, float]
    monotonic: float          # when the bytes reached us — batched, unreliable
    # Reconstructed from the device's own 100 Hz frame counter. Use THIS to
    # place a sample in time: `monotonic` is up to 20 ms late because the CH340
    # delivers frames in pairs (measured: p95 gap 20.07 ms, 50.2% of gaps under
    # 1 ms). At 30 deg/s that is 0.6 deg of heading on a camera frame.
    t_grid: float = 0.0
    seq: int = 0              # frames since start, immune to the 0-255 wrap

    @property
    def accel_magnitude_mg(self) -> float:
        x, y, z = self.accel_mg
        return math.sqrt(x * x + y * y + z * z)

    def up_sensor(self) -> tuple[float, float, float] | None:
        """Unit vector along the sensor's measured specific force.

        At rest an accelerometer reads the normal force holding it up, so
        this points UP (away from the ground) — the sign that trips people
        up. `down_sensor` is its negation.
        """
        magnitude = self.accel_magnitude_mg
        if magnitude < 1e-6:
            return None
        x, y, z = self.accel_mg
        return (x / magnitude, y / magnitude, z / magnitude)

    def down_sensor(self) -> tuple[float, float, float] | None:
        up = self.up_sensor()
        return None if up is None else (-up[0], -up[1], -up[2])

    def euler_tilt_deg(self) -> float:
        """Tilt from level according to the fused Euler angles.

        Convention-independent (it's an angle, not an axis), which is what
        makes it usable as a cross-check on the accelerometer despite the two
        living in differently-oriented frames.
        """
        pitch = math.radians(self.pitch_deg)
        roll = math.radians(self.roll_deg)
        cos_tilt = math.cos(pitch) * math.cos(roll)
        return math.degrees(math.acos(max(-1.0, min(1.0, cos_tilt))))

    def accel_tilt_deg(self) -> float:
        """Tilt from level implied by the accelerometer, for the same check.

        Uses the sensor axis that reads ~1 g at rest, so it does not depend
        on which way round the board is mounted.
        """
        magnitude = self.accel_magnitude_mg
        if magnitude < 1e-6:
            return float("nan")
        vertical = max(abs(v) for v in self.accel_mg)
        return math.degrees(math.acos(max(-1.0, min(1.0, vertical / magnitude))))


def checksum(frame: bytes) -> int:
    return sum(frame[2:18]) & 0xFF


def parse_frame(frame: bytes, monotonic: float | None = None) -> ImuSample | None:
    """Decode one 19-byte frame, or None if it isn't a valid RVC frame."""
    if len(frame) != FRAME_LEN or frame[0:2] != HEADER:
        return None
    if checksum(frame) != frame[18]:
        return None

    def i16(offset: int) -> int:
        value = frame[offset] | (frame[offset + 1] << 8)
        return value - 0x10000 if value & 0x8000 else value

    return ImuSample(
        index=frame[2],
        yaw_deg=i16(3) / 100.0,
        pitch_deg=i16(5) / 100.0,
        roll_deg=i16(7) / 100.0,
        accel_mg=(float(i16(9)), float(i16(11)), float(i16(13))),
        monotonic=time.monotonic() if monotonic is None else monotonic,
    )


def iter_frames(buffer: bytes) -> tuple[list[ImuSample], bytes, int]:
    """Pull every complete valid frame out of `buffer`.

    Returns `(samples, remainder, resyncs)`. Resynchronising matters: a
    USB-serial bridge drops bytes under load, and a decoder that only ever
    advances by whole frames stays permanently misaligned after a single
    lost byte. On a checksum failure this advances by ONE byte and looks for
    the next header, so the stream re-locks within a frame instead of
    producing garbage forever.

    `resyncs` counts those failures. It is the link-health signal: on a
    healthy RVC link it stays at zero (measured on this rig: 501 frames in
    5 s, 0 resyncs, 0 gaps in the frame index), so any sustained non-zero
    rate means the wiring or the adapter is degrading — the exact failure
    the SHTP link died of, which went unnoticed for a session because
    nothing counted it.
    """
    samples: list[ImuSample] = []
    cursor = 0
    resyncs = 0
    length = len(buffer)
    while True:
        start = buffer.find(HEADER, cursor)
        if start < 0:
            # A header may straddle the read boundary — keep the last byte.
            return samples, buffer[max(cursor, length - 1):], resyncs
        if start + FRAME_LEN > length:
            return samples, buffer[start:], resyncs
        sample = parse_frame(buffer[start:start + FRAME_LEN])
        if sample is None:
            cursor = start + 1  # false header, resync
            resyncs += 1
            continue
        samples.append(sample)
        cursor = start + FRAME_LEN


def kabsch_rotation(source: Any, target: Any) -> Any:
    """Rotation R minimising sum ||R*source_i - target_i|| (Wahba's problem).

    Needs at least two non-parallel pairs; used by the calibration to solve
    for the sensor->camera rotation from pairs of unit vectors that point at
    the same physical direction (up) in the two frames.
    """
    import numpy as np

    src = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if len(src) < 2 or len(src) != len(dst):
        raise ValueError("need >= 2 matching vector pairs")
    covariance = src.T @ dst
    u, _, vt = np.linalg.svd(covariance)
    # Guard against a reflection: SVD alone can return det = -1, which is a
    # mirror, not a rotation, and would silently flip handedness.
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ correction @ u.T


def fit_tilt_model(
    pitch_roll_deg: Any,
    up_sensor: Any,
    reference_pitch_deg: float,
    reference_roll_deg: float,
    reference_up: Any,
) -> dict[str, Any] | None:
    """Small-angle linear fit: (Δpitch, Δroll) [rad] -> Δ(accelerometer's up).

    This is the piece `ImuIntegrator.current_up_sensor` needs to track a
    chassis that is actually tilting instead of assuming it stays at the
    pose gravity was measured in — see its docstring for why. It uses the
    SAME multi-attitude data `imu_calibrate.py`'s `full` mode already
    collects for `cam_from_imu` (measured directions in, measured
    directions out) rather than assuming any particular relationship
    between the fused Euler frame and the raw accelerometer's — the two
    are demonstrably not the same axes.

    `pitch_roll_deg` and `up_sensor` must be the same length and in the
    same attitude order. Returns None when there isn't enough independent
    attitude spread to fit both a pitch and a roll sensitivity (propping
    the rig up the same way twice constrains only one direction).
    """
    import numpy as np

    pr = np.asarray(pitch_roll_deg, dtype=np.float64).reshape(-1, 2)
    up = np.asarray(up_sensor, dtype=np.float64).reshape(-1, 3)
    if len(pr) < 3 or len(pr) != len(up):
        return None

    reference = np.array([reference_pitch_deg, reference_roll_deg], dtype=np.float64)
    d_pr = np.radians(pr - reference)
    ref_up = np.asarray(reference_up, dtype=np.float64)
    d_up = up - ref_up

    # A robot propped only ever the same way leaves d_pr rank-1: any fit
    # would be extrapolating along an axis nothing measured.
    singular_values = np.linalg.svd(d_pr, compute_uv=False)
    if len(singular_values) < 2 or singular_values[1] < 1e-3:
        return None

    sensitivity, *_ = np.linalg.lstsq(d_pr, d_up, rcond=None)  # (2, 3)

    predicted_up = ref_up + d_pr @ sensitivity
    norms = np.linalg.norm(predicted_up, axis=1, keepdims=True)
    predicted_up = predicted_up / np.clip(norms, 1e-9, None)
    cos_residual = np.clip(np.sum(predicted_up * up, axis=1), -1.0, 1.0)
    residual_deg = np.degrees(np.arccos(cos_residual))

    return {
        "pitch_ref_deg": float(reference_pitch_deg),
        "roll_ref_deg": float(reference_roll_deg),
        "up_ref": [float(v) for v in ref_up],
        "sensitivity": [[float(v) for v in row] for row in sensitivity],
        "fit_residual_deg_mean": round(float(np.mean(residual_deg)), 3),
        "fit_residual_deg_max": round(float(np.max(residual_deg)), 3),
        "attitudes_used": int(len(pr)),
    }


def angle_between_deg(a: Any, b: Any) -> float:
    import numpy as np

    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(va @ vb / denom, -1.0, 1.0))))


def load_calibration(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable IMU calibration %s: %s", path, exc)
        return None


def tilt_model_from_calibration(calibration: dict[str, Any] | None) -> dict[str, Any] | None:
    """Unpack the `tilt_model` block `imu_calibrate.py`'s `full` mode writes
    (see `ImuIntegrator.current_up_sensor`). Absent on `reference`-mode
    calibrations, or on `full`-mode ones from before this fit existed —
    both fall back to the static reference vector, same as always."""
    model = (calibration or {}).get("tilt_model")
    if not model:
        return None
    try:
        return {
            "pitch_ref_deg": float(model["pitch_ref_deg"]),
            "roll_ref_deg": float(model["roll_ref_deg"]),
            "up_ref": [float(v) for v in model["up_ref"]],
            "sensitivity": [[float(v) for v in row] for row in model["sensitivity"]],
        }
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Malformed tilt_model in IMU calibration, ignoring: %s", exc)
        return None


G_MS2 = 9.80665


class ImuIntegrator:
    """Preintegrates motion between keyframes.

    Inertial position diverges when nothing corrects it — the error grows as
    0.5*a*t^2, so a residual acceleration of a few hundredths of a m/s^2
    becomes tens of metres over a minute. That is why inertial-only
    navigation does not exist. It is *not* why the accelerometer is useless:
    over the fraction of a second between two keyframes the same error is
    millimetres, and the next camera frame resets it. This is the mechanism
    every visual-inertial system runs on — PROVIDED gravity is removed
    correctly at every sample, which is the one thing this class got wrong
    until now (see `tilt_model` below).

    What that buys, and why it matters more than a drawn trajectory:
    **metric scale**. Vision recovers motion only up to an unknown factor,
    which the pipeline currently pins with monocular depth that carries its
    own scale error. The accelerometer measures metres. The ratio of the two
    displacement magnitudes over many intervals is the scale factor — and
    magnitudes are enough, so this works without knowing the rotation about
    gravity (yaw never enters it).

    **Why a single static reference vector was not enough.** A rig that
    genuinely never tilted could subtract one constant gravity vector
    forever. Measured on a real drive (session_20260726_193117): the chassis
    swings ~3 deg (p10..p90, peaking at 10.8 deg) around the pose the
    reference was captured in — acceleration/braking pitch, floor bumps.
    Subtracting a *constant* from a rig that is actually tilting leaks
    9.81*sin(3 deg) = 0.52 m/s^2 into "linear" acceleration, fourteen times
    the budget this design was sized against; over 215 s that integrated to
    a 78 m/s "velocity". Removing a bigger constant does not help — the leak
    tracks the chassis, not a fixed offset.

    **The fix: `tilt_model`.** The one independent tilt signal this sensor
    offers is its own fused pitch/roll — it uses the on-chip gyroscope
    internally, so (unlike the raw accelerometer) it is not fooled by
    linear acceleration into reporting the wrong tilt. The fused Euler
    frame and the raw-accelerometer frame are demonstrably NOT the same
    axes (imu_calibrate.py's rest check sees the same physical tilt as two
    different vectors), so pitch/roll cannot be rotated into the
    accelerometer frame by assumption — `imu_calibrate.py`'s `full` mode
    fits that relationship empirically instead (a small-angle linear
    regression from measured (Δpitch, Δroll) to the accelerometer's own
    measured Δup, across the same multi-attitude data it already collects
    for `cam_from_imu`). `tilt_model`, when present, is that fit: it lets
    `linear_accel` rebuild gravity's current direction from live pitch/roll
    instead of assuming the rig never left the pose gravity was measured
    in. Without it (mode=`reference`, no book-propping run yet), behaviour
    is unchanged: the static reference vector, and the honest 78 m/s result
    is why `poses.scale_source` still defaults to `depth`.

    Gravity is removed using directions measured at calibration rather than
    any datasheet axis convention. Everything is accumulated in the SENSOR
    frame; only lengths are consumed downstream, which is exactly what
    keeps the unobservable spin about gravity out of the answer.
    """

    def __init__(
        self,
        up_sensor: tuple[float, float, float] | None = None,
        tilt_model: dict[str, Any] | None = None,
    ) -> None:
        self.up_sensor = up_sensor
        self.tilt_model = tilt_model
        self.reset()

    def reset(self) -> None:
        self._dt = 0.0
        self._samples = 0
        self._dv = [0.0, 0.0, 0.0]     # velocity change, sensor frame, m/s
        self._dp = [0.0, 0.0, 0.0]     # position change, sensor frame, m
        self._peak_linear = 0.0        # m/s^2, for shake/blur rejection
        self._sum_linear = 0.0
        self._last: ImuSample | None = None
        self._zupts = 0
        # Velocity is deliberately NOT reset here. It is a property of the
        # vehicle, not of the segment: a robot rolling at a steady speed has
        # near-zero acceleration, so a segment that restarts from v=0 measures
        # almost no displacement and the scale estimate collapses. Velocity
        # only goes to zero when something says the rig actually stopped —
        # see `zero_velocity`.
        self._vel = [0.0, 0.0, 0.0]

    def set_reference(self, up_sensor: tuple[float, float, float] | None) -> None:
        self.up_sensor = up_sensor

    def set_tilt_model(self, tilt_model: dict[str, Any] | None) -> None:
        self.tilt_model = tilt_model

    def current_up_sensor(self, sample: ImuSample) -> tuple[float, float, float] | None:
        """Gravity's direction in the sensor frame for THIS sample.

        Static reference by default. With `tilt_model` fitted, this instead
        rebuilds the direction from the sample's own fused pitch/roll — see
        the class docstring for why that is the only way to track a chassis
        that is actually tilting instead of assuming it never left the
        pose the reference was measured in.
        """
        if self.tilt_model is None:
            return self.up_sensor
        model = self.tilt_model
        d_pitch = math.radians(sample.pitch_deg - model["pitch_ref_deg"])
        d_roll = math.radians(sample.roll_deg - model["roll_ref_deg"])
        sensitivity = model["sensitivity"]  # (2, 3): [d_pitch, d_roll] @ sensitivity -> delta up
        up_ref = model["up_ref"]
        delta = [
            d_pitch * sensitivity[0][k] + d_roll * sensitivity[1][k]
            for k in range(3)
        ]
        candidate = [up_ref[k] + delta[k] for k in range(3)]
        norm = math.sqrt(sum(v * v for v in candidate))
        if norm < 1e-6:
            return self.up_sensor
        return (candidate[0] / norm, candidate[1] / norm, candidate[2] / norm)

    def linear_accel(self, sample: ImuSample) -> tuple[float, float, float] | None:
        """Acceleration with gravity removed, in m/s^2, sensor frame."""
        up = self.current_up_sensor(sample)
        if up is None:
            return None
        ux, uy, uz = up
        # Specific force in m/s^2; at rest this is +1 g along `up`.
        ax = sample.accel_mg[0] / 1000.0 * G_MS2
        ay = sample.accel_mg[1] / 1000.0 * G_MS2
        az = sample.accel_mg[2] / 1000.0 * G_MS2
        return (ax - ux * G_MS2, ay - uy * G_MS2, az - uz * G_MS2)

    def add(self, sample: ImuSample) -> None:
        previous = self._last
        self._last = sample
        if previous is None:
            return
        dt = sample.monotonic - previous.monotonic
        # Guard against a stalled or rewound clock, and against a gap so long
        # that integrating across it is meaningless.
        if not (0.0 < dt < 0.5):
            return
        linear = self.linear_accel(sample)
        if linear is None:
            return

        magnitude = math.sqrt(sum(v * v for v in linear))
        self._peak_linear = max(self._peak_linear, magnitude)
        self._sum_linear += magnitude
        self._samples += 1
        self._dt += dt
        for k in range(3):
            # Trapezoid on position: using the mid-segment velocity rather
            # than the end value keeps a steady acceleration from being
            # over-counted by a full 0.5*a*dt^2 every step.
            self._dp[k] += self._vel[k] * dt + 0.5 * linear[k] * dt * dt
            self._vel[k] += linear[k] * dt
            self._dv[k] += linear[k] * dt

    def zero_velocity(self) -> None:
        """Zero-velocity update: declare the rig stopped.

        Velocity carried across segments would otherwise drift without
        bound — an acceleration error of a, unopposed, becomes a velocity
        error of a*t and a position error of 0.5*a*t^2. Pinning velocity to
        zero whenever the rig is known to be still is what keeps that
        bounded; it is the same trick pedestrian dead reckoning uses at each
        footfall.

        The evidence has to come from outside the accelerometer, because a
        rig moving at constant speed and a rig standing still both read zero
        linear acceleration. Here it comes from the camera: the recorder
        already measures how far the image shifted between keyframes for its
        keyframe selector, and an image that did not move means a rig that
        did not move.
        """
        self._vel = [0.0, 0.0, 0.0]
        self._zupts += 1

    def cut(self) -> dict[str, Any]:
        """Take the segment accumulated so far and start a new one."""
        segment = {
            "dt_s": round(self._dt, 4),
            "samples": self._samples,
            "delta_p_m": [round(v, 5) for v in self._dp],
            "delta_v_ms": [round(v, 5) for v in self._dv],
            "distance_m": round(math.sqrt(sum(v * v for v in self._dp)), 5),
            "speed_ms": round(math.sqrt(sum(v * v for v in self._vel)), 5),
            "peak_linear_accel_ms2": round(self._peak_linear, 4),
            "mean_linear_accel_ms2": round(
                self._sum_linear / self._samples if self._samples else 0.0, 4
            ),
            "zupts": self._zupts,
        }
        last, velocity = self._last, list(self._vel)
        self.reset()
        # Continuity across the cut: the next segment starts from the same
        # sample and the same velocity this one ended at.
        self._last = last
        self._vel = velocity
        return segment


class RvcReader:
    """Background reader keeping a short window of recent samples.

    A thread rather than on-demand reads: the port streams at 100 Hz whether
    or not anyone is listening, so a reader that opened the port per keyframe
    would hand back whatever stale bytes had piled up in the kernel buffer.
    """

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baud: int = DEFAULT_BAUD,
        calibration_path: Path | None = None,
        window_s: float = SAMPLE_WINDOW_S,
    ) -> None:
        self.port = port
        self.baud = baud
        self.window_s = window_s
        self.calibration = load_calibration(calibration_path) if calibration_path else None
        self._serial: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._samples: list[ImuSample] = []
        # Timestamps come from the frame index, never from arrival time.
        self._clock = FrameClock()
        self._seq = 0
        self._seq_last_index: int | None = None
        self._frames_ok = 0
        self._frames_bad = 0
        self._dropped = 0        # frames the sensor sent that never reached us
        self._last_index: int | None = None
        self._last_rx = 0.0      # monotonic time of the most recent valid frame
        self._started_at = 0.0
        self._yaw_unwrapped: float | None = None
        self._yaw_last_raw: float | None = None
        self._suppressed = 0  # reads withheld because the rig left its reference pose
        reference_up = (self.calibration or {}).get("up_imu_reference")
        self.integrator = ImuIntegrator(
            tuple(reference_up) if reference_up else None,
            tilt_model=tilt_model_from_calibration(self.calibration),
        )

    # ------------------------------------------------------------ lifecycle

    def start(self, wait_s: float = 2.0) -> bool:
        """Open the port and confirm the sensor is actually streaming RVC.

        Opening a serial port succeeds against anything — an unpowered
        sensor, a board left in SHTP mode, the wrong adapter. Returning
        True on that alone is how a dead IMU gets silently carried into a
        recording session. This waits for one checksum-valid frame before
        claiming success; the sensor emits at 100 Hz, so `wait_s` of 2.0 is
        two hundred chances.
        """
        try:
            import serial
        except ImportError:
            logger.error("pyserial not installed — no IMU. pip install pyserial")
            return False
        try:
            self._serial = serial.Serial(self.port, self.baud, timeout=0.2)
            self._serial.reset_input_buffer()
        except Exception as exc:
            logger.error("Cannot open IMU port %s: %s", self.port, exc)
            self._serial = None
            return False
        self._stop.clear()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="imu-rvc", daemon=True)
        self._thread.start()

        deadline = time.monotonic() + max(wait_s, 0.0)
        while time.monotonic() < deadline:
            if self.latest() is not None:
                logger.info("IMU reader started on %s @ %d", self.port, self.baud)
                return True
            time.sleep(0.05)
        logger.error(
            "No valid RVC frame on %s within %.1fs. The port opened, so the adapter is "
            "there — check the sensor is powered and its PS1/PS0 jumpers select UART-RVC "
            "(not SHTP).", self.port, wait_s,
        )
        self.stop()
        return False

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    def __enter__(self) -> "RvcReader":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    # --------------------------------------------------------------- thread

    def _run(self) -> None:
        buffer = b""
        while not self._stop.is_set():
            try:
                chunk = self._serial.read(256)
            except Exception as exc:
                logger.warning("IMU read failed: %s", exc)
                break
            if not chunk:
                continue
            buffer += chunk
            samples, buffer, resyncs = iter_frames(buffer)
            if len(buffer) > 4 * FRAME_LEN:  # nothing parseable — drop the junk
                buffer = buffer[-FRAME_LEN:]
            if resyncs:
                with self._lock:
                    self._frames_bad += resyncs
            if not samples:
                continue
            now = time.monotonic()
            with self._lock:
                self._frames_ok += len(samples)
                self._last_rx = now
                for sample in samples:
                    # The RVC frame index increments by one per frame and
                    # wraps at 256. Any other step means frames the sensor
                    # emitted never arrived — a loss the checksum cannot
                    # see, because each surviving frame is individually
                    # perfect. Measured healthy on this rig: 500/500 gaps
                    # of exactly 1.
                    if self._last_index is not None:
                        gap = (sample.index - self._last_index) % 256
                        if gap != 1:
                            self._dropped += (gap - 1) % 256
                    self._last_index = sample.index
                    # Continuous yaw: the wire value wraps at +/-180, and a
                    # consumer differencing raw values across the wrap sees
                    # a 360 deg jolt. Unwrapping here means every consumer
                    # gets it right instead of each re-deriving it.
                    raw = sample.yaw_deg
                    if self._yaw_unwrapped is None:
                        self._yaw_unwrapped = raw
                    else:
                        self._yaw_unwrapped += (raw - self._yaw_last_raw + 180.0) % 360.0 - 180.0
                    self._yaw_last_raw = raw
                for sample in samples:
                    # The wire index wraps at 256; turn it into a running count
                    # so the clock model sees a straight line. Gaps are added
                    # too, so a dropped frame shifts time forward by the right
                    # amount instead of compressing the timeline.
                    if self._seq_last_index is None:
                        self._seq += 1
                    else:
                        step = (sample.index - self._seq_last_index) & 0xFF
                        self._seq += step if step else 256
                    self._seq_last_index = sample.index
                    sample.seq = self._seq
                    sample.t_grid = self._clock.update(self._seq, sample.monotonic)
                self._samples.extend(samples)
                cutoff = now - self.window_s
                self._samples = [s for s in self._samples if s.monotonic >= cutoff]
                # Integration has to happen here, on every sample: the
                # rolling window only keeps 0.5 s, but a keyframe interval can
                # be several seconds, and motion between keyframes is exactly
                # what the scale estimate needs.
                for sample in samples:
                    self.integrator.add(sample)

    # ---------------------------------------------------------------- reads

    def latest(self) -> ImuSample | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def window(self) -> list[ImuSample]:
        with self._lock:
            return list(self._samples)

    @property
    def mount_offsets_deg(self) -> tuple[float, float]:
        """Pitch/roll of the MOUNT itself, subtracted from every reported angle.

        Measured by acceptance test E and stored as
        `rvc_acceptance.pitch_offset_deg` / `roll_offset_deg`. On this rig:
        1.39 and 1.49 degrees, i.e. 2.04 degrees of combined tilt.

        Not cosmetic. The whole value of RVC here is that pitch and roll give
        a vertical that never drifts, and everything downstream leans on it —
        levelling the point cloud, finding the floor, measuring object height.
        A constant 2.04 degree lie in that vertical tips the floor by 14 cm
        across a 4 m room, and does it consistently enough to look like a real
        sloping floor rather than an error.
        """
        # getattr, not attribute access: a reader built without a calibration
        # file must still report angles, just uncorrected ones. Refusing to
        # work at all because the mount was never measured would be worse than
        # a 2 degree tilt.
        section = (getattr(self, "calibration", None) or {}).get("rvc_acceptance") or {}
        return (float(section.get("pitch_offset_deg", 0.0)),
                float(section.get("roll_offset_deg", 0.0)))

    def level_angles_deg(self, pitch_deg: float, roll_deg: float) -> tuple[float, float]:
        """Angles with the mount offset removed — what the WORLD is doing."""
        pitch_off, roll_off = self.mount_offsets_deg
        return (pitch_deg - pitch_off, roll_deg - roll_off)

    def at_time(self, t_query: float) -> dict[str, Any] | None:
        """Orientation at an arbitrary instant, interpolated on the frame clock.

        This is what ties the IMU to the camera. A keyframe is exposed at some
        moment; the nearest IMU sample can be up to 10 ms away and the arrival
        time of that sample up to 20 ms wrong. Interpolating on `t_grid` — the
        reconstructed device clock — removes both.

        `t_query` must be on the same monotonic base as the samples, i.e. the
        camera's `SensorTimestamp`, NOT a `time.monotonic()` taken after the
        frame was processed: that one includes capture, transfer and whatever
        inference ran in between.

        Returns None rather than the nearest sample when the query falls
        outside the buffered window. A silently-extrapolated orientation is
        indistinguishable from a real one downstream, and wrong.
        """
        with self._lock:
            samples = list(self._samples)
        if len(samples) < 2:
            return None
        if t_query < samples[0].t_grid or t_query > samples[-1].t_grid:
            return None

        low, high = 0, len(samples) - 1
        while high - low > 1:
            mid = (low + high) // 2
            if samples[mid].t_grid <= t_query:
                low = mid
            else:
                high = mid
        before, after = samples[low], samples[high]
        span = after.t_grid - before.t_grid
        weight = 0.0 if span <= 0 else (t_query - before.t_grid) / span

        def lerp(a: float, b: float) -> float:
            return a + weight * (b - a)

        # Yaw is interpolated on the SHORT way round, so a query landing on the
        # +/-180 wrap does not average 179 and -179 into 0.
        yaw_step = (after.yaw_deg - before.yaw_deg + 180.0) % 360.0 - 180.0
        pitch_level, roll_level = self.level_angles_deg(
            lerp(before.pitch_deg, after.pitch_deg),
            lerp(before.roll_deg, after.roll_deg))
        return {
            "yaw_deg": YAW_SIGN * (before.yaw_deg + weight * yaw_step),
            "pitch_deg": pitch_level,
            "roll_deg": roll_level,
            "t_grid": t_query,
            # How far apart the two samples used were. Above ~30 ms the
            # interpolation spanned a dropout and the answer is coarse — worth
            # recording next to the value rather than hiding.
            "interp_gap_ms": round(span * 1000.0, 2),
            "seq": before.seq,
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            elapsed = max(now - self._started_at, 1e-6) if self._started_at else 0.0
            age = (now - self._last_rx) if self._last_rx else float("inf")
            total = self._frames_ok + self._dropped
            return {
                "frames_ok": self._frames_ok,
                "frames_bad": self._frames_bad,     # checksum/resync failures
                "frames_dropped": self._dropped,    # gaps in the frame index
                "delivery_frac": round(self._frames_ok / total, 4) if total else 0.0,
                "rate_hz": round(self._frames_ok / elapsed, 1) if elapsed else 0.0,
                "age_s": round(age, 3) if math.isfinite(age) else None,
                "healthy": self._is_healthy(now),
                "window": len(self._samples),
                "calibrated": self.calibration is not None,
                "mode": (self.calibration or {}).get("mode"),
                "gravity_mode": (
                    "dynamic_tilt_model" if self.integrator.tilt_model else "static_reference"
                ),
                "suppressed_off_reference": self._suppressed,
            }

    def _is_healthy(self, now: float, max_age_s: float = 0.5) -> bool:
        """Fresh data, not merely data. Callers of `latest()` otherwise get
        the last sample forever after the cable is pulled, and a heading
        that is frozen looks exactly like a heading that is steady."""
        return bool(self._last_rx) and (now - self._last_rx) <= max_age_s

    def is_healthy(self, max_age_s: float = 0.5) -> bool:
        with self._lock:
            return self._is_healthy(time.monotonic(), max_age_s)

    # ---------------------------------------------------------- orientation

    def read_orientation(self, max_age_s: float = 0.5) -> dict[str, Any] | None:
        """Attitude for navigation: the whole point of this link.

        Returns None when the stream is stale rather than a frozen last
        value — a navigation filter must be told "no measurement" so its
        uncertainty grows, not handed a stale one it will treat as fresh
        evidence.

        `yaw_deg` is CONTINUOUS (unwrapped past +/-180) so consumers can
        difference it freely; `yaw_wrapped_deg` is the raw wire value.
        Pitch/roll are the window median, not the instantaneous sample —
        a single bump would otherwise show up as a tilt spike.
        """
        with self._lock:
            now = time.monotonic()
            if not self._is_healthy(now, max_age_s) or not self._samples:
                return None
            sample = self._samples[-1]
            yaw_continuous = self._yaw_unwrapped
            pitch = statistics.median(s.pitch_deg for s in self._samples)
            roll = statistics.median(s.roll_deg for s in self._samples)
            age = now - self._last_rx
        return {
            # Normalized to the plan frame's convention (CCW positive). This
            # is the one consumers should use; the raw sensor value is kept
            # alongside for diagnostics only.
            "yaw_deg": round(YAW_SIGN * float(yaw_continuous), 3),
            "yaw_sensor_deg": round(float(yaw_continuous), 3),
            "yaw_wrapped_deg": round(sample.yaw_deg, 2),
            "pitch_deg": round(float(pitch), 2),
            "roll_deg": round(float(roll), 2),
            "tilt_deg": round(sample.accel_tilt_deg(), 2),
            "age_s": round(age, 3),
        }

    def averaged_up_sensor(self) -> tuple[float, float, float] | None:
        """Low-passed unit 'up' over the current window.

        Median per axis, not mean: a single jolt (the chassis hitting a
        threshold) would drag a mean but leaves a median alone.
        """
        samples = self.window()
        vectors = [s.up_sensor() for s in samples]
        vectors = [v for v in vectors if v is not None]
        if not vectors:
            return None
        axes = [statistics.median(v[k] for v in vectors) for k in range(3)]
        norm = math.sqrt(sum(a * a for a in axes))
        if norm < 1e-6:
            return None
        return (axes[0] / norm, axes[1] / norm, axes[2] / norm)

    def averaged_pitch_roll_deg(self) -> tuple[float, float] | None:
        """Median (pitch, roll) over the current window — same rationale as
        `averaged_up_sensor`. Used by `imu_calibrate.py`'s `full` mode to
        pair a steady attitude reading with the accelerometer's `up` at the
        same instant."""
        samples = self.window()
        if not samples:
            return None
        pitch = statistics.median(s.pitch_deg for s in samples)
        roll = statistics.median(s.roll_deg for s in samples)
        return (pitch, roll)

    def read_motion(self) -> dict[str, Any] | None:
        """Everything the IMU knows about this keyframe, and closes the
        preintegration segment that ended with it.

        Deliberately raw: the segment's displacement magnitude, the attitude,
        and the shake statistics all travel to the Mac, where they are fused
        with the poses. The Pi does not try to decide anything from them.
        """
        sample = self.latest()
        if sample is None or not self.is_healthy():
            return None
        with self._lock:
            segment = self.integrator.cut()
        motion: dict[str, Any] = {
            # Same normalization as read_orientation — see YAW_SIGN.
            "yaw_deg": round(YAW_SIGN * sample.yaw_deg, 2),
            "yaw_sensor_deg": round(sample.yaw_deg, 2),
            "pitch_deg": round(sample.pitch_deg, 2),
            "roll_deg": round(sample.roll_deg, 2),
            "tilt_deg": round(sample.accel_tilt_deg(), 2),
            "accel_mg": [round(v, 1) for v in sample.accel_mg],
            "segment": segment,
            "gravity_mode": (
                "dynamic_tilt_model" if self.integrator.tilt_model else "static_reference"
            ),
        }
        gravity = self.read()
        if gravity is not None:
            motion["gravity_camera"] = [round(v, 5) for v in gravity]
        if segment is not None:
            # The SAME doubt that withholds gravity has to reach the distance,
            # and it matters far more there. `read()` returns None when the rig
            # has tilted outside the pose the calibration was measured in; the
            # integrator meanwhile keeps subtracting a gravity vector it can no
            # longer justify, and that residual is integrated TWICE.
            #
            # The arithmetic, at this rig's ~1.07 s keyframe interval:
            #     3 deg of gravity error -> 0.51 m/s^2 -> 0.29 m of phantom
            #     motion per interval, from standing still.
            # Measured on session_20260807_171938, where gravity was withheld
            # for all 143 keyframes: the IMU reported 87.9 m of path around one
            # 4 x 3.6 m room, median 0.357 m per interval — i.e. the signal was
            # essentially all gravity-subtraction error, and it inflated the
            # reconstruction's metric scale by ~17x.
            #
            # Orientation is unaffected: yaw comes from the sensor's own fusion
            # and drifts 0.03 deg/min. Only the doubly-integrated distance is
            # this fragile.
            segment["gravity_ok"] = gravity is not None
        return motion

    def read(self) -> list[float] | None:
        """Gravity (pointing DOWN) in the CAMERA frame, or None.

        This is the contract `scene_recorder.gravity_source` expects and what
        `SceneSession.imu_up_vector()` on the Mac consumes. Two calibration
        modes, both produced by `imu_calibrate.py`:

        * **reference** — the normal outcome for a robot that drives level.
          Gravity in the camera frame is a constant for a rig that only
          translates and yaws (yaw turns about the gravity axis itself), so
          the calibration measured it once against a plumb wall board. Here
          the IMU's job is to confirm the rig is still in that attitude: if
          it has tilted more than `tilt_tolerance_deg` away, this returns
          None and the Mac falls back to its floor-plane fit rather than
          being handed a vector that no longer describes reality.

        * **full** — a rotation solved from several rig attitudes, so
          gravity is tracked live however the rig is tilted.

        Returns None with no calibration at all, rather than guessing an
        axis mapping: the Mac prefers an IMU vector over its own floor fit,
        so a confidently wrong one is worse than none.
        """
        if not self.calibration:
            return None
        import numpy as np

        up_sensor = self.averaged_up_sensor()
        if up_sensor is None:
            return None
        sensor = np.asarray(up_sensor, dtype=np.float64)

        rotation = self.calibration.get("cam_from_imu")
        if self.calibration.get("mode") == "full" and rotation:
            up_camera = np.asarray(rotation, dtype=np.float64) @ sensor
        else:
            reference_imu = self.calibration.get("up_imu_reference")
            reference_cam = self.calibration.get("up_camera_reference")
            if not reference_imu or not reference_cam:
                return None
            drift = angle_between_deg(sensor, np.asarray(reference_imu, dtype=np.float64))
            tolerance = float(self.calibration.get("tilt_tolerance_deg", 4.0))
            if not math.isfinite(drift) or drift > tolerance:
                self._suppressed += 1
                return None
            up_camera = np.asarray(reference_cam, dtype=np.float64)

        norm = float(np.linalg.norm(up_camera))
        if norm < 1e-6:
            return None
        down_camera = -up_camera / norm
        return [float(v) for v in down_camera]
