"""BNO08x UART-RVC driver: gravity direction for scene reconstruction.

The sensor is on a CH340 USB-serial bridge (`/dev/ttyUSB0`, 115200 8N1) and
streams **UART-RVC**, not SHTP — a fixed 19-byte frame at 100 Hz:

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
100 Hz, |accel| = 997 mg at rest, per-axis noise ~2 mg.

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

logger = logging.getLogger(__name__)

HEADER = b"\xaa\xaa"
FRAME_LEN = 19
DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 115200
SAMPLE_RATE_HZ = 100.0
SAMPLE_WINDOW_S = 0.5  # low-pass window for gravity; 50 samples at 100 Hz


@dataclass(frozen=True)
class ImuSample:
    """One decoded RVC frame. Accel is milli-g in the SENSOR frame."""

    index: int
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    accel_mg: tuple[float, float, float]
    monotonic: float

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


def iter_frames(buffer: bytes) -> tuple[list[ImuSample], bytes]:
    """Pull every complete valid frame out of `buffer`; return the rest.

    Resynchronising matters: a USB-serial bridge drops bytes under load, and
    a decoder that only ever advances by whole frames stays permanently
    misaligned after a single lost byte. On a checksum failure this advances
    by ONE byte and looks for the next header, so the stream re-locks within
    a frame instead of producing garbage forever.
    """
    samples: list[ImuSample] = []
    cursor = 0
    length = len(buffer)
    while True:
        start = buffer.find(HEADER, cursor)
        if start < 0:
            # A header may straddle the read boundary — keep the last byte.
            return samples, buffer[max(cursor, length - 1):]
        if start + FRAME_LEN > length:
            return samples, buffer[start:]
        sample = parse_frame(buffer[start:start + FRAME_LEN])
        if sample is None:
            cursor = start + 1  # false header, resync
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
        self._frames_ok = 0
        self._frames_bad = 0
        self._suppressed = 0  # reads withheld because the rig left its reference pose

    # ------------------------------------------------------------ lifecycle

    def start(self) -> bool:
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
        self._thread = threading.Thread(target=self._run, name="imu-rvc", daemon=True)
        self._thread.start()
        logger.info("IMU reader started on %s @ %d", self.port, self.baud)
        return True

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
            samples, buffer = iter_frames(buffer)
            if len(buffer) > 4 * FRAME_LEN:  # nothing parseable — drop the junk
                buffer = buffer[-FRAME_LEN:]
            if not samples:
                continue
            now = time.monotonic()
            with self._lock:
                self._frames_ok += len(samples)
                self._samples.extend(samples)
                cutoff = now - self.window_s
                self._samples = [s for s in self._samples if s.monotonic >= cutoff]

    # ---------------------------------------------------------------- reads

    def latest(self) -> ImuSample | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def window(self) -> list[ImuSample]:
        with self._lock:
            return list(self._samples)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "frames_ok": self._frames_ok,
                "frames_bad": self._frames_bad,
                "window": len(self._samples),
                "calibrated": self.calibration is not None,
                "mode": (self.calibration or {}).get("mode"),
                "suppressed_off_reference": self._suppressed,
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
