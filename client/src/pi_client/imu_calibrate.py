"""Measure which way is DOWN in the camera's frame, using a plumb wall board.

Designed around what this rig can actually do: a robot that drives in the
X-Y plane. It does not tilt on command and it certainly does not turn over.

**Why the reference comes from a board on a WALL.** Two candidates give a
camera-frame gravity direction. A board lying on the floor gives it as the
plane's normal, which needs no care about how the board is turned — but the
camera sits ~40 cm up and pitched upward, so it sees the floor at a grazing
angle where `solvePnP` is badly conditioned. A board taped flat to a
vertical wall is seen face-on, where the pose is precise; the price is that
the board must hang *plumb*, since we then read gravity off its in-plane
vertical axis instead of its normal.

**Why the robot does not need to be tilted.** For a rig that only translates
and yaws, gravity in the camera frame is a constant: yaw is a rotation about
the gravity axis itself, so it leaves that axis unchanged. Driving the robot
around therefore produces the same vector pair over and over, and no amount
of it constrains the rotation about gravity. Which is fine — because a
constant is exactly what is needed. Measuring it once, well, beats the Mac's
floor-plane fit, and that is the whole job.

So there are two modes, and the common one needs nothing unusual:

  * `reference` — the robot in its normal driving pose. One accurate static
    measurement of down-in-camera, plus the IMU reading taken at the same
    instant. At record time the IMU is used to *verify* the rig is still in
    that pose: if its tilt has drifted from the reference by more than
    `tilt_tolerance_deg`, the recorder reports no gravity for those frames
    and the Mac falls back to its floor fit, instead of quietly emitting a
    stale vector.

  * `full` — optional, and only worth it if you want gravity tracked while
    the rig is off level. Needs the rig at genuinely different attitudes,
    which for this robot means propping one side up: slide a book (2-4 cm is
    plenty) under the left wheels, capture, move it to the front, capture
    again. About 10 deg of total spread is enough — the residual error when
    mapping a later 5 deg deviation is then well under a degree. The script
    accepts each new attitude automatically; you never have to tell it
    anything.

Ctrl+C at any point finishes with whatever was collected: if there is not
enough attitude spread for `full`, it writes a `reference` calibration
rather than a badly-conditioned rotation that would look authoritative.

Usage (board taped to a wall, hanging plumb, camera looking at it):

    ./scripts/run_imu_calibrate.sh --square-mm 24
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "client/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import numpy as np

from pi_client.camera import CameraFocusOptions
from pi_client.camera_session import CaptureSettings, make_capture_source
from pi_client.imu_rvc import (
    DEFAULT_BAUD,
    DEFAULT_PORT,
    RvcReader,
    angle_between_deg,
    fit_tilt_model,
    kabsch_rotation,
)
from pi_client.imu_shtp_motion import ShtpMotionReader
from pi_client.scene_calibrate import CANDIDATE_PATTERNS, PreviewUplink, find_board
from shared.config import load_config

logger = logging.getLogger(__name__)

# In OpenCV camera axes (X right, Y down, Z forward) the camera's own "up".
CAMERA_UP = np.array([0.0, -1.0, 0.0])
# ...and where it is looking, which the yaw check flattens into the
# horizontal plane to measure heading against the board.
CAMERA_FORWARD = np.array([0.0, 0.0, 1.0])

# The board's vertical axis must land within this of the camera's up, or the
# board is not hanging the way this procedure assumes (or the camera is
# rolled over). Generous, because the camera is deliberately pitched up.
MAX_BOARD_UP_TILT_DEG = 55.0
# On a vertical wall the board's normal is horizontal, i.e. perpendicular to
# gravity. If it isn't, the board is lying on a table, not hanging.
MAX_NORMAL_TILT_DEG = 35.0

MIN_NEW_ANGLE_DEG = 4.0     # a new attitude has to actually be new
# The yaw check fits a slope; the turn has to be wide enough that the slope
# means something. The first real run swept 15.8 deg and the gain came out
# with an uncertainty larger than the quantity being tested.
MIN_YAW_SPAN_DEG = 25.0
# How far apart adjacent inner corners must be in the IMAGE. This is what
# actually limits corner accuracy, and unlike a distance threshold it does
# not depend on the intrinsics being right — which on this rig they may not
# be (see check_intrinsics_sanity). Below ~12 px subpixel refinement has
# little to work with; at the rig's true ~75 deg FOV that is about 2 m with
# the 25 mm board, so it only bites when the operator is genuinely far off.
MIN_CORNER_PITCH_PX = 12.0
FULL_MODE_SPREAD_DEG = 8.0  # below this, rotation about gravity is guesswork
GOOD_RESIDUAL_DEG = 3.0
USABLE_RESIDUAL_DEG = 8.0
DEFAULT_TILT_TOLERANCE_DEG = 4.0

# The mounting as described by hand: the IMU's accelerometer X points
# forward and Y points left, so Z points up. In camera axes that maps
# IMU X -> +Z, Y -> -X, Z -> -Y. Nothing depends on this being right; it is
# reported as a difference so a rotated bracket shows up as a number.
NOMINAL_CAM_FROM_IMU = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure gravity in the camera frame from a plumb wall board"
    )
    parser.add_argument("--driver", choices=["rvc", "shtp"], default="rvc",
                        help="rvc: imu_rvc.RvcReader (115200 UART-RVC) - the current "
                             "hardware jumper state. shtp: imu_shtp_motion."
                             "ShtpMotionReader (raw gyro, 3 Mbaud UART-SHTP). Mutually "
                             "exclusive on the physical sensor - a calibration "
                             "measured with one driver's accelerometer sign "
                             "convention must not be fed to the other (confirmed "
                             "this session: RVC reads ~+1g at rest, SHTP ~-1g).")
    parser.add_argument("--output", type=Path, default=None,
                        help="default: config/imu_calibration.json (rvc) or "
                             "config/imu_calibration_shtp.json (shtp)")
    parser.add_argument("--intrinsics", type=Path, default=REPO_ROOT / "config/scene_intrinsics.json")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=864)
    parser.add_argument("--cols", type=int, default=None, help="Inner corners per row (default: auto)")
    parser.add_argument("--rows", type=int, default=None, help="Inner corners per column (default: auto)")
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--views", type=int, default=20,
                        help="Stop after this many accepted views")
    parser.add_argument("--reference-views", type=int, default=8,
                        help="Views averaged into the reference measurement")
    parser.add_argument("--rest-seconds", type=float, default=5.0)
    parser.add_argument("--settle-seconds", type=float, default=8.0,
                        help="Once the reference is collected, finish automatically "
                             "after this long with no new rig attitude")
    parser.add_argument("--tilt-tolerance-deg", type=float, default=DEFAULT_TILT_TOLERANCE_DEG,
                        help="How far the rig may drift from the reference attitude before "
                             "the recorder stops trusting the reference vector")
    parser.add_argument("--lens-position", type=float, default=None,
                        help="Manual focus; default: the value in intrinsics.json")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port-tcp", type=int, default=None)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--yaw-check", type=float, default=0.0, metavar="SECONDS",
                        help="After the gravity calibration, spend SECONDS swinging the "
                             "robot left/right in front of the board to VALIDATE the yaw "
                             "the navigation filter runs on. 45-60 is plenty. This cannot "
                             "improve the gravity calibration - rotating about gravity "
                             "leaves the gravity vector unchanged by construction - but it "
                             "measures the yaw gain, sign, per-sample error and drift under "
                             "MOTION, which nothing on this rig had checked (the 0.03 deg/min "
                             "bench figure was measured stationary).")
    parser.add_argument("--yaw-check-only", action="store_true",
                        help="Run only the yaw check against the EXISTING calibration and "
                             "print the report; write nothing. Use this to re-validate after "
                             "moving the sensor without redoing the whole procedure.")
    return parser.parse_args()


def board_axes_camera(rvec: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The board's three axes as unit vectors in the camera frame."""
    import cv2

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return rotation[:, 0], rotation[:, 1], rotation[:, 2]


def board_up_camera(rvec: Any) -> tuple[np.ndarray | None, str]:
    """Up (gravity, negated) in the camera frame from a plumb wall board.

    Which of the board's two in-plane axes runs vertically, and which way
    round it points, both depend on the corner ordering the detector happened
    to produce — that flips with viewing angle and is not worth trusting.
    Resolve it physically instead: of the four candidates (+/- each in-plane
    axis), the vertical one is whichever sits closest to the camera's own up.
    That is unambiguous as long as the camera isn't rolled onto its side,
    which a wheeled robot is not.

    Returns (up, reason); up is None when the board clearly isn't hanging on
    a wall the way this procedure needs.
    """
    axis_x, axis_y, normal = board_axes_camera(rvec)

    # A board on a vertical wall has a horizontal normal. One lying flat has
    # a vertical one — catching that here beats silently reading gravity off
    # the wrong axis.
    normal_tilt = abs(90.0 - angle_between_deg(normal, CAMERA_UP))
    if normal_tilt > MAX_NORMAL_TILT_DEG:
        return None, f"board is not on a vertical surface (normal {normal_tilt:.0f} deg off)"

    candidates = [axis_x, -axis_x, axis_y, -axis_y]
    best = min(candidates, key=lambda v: angle_between_deg(v, CAMERA_UP))
    offset = angle_between_deg(best, CAMERA_UP)
    if offset > MAX_BOARD_UP_TILT_DEG:
        return None, f"no board axis points up (closest {offset:.0f} deg)"
    return best / (np.linalg.norm(best) or 1.0), "ok"


def camera_yaw_about_up_deg(rvec: Any, up_camera: np.ndarray) -> float | None:
    """The camera's heading relative to the board, measured about gravity.

    This is the ground truth the yaw check needs. `up_camera` comes from the
    board itself (`board_up_camera`), so the angle is taken about the true
    vertical rather than about the camera's own tilted axis — a 6 deg
    camera pitch would otherwise smear into the yaw reading.

    Both the board's normal and the optical axis are flattened into the
    horizontal plane and the SIGNED angle between them is returned, so left
    and right are distinguishable. None when the geometry degenerates
    (camera looking straight up/down the board normal), which cannot happen
    with a wall board and a wheeled robot but is cheap to guard.

    **The normal is negated first, and that is not cosmetic.** A board on a
    wall faces the robot, so its outward normal points back along the
    optical axis and the un-negated angle sits at +/-180 — exactly the
    `atan2` discontinuity. Measured that way the series flips between
    +179.9 and -179.9 on sensor noise alone, and `unwrap_deg` then banks
    every flip as real rotation. That produced a smooth, confident and
    completely wrong `gain: -2.19` on the first real run (offset_deg
    -174.4 in that report is the fingerprint). Negating puts the operating
    point at 0, where wrapping cannot reach.
    """
    _axis_x, _axis_y, outward = board_axes_camera(rvec)
    normal = -np.asarray(outward, dtype=np.float64)
    up = np.asarray(up_camera, dtype=np.float64)
    up = up / (np.linalg.norm(up) or 1.0)

    def flatten(v: np.ndarray) -> np.ndarray | None:
        horizontal = v - float(v @ up) * up
        norm = float(np.linalg.norm(horizontal))
        return horizontal / norm if norm > 1e-6 else None

    reference = flatten(np.asarray(normal, dtype=np.float64))
    optical = flatten(np.asarray(CAMERA_FORWARD, dtype=np.float64))
    if reference is None or optical is None:
        return None
    sin_term = float(np.cross(reference, optical) @ up)
    cos_term = float(reference @ optical)
    return float(np.degrees(np.arctan2(sin_term, cos_term)))


def unwrap_deg(series: list[float]) -> list[float]:
    """Continuous angle series from one that wraps at +/-180."""
    if not series:
        return []
    out = [series[0]]
    for previous, current in zip(series, series[1:]):
        out.append(out[-1] + (current - previous + 180.0) % 360.0 - 180.0)
    return out


def fit_yaw_relation(
    camera_yaw_deg: list[float],
    imu_yaw_deg: list[float],
    elapsed_s: list[float] | None = None,
) -> dict[str, Any] | None:
    """Compare IMU yaw against the board's PnP yaw over a real rotation.

    Deliberately a VALIDATION, not a calibration input. Rotating about
    gravity leaves the gravity vector in the camera frame unchanged, so
    none of this can refine `up_camera_reference`/`up_imu_reference` — it
    measures whether the yaw the navigation filter leans on is actually
    trustworthy, which nothing on this rig had ever checked.

    Fits `imu = gain * camera + offset` by least squares:

    * `gain` should be +/-1. The SIGN is information, not noise: the RVC
      yaw convention and the camera's need not agree, and a robot that
      turns left while the filter thinks it turned right is a failure the
      gravity calibration cannot see. Magnitude away from 1 means the
      sensor's yaw axis is not aligned with the true vertical.
    * `residual_rms_deg` is the honest per-sample yaw error, and what
      `pi_navigation --imu-yaw-sigma-deg` should be set from.
    * `drift_deg_per_min` regresses the residual against time. This is the
      number `Ekf2DHeading.predict`'s `bias_walk_per_s` was guessed at:
      the 0.03 deg/min bench figure was measured STATIONARY, and drift
      under motion is what actually matters.

    Returns None when the rotation covered is too small to fit anything —
    refusing is better than reporting a gain fitted to sensor noise.
    """
    if len(camera_yaw_deg) != len(imu_yaw_deg) or len(camera_yaw_deg) < 8:
        return None

    raw = np.asarray(camera_yaw_deg, dtype=np.float64)
    # Refuse to unwrap a series that lives on the discontinuity. If the
    # board-relative yaw sits near +/-180, every noise flip becomes a fake
    # 360 deg step and the fit below is smooth, confident garbage — the
    # exact failure that produced gain -2.19 on the first real run.
    if float(np.mean(np.abs(raw) > 150.0)) > 0.2:
        logger.error(
            "Board-relative yaw is sitting at +/-180 (median |yaw| = %.0f deg) — that is "
            "the atan2 discontinuity and no fit from it can be trusted. This means the "
            "board normal is pointing the wrong way; it is a code bug, not an operator "
            "mistake.", float(np.median(np.abs(raw))),
        )
        return None

    camera = np.asarray(unwrap_deg(camera_yaw_deg), dtype=np.float64)
    imu = np.asarray(unwrap_deg(imu_yaw_deg), dtype=np.float64)
    span = float(camera.max() - camera.min())
    if span < MIN_YAW_SPAN_DEG:
        return None

    def solve(mask: np.ndarray) -> tuple[float, float, np.ndarray]:
        design = np.column_stack([camera[mask], np.ones(int(mask.sum()))])
        (g, o), *_ = np.linalg.lstsq(design, imu[mask], rcond=None)
        return float(g), float(o), imu - (g * camera + o)

    # One robust pass: planar PnP is two-fold ambiguous near head-on, so a
    # handful of pose flips is expected and must not drag the slope.
    keep = np.ones(len(camera), dtype=bool)
    gain, offset, residual = solve(keep)
    scale = float(np.median(np.abs(residual - np.median(residual)))) * 1.4826
    if scale > 1e-6:
        keep = np.abs(residual - np.median(residual)) <= 3.0 * scale
        if int(keep.sum()) >= 8:
            gain, offset, residual = solve(keep)

    kept_residual = residual[keep]
    centred = camera[keep] - camera[keep].mean()
    sxx = float(centred @ centred)
    resid_std = float(np.std(kept_residual, ddof=2)) if int(keep.sum()) > 2 else float("inf")
    gain_se = resid_std / math.sqrt(sxx) if sxx > 1e-9 else float("inf")

    report: dict[str, Any] = {
        "samples": len(camera),
        "samples_rejected": int((~keep).sum()),
        "yaw_span_deg": round(span, 2),
        "gain": round(float(gain), 4),
        "gain_std_err": round(gain_se, 4) if math.isfinite(gain_se) else None,
        "offset_deg": round(float(offset), 2),
        "residual_rms_deg": round(float(np.sqrt(np.mean(kept_residual ** 2))), 3),
        "residual_max_deg": round(float(np.max(np.abs(kept_residual))), 3),
    }
    if elapsed_s is not None and len(elapsed_s) == len(camera):
        t = np.asarray(elapsed_s, dtype=np.float64)
        if float(t.max() - t.min()) > 5.0:
            slope, _ = np.polyfit(t, residual, 1)
            report["drift_deg_per_min"] = round(float(slope) * 60.0, 3)
            report["duration_s"] = round(float(t.max() - t.min()), 1)

    problems = []
    if abs(abs(gain) - 1.0) > 0.05:
        # Only call the axis wrong when the fit is precise enough to say so.
        # An imprecise gain is an inconclusive run, not a hardware fault —
        # reporting the two as the same thing sends you chasing wiring that
        # is fine.
        if math.isfinite(gain_se) and gain_se < 0.05:
            problems.append(
                f"gain {gain:+.3f} +/- {gain_se:.3f} is not +/-1 — IMU yaw axis is not vertical"
            )
        else:
            problems.append(
                f"gain {gain:+.3f} is off, but its own uncertainty is +/-{gain_se:.2f} — "
                f"the run is inconclusive, not a diagnosis. Turn through a wider angle "
                f"(swept only {span:.0f} deg) from further back."
            )
    if report["residual_rms_deg"] > 3.0:
        problems.append(f"residual {report['residual_rms_deg']:.1f} deg is too large to navigate on")
    report["sign"] = "agrees" if gain > 0 else "inverted"
    report["ok"] = not problems
    report["problems"] = problems
    return report


def spread_deg(vectors: list[np.ndarray]) -> float:
    """Largest angle between any two vectors in the set."""
    worst = 0.0
    for i, a in enumerate(vectors):
        for b in vectors[i + 1:]:
            worst = max(worst, angle_between_deg(a, b))
    return worst


def average_unit(vectors: list[np.ndarray]) -> np.ndarray:
    mean = np.mean(np.asarray(vectors, dtype=np.float64), axis=0)
    return mean / (np.linalg.norm(mean) or 1.0)


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """How far apart two rotations are, as a single angle."""
    relative = np.asarray(a) @ np.asarray(b).T
    cos_theta = (float(np.trace(relative)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def corner_pitch_px(corners: Any, pattern: tuple[int, int]) -> float:
    """Median spacing between horizontally adjacent inner corners, in pixels.

    The intrinsics-free measure of "is the board big enough in frame to
    localize accurately". Preferred over a distance threshold because
    distance comes out of solvePnP, which is exactly what a wrong focal
    length corrupts.
    """
    cols, rows = pattern
    grid = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    return float(np.median(np.linalg.norm(np.diff(grid, axis=1), axis=2)))


def check_intrinsics_sanity(
    camera_matrix: Any, width: int, height: int, expected_hfov_deg: float = 75.0,
) -> list[str]:
    """Warn when the loaded intrinsics disagree with this camera.

    Everything the calibration and the yaw check produce flows through
    `solvePnP`, so a wrong focal length silently biases the gravity
    reference AND the yaw gain — for a planar target, focal error and
    out-of-plane rotation are strongly coupled, so it is not just distances
    that go wrong.

    Measured on this project: `config/scene_intrinsics.json` claims fx 534
    at 1536x864, i.e. a 110 deg horizontal field, while VGGT's refined
    intrinsics (75.6 deg) and COLMAP/DA3 (75.7/74.4 deg) agree on ~75 deg —
    fx should be ~991. Its `cy` also sits 68 px off the frame centre. That
    file was almost certainly calibrated at a different capture resolution.
    """
    problems: list[str] = []
    fx = float(camera_matrix[0][0])
    cx, cy = float(camera_matrix[0][2]), float(camera_matrix[1][2])
    hfov = 2.0 * math.degrees(math.atan(width / 2.0 / fx)) if fx > 1e-6 else 0.0
    if abs(hfov - expected_hfov_deg) > 12.0:
        problems.append(
            f"intrinsics imply a {hfov:.0f} deg horizontal field, but this camera measures "
            f"~{expected_hfov_deg:.0f} deg (fx should be ~{width / 2 / math.tan(math.radians(expected_hfov_deg / 2)):.0f}, "
            f"not {fx:.0f}) — solvePnP angles and distances will both be biased"
        )
    for name, value, centre in (("cx", cx, width / 2.0), ("cy", cy, height / 2.0)):
        if abs(value - centre) > 0.1 * centre:
            problems.append(
                f"{name}={value:.0f} is {abs(value - centre):.0f} px off the frame centre "
                f"({centre:.0f}) — the intrinsics were probably calibrated at a different "
                f"capture resolution than {width}x{height}"
            )
    return problems


def lock_pattern(
    cv2: Any, source: Any, candidates: list[tuple[int, int]], stop: dict[str, bool],
    timeout_s: float = 30.0,
) -> tuple[int, int] | None:
    """Find which chessboard is on the wall. Needed by `--yaw-check-only`,
    which skips the calibration loop where this normally happens."""
    deadline = time.monotonic() + timeout_s
    while not stop["flag"] and time.monotonic() < deadline:
        bgr = source.capture_bgr()
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for candidate in candidates:
            found, _ = find_board(cv2, gray, candidate)
            if found:
                logger.info("Locked pattern: %dx%d inner corners", *candidate)
                return candidate
    return None


def run_yaw_check(
    cv2: Any,
    source: Any,
    reader: Any,
    camera_matrix: Any,
    dist: Any,
    pattern: tuple[int, int],
    seconds: float,
    square_mm: float,
    stop: dict[str, bool],
    preview: Any = None,
) -> dict[str, Any] | None:
    """Drive/turn in front of the board while both yaw sources are logged.

    The operator swings the robot left and right (and may roll it forward
    and back) with the board in view. Every frame that resolves a board
    pose contributes one (camera yaw, IMU yaw, distance) triple; the fit
    happens afterwards in `fit_yaw_relation`.

    Distance is logged so a yaw error that only appears close to the board
    can be told apart from one that grows with time — the first would be
    PnP geometry, the second real IMU drift.
    """
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= square_mm / 1000.0

    camera_yaw: list[float] = []
    imu_yaw: list[float] = []
    elapsed: list[float] = []
    distance_m: list[float] = []
    pitch_px: list[float] = []

    logger.info("YAW CHECK (%.0f s): stand the robot roughly 0.7-1.0 m from the board — "
                "far enough that the squares are not blurred by the lens being parked at "
                "~1 m, close enough that each square is still tens of pixels across. Then "
                "swing LEFT and RIGHT through at least %.0f deg of TOTAL sweep, keeping "
                "the board in view. Rolling forward and back as well is welcome; it is a "
                "control, not a requirement.", seconds, MIN_YAW_SPAN_DEG)
    started = time.monotonic()
    while not stop["flag"] and time.monotonic() - started < seconds:
        bgr = source.capture_bgr()
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        found, corners = find_board(cv2, gray, pattern)
        message = "board not visible"
        if found:
            ok, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist)
            cam_up, reason = board_up_camera(rvec) if ok else (None, "pose failed")
            orientation = reader.read_orientation() if hasattr(reader, "read_orientation") else None
            if cam_up is not None and orientation is not None:
                yaw = camera_yaw_about_up_deg(rvec, cam_up)
                if yaw is not None:
                    camera_yaw.append(yaw)
                    imu_yaw.append(float(orientation["yaw_deg"]))
                    elapsed.append(time.monotonic() - started)
                    distance_m.append(float(np.linalg.norm(np.asarray(tvec).reshape(3))))
                    pitch_px.append(corner_pitch_px(corners, pattern))
                    swept = max(camera_yaw) - min(camera_yaw)
                    small = pitch_px[-1] < MIN_CORNER_PITCH_PX
                    message = (
                        f"{len(camera_yaw)} samples, swept {swept:.0f}/{MIN_YAW_SPAN_DEG:.0f} deg, "
                        f"{pitch_px[-1]:.0f} px/square"
                        + (" — TOO FAR, move closer" if small else "")
                    )
            elif cam_up is None:
                message = reason
            else:
                message = "board found, waiting for a fresh IMU reading"
            cv2.drawChessboardCorners(bgr, pattern, corners, found)

        if preview is not None:
            remaining = seconds - (time.monotonic() - started)
            draw_status(cv2, bgr, len(camera_yaw), 0, 0.0,
                        f"YAW CHECK {remaining:.0f}s — swing left/right — {message}")
            ok_jpeg, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok_jpeg:
                preview.send(encoded.tobytes(), bgr.shape[1], bgr.shape[0])

    # Always dump the raw pairs. A yaw check that fails is worth diagnosing
    # offline; re-running blind costs the operator far more than this file.
    if camera_yaw:
        raw_dir = REPO_ROOT / "data/imu_tests"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"yaw_check_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with raw_path.open("w", encoding="utf-8") as handle:
            handle.write("t_s,camera_yaw_deg,imu_yaw_deg,distance_m,corner_pitch_px\n")
            for t, c, i, d, p in zip(elapsed, camera_yaw, imu_yaw, distance_m, pitch_px):
                handle.write(f"{t:.3f},{c:.4f},{i:.4f},{d:.4f},{p:.2f}\n")
        logger.info("Raw yaw-check samples: %s", raw_path)

    report = fit_yaw_relation(camera_yaw, imu_yaw, elapsed)
    if report is None:
        logger.warning(
            "Yaw check inconclusive: %d samples over %.0f deg (need %.0f). Refusing to "
            "report a gain whose uncertainty exceeds what it is meant to measure.",
            len(camera_yaw),
            (max(camera_yaw) - min(camera_yaw)) if len(camera_yaw) > 1 else 0.0,
            MIN_YAW_SPAN_DEG,
        )
        return None
    if distance_m:
        # Reported, not gated on: distance comes out of solvePnP and is
        # therefore only as good as the intrinsics, which on this rig are
        # suspect (check_intrinsics_sanity).
        report["distance_m_min"] = round(float(min(distance_m)), 2)
        report["distance_m_max"] = round(float(max(distance_m)), 2)
    if pitch_px:
        report["corner_pitch_px_median"] = round(float(np.median(pitch_px)), 1)
        if float(np.median(pitch_px)) < MIN_CORNER_PITCH_PX:
            report["problems"].append(
                f"only {np.median(pitch_px):.0f} px between corners (want >= "
                f"{MIN_CORNER_PITCH_PX:.0f}) — too far from the board for the corners "
                f"to localize accurately"
            )
            report["ok"] = False
    return report


def measure_rest(reader: RvcReader, seconds: float) -> dict[str, Any] | None:
    """Confirm the sensor is sane before trusting anything it says."""
    logger.info("Rest check: leave the robot still for %.0f s ...", seconds)
    # The reader keeps only a 0.5 s window, so accumulate across the whole
    # period — the point is to catch drift and vibration over seconds.
    deadline = time.monotonic() + seconds
    seen: dict[float, Any] = {}
    while time.monotonic() < deadline:
        time.sleep(0.1)
        for sample in reader.window():
            seen[sample.monotonic] = sample
    samples = [seen[k] for k in sorted(seen)]
    if len(samples) < 10:
        logger.error("No IMU data (%d samples) — is %s the right port?", len(samples), reader.port)
        return None

    axes = np.asarray([s.accel_mg for s in samples], dtype=np.float64)
    magnitude = float(np.median(np.linalg.norm(axes, axis=1)))
    noise = [float(np.std(axes[:, k])) for k in range(3)]
    euler_tilt = float(np.median([s.euler_tilt_deg() for s in samples]))
    accel_tilt = float(np.median([s.accel_tilt_deg() for s in samples]))

    logger.info("  |accel| = %.0f mg (expect ~1000), per-axis noise %.1f %.1f %.1f mg",
                magnitude, *noise)
    logger.info("  tilt from level: %.2f deg (accel) vs %.2f deg (fused euler)",
                accel_tilt, euler_tilt)
    if not 900 <= magnitude <= 1100:
        logger.error("  |accel| is %.0f mg, not ~1 g — the scale or the frame decode is wrong.",
                     magnitude)
        return None
    if abs(euler_tilt - accel_tilt) > 5.0:
        logger.warning("  the two tilt estimates disagree by %.1f deg — something was moving; "
                       "redo the rest check with the robot switched off.",
                       abs(euler_tilt - accel_tilt))
    if max(noise) > 50:
        logger.warning("  noisy (%.0f mg) — vibration widens the result but does not stop it.",
                       max(noise))
    return {
        "accel_magnitude_mg": round(magnitude, 1),
        "noise_mg": [round(v, 2) for v in noise],
        "tilt_accel_deg": round(accel_tilt, 2),
        "tilt_euler_deg": round(euler_tilt, 2),
        "samples": len(samples),
    }


def measure_rest_shtp(reader: ShtpMotionReader, seconds: float) -> dict[str, Any] | None:
    """SHTP equivalent of measure_rest() above. Raw accel here is m/s^2, not
    milli-g (RVC's unit), and there is no independent on-chip fused Euler to
    cross-check against — SHTP mode as used by this project only has raw
    accel+gyro reports enabled (see imu_shtp_uart.py), not the onboard
    Rotation Vector — so this reports accel-only tilt and skips the
    RVC-style two-estimate agreement check rather than fake one."""
    logger.info("Rest check: leave the robot still for %.0f s ...", seconds)
    deadline = time.monotonic() + seconds
    seen: dict[float, tuple[float, float, float]] = {}
    while time.monotonic() < deadline:
        time.sleep(0.1)
        for t, accel in reader.window():
            seen[t] = accel
    samples = [seen[k] for k in sorted(seen)]
    if len(samples) < 10:
        logger.error("No IMU data (%d samples) — is %s the right port?", len(samples), reader.port)
        return None

    axes = np.asarray(samples, dtype=np.float64)
    magnitude = float(np.median(np.linalg.norm(axes, axis=1)))
    noise = [float(np.std(axes[:, k])) for k in range(3)]
    vertical = np.max(np.abs(axes), axis=1)
    accel_tilt = float(np.median(np.degrees(np.arccos(
        np.clip(vertical / np.maximum(np.linalg.norm(axes, axis=1), 1e-9), -1.0, 1.0)
    ))))

    logger.info("  |accel| = %.2f m/s^2 (expect ~9.81), per-axis noise %.3f %.3f %.3f m/s^2",
                magnitude, *noise)
    logger.info("  tilt from level (accel only, no independent fused-euler cross-check "
                "under SHTP): %.2f deg", accel_tilt)
    if not 9.0 <= magnitude <= 10.6:
        logger.error("  |accel| is %.2f m/s^2, not ~9.81 — the scale or the frame decode is wrong.",
                     magnitude)
        return None
    if max(noise) > 0.5:
        logger.warning("  noisy (%.2f m/s^2) — vibration widens the result but does not stop it.",
                       max(noise))
    return {
        "accel_magnitude_ms2": round(magnitude, 3),
        "noise_ms2": [round(v, 4) for v in noise],
        "tilt_accel_deg": round(accel_tilt, 2),
        "samples": len(samples),
    }


def log_yaw_report(report: dict[str, Any]) -> None:
    logger.info("Yaw check: %d samples over %.0f deg of turn, %.1f-%.1f m from the board",
                report["samples"], report["yaw_span_deg"],
                report.get("distance_m_min", float("nan")),
                report.get("distance_m_max", float("nan")))
    logger.info("  gain %+.4f (%s the camera's sense)  offset %+.1f deg",
                report["gain"], report["sign"], report["offset_deg"])
    logger.info("  per-sample yaw error: %.2f deg RMS, %.2f deg worst",
                report["residual_rms_deg"], report["residual_max_deg"])
    if "drift_deg_per_min" in report:
        logger.info("  drift under motion: %+.2f deg/min over %.0f s "
                    "(bench figure, stationary, was 0.03)",
                    report["drift_deg_per_min"], report["duration_s"])
    for problem in report["problems"]:
        logger.error("  PROBLEM: %s", problem)
    if report["ok"]:
        logger.info("  -> set pi_navigation --imu-yaw-sigma-deg to about %.1f",
                    max(0.5, report["residual_rms_deg"]))


def draw_status(cv2, frame, views: int, target: int, spread: float, message: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 96), (0, 0, 0), -1)
    cv2.putText(frame, f"IMU  views {views}/{target}   attitude spread {spread:.1f} deg",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    ready = "REFERENCE OK" if views else "need the board in view"
    extra = "  (+ full tilt tracking)" if spread >= FULL_MODE_SPREAD_DEG else ""
    cv2.putText(frame, ready + extra, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (60, 220, 60) if views else (60, 200, 240), 2, cv2.LINE_AA)
    cv2.putText(frame, message, (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)


def main() -> int:
    import cv2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("pi_client.network").setLevel(logging.WARNING)
    args = parse_args()
    if args.driver == "shtp":
        if args.output is None:
            args.output = REPO_ROOT / "config/imu_calibration_shtp.json"
        if args.port is None:
            args.port = "/dev/ttyUSB0"
        if args.baud is None:
            args.baud = 3_000_000
    else:
        if args.output is None:
            args.output = REPO_ROOT / "config/imu_calibration.json"
        if args.port is None:
            args.port = DEFAULT_PORT
        if args.baud is None:
            args.baud = DEFAULT_BAUD

    if not args.intrinsics.exists():
        logger.error("No camera intrinsics at %s — run ./scripts/run_scene_calibrate.sh first. "
                     "solvePnP needs a calibrated camera.", args.intrinsics)
        return 2
    intrinsics = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    camera_matrix = np.array([
        [intrinsics["fx"], 0.0, intrinsics["cx"]],
        [0.0, intrinsics["fy"], intrinsics["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist = np.asarray(intrinsics.get("dist") or [0, 0, 0, 0, 0], dtype=np.float64)
    for problem in check_intrinsics_sanity(
        camera_matrix, int(intrinsics.get("width", args.width)),
        int(intrinsics.get("height", args.height)),
    ):
        logger.error("INTRINSICS: %s", problem)
    lens_position = args.lens_position
    if lens_position is None:
        lens_position = float(intrinsics.get("lens_position", 1.0))
        logger.info("Using LensPosition %.2f from the camera calibration (must match).",
                    lens_position)

    config = load_config(REPO_ROOT / "config/default.json", REPO_ROOT / ".env")
    host = args.host or config.get("client", {}).get("server_host", "127.0.0.1")
    tcp_port = args.port_tcp or int(config.get("server", {}).get("port", 8765))
    device_id = str(config.get("client", {}).get("device_id", "raspberry_pi_01"))
    preview = None if args.no_preview else PreviewUplink(host, tcp_port, device_id)

    if args.driver == "shtp":
        reader = ShtpMotionReader(port=args.port, baud=args.baud)
    else:
        reader = RvcReader(port=args.port, baud=args.baud)
    if not reader.start():
        return 2

    stop = {"flag": False}

    def handle_signal(_signum: int, _frame: Any) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    up_camera: list[np.ndarray] = []
    up_imu: list[np.ndarray] = []
    pitch_roll: list[tuple[float, float]] = []
    yaw_report: dict[str, Any] | None = None
    pattern: tuple[int, int] | None = None
    source = None

    try:
        time.sleep(1.0)  # let the sample window fill
        rest = (
            measure_rest_shtp(reader, args.rest_seconds) if args.driver == "shtp"
            else measure_rest(reader, args.rest_seconds)
        )
        if rest is None:
            return 2

        source = make_capture_source(
            "camera",
            CaptureSettings(
                width=args.width, height=args.height, fps=5.0, jpeg_quality=90,
                focus_options=CameraFocusOptions(
                    autofocus_mode="manual", autofocus_range="normal",
                    autofocus_speed="normal", lens_position=lens_position,
                ),
            ),
            REPO_ROOT / "data/cat.jpg",
        )
        source.start()

        pattern = (args.cols, args.rows) if args.cols and args.rows else None
        candidates = [pattern] if pattern else list(CANDIDATE_PATTERNS)

        if args.yaw_check_only:
            # Validate the yaw the navigation filter runs on, against the
            # calibration already on disk. Writes nothing.
            locked = pattern or lock_pattern(cv2, source, candidates, stop)
            if locked is None:
                logger.error("Board never came into view — nothing to validate against.")
                return 1
            report = run_yaw_check(
                cv2, source, reader, camera_matrix, dist, locked,
                max(args.yaw_check, 45.0), args.square_mm, stop, preview,
            )
            if report is None:
                return 1
            log_yaw_report(report)
            return 0 if report["ok"] else 1

        logger.info("Tape the chessboard FLAT ON A WALL, hanging straight (plumb), and "
                    "park the robot facing it so the whole board is in frame.")
        logger.info("Stay put: that alone produces a usable calibration. To also track "
                    "tilt, slide a 2-4 cm book under one side of the robot, wait, then "
                    "move it to another side. Preview: http://%s:8080/ -> Live tab.", host)

        message = "looking for the board"
        # Once the reference is in hand the run is already useful, and for a
        # robot that cannot tilt no further attitude will ever arrive. Ending
        # on a quiet timer means the common case finishes by itself instead
        # of sitting there looking stuck; propping a book under a wheel
        # restarts the clock, so the optional `full` path still works.
        last_accept = time.monotonic()
        while not stop["flag"] and len(up_camera) < args.views:
            if (len(up_camera) >= args.reference_views
                    and time.monotonic() - last_accept >= args.settle_seconds):
                logger.info("No new rig attitude for %.0f s — finishing.", args.settle_seconds)
                break
            bgr = source.capture_bgr()
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            found, corners = False, None
            for candidate in candidates:
                found, corners = find_board(cv2, gray, candidate)
                if found:
                    if pattern is None:
                        pattern = candidate
                        candidates = [candidate]
                        logger.info("Locked pattern: %dx%d inner corners "
                                    "(= a %dx%d square board)",
                                    candidate[0], candidate[1],
                                    candidate[0] + 1, candidate[1] + 1)
                    break

            if found and pattern is not None:
                objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
                objp *= args.square_mm / 1000.0
                ok, rvec, _tvec = cv2.solvePnP(objp, corners, camera_matrix, dist)
                cam_up, reason = board_up_camera(rvec) if ok else (None, "pose failed")
                imu_up = reader.averaged_up_sensor()
                imu_attitude = reader.averaged_pitch_roll_deg()
                if cam_up is not None and imu_up is not None and imu_attitude is not None:
                    imu_vec = np.asarray(imu_up, dtype=np.float64)
                    novelty = min(
                        (angle_between_deg(imu_vec, prev) for prev in up_imu),
                        default=180.0,
                    )
                    if novelty >= MIN_NEW_ANGLE_DEG or len(up_camera) < args.reference_views:
                        up_camera.append(cam_up)
                        up_imu.append(imu_vec)
                        pitch_roll.append(imu_attitude)
                        last_accept = time.monotonic()
                        message = (f"view {len(up_camera)} accepted "
                                   f"(attitude {novelty:.1f} deg from the nearest earlier one)")
                        logger.info("View %d/%d  cam_up=[%+.3f %+.3f %+.3f]  "
                                    "imu_up=[%+.3f %+.3f %+.3f]",
                                    len(up_camera), args.views, *cam_up, *imu_vec)
                    else:
                        remaining = args.settle_seconds - (time.monotonic() - last_accept)
                        message = (f"reference collected — finishing in {max(0.0, remaining):.0f} s. "
                                   "Prop one side up on a book now if you want tilt tracking.")
                elif cam_up is None:
                    message = reason
                else:
                    message = "board found, waiting for a steady IMU reading"
                cv2.drawChessboardCorners(bgr, pattern, corners, found)
            else:
                message = "board not visible — tape it flat on the wall, facing the robot"

            if preview is not None:
                draw_status(cv2, bgr, len(up_camera), args.views,
                            spread_deg(up_imu) if len(up_imu) > 1 else 0.0, message)
                ok_jpeg, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok_jpeg:
                    preview.send(encoded.tobytes(), bgr.shape[1], bgr.shape[0])

        if args.yaw_check > 0 and pattern is not None and not stop["flag"]:
            yaw_report = run_yaw_check(
                cv2, source, reader, camera_matrix, dist, pattern,
                args.yaw_check, args.square_mm, stop, preview,
            )
    finally:
        if source is not None:
            try:
                source.stop()
            except Exception:
                pass
        reader.stop()
        if preview is not None:
            preview.close()

    # ------------------------------------------------------------- solve
    if not up_camera:
        logger.error("No usable views — nothing written. The board must be flat on a "
                     "VERTICAL surface and fully inside the frame.")
        return 1

    reference_cam = average_unit(up_camera[: args.reference_views])
    reference_imu = average_unit(up_imu[: args.reference_views])
    spread = spread_deg(up_imu) if len(up_imu) > 1 else 0.0

    payload: dict[str, Any] = {
        "driver": args.driver,
        "up_camera_reference": [float(v) for v in reference_cam],
        "up_imu_reference": [float(v) for v in reference_imu],
        "tilt_tolerance_deg": args.tilt_tolerance_deg,
        "views": len(up_camera),
        "spread_deg": round(spread, 2),
        # Validation, not an input to the solve above — see fit_yaw_relation.
        # Stored alongside so a calibration file carries the evidence that the
        # yaw the navigation filter leans on was actually checked, and when.
        "yaw_check": yaw_report,
        "rest": rest,
        "port": args.port,
        "baud": args.baud,
        "square_mm": args.square_mm,
        "pattern": list(pattern) if pattern else None,
        "lens_position": lens_position,
        "calibrated_at": time.time(),
    }

    # How steady the reference itself was: scatter of the views that went
    # into it. A shaky number here means the robot or the board moved.
    reference_scatter = max(
        (angle_between_deg(v, reference_cam) for v in up_camera[: args.reference_views]),
        default=0.0,
    )
    payload["reference_scatter_deg"] = round(reference_scatter, 2)

    down_camera = -reference_cam
    logger.info("Gravity (down) in the camera frame: [%+.4f %+.4f %+.4f]",
                *down_camera)
    camera_pitch = 90.0 - angle_between_deg(np.array([0.0, 0.0, 1.0]), reference_cam)
    logger.info("Camera optical axis sits %.1f deg above horizontal "
                "(the recording guide wants +15 to +25).", camera_pitch)
    payload["camera_pitch_deg"] = round(camera_pitch, 1)
    if reference_scatter > 2.0:
        logger.warning("The reference views scatter by %.1f deg — the robot or the board "
                       "was moving. Redo it with both held still.", reference_scatter)

    if spread >= FULL_MODE_SPREAD_DEG:
        rotation = kabsch_rotation(up_imu, up_camera)
        residuals = [angle_between_deg(rotation @ i, c) for i, c in zip(up_imu, up_camera)]
        mean_residual = float(np.mean(residuals))
        max_residual = float(np.max(residuals))
        payload.update({
            "mode": "full",
            "cam_from_imu": [[float(v) for v in row] for row in rotation],
            "residual_deg_mean": round(mean_residual, 3),
            "residual_deg_max": round(max_residual, 3),
            "off_nominal_deg": round(rotation_angle_deg(rotation, NOMINAL_CAM_FROM_IMU), 1),
        })
        # The unobservable part is the spin about gravity, and its
        # uncertainty grows as 1/sin(spread). Report what that costs at a
        # realistic runtime deviation instead of leaving it implicit.
        spin_error = np.degrees(np.arcsin(min(1.0, np.radians(mean_residual)
                                              / max(np.sin(np.radians(spread)), 1e-6))))
        cost_at_5deg = 5.0 * 2 * abs(np.sin(np.radians(spin_error) / 2))
        payload["spin_uncertainty_deg"] = round(float(spin_error), 1)
        if mean_residual <= GOOD_RESIDUAL_DEG:
            logger.info("Full tilt tracking: residual mean %.2f deg, max %.2f deg over "
                        "%.1f deg of attitude spread — good.",
                        mean_residual, max_residual, spread)
        elif mean_residual <= USABLE_RESIDUAL_DEG:
            logger.info("Full tilt tracking: residual mean %.2f deg (ideal < %.0f) — usable.",
                        mean_residual, GOOD_RESIDUAL_DEG)
        else:
            logger.warning("Full tilt tracking: residual mean %.2f deg is too high to trust; "
                           "the board was probably not plumb. Falling back to reference mode.",
                           mean_residual)
            payload["mode"] = "reference"
            payload.pop("cam_from_imu", None)
        if payload["mode"] == "full":
            logger.info("  a later 5 deg tilt of the robot maps with about %.2f deg of error.",
                        cost_at_5deg)
            logger.info("  measured mounting differs from the described one "
                        "(accel X forward, Y left) by %.1f deg.", payload["off_nominal_deg"])
    else:
        payload["mode"] = "reference"
        logger.info("Attitude spread was %.1f deg (need %.0f for cam_from_imu's 3D rotation "
                    "fit) — writing a REFERENCE calibration for gravity. That is the expected "
                    "outcome for a robot that drives level: gravity in the camera frame is a "
                    "constant, and this is a direct measurement of it. The recorder will emit "
                    "it for every keyframe and withhold it if the rig ever tilts more than "
                    "%.0f deg off this pose.",
                    spread, FULL_MODE_SPREAD_DEG, args.tilt_tolerance_deg)

    # Independent of cam_from_imu and of the 8 deg full-mode gate above:
    # this is the fix for the metric-scale bug (docs/scene3d.md,
    # poses_step.imu_scale_samples) — a live tilt->gravity model fit from
    # the same multi-attitude data. It's a much lower bar than
    # cam_from_imu's 3D rotation (a 2D linear regression, not a Wahba
    # solve), so it doesn't need FULL_MODE_SPREAD_DEG of spread — just
    # enough that pitch and roll both varied a little (fit_tilt_model's own
    # rank check catches "not enough", e.g. the rig only ever pitched and
    # never rolled). A few seconds of the rig gently rocking in front of
    # the board — the same small wobble it actually does while driving —
    # is exactly the right input, and is usually well under 8 deg. Uses
    # every accepted view, including the "reference" ones: if the rig was
    # already moving a little while those were collected (rather than
    # perfectly still), that's still usable signal, not just noise to
    # average away — only fit_tilt_model's own count/rank checks decide
    # whether there's enough of it.
    if len(up_camera) >= 3:
        reference_pitch_deg, reference_roll_deg = np.mean(
            pitch_roll[: args.reference_views], axis=0
        )
        tilt_model = fit_tilt_model(
            pitch_roll, up_imu, reference_pitch_deg, reference_roll_deg, reference_imu
        )
        if tilt_model is None:
            logger.warning("Not enough independent attitude spread to fit a tilt model — "
                            "need the rig's pitch AND roll to both vary a little (not just "
                            "one of them), e.g. gently rock it in front of the board rather "
                            "than tilting it the same way every time. ImuIntegrator will keep "
                            "using the static reference vector.")
        elif tilt_model["fit_residual_deg_mean"] <= GOOD_RESIDUAL_DEG:
            payload["tilt_model"] = tilt_model
            logger.info("Tilt model: residual mean %.2f deg, max %.2f deg over %d attitudes — "
                        "good. ImuIntegrator will now track gravity dynamically instead of "
                        "assuming a constant.", tilt_model["fit_residual_deg_mean"],
                        tilt_model["fit_residual_deg_max"], tilt_model["attitudes_used"])
        elif tilt_model["fit_residual_deg_mean"] <= USABLE_RESIDUAL_DEG:
            payload["tilt_model"] = tilt_model
            logger.info("Tilt model: residual mean %.2f deg (ideal < %.0f) — usable.",
                        tilt_model["fit_residual_deg_mean"], GOOD_RESIDUAL_DEG)
        else:
            logger.warning("Tilt model: residual mean %.2f deg is too high to trust — "
                            "ImuIntegrator will keep using the static reference vector.",
                            tilt_model["fit_residual_deg_mean"])
    else:
        logger.info("Only %d view(s) collected — need at least 3 to fit a tilt model.",
                    len(up_camera))

    if yaw_report is not None:
        log_yaw_report(yaw_report)
    elif args.yaw_check > 0:
        logger.warning("Yaw check produced no usable fit — the calibration below is "
                       "still valid, the yaw simply went unverified.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Сохранить чужие разделы файла, а не переписать его целиком.
    #
    # Этот файл делят два измерителя: здесь живёт гравитационная калибровка, а
    # `scripts/imu_acceptance.py` кладёт рядом `rvc_acceptance` — смещение
    # крепления, шумы и дрейф из приёмочных тестов. Полная перезапись стирала
    # их молча: калибровка отрабатывала успешно, печатала красивый JSON, и
    # измеренные 1.39/1.49 градуса перекоса просто исчезали. Обнаружилось это
    # только потому, что в выводе не хватало одной строки.
    #
    # Ключи, которые пишет эта функция, обновляются; всё остальное остаётся.
    merged = {}
    if args.output.exists():
        try:
            merged = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            merged = {}
        if merged:
            args.output.with_suffix(".json.bak").write_text(
                json.dumps(merged, indent=2), encoding="utf-8")
    merged.update(payload)
    args.output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    logger.info("Saved %s (mode: %s)", args.output, payload["mode"])
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
