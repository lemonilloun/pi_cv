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
from pi_client.scene_calibrate import CANDIDATE_PATTERNS, PreviewUplink, find_board
from shared.config import load_config

logger = logging.getLogger(__name__)

# In OpenCV camera axes (X right, Y down, Z forward) the camera's own "up".
CAMERA_UP = np.array([0.0, -1.0, 0.0])

# The board's vertical axis must land within this of the camera's up, or the
# board is not hanging the way this procedure assumes (or the camera is
# rolled over). Generous, because the camera is deliberately pitched up.
MAX_BOARD_UP_TILT_DEG = 55.0
# On a vertical wall the board's normal is horizontal, i.e. perpendicular to
# gravity. If it isn't, the board is lying on a table, not hanging.
MAX_NORMAL_TILT_DEG = 35.0

MIN_NEW_ANGLE_DEG = 4.0     # a new attitude has to actually be new
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
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "config/imu_calibration.json")
    parser.add_argument("--intrinsics", type=Path, default=REPO_ROOT / "config/scene_intrinsics.json")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
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
    pattern: tuple[int, int] | None = None
    source = None

    try:
        time.sleep(1.0)  # let the sample window fill
        rest = measure_rest(reader, args.rest_seconds)
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
        "up_camera_reference": [float(v) for v in reference_cam],
        "up_imu_reference": [float(v) for v in reference_imu],
        "tilt_tolerance_deg": args.tilt_tolerance_deg,
        "views": len(up_camera),
        "spread_deg": round(spread, 2),
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
    # is exactly the right input, and is usually well under 8 deg.
    if len(up_camera) > args.reference_views:
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
        logger.info("Only the %d reference views were collected (all the same pose) — "
                    "no rocking/tilting happened, so there's nothing to fit a tilt model from.",
                    len(up_camera))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved %s (mode: %s)", args.output, payload["mode"])
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
