"""Measure the rotation between the IMU and the camera, using gravity.

The pipeline needs exactly one thing from the IMU: which way is DOWN, in the
camera's frame. That is a direction — two degrees of freedom — so this does
not need a full six-DoF hand-eye calibration, and it does not need the IMU's
axis convention to be looked up anywhere. It needs pairs of vectors that
point at the same physical direction in the two frames, and enough different
rig orientations to pin the rotation down.

The trick is where the camera's "up" comes from: **the chessboard lying flat
on the floor**. `solvePnP` gives the board's pose in the camera frame, and
the board's normal is the floor normal, which is up. The same instant, the
accelerometer reports up in its own frame. Tilt the rig around, collect the
pairs, and solve Wahba's problem (`kabsch_rotation`) for the rotation that
maps one onto the other.

Everything else is a check that the answer is trustworthy:

  * a rest stage that confirms the sensor really reads 1 g and is quiet, and
    that the fused Euler tilt agrees with the accelerometer tilt (they use
    different axis conventions, but the tilt ANGLE is convention-free, so
    disagreement there means something is genuinely wrong);
  * an angular-spread requirement, because pairs taken at one orientation
    are degenerate — they constrain a rotation about the shared axis not at
    all, and would produce a confident, wrong matrix;
  * a residual: after solving, every pair is re-projected and the angular
    error reported. A good result is a couple of degrees.

Usage (on the Pi, board flat on the floor, camera looking at it):

    ./scripts/run_imu_calibrate.sh --square-mm 24

Writes config/imu_calibration.json, which `imu_rvc.RvcReader` reads. Without
that file the reader reports no gravity at all rather than guessing.
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
    kabsch_rotation,
)
from pi_client.scene_calibrate import CANDIDATE_PATTERNS, PreviewUplink, find_board
from shared.config import load_config

logger = logging.getLogger(__name__)

# A pair only earns its place if the rig is pointing somewhere new. Below
# this the pair adds noise, not information.
MIN_NEW_ANGLE_DEG = 8.0
# The whole set must span at least this much, or the rotation about the
# common axis is unconstrained and the fit is meaningless however small its
# residual looks.
MIN_SPREAD_DEG = 25.0
GOOD_RESIDUAL_DEG = 3.0
USABLE_RESIDUAL_DEG = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate IMU orientation against the camera")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "config/imu_calibration.json")
    parser.add_argument("--intrinsics", type=Path, default=REPO_ROOT / "config/scene_intrinsics.json")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=864)
    parser.add_argument("--cols", type=int, default=None, help="Inner corners per row (default: auto)")
    parser.add_argument("--rows", type=int, default=None, help="Inner corners per column (default: auto)")
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--pairs", type=int, default=12, help="Stop after this many accepted pairs")
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--rest-seconds", type=float, default=5.0)
    parser.add_argument("--lens-position", type=float, default=None,
                        help="Manual focus; default: the value in intrinsics.json")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port-tcp", type=int, default=None)
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def board_normal_camera(rvec: Any, tvec: Any) -> np.ndarray:
    """Unit floor normal (pointing UP) in the camera frame.

    The board's own +Z is its normal, but whether that comes out of the front
    or back face depends on corner ordering, which flips with viewing angle.
    Disambiguate physically instead of trusting the ordering: the board is on
    the floor and the camera is above it, so the up normal must point back
    towards the camera. `-tvec` is the board-to-camera direction.
    """
    import cv2

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    normal = rotation @ np.array([0.0, 0.0, 1.0])
    normal = normal / (np.linalg.norm(normal) or 1.0)
    to_camera = -np.asarray(tvec, dtype=np.float64).reshape(3)
    if float(normal @ to_camera) < 0:
        normal = -normal
    return normal


def spread_deg(vectors: list[np.ndarray]) -> float:
    """Largest angle between any two vectors in the set."""
    worst = 0.0
    for i, a in enumerate(vectors):
        for b in vectors[i + 1:]:
            worst = max(worst, angle_between_deg(a, b))
    return worst


def measure_rest(reader: RvcReader, seconds: float) -> dict[str, Any] | None:
    """Confirm the sensor is sane before trusting anything it says."""
    logger.info("Rest check: hold the rig still for %.0f s ...", seconds)
    # The reader only keeps a 0.5 s window, so accumulate across the whole
    # rest period here — the point is to catch drift and vibration over
    # seconds, which half a second of samples cannot show.
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
        logger.warning("  the two tilt estimates disagree by %.1f deg — the rig was probably "
                       "moving during the rest check; redo it on a still surface.",
                       abs(euler_tilt - accel_tilt))
    if max(noise) > 50:
        logger.warning("  noisy (%.0f mg) — vibration will not stop the calibration but "
                       "will widen the residual.", max(noise))
    return {
        "accel_magnitude_mg": round(magnitude, 1),
        "noise_mg": [round(v, 2) for v in noise],
        "tilt_accel_deg": round(accel_tilt, 2),
        "tilt_euler_deg": round(euler_tilt, 2),
        "samples": len(samples),
    }


def draw_status(cv2, frame, accepted: int, target: int, spread: float, message: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"IMU CALIBRATE  pairs {accepted}/{target}  spread {spread:.0f} deg "
                       f"(need {MIN_SPREAD_DEG:.0f})",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, message, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (60, 220, 60) if accepted else (200, 200, 60), 2, cv2.LINE_AA)


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

    try:
        time.sleep(1.0)  # let the window fill
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

        pattern: tuple[int, int] | None = (
            (args.cols, args.rows) if args.cols and args.rows else None
        )
        candidates = [pattern] if pattern else list(CANDIDATE_PATTERNS)

        logger.info("Put the chessboard FLAT ON THE FLOOR and point the camera at it.")
        logger.info("Then slowly tilt and roll the rig — keep the board in view — until "
                    "the spread reaches %.0f deg. Preview: http://%s:8080/ -> Live tab.",
                    MIN_SPREAD_DEG, host)

        up_camera: list[np.ndarray] = []
        up_imu: list[np.ndarray] = []
        message = "looking for the board"

        while not stop["flag"] and len(up_camera) < args.pairs:
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
                        logger.info("Locked pattern: %dx%d inner corners", *candidate)
                    break

            if found and pattern is not None:
                objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
                objp *= args.square_mm / 1000.0
                ok, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist)
                imu_up = reader.averaged_up_sensor()
                if ok and imu_up is not None:
                    cam_up = board_normal_camera(rvec, tvec)
                    imu_vec = np.asarray(imu_up, dtype=np.float64)
                    novelty = min(
                        (angle_between_deg(imu_vec, prev) for prev in up_imu),
                        default=180.0,
                    )
                    if novelty >= MIN_NEW_ANGLE_DEG:
                        up_camera.append(cam_up)
                        up_imu.append(imu_vec)
                        message = f"accepted (new angle {novelty:.0f} deg)"
                        logger.info("Pair %d/%d  cam_up=[%+.3f %+.3f %+.3f]  "
                                    "imu_up=[%+.3f %+.3f %+.3f]  new angle %.0f deg",
                                    len(up_camera), args.pairs, *cam_up, *imu_vec, novelty)
                    else:
                        message = f"tilt further (only {novelty:.0f} deg from a previous pose)"
                else:
                    message = "board found, waiting for a steady IMU reading"
                cv2.drawChessboardCorners(bgr, pattern, corners, found)
            else:
                message = "board not visible — put it flat on the floor, in view"

            if preview is not None:
                draw_status(cv2, bgr, len(up_camera), args.pairs,
                            spread_deg(up_imu) if len(up_imu) > 1 else 0.0, message)
                ok_jpeg, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok_jpeg:
                    preview.send(encoded.tobytes(), bgr.shape[1], bgr.shape[0])

        source.stop()
    finally:
        reader.stop()
        if preview is not None:
            preview.close()

    # ------------------------------------------------------------- solve
    if len(up_camera) < args.min_pairs:
        logger.error("Only %d pairs (need %d) — nothing written.", len(up_camera), args.min_pairs)
        return 1
    spread = spread_deg(up_imu)
    if spread < MIN_SPREAD_DEG:
        logger.error(
            "Orientations only span %.0f deg (need %.0f). Every pair points nearly the same "
            "way, so the rotation about that axis is unconstrained — a fit from this data "
            "would look confident and be wrong. Redo it tilting the rig much more.",
            spread, MIN_SPREAD_DEG,
        )
        return 1

    rotation = kabsch_rotation(up_imu, up_camera)
    residuals = [
        angle_between_deg(rotation @ imu, cam) for imu, cam in zip(up_imu, up_camera)
    ]
    mean_residual = float(np.mean(residuals))
    max_residual = float(np.max(residuals))

    # What the camera is actually doing in the room, in plain terms: the
    # angle between its optical axis and the horizontal plane, in the rig's
    # current pose. This is the number the recording guide talks about
    # ("tilt the camera up 15-25 deg"), so report it rather than make the
    # user infer it from a matrix.
    resting_up = reader.averaged_up_sensor()
    camera_pitch = None
    if resting_up is not None:
        up_cam_now = rotation @ np.asarray(resting_up)
        optical_axis = np.array([0.0, 0.0, 1.0])  # camera +Z, OpenCV convention
        camera_pitch = 90.0 - angle_between_deg(optical_axis, up_cam_now)

    if mean_residual <= GOOD_RESIDUAL_DEG:
        logger.info("Residual: mean %.2f deg, max %.2f deg — good.", mean_residual, max_residual)
    elif mean_residual <= USABLE_RESIDUAL_DEG:
        logger.info("Residual: mean %.2f deg, max %.2f deg — USABLE (ideal < %.0f). "
                    "Saved; the Mac averages gravity over the whole session, so a few "
                    "degrees here is not a problem.",
                    mean_residual, max_residual, GOOD_RESIDUAL_DEG)
    else:
        logger.warning("Residual: mean %.2f deg, max %.2f deg — too high to trust (< %.0f wanted). "
                       "Usual cause: the board was not actually flat on the floor, or the rig "
                       "moved while a pair was captured. Saving anyway, but redo it.",
                       mean_residual, max_residual, USABLE_RESIDUAL_DEG)
    if camera_pitch is not None:
        logger.info("Camera optical axis is %.1f deg above horizontal in the current pose "
                    "(the recording guide wants +15 to +25).", camera_pitch)

    payload = {
        "cam_from_imu": [[float(v) for v in row] for row in rotation],
        "pairs": len(up_camera),
        "spread_deg": round(spread, 1),
        "residual_deg_mean": round(mean_residual, 3),
        "residual_deg_max": round(max_residual, 3),
        "camera_pitch_deg": None if camera_pitch is None else round(camera_pitch, 1),
        "rest": rest,
        "port": args.port,
        "baud": args.baud,
        "square_mm": args.square_mm,
        "pattern": list(pattern) if pattern else None,
        "lens_position": lens_position,
        "calibrated_at": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved %s", args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
