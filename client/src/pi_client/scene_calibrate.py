"""One-time camera calibration for scene recording (chessboard).

Camera Module 3 has autofocus — intrinsics DRIFT with focus, so the lens is
locked at a fixed LensPosition here and the SAME value must be used for
every recording (scene_recorder.py reads it back from intrinsics.json).

Usage (print a 9x6 chessboard, 25 mm squares, glue it to something rigid):

    ./scripts/run_scene_calibrate.sh --lens-position 2.0

Move the board around the view (angles, corners, distances 0.5-2 m); a
frame is accepted automatically whenever corners are found and ~1 s passed.
Stops after --frames boards, writes config/scene_intrinsics.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "client/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import numpy as np

from pi_client.camera import CameraFocusOptions
from pi_client.camera_session import CaptureSettings, make_capture_source


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the Pi camera (chessboard)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "config/scene_intrinsics.json")
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=864)
    parser.add_argument("--cols", type=int, default=9, help="Inner corners per row")
    parser.add_argument("--rows", type=int, default=6, help="Inner corners per column")
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--frames", type=int, default=35)
    parser.add_argument("--lens-position", type=float, default=2.0,
                        help="Manual focus in 1/m; MUST be reused for recording")
    return parser.parse_args()


def main() -> int:
    import cv2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    source = make_capture_source(
        "camera",
        CaptureSettings(
            width=args.width, height=args.height, fps=5.0, jpeg_quality=95,
            focus_options=CameraFocusOptions(
                autofocus_mode="manual", autofocus_range="normal",
                autofocus_speed="normal", lens_position=args.lens_position,
            ),
        ),
        REPO_ROOT / "data/cat.jpg",
    )
    source.start()

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_mm / 1000.0  # meters

    obj_points, img_points = [], []
    last_accept = 0.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    logger.info(
        "Show the %dx%d board; need %d accepted frames (Ctrl+C to abort)",
        args.cols, args.rows, args.frames,
    )
    try:
        while len(obj_points) < args.frames:
            bgr = source.capture_bgr()
            if time.monotonic() - last_accept < 1.0:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if not found:
                continue
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)
            last_accept = time.monotonic()
            logger.info("Accepted %d/%d", len(obj_points), args.frames)
    finally:
        source.stop()

    if len(obj_points) < 10:
        logger.error("Only %d boards captured — not enough, aborting", len(obj_points))
        return 1

    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, (args.width, args.height), None, None
    )
    logger.info("RMS reprojection error: %.3f px (target < 0.5)", rms)

    intrinsics = {
        "width": args.width,
        "height": args.height,
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "dist": [float(v) for v in dist.reshape(-1)[:5]],
        "lens_position": args.lens_position,
        "rms_px": float(rms),
        "boards": len(obj_points),
        "calibrated_at": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
    logger.info("Saved %s", args.output)
    print(json.dumps(intrinsics, indent=2))
    return 0 if rms < 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
