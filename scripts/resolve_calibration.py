#!/usr/bin/env python3
"""Re-solve the camera calibration from saved board detections. No camera.

`scene_calibrate.py` writes every detected board's corners to
`config/scene_intrinsics_captures.npz`. This script solves from that file, so
a question about focal length, principal point or distortion can be answered
by re-running the solver with different constraints instead of by another
capture session with the board and the robot.

That matters because the failure mode we hit is a *solver* problem, not a
capture problem. On 2026-08-03 a capture with 30 boards, full 3x3 frame
coverage, three distance buckets and RMS 0.484 px produced fx=1098 (69.9 deg)
when a tape measure said 968 (76.9 deg). Focal length and distance are nearly
degenerate for head-on views, so the solve slid along that pair. The fix is
to pin the focal length with the independent measurement and let the solver
re-fit everything else around it — which needs the corners, not the camera.

Typical use, after `run_focal_check.sh` has measured the true focal length:

    python3 scripts/resolve_calibration.py --fix-fx 967.7

Or just inspect what the saved data supports, changing nothing:

    python3 scripts/resolve_calibration.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "client/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--captures", type=Path,
                        default=REPO_ROOT / "config/scene_intrinsics_captures.npz")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "config/scene_intrinsics.json")
    parser.add_argument("--fix-fx", type=float, default=None,
                        help="Pin the focal length to this value (pixels) and re-fit "
                             "the principal point and distortion around it. Use the "
                             "`fx_implied` number that run_focal_check.sh measured "
                             "against a tape.")
    parser.add_argument("--fix-principal-point", action="store_true",
                        help="Also hold cx/cy at the frame centre. Worth trying if "
                             "the principal point comes out implausibly far off.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solve and print, write nothing.")
    return parser.parse_args()


def main() -> int:
    import cv2

    from pi_client.scene_calibrate import board_tilt_spread_deg, check_calibration_sanity

    args = parse_args()
    if not args.captures.exists():
        print(f"No saved detections at {args.captures}.\n"
              f"Run ./scripts/run_scene_calibrate.sh once; it writes this file "
              f"alongside the intrinsics, and every later re-solve reads it.",
              file=sys.stderr)
        return 2

    data = np.load(args.captures)
    obj_points = [np.asarray(p, dtype=np.float32) for p in data["object_points"]]
    img_points = [np.asarray(p, dtype=np.float32) for p in data["image_points"]]
    width, height = (int(v) for v in data["image_size"])
    pattern = tuple(int(v) for v in data["pattern"])
    print(f"{len(obj_points)} boards, pattern {pattern[0]}x{pattern[1]}, {width}x{height}")

    flags = 0
    camera_matrix = None
    dist_init = None
    if args.fix_fx is not None:
        # An initial guess plus CALIB_FIX_FOCAL_LENGTH: OpenCV keeps fx/fy at
        # the supplied values and optimizes only cx, cy and the distortion
        # coefficients. Because the focal length is no longer free to absorb
        # error, those remaining parameters come out consistent with the
        # measured lens rather than with the degenerate solve.
        camera_matrix = np.array([
            [args.fix_fx, 0.0, width / 2.0],
            [0.0, args.fix_fx, height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist_init = np.zeros(5, dtype=np.float64)
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_USE_INTRINSIC_GUESS
        print(f"Pinning fx = fy = {args.fix_fx:.1f} px and re-fitting the rest")
    if args.fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        print("Holding the principal point at the frame centre")

    rms, K, dist, rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, (width, height), camera_matrix, dist_init, flags=flags
    )

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    hfov = 2.0 * math.degrees(math.atan(width / 2.0 / fx))
    tilt_spread = board_tilt_spread_deg(rvecs)
    print(f"  RMS      {rms:.3f} px")
    print(f"  fx/fy    {fx:.1f} / {fy:.1f}   -> {hfov:.1f} deg horizontal")
    print(f"  cx/cy    {cx:.1f} / {cy:.1f}   (centre {width/2:.0f} / {height/2:.0f})")
    print(f"  dist     {[round(float(v), 5) for v in dist.reshape(-1)[:5]]}")
    print(f"  board orientation spread {tilt_spread:.1f} deg")

    problems = check_calibration_sanity(
        fx, fy, cx, cy, width, height,
        tilt_spread_deg=None if args.fix_fx is not None else tilt_spread,
    )
    # With the focal length pinned to an external measurement, a small
    # orientation spread is no longer a problem: the degenerate direction is
    # constrained by the tape rather than by the geometry of the capture.
    for problem in problems:
        print(f"  PROBLEM: {problem}")

    intrinsics = {
        "width": width, "height": height,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "dist": [float(v) for v in dist.reshape(-1)[:5]],
        "lens_position": float(data["lens_position"]),
        "rms_px": float(rms),
        "boards": len(obj_points),
        "pattern": list(pattern),
        "square_mm": float(data["square_mm"]),
        "board_tilt_spread_deg": round(tilt_spread, 1),
        "focal_pinned_to": args.fix_fx,
        "resolved_from_captures": str(args.captures),
        "calibrated_at": time.time(),
        "sanity_problems": problems,
    }
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    if problems:
        print("\nRefusing to write while problems remain (use scene_calibrate --force "
              "semantics deliberately not offered here — fix the input instead)",
              file=sys.stderr)
        return 3
    args.output.write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
