"""Decide the focal length against a tape measure, not against another model.

Three estimates of this camera's fx at 1536 px wide currently disagree:

    chessboard calibration (2026-08-03)   1098   -> 69.9 deg horizontal
    VGGT refined intrinsics                991   -> 75.6 deg
    COLMAP / DA3 (recorded in CLAUDE.md)  1001   -> 75.0 deg

A 10% spread is far outside noise, and it matters: every solvePnP distance,
the reconstruction's geometry, and the metric scale of the floor plan all
inherit fx linearly. The previous round shipped a 5-6x scale error precisely
because no one measured against something physical.

Neither of the disagreeing methods can settle it, because both are estimates
from the same class of evidence — one fits a board, the other fits a scene,
and both can be wrong in ways that look self-consistent. A tape measure is a
different kind of evidence entirely.

**The method.** Put the calibration board flat against a wall, park the robot
so the whole board is in frame, and measure the camera-to-board distance with
a tape. `solvePnP` then reports its own idea of that distance using whatever
fx is in `config/scene_intrinsics.json`. For a fixed observed board size the
reported distance scales linearly with the assumed focal length:

    d_measured = d_true * fx_assumed / fx_true
    =>  fx_true = fx_assumed * d_true / d_measured

so one careful tape reading pins fx directly. The two candidate focal lengths
predict distances 10.8% apart, which dwarfs the centimetre-level error of a
tape held straight. Work at **0.4-0.6 m**: close enough that the board is 50+
pixels per square (good corner localization, which is what actually limits
solvePnP), far enough that 1 cm of tape error is only ~2% against a 10.8%
effect.

Measure to the CAMERA LENS, not to the front of the chassis: on this rig
those differ by several centimetres, which is most of the effect being
measured.

    ./scripts/run_focal_check.sh --distance-m 0.50
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
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
from pi_client.scene_calibrate import CANDIDATE_PATTERNS, find_board

logger = logging.getLogger(__name__)

# A tape reading is good to ~1 cm, so the distance floor is set by when that
# 1 cm stops being small against the ~10% disagreement being tested: at 0.5 m
# it is 2% (a 5:1 margin), at 0.35 m it is 2.9%. Still fine. An earlier
# version put this at 0.6 m on the reasoning that 2% was "close to" 10% —
# that was simply wrong arithmetic, and it pushed the operator further away
# for no benefit.
#
# The real constraint runs the OTHER way: further back, the board shrinks and
# corner localization degrades, which hurts solvePnP more than a centimetre
# of tape ever could. At 1.0 m this board is 26 px per square; at 0.5 m it is
# 53. So the gate that matters is the board's size in the IMAGE, not the
# distance — see MIN_CORNER_PITCH_PX below.
MIN_CHECK_DISTANCE_M = 0.35

# Pixels between adjacent inner corners. Below ~15 px subpixel refinement has
# little to work with. Intrinsics-free, unlike a distance threshold — which
# matters here, since the intrinsics are the thing under suspicion.
MIN_CORNER_PITCH_PX = 15.0

# Physically possible focal lengths for this camera at 1536 px wide. The
# sensor is an imx708_WIDE; 400 px would be a 125 deg fisheye and 1600 px a
# 51 deg short telephoto. Anything outside says the ARITHMETIC INPUT is
# wrong, and the only free input is the operator's tape reading.
#
# This guard exists because on 2026-08-03 the tool confidently blamed the
# calibration for a 48% error when the real cause was a mistyped distance:
# the board measured 48.7 px per square in both runs, i.e. it had not moved,
# while the two runs claimed 0.50 m and 1.00 m. Board size depends only on
# distance, so the two claims could not both be true — and the tool had every
# number it needed to notice, and did not.
PLAUSIBLE_FX_RANGE = (400.0, 1600.0)


def implied_fx(fx_assumed: float, measured_distance_m: float, true_distance_m: float) -> float:
    """The focal length that would have made solvePnP report the tape reading.

    Linear because a pinhole projection scales the image of a fixed object by
    fx/distance: halve the focal length and the same board fills half as many
    pixels, which solvePnP can only explain by placing it twice as far away.
    """
    if measured_distance_m <= 0:
        raise ValueError("measured distance must be positive")
    return fx_assumed * true_distance_m / measured_distance_m


def corner_pitch_px(corners: Any, pattern: tuple[int, int]) -> float:
    """Median spacing between horizontally adjacent inner corners."""
    cols, rows = pattern
    grid = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    return float(np.median(np.linalg.norm(np.diff(grid, axis=1), axis=2)))


def summarize(
    distances_m: list[float],
    fx_assumed: float,
    true_distance_m: float,
    pitches_px: list[float] | None = None,
    square_m: float = 0.024,
) -> dict[str, Any]:
    """Median over many frames, plus the spread that says whether to believe it.

    Median rather than mean: a frame where the detector snapped to a reflection
    or a partially occluded board produces an outlier, and one such frame would
    drag a mean by more than the effect being measured.
    """
    if not distances_m:
        return {"ok": False, "reason": "the board was never detected"}
    measured = statistics.median(distances_m)
    spread = (max(distances_m) - min(distances_m)) / measured if measured else float("inf")
    fx_true = implied_fx(fx_assumed, measured, true_distance_m)
    error_frac = (measured - true_distance_m) / true_distance_m

    problems = []
    if spread > 0.05:
        problems.append(
            f"distance varied {spread:.1%} across frames — hold the robot still, "
            f"and check the whole board stays in view"
        )
    if true_distance_m < MIN_CHECK_DISTANCE_M:
        problems.append(
            f"too close to measure reliably: at {true_distance_m:.2f} m a 1 cm tape "
            f"error is {1.0 / (true_distance_m * 100):.1%} of the reading"
        )
    if pitches_px:
        pitch = statistics.median(pitches_px)
        # Intrinsics-free cross-check: the board's size in pixels and its true
        # distance fix the focal length by similar triangles alone, with no
        # solver involved. If that lands outside what this lens can be, the
        # suspect input is the tape reading, not the calibration.
        if square_m > 0:
            lo, hi = PLAUSIBLE_FX_RANGE
            direct_fx = pitch * true_distance_m / square_m
            if not (lo <= direct_fx <= hi):
                likely_m = square_m * (lo + hi) / 2 / pitch
                problems.append(
                    f"the board measures {pitch:.0f} px per square, which at "
                    f"{true_distance_m:.2f} m implies fx={direct_fx:.0f} — impossible "
                    f"for this lens. CHECK THE TAPE READING: that board size is "
                    f"consistent with roughly {likely_m:.2f} m, not "
                    f"{true_distance_m:.2f} m. Nothing is wrong with the calibration"
                )
        if pitch < MIN_CORNER_PITCH_PX:
            problems.append(
                f"only {pitch:.0f} px between corners — move CLOSER to the board; "
                f"small corners hurt solvePnP more than a centimetre of tape does"
            )
    return {
        "ok": not problems,
        "frames": len(distances_m),
        "true_distance_m": round(true_distance_m, 4),
        "measured_distance_m": round(measured, 4),
        "distance_error_frac": round(error_frac, 4),
        "spread_frac": round(spread, 4),
        "corner_pitch_px": round(statistics.median(pitches_px), 1) if pitches_px else None,
        "fx_assumed": round(fx_assumed, 1),
        "fx_implied": round(fx_true, 1),
        "hfov_implied_deg": round(2 * math.degrees(math.atan(768.0 / fx_true)), 1),
        "problems": problems,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pin the camera's focal length against a tape measure"
    )
    parser.add_argument("--distance-m", type=float, required=True,
                        help="Tape-measured distance from the CAMERA LENS to the "
                             "board's surface. On this rig the lens and the front "
                             "of the chassis differ by centimetres, which is most "
                             "of the effect being measured.")
    parser.add_argument("--intrinsics", type=Path,
                        default=REPO_ROOT / "config/scene_intrinsics.json")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=864)
    parser.add_argument("--square-mm", type=float, default=24.0)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--lens-position", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    import cv2
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    if not args.intrinsics.exists():
        logger.error("No intrinsics at %s — run ./scripts/run_scene_calibrate.sh first.",
                     args.intrinsics)
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
    distances: list[float] = []
    pitches: list[float] = []

    logger.info("Board flat on a wall, whole board in frame, robot still. "
                "Collecting %d detections...", args.frames)
    try:
        deadline = time.monotonic() + 60.0
        while len(distances) < args.frames and time.monotonic() < deadline:
            bgr = source.capture_bgr()
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            for candidate in candidates:
                found, corners = find_board(cv2, gray, candidate)
                if not found:
                    continue
                if pattern is None:
                    pattern, candidates = candidate, [candidate]
                    logger.info("Locked pattern %dx%d", *candidate)
                objp = np.zeros((candidate[0] * candidate[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:candidate[0], 0:candidate[1]].T.reshape(-1, 2)
                objp *= args.square_mm / 1000.0
                ok, _rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist)
                if ok:
                    distances.append(float(np.linalg.norm(np.asarray(tvec).reshape(3))))
                    pitches.append(corner_pitch_px(corners, candidate))
                break
    finally:
        try:
            source.stop()
        except Exception:
            pass

    report = summarize(distances, float(intrinsics["fx"]), args.distance_m, pitches,
                       square_m=args.square_mm / 1000.0)
    if not report.get("frames"):
        logger.error("%s", report["reason"])
        return 1

    logger.info("Board seen in %d frames", report["frames"])
    logger.info("  board size     %.0f px per square", report["corner_pitch_px"] or 0)
    logger.info("  tape says      %.3f m", report["true_distance_m"])
    logger.info("  solvePnP says  %.3f m  (%+.1f%%, spread %.1f%% across frames)",
                report["measured_distance_m"], 100 * report["distance_error_frac"],
                100 * report["spread_frac"])
    logger.info("  fx in use      %.1f", report["fx_assumed"])
    logger.info("  fx implied     %.1f  -> %.1f deg horizontal",
                report["fx_implied"], report["hfov_implied_deg"])
    for problem in report["problems"]:
        logger.warning("  %s", problem)
    if report["ok"] and abs(report["distance_error_frac"]) < 0.03:
        logger.info("  -> the calibration's fx agrees with the tape to within 3%%. Keep it.")
    elif report["ok"]:
        logger.warning(
            "  -> fx and the tape disagree by %.0f%%. Before touching the "
            "calibration, re-check the tape reading — a mistyped distance produces "
            "exactly this, and the board's size in pixels (printed above) is an "
            "independent witness to how far away it really was. If the tape is "
            "right, re-solve offline with the measured focal length:\n"
            "    python3 scripts/resolve_calibration.py --fix-fx %.1f",
            100 * abs(report["distance_error_frac"]), report["fx_implied"])
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
