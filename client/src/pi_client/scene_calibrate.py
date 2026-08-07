"""One-time camera calibration for scene recording (chessboard), with a
live annotated preview streamed to the Mac panel's Live tab so you can see
exactly what the camera sees while you hold the board.

Camera Module 3 has autofocus — intrinsics DRIFT with focus, so the lens is
locked at a fixed LensPosition here and the SAME value must be used for
every recording (scene_recorder.py reads it back from intrinsics.json).

Usage:

    ./scripts/run_scene_calibrate.sh --square-mm 24 --lens-position 1.0

Open http://<mac>:8080/ -> Live tab and watch the board: detected corners
are drawn in real time, a counter shows accepted/target boards, and a 3x3
grid + near/mid/far indicators show (informationally, not as a gate) which
parts of the frame and which distances you've sampled so far. A frame is
accepted whenever the board has actually moved (position, angle or
distance) since the last accepted one — including near the frame edges,
which is exactly where you want samples for good distortion correction.
Ctrl+C finishes early with whatever was captured (if enough); the script
also stops automatically once --frames boards are accepted. Writes
config/scene_intrinsics.json.
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

from pi_client.camera import CameraFocusOptions, CameraFrame
from pi_client.camera_session import CaptureSettings, make_capture_source
from pi_client.network import ClientConnectionError, PiClient
from pi_client.protocol import make_camera_stream_frame_message
from shared.config import load_config


logger = logging.getLogger(__name__)

# Inner-corner counts of common printable chessboard templates. Detection
# tries each until one locks on, so you don't have to know your template's
# exact layout — just count the SQUARES on your printout (both colors) and
# pass --cols/--rows explicitly only if none of these match.
CANDIDATE_PATTERNS = [
    (9, 6), (6, 9), (8, 6), (6, 8), (7, 6), (6, 7),
    (9, 7), (7, 9), (8, 5), (5, 8), (7, 5), (5, 7),
]

COVERAGE_GRID = 3  # 3x3 regions of the frame to encourage spreading boards out
GOOD_RMS_PX = 0.5    # nothing to improve below this
USABLE_RMS_PX = 1.5  # above this the intrinsics are not trustworthy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the Pi camera (chessboard)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "config/scene_intrinsics.json")
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=864)
    parser.add_argument("--cols", type=int, default=None, help="Inner corners per row (default: auto-detect)")
    parser.add_argument("--rows", type=int, default=None, help="Inner corners per column (default: auto-detect)")
    parser.add_argument("--square-mm", type=float, default=25.0, help="Measured square size, edge to edge")
    parser.add_argument("--frames", type=int, default=30, help="Stop automatically after this many accepted boards")
    parser.add_argument("--min-frames", type=int, default=12, help="Minimum to compute calibration at all")
    parser.add_argument("--lens-position", type=float, default=1.0,
                         help="Manual focus in 1/m; MUST be reused for recording. "
                              "1.0 = 1m focus, a reasonable middle ground for room-scale walkthroughs.")
    parser.add_argument("--host", default=None, help="Mac server host for the live preview")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--expected-hfov-deg", type=float, default=EXPECTED_HFOV_DEG,
                        help="Horizontal field this camera should have. Used only as a "
                             "sanity gate on the solved fx — measured on this rig at "
                             "~75 deg by VGGT, COLMAP and DA3 independently.")
    parser.add_argument("--force", action="store_true",
                        help="Write the calibration even if it fails the structural "
                             "sanity gates. Only when you know why the numbers look "
                             "the way they do.")
    return parser.parse_args()


class PreviewUplink:
    """Best-effort live preview to the Mac panel — same mechanism the scene
    recorder uses (a plain camera_stream_frame). Never blocks calibration:
    on any failure it just stops trying until the retry window passes."""

    def __init__(self, host: str, port: int, device_id: str, retry_s: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.device_id = device_id
        self.retry_s = retry_s
        self.healthy = False
        self._client: PiClient | None = None
        self._last_attempt = 0.0

    def _ensure(self) -> bool:
        if self.healthy and self._client is not None:
            return True
        if time.monotonic() - self._last_attempt < self.retry_s:
            return False
        self._last_attempt = time.monotonic()
        try:
            client = PiClient(self.host, self.port, timeout_seconds=4.0)
            client.connect()
            self._client = client
            self.healthy = True
            logger.info("Live preview connected to %s:%s (Live tab)", self.host, self.port)
            return True
        except (ClientConnectionError, OSError) as exc:
            logger.debug("Preview uplink unavailable: %s", exc)
            self._client = None
            self.healthy = False
            return False

    def send(self, jpeg: bytes, width: int, height: int) -> None:
        if not self._ensure():
            return
        assert self._client is not None
        try:
            frame = CameraFrame(
                frame_id=str(time.monotonic_ns()), width=width, height=height,
                image_format="jpeg", content_type="image/jpeg", data=jpeg,
            )
            message = make_camera_stream_frame_message(
                device_id=self.device_id, frame=frame, session_id="scene_calib",
                frame_index=0, fps=5.0, jpeg_quality=75, save_frame=False,
                view="camera", mode="scene_calib",
            )
            response, _ = self._client.request(message, jpeg)
            if response.type == "error":
                raise ClientConnectionError(str(response.payload.get("error")))
        except (ClientConnectionError, OSError) as exc:
            logger.debug("Preview send failed: %s", exc)
            self._client = None
            self.healthy = False

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None


# Structural gates on the solved camera. These are NOT about precision — they
# catch a calibration that is *wrong* rather than merely imprecise, which is a
# different failure and needs a different response.
#
# Why they exist: `config/scene_intrinsics.json` shipped fx=534.5 at 1536x864
# (a 110 deg horizontal field) with cy=500 in an 864-tall frame — the principal
# point 68 px off centre. Three independent sources (VGGT-refined 75.6 deg,
# COLMAP 75.7, DA3 74.4) put the real field at ~75 deg, fx ~991. That file was
# almost certainly solved against frames at a different capture resolution.
# Nothing caught it, and every solvePnP, every reconstruction and every metric
# distance downstream silently inherited the error for weeks.
#
# A well-conditioned calibration on this camera puts the principal point within
# a few percent of centre and fx within a couple of percent of fy. When it does
# not, the usual causes are a resolution/crop mismatch or boards presented at a
# single distance (focal length and distance are degenerate in that case).
# Focal length is constrained mainly by how much the board's ORIENTATION
# varies, not by how far away it is. Viewed head-on, a change in focal length
# and a change in distance produce almost the same image, so the solve can
# slide along that pair and still report an excellent reprojection error.
#
# Measured on this rig 2026-08-03: a board taped to a wall, photographed by a
# robot that can only drive on the floor and yaw, is fronto-parallel in every
# single view. The solve returned fx=1098 (69.9 deg) with RMS 0.484 px, full
# 3x3 frame coverage and three distance buckets — and a tape measure put the
# true fx at 968 (76.9 deg), agreeing with VGGT's 991 and COLMAP's 1001.
# Nothing in the usual quality metrics caught a 13% error.
MIN_BOARD_TILT_SPREAD_DEG = 25.0
# Intrinsics-free live version of the same requirement, from `board_skew`.
# +/-0.13 of trapezoid asymmetry corresponds to roughly 30 deg of obliquity
# on this board at working distance — comfortably inside what the robot can
# reach by parking off to one side of a wall-mounted board.
MIN_SKEW_SPAN = 0.26

MAX_PRINCIPAL_OFFSET_FRAC = 0.05
MAX_FOCAL_ASYMMETRY = 0.02
EXPECTED_HFOV_DEG = 75.0
MAX_HFOV_DEVIATION_DEG = 12.0


def board_skew(corners: Any, pattern: tuple[int, int]) -> tuple[float, float]:
    """How obliquely the board is being viewed, WITHOUT needing intrinsics.

    A chessboard seen head-on projects to a rectangle; seen from the side it
    projects to a trapezoid, with the near edge longer than the far one. The
    relative difference between opposite edges is therefore a direct,
    calibration-free measure of obliquity — which matters because obliquity
    is exactly what the focal length needs and we cannot use `solvePnP` to
    measure it (that would need the intrinsics we are trying to find).

    Returns `(horizontal, vertical)`, each in roughly [-1, 1]; 0 means
    perfectly head-on. Horizontal skew comes from viewing the board from the
    left or right, which on this rig the ROBOT produces by parking off to one
    side and turning to face the wall — no need to touch the board.
    """
    cols, rows = pattern
    grid = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)

    def edge_len(points: Any) -> float:
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    left, right = edge_len(grid[:, 0]), edge_len(grid[:, -1])
    top, bottom = edge_len(grid[0, :]), edge_len(grid[-1, :])
    horizontal = (right - left) / (right + left) if (right + left) else 0.0
    vertical = (bottom - top) / (bottom + top) if (bottom + top) else 0.0
    return horizontal, vertical


def board_tilt_spread_deg(rvecs: Any) -> float:
    """Largest angle between any two board orientations in the capture set.

    Each `rvec` rotates the board into the camera frame, so its third column
    is the board's normal as seen by the camera. The spread of those normals
    is exactly the "did the board actually turn?" question that decides
    whether focal length is observable.
    """
    import cv2

    normals = []
    for rvec in rvecs or []:
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        normals.append(rotation[:, 2] / (np.linalg.norm(rotation[:, 2]) or 1.0))
    worst = 0.0
    for i, a in enumerate(normals):
        for b in normals[i + 1:]:
            cos = float(np.clip(a @ b, -1.0, 1.0))
            worst = max(worst, math.degrees(math.acos(cos)))
    return worst


def check_calibration_sanity(
    fx: float, fy: float, cx: float, cy: float, width: int, height: int,
    expected_hfov_deg: float = EXPECTED_HFOV_DEG,
    tilt_spread_deg: float | None = None,
) -> list[str]:
    """Structural problems with a solved camera; empty list means it is sane.

    Deliberately separate from the RMS check. A high RMS means "the boards
    moved" and the result is still usable; a failure here means the numbers
    describe a different camera than the one that took the pictures, and using
    them is worse than having no calibration at all.
    """
    problems: list[str] = []
    if fx <= 0 or fy <= 0:
        return ["focal length is non-positive — the solve did not converge"]

    hfov = 2.0 * math.degrees(math.atan(width / 2.0 / fx))
    if abs(hfov - expected_hfov_deg) > MAX_HFOV_DEVIATION_DEG:
        want = width / 2.0 / math.tan(math.radians(expected_hfov_deg / 2.0))
        problems.append(
            f"implied horizontal field {hfov:.0f} deg, expected ~{expected_hfov_deg:.0f} "
            f"(fx {fx:.0f}, expected ~{want:.0f}) — usually a capture-resolution mismatch"
        )
    for name, value, centre, extent in (
        ("cx", cx, width / 2.0, width), ("cy", cy, height / 2.0, height),
    ):
        if abs(value - centre) > MAX_PRINCIPAL_OFFSET_FRAC * extent:
            problems.append(
                f"{name}={value:.0f} is {abs(value - centre):.0f} px off centre "
                f"({centre:.0f}) — more than {MAX_PRINCIPAL_OFFSET_FRAC:.0%} of the frame"
            )
    if abs(fx - fy) / fx > MAX_FOCAL_ASYMMETRY:
        problems.append(
            f"fx={fx:.0f} and fy={fy:.0f} differ by {abs(fx - fy) / fx:.1%} — "
            f"square pixels should agree within {MAX_FOCAL_ASYMMETRY:.0%}"
        )
    if tilt_spread_deg is not None and tilt_spread_deg < MIN_BOARD_TILT_SPREAD_DEG:
        problems.append(
            f"the board was viewed almost head-on throughout ({tilt_spread_deg:.0f} deg "
            f"of orientation spread, want >= {MIN_BOARD_TILT_SPREAD_DEG:.0f}). That leaves "
            f"the focal length nearly unconstrained — the solve trades it off against "
            f"distance and reports a low RMS while being badly wrong. FIX: leave the "
            f"board on the wall and DRIVE THE ROBOT to strongly different angles — park "
            f"well off to the left, turn to face the board, capture; repeat from the "
            f"right and from several distances. Obliquity is what constrains fx, and "
            f"the robot's own yaw produces plenty of it"
        )
    return problems


def find_board(cv2: Any, gray: np.ndarray, pattern: tuple[int, int]):
    """Best available detector: findChessboardCornersSB (OpenCV >= 4.x) is
    faster and more robust to glare/perspective and already sub-pixel
    accurate; fall back to the classic detector + cornerSubPix."""
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern)
        return found, corners
    found, corners = cv2.findChessboardCorners(
        gray, pattern,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def coverage_cell(cx: float, cy: float, width: int, height: int) -> tuple[int, int]:
    col = min(COVERAGE_GRID - 1, int(cx / width * COVERAGE_GRID))
    row = min(COVERAGE_GRID - 1, int(cy / height * COVERAGE_GRID))
    return row, col


def draw_status(
    cv2, frame: np.ndarray, accepted: int, target: int, pattern: tuple[int, int],
    covered: set[tuple[int, int]], size_buckets: set[str], status_text: str,
    status_color: tuple[int, int, int],
    skew_now: float | None = None, skew_lo: float = 0.0, skew_hi: float = 0.0,
) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"CALIBRATE  boards {accepted}/{target}  pattern {pattern[0]}x{pattern[1]}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, status_text, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

    # 3x3 coverage grid, bottom-right corner: green = has a sample, red = needs one.
    cell = 26
    pad = 14
    gx0, gy0 = w - COVERAGE_GRID * cell - pad, h - COVERAGE_GRID * cell - pad
    for row in range(COVERAGE_GRID):
        for col in range(COVERAGE_GRID):
            x0, y0 = gx0 + col * cell, gy0 + row * cell
            done = (row, col) in covered
            color = (60, 200, 60) if done else (50, 50, 210)
            cv2.rectangle(frame, (x0, y0), (x0 + cell - 2, y0 + cell - 2), color, -1)
            cv2.rectangle(frame, (x0, y0), (x0 + cell - 2, y0 + cell - 2), (20, 20, 20), 1)
    cv2.putText(frame, "coverage", (gx0, gy0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)

    # Obliquity bar. This is the one thing that actually constrains the focal
    # length, and it is the one thing the previous capture had none of, so it
    # gets the widest, most legible widget on screen rather than a footnote.
    span = skew_hi - skew_lo
    bar_w, bar_h, bx, by = w - 2 * pad, 22, pad, h - 78
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (35, 35, 35), -1)

    def to_x(value: float) -> int:
        return int(bx + bar_w * (max(-0.5, min(0.5, value)) + 0.5))

    cv2.line(frame, (to_x(0.0), by), (to_x(0.0), by + bar_h), (90, 90, 90), 1)
    if span > 0:
        done = span >= MIN_SKEW_SPAN
        cv2.rectangle(frame, (to_x(skew_lo), by + 4), (to_x(skew_hi), by + bar_h - 4),
                      (60, 200, 60) if done else (0, 165, 235), -1)
    if skew_now is not None:
        cv2.line(frame, (to_x(skew_now), by - 4), (to_x(skew_now), by + bar_h + 4),
                 (255, 255, 255), 2)
    cv2.putText(frame,
                "ANGLE  span %.2f / %.2f  %s" % (
                    span, MIN_SKEW_SPAN,
                    "OK" if span >= MIN_SKEW_SPAN else "drive further to the SIDE and turn to face the board"),
                (bx, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (60, 200, 60) if span >= MIN_SKEW_SPAN else (0, 165, 235), 1, cv2.LINE_AA)

    dist_labels = {"near": "NEAR", "mid": "MID", "far": "FAR"}
    x = pad
    for key, label in dist_labels.items():
        color = (60, 200, 60) if key in size_buckets else (90, 90, 90)
        cv2.putText(frame, label, (x, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        x += 70


def main() -> int:
    import cv2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("pi_client.network").setLevel(logging.WARNING)
    args = parse_args()

    config = load_config(REPO_ROOT / "config/default.json", REPO_ROOT / ".env")
    host = args.host or config.get("client", {}).get("server_host", "127.0.0.1")
    port = args.port or int(config.get("server", {}).get("port", 8765))
    device_id = str(config.get("client", {}).get("device_id", "raspberry_pi_01"))
    preview = None if args.no_preview else PreviewUplink(host, port, device_id)

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

    stop = {"flag": False}

    def handle_signal(signum: int, _frame: Any) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    pattern: tuple[int, int] | None = (args.cols, args.rows) if args.cols and args.rows else None
    candidates = [pattern] if pattern else list(CANDIDATE_PATTERNS)

    obj_points, img_points = [], []
    skews: list[float] = []
    skew_now: float | None = None
    covered: set[tuple[int, int]] = set()  # informational only — never blocks acceptance
    size_buckets: set[str] = set()
    last_accept_time = 0.0
    last_accept_pos: tuple[float, float] | None = None
    last_accept_frac: float | None = None

    # Only requirement to accept a frame: the board actually MOVED (position
    # or apparent size changed meaningfully) since the last accepted one, or
    # enough time passed that we accept it anyway so the run can never get
    # stuck. No screen-zone quotas, no edge rejection — corner points near
    # the image border are exactly what constrains lens distortion best, so
    # they are welcome, not penalized.
    MIN_GAP_S = 0.6
    MIN_MOVE_FRAC = 0.05
    MIN_SIZE_CHANGE = 0.08
    FALLBACK_ACCEPT_S = 2.5

    logger.info(
        "Live preview: http://%s:8080/ -> Live tab. Need %d boards (min %d), square=%.1fmm.",
        host, args.frames, args.min_frames, args.square_mm,
    )
    if pattern:
        logger.info("Pattern fixed at %dx%d inner corners", *pattern)
    else:
        logger.info("Auto-detecting pattern from: %s", candidates)

    try:
        while not stop["flag"] and len(obj_points) < args.frames:
            bgr = source.capture_bgr()
            h, w = bgr.shape[:2]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            found, corners = False, None
            if pattern is not None:
                found, corners = find_board(cv2, gray, pattern)
            else:
                for cand in candidates:
                    found, corners = find_board(cv2, gray, cand)
                    if found:
                        pattern = cand
                        logger.info("Locked pattern: %dx%d inner corners", *pattern)
                        break

            status_text = "no board detected"
            status_color = (0, 0, 220)
            just_accepted = False

            if found and corners is not None:
                cv2.drawChessboardCorners(bgr, pattern, corners, found)
                pts = corners.reshape(-1, 2)
                cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                span = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())
                frac = span / w
                bucket = "near" if frac > 0.55 else "far" if frac < 0.28 else "mid"
                diag = math.hypot(w, h)

                now = time.monotonic()
                dt = now - last_accept_time
                moved = last_accept_pos is None or (
                    math.hypot(cx - last_accept_pos[0], cy - last_accept_pos[1]) / diag >= MIN_MOVE_FRAC
                    or (last_accept_frac is not None and abs(frac - last_accept_frac) >= MIN_SIZE_CHANGE)
                )

                if dt < MIN_GAP_S:
                    status_text = "board detected - hold steady..."
                    status_color = (0, 165, 255)
                elif moved or dt >= FALLBACK_ACCEPT_S:
                    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
                    objp *= args.square_mm / 1000.0
                    obj_points.append(objp)
                    img_points.append(corners)
                    last_accept_time = now
                    last_accept_pos = (cx, cy)
                    last_accept_frac = frac
                    just_accepted = True
                    status_text = f"ACCEPTED {len(obj_points)}/{args.frames}"
                    status_color = (60, 220, 60)

                    # Informational coverage map: every cell touched by ANY
                    # corner point of this sample counts, so a large board
                    # tilted toward an edge correctly credits that edge even
                    # though its center stayed near the middle of the frame.
                    for px, py in pts:
                        covered.add(coverage_cell(float(px), float(py), w, h))
                    size_buckets.add(bucket)
                    skews.append(board_skew(corners, pattern)[0])

                    logger.info(
                        "Accepted %d/%d  regions=%d/9  distances=%s",
                        len(obj_points), args.frames, len(covered), sorted(size_buckets),
                    )
                else:
                    status_text = "move the board (position / angle / distance) for the next shot"
                    status_color = (0, 165, 255)

            if found and pattern is not None:
                skew_now = board_skew(corners, pattern)[0]
            draw_status(cv2, bgr, len(obj_points), args.frames, pattern or (0, 0),
                        covered, size_buckets, status_text, status_color,
                        skew_now=skew_now,
                        skew_lo=min(skews) if skews else 0.0,
                        skew_hi=max(skews) if skews else 0.0)

            if preview is not None:
                send_frame = bgr
                if just_accepted:
                    send_frame = bgr.copy()
                    cv2.rectangle(send_frame, (0, 0), (w - 1, h - 1), (60, 220, 60), 10)
                ok, encoded = cv2.imencode(".jpg", send_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok:
                    preview.send(encoded.tobytes(), w, h)
    finally:
        source.stop()

    if len(obj_points) < args.min_frames:
        logger.error("Only %d boards captured (need >= %d) — aborting, run again", len(obj_points), args.min_frames)
        if preview is not None:
            preview.close()
        return 1

    assert pattern is not None
    logger.info("Computing calibration from %d boards...", len(obj_points))
    rms, K, dist, rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, (args.width, args.height), None, None
    )

    # Two thresholds, not one. Below GOOD_RMS_PX the calibration is as good as
    # this setup gets; between GOOD and USABLE it is still perfectly usable for
    # reconstruction (a ~1 px reprojection error on a 1536 px sensor is well
    # under the error monocular depth contributes) — it just means the boards
    # moved a little while they were captured. Only above USABLE_RMS_PX is
    # there a real reason to redo it. Reporting one strict target made a
    # successful run read like a failure.
    passed = rms < GOOD_RMS_PX
    usable = rms < USABLE_RMS_PX
    if passed:
        logger.info("RMS reprojection error: %.3f px — good", rms)
    elif usable:
        logger.info(
            "RMS reprojection error: %.3f px — USABLE (ideal is < %.1f). "
            "Calibration saved and safe to use. To tighten it, hold the board "
            "still for ~1.5 s per capture; this error level is normal when "
            "boards are grabbed about once a second by hand.",
            rms, GOOD_RMS_PX,
        )
    else:
        logger.warning(
            "RMS reprojection error: %.3f px — too high to trust (want < %.1f). "
            "Usual causes: motion blur, a board that isn't flat, or a wrong "
            "--square-mm/pattern. Saving anyway so nothing is lost, but redo it.",
            rms, USABLE_RMS_PX,
        )
    logger.info(
        "Pattern %dx%d INNER CORNERS = a %dx%d SQUARE board; square %.1f mm "
        "(square size sets world scale only — it does not affect fx/fy/cx/cy).",
        pattern[0], pattern[1], pattern[0] + 1, pattern[1] + 1, args.square_mm,
    )
    if not covered.issuperset({(r, c) for r in range(COVERAGE_GRID) for c in range(COVERAGE_GRID)}):
        missing = COVERAGE_GRID * COVERAGE_GRID - len(covered)
        logger.warning("%d/%d frame regions never had a board — distortion at those edges is less certain",
                        missing, COVERAGE_GRID * COVERAGE_GRID)
    if len(size_buckets) < 2:
        logger.warning("Only one distance range covered (%s) — vary near/far more next time", sorted(size_buckets))

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
        "pattern": list(pattern),
        "square_mm": args.square_mm,
        "coverage_cells": len(covered),
        "distance_buckets": sorted(size_buckets),
        "calibrated_at": time.time(),
    }
    # Save the raw detections BEFORE judging the solve. This is the promise
    # that the capture never has to be repeated: every question about focal
    # length, principal point or distortion can be re-answered offline from
    # these corners with different solver constraints
    # (scripts/resolve_calibration.py), with no camera, no board and no
    # operator. Not doing this is why the last three rounds each needed a
    # fresh capture session.
    captures_path = args.output.with_name(args.output.stem + "_captures.npz")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        captures_path,
        object_points=np.asarray(obj_points, dtype=np.float32),
        image_points=np.asarray(img_points, dtype=np.float32),
        pattern=np.asarray(pattern, dtype=np.int32),
        image_size=np.asarray([args.width, args.height], dtype=np.int32),
        square_mm=np.float32(args.square_mm),
        lens_position=np.float32(args.lens_position),
        skews=np.asarray(skews, dtype=np.float32),
    )
    logger.info("Saved %d board detections to %s — any future re-solve reads "
                "this file; the camera is never needed again.",
                len(obj_points), captures_path)

    tilt_spread = board_tilt_spread_deg(rvecs)
    intrinsics["board_tilt_spread_deg"] = round(tilt_spread, 1)
    problems = check_calibration_sanity(
        intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"],
        args.width, args.height, expected_hfov_deg=args.expected_hfov_deg,
        tilt_spread_deg=tilt_spread,
    )
    intrinsics["sanity_problems"] = problems

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if problems and not args.force:
        # Refuse the canonical path but never lose the data: a rejected solve
        # is still the evidence you need to work out what went wrong, and
        # deleting it would just force another capture session.
        rejected = args.output.with_suffix(".rejected.json")
        rejected.write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
        for problem in problems:
            logger.error("REJECTED: %s", problem)
        logger.error(
            "Not writing %s — these are structural problems, not precision ones, and "
            "everything downstream (solvePnP, reconstruction, every metric distance) "
            "would silently inherit them. Saved to %s instead.",
            args.output, rejected,
        )
        logger.error(
            "NO RE-SHOOT NEEDED. The %d board detections are saved in %s. Measure the "
            "focal length against a tape once:\n"
            "    ./scripts/run_focal_check.sh --distance-m 0.50\n"
            "then pin it and re-fit everything else from those saved corners:\n"
            "    python3 scripts/resolve_calibration.py --fix-fx <fx_implied>\n"
            "That writes %s and needs no camera, no board and no operator.",
            len(obj_points), captures_path, args.output,
        )
        if preview is not None:
            preview.close()
        return 3

    args.output.write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
    if problems:
        logger.warning("Saved %s despite %d sanity problem(s) — --force was given.",
                       args.output, len(problems))
    else:
        logger.info("Saved %s", args.output)
    print(json.dumps(intrinsics, indent=2))

    if preview is not None:
        # The camera is already stopped; render a synthetic result card
        # instead of a live frame so the outcome is unmistakable.
        final = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        color = (60, 220, 60) if passed else ((0, 190, 220) if usable else (0, 60, 220))
        cv2.rectangle(final, (0, 0), (args.width - 1, args.height - 1), color, 16)
        title = ("CALIBRATION OK" if passed
                 else ("CALIBRATION USABLE — saved" if usable
                       else "CALIBRATION WEAK — redo it"))
        cv2.putText(final, title, (60, args.height // 2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
        cv2.putText(final, f"RMS {rms:.3f} px  ({len(obj_points)} boards, pattern {pattern[0]}x{pattern[1]})",
                    (60, args.height // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(final, f"lens_position={args.lens_position} — reuse this for run_scene_recorder.sh",
                    (60, args.height // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", final, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            for _ in range(3):
                preview.send(encoded.tobytes(), args.width, args.height)
                time.sleep(0.3)
        preview.close()

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
