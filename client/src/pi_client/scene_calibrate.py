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
    covered: set[tuple[int, int]], size_buckets: set[str], status_text: str, status_color: tuple[int, int, int],
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

                    logger.info(
                        "Accepted %d/%d  regions=%d/9  distances=%s",
                        len(obj_points), args.frames, len(covered), sorted(size_buckets),
                    )
                else:
                    status_text = "move the board (position / angle / distance) for the next shot"
                    status_color = (0, 165, 255)

            draw_status(cv2, bgr, len(obj_points), args.frames, pattern or (0, 0),
                        covered, size_buckets, status_text, status_color)

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
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, (args.width, args.height), None, None
    )

    passed = rms < 0.5
    logger.info("RMS reprojection error: %.3f px (target < 0.5)", rms)
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
    logger.info("Saved %s", args.output)
    print(json.dumps(intrinsics, indent=2))

    if preview is not None:
        # The camera is already stopped; render a synthetic result card
        # instead of a live frame so the outcome is unmistakable.
        final = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        color = (60, 220, 60) if passed else (0, 60, 220)
        cv2.rectangle(final, (0, 0), (args.width - 1, args.height - 1), color, 16)
        title = "CALIBRATION OK" if passed else "CALIBRATION WEAK — consider redoing"
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
