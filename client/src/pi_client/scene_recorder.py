"""Scene recording client for the Pi: walk the camera around a room; every
keyframe is STREAMED to the Mac server over the project TCP protocol as it
is captured (scene_session_start / scene_keyframe / scene_session_end), so
nothing accumulates on the Pi's microSD. The Mac materializes the exact
session layout the pipeline consumes (docs/scene3d.md).

The local disk is only a FALLBACK: if the server is unreachable a keyframe
is written under data/scene_sessions/ instead (and the uplink keeps
retrying), so a Wi-Fi hiccup never loses data — merge leftovers later with
rsync if that happens.

A live preview (annotated with detections) is also pushed for the panel's
Live tab, so you can see framing/blur/detections while walking.

Per frame (~2-3 fps target, CPU stays light — the NPU does the work):
  camera -> yolov8-seg hef (Hailo-8, host decode) -> greedy tracker
         -> keyframe selector (min interval + blur gate)
         -> keyframe: rgb.jpg + masks.png (uint16) + meta.json with
            track_id and a CLIP embedding per detection (CLIP hef on the
            same NPU via the model scheduler) -> uplink to the Mac

Camera controls are FIXED for the whole session (manual focus at the
calibrated LensPosition) — drifting intrinsics would hurt COLMAP.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "client/src"), str(REPO_ROOT / "server/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import numpy as np

from pi_client.camera import CameraFocusOptions, CameraFrame
from pi_client.camera_session import CaptureSettings, make_capture_source
from pi_client.cv_models import COCO80_CLASS_NAMES
from pi_client.hailo_infer import HailoMultiModel
from pi_client.network import ClientConnectionError, PiClient
from pi_client.protocol import (
    make_camera_stream_frame_message,
    make_scene_keyframe_message,
    make_scene_session_end_message,
    make_scene_session_start_message,
)
from pi_client.imu_rvc_math import motion_state
from pi_client.seg_postprocess import SEG_ARCHS, letterbox, yolov8_seg_postprocess
from shared.config import load_config

# Pure-python tracker shared with the Mac monitoring stack (no server
# runtime involved — just tested association code).
from mac_server.monitoring.tracker import GreedyTracker, TrackerConfig


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a scene session (Pi + Hailo)")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/scene_sessions",
                        help="Local FALLBACK dir (used only when the uplink is down)")
    parser.add_argument("--intrinsics", type=Path, default=REPO_ROOT / "config/scene_intrinsics.json")
    parser.add_argument("--seg-hef", type=Path,
                        default=REPO_ROOT / "models/indoor_yolo11l_seg.hef",
                        help="Default is the indoor fine-tune (31 room classes). "
                             "Measured against the old COCO hef on the same 53 "
                             "keyframes: 112 detections vs 16, 91 ms vs 72. Pass "
                             "models/yolov8m_seg_h8.hef with --seg-arch yolov8_seg "
                             "to reproduce an old session.")
    parser.add_argument("--seg-arch", choices=sorted(SEG_ARCHS), default="indoor_ade20k",
                        help="Output layout of --seg-hef. Verify a new hef with "
                             "scripts/probe_hef.py before trusting this flag: it "
                             "derives the arch from the blobs, and a mismatch here "
                             "is caught loudly by order_endnodes rather than "
                             "producing wrong boxes quietly.")
    parser.add_argument("--seg-classes", type=Path,
                        default=REPO_ROOT / "config/seg_classes_indoor.json",
                        help="Class names for a fine-tuned --seg-arch, in the "
                             "order baked into the weights. Ignored for the "
                             "COCO and class-agnostic archs.")
    parser.add_argument("--clip-hef", type=Path, default=REPO_ROOT / "models/clip_resnet_50x4_h8.hef")
    parser.add_argument("--no-clip", action="store_true", help="Skip CLIP embeddings")
    parser.add_argument("--source", choices=["camera", "synthetic", "replay"], default="camera")
    parser.add_argument("--synthetic-image", type=Path, default=REPO_ROOT / "data/cat.jpg")
    parser.add_argument("--replay-session", type=Path, default=None,
                        help="With --source replay: a recorded session dir whose "
                             "keyframes are re-fed through the pipeline. Fixed "
                             "pixels make perception changes comparable without "
                             "a robot; note the frames are the saved JPEGs, so "
                             "near-threshold detections will not match the "
                             "original run exactly.")
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=864)
    parser.add_argument("--fps", type=float, default=3.0, help="Capture/inference loop rate")
    # Keyframes are chosen by parallax, not by the clock — see KeyframeSelector.
    parser.add_argument("--keyframe-min-interval", type=float, default=0.2,
                        help="Never emit keyframes faster than this")
    parser.add_argument("--keyframe-max-interval", type=float, default=3.0,
                        help="Emit anyway after this long, even without motion")
    parser.add_argument("--keyframe-shift-px", type=float, default=12.0,
                        help="Median feature displacement (in a 480x270 image) "
                             "that counts as a genuinely new viewpoint")
    parser.add_argument("--blur-threshold", type=float, default=100.0, help="Variance-of-Laplacian gate")
    parser.add_argument("--max-keyframe-omega-dps", type=float, default=15.0,
                        help="Не выдавать ключевой кадр, если камера поворачивается "
                             "быстрее (руководство §7.5: при выдержке 1/30 с выше "
                             "этого начинается смаз). 0 — выключить.")
    parser.add_argument("--confidence", type=float, default=0.4,
                        help="Proposal score gate. One value used to drive both "
                             "this and the tracker; see --track-min-confidence.")
    parser.add_argument("--track-min-confidence", type=float, default=None,
                        help="Separate gate for the tracker (default: --confidence). "
                             "They are different jobs: a proposal worth embedding "
                             "and reconstructing can be far weaker than one worth "
                             "asserting an identity about across frames — and with "
                             "a class-agnostic arch, every proposal scores ~1.0, so "
                             "a shared value silently stops gating anything.")
    parser.add_argument("--max-detections", type=int, default=30)
    # Proposal gates. Inert for COCO (a detector that fires at all fires on
    # something bounded), and load-bearing for class-agnostic proposals, which
    # happily return the wall, the floor, and a 4-pixel speck.
    parser.add_argument("--min-mask-area-frac", type=float, default=0.0,
                        help="Reject masks below this fraction of the frame")
    parser.add_argument("--max-mask-area-frac", type=float, default=1.0,
                        help="Reject masks above this fraction of the frame — a "
                             "proposal covering most of the view is a wall or "
                             "floor, not an object")
    parser.add_argument("--reject-edge-count", type=int, default=0,
                        help="Reject a box touching at least this many frame "
                             "edges; 0 = off. A box on 3+ edges is usually "
                             "background the segmenter wrapped a rectangle "
                             "around. Note 2 is legitimate — a cabinet in a "
                             "corner touches two.")
    parser.add_argument("--lens-position", type=float, default=None,
                        help="Manual focus (1/m); default: from intrinsics.json")
    parser.add_argument("--max-keyframes", type=int, default=2000)
    parser.add_argument("--host", default=None, help="Mac server host (default: from .env/config)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-imu", action="store_true",
                        help="Ignore the IMU even if it is connected and calibrated")
    parser.add_argument("--imu-driver", choices=["rvc", "shtp"], default="rvc",
                        help="rvc: imu_rvc.RvcReader (115200 UART-RVC, on-chip fused "
                             "attitude, no raw gyro) - the current hardware jumper state "
                             "and the measured-healthy link (100.2 Hz, 0 resyncs, 0 "
                             "dropped frames). shtp: imu_shtp_motion.ShtpMotionReader "
                             "(raw gyro, 3 Mbaud UART-SHTP) - kept for the case the "
                             "jumpers are flipped back, but that link measured ~25% "
                             "well-formed frames on this adapter. The two modes are "
                             "mutually exclusive on the physical sensor - check the "
                             "jumpers before switching this flag.")
    parser.add_argument("--imu-port", default="/dev/ttyUSB0")
    parser.add_argument("--zupt-shift-px", type=float, default=2.0,
                        help="Image shift below which the rig counts as stopped, "
                             "so the IMU's velocity is pinned back to zero")
    parser.add_argument("--imu-calibration", type=Path, default=None,
                        help="default: config/imu_calibration.json (rvc) or "
                             "config/imu_calibration_shtp.json (shtp) - the two "
                             "drivers read the accelerometer with different sign "
                             "conventions (confirmed this session), so a "
                             "calibration measured with one driver must not be "
                             "reused with the other.")
    parser.add_argument("--offline", action="store_true",
                        help="No uplink: write everything locally (old behavior)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Uplink keyframes but skip the live preview stream")
    return parser.parse_args()


def load_intrinsics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable intrinsics %s: %s", path, exc)
        return None


def load_class_names(path: Path, expected: int) -> list[str]:
    """Class names for a fine-tuned arch, in the order baked into the weights.

    Refuses on a count mismatch rather than padding or truncating. A names list
    one entry short of the model's class count does not fail — it silently
    relabels everything past that point, and the output stays perfectly
    plausible ("door" where the model said "window"), so nothing downstream can
    catch it.
    """
    config = json.loads(path.read_text(encoding="utf-8"))
    names = [str(entry["name"]) for entry in config["classes"]]
    if len(names) != expected:
        raise ValueError(
            f"{path} lists {len(names)} classes but the architecture has "
            f"{expected}; the weights and this file disagree, so every label "
            f"past the first difference would be wrong")
    return names


def gate_proposals(
    detections: list[dict[str, Any]],
    masks: np.ndarray | None,
    frame_shape: tuple[int, int],
    *,
    min_area_frac: float = 0.0,
    max_area_frac: float = 1.0,
    reject_edge_count: int = 0,   # 0 = off; 1..4 = reject at that many edges
    edge_tol_px: float = 2.0,
) -> tuple[list[int], dict[str, int]]:
    """Which proposals are worth spending the rest of the pipeline on.

    Returns the indices to keep and a count per rejection reason. The counts
    are the point: a gate that quietly discards half the proposals is
    indistinguishable from a detector that never found them, and the two call
    for opposite fixes. They are summed into `session_meta.json`.

    Area is measured on the mask, not the box — a box is a rectangle around
    something that may be a thin diagonal sliver, and it is the mask that gets
    reconstructed. Edge contact is measured on the box, because a mask can
    legitimately touch an edge (a table running out of frame) while a box
    pinned to three or four edges means the segmenter drew a rectangle around
    the background.

    Defaults are inert (0.0 / 1.0 / 4), so turning this on is an explicit act.
    """
    reasons: dict[str, int] = {}
    keep: list[int] = []
    frame_h, frame_w = float(frame_shape[0]), float(frame_shape[1])

    def reject(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    for idx, det in enumerate(detections):
        if masks is not None and idx < len(masks):
            mask = masks[idx]
            area_frac = float((mask > 0.5).sum()) / float(mask.shape[0] * mask.shape[1])
            if area_frac < min_area_frac:
                reject("mask_too_small")
                continue
            if area_frac > max_area_frac:
                reject("mask_too_large")
                continue

        bbox = det.get("bbox_xyxy")
        if bbox and len(bbox) == 4 and reject_edge_count > 0:
            x0, y0, x1, y1 = (float(v) for v in bbox)
            edges = (
                x0 <= edge_tol_px,
                y0 <= edge_tol_px,
                x1 >= frame_w - edge_tol_px,
                y1 >= frame_h - edge_tol_px,
            )
            if sum(edges) >= reject_edge_count:
                reject("touches_frame_edges")
                continue
        keep.append(idx)
    return keep, reasons


def variance_of_laplacian(gray: np.ndarray) -> float:
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class KeyframeSelector:
    """Decides which captured frames become keyframes.

    The old rule was "every 0.5 s if not blurred", which gets both ends
    wrong on a robot that stops and starts: standing still emits a stream of
    near-identical frames (zero baseline — nothing to triangulate, and SfM
    pays O(N^2) for them anyway), while a fast pan leaves a gap wide enough
    that feature matching cannot bridge it.

    What actually matters for reconstruction is *parallax*: how far the
    scene has shifted since the last keyframe. So the rule is:

      * emit when the image has moved by `min_shift_px` since the last
        keyframe (real new viewpoint), or when `max_interval_s` has passed
        anyway (so a stationary robot still records something);
      * never emit blurrier than `blur_threshold`;
      * never emit faster than `min_interval_s`.

    Keeping the decision separate from the flow computation makes it
    unit-testable without a camera.
    """

    def __init__(
        self,
        min_interval_s: float = 0.2,
        max_interval_s: float = 3.0,
        min_shift_px: float = 12.0,
        blur_threshold: float = 100.0,
    ) -> None:
        self.min_interval_s = min_interval_s
        self.max_interval_s = max_interval_s
        self.min_shift_px = min_shift_px
        self.blur_threshold = blur_threshold

    def decide(
        self,
        elapsed_s: float,
        shift_px: float | None,
        sharpness: float,
        is_first: bool = False,
    ) -> tuple[bool, str]:
        """Returns (emit, reason) — the reason goes into the log so a
        disappointing session can be explained after the fact."""
        if is_first:
            return (sharpness >= self.blur_threshold, "first")
        if elapsed_s < self.min_interval_s:
            return False, "too_soon"
        if sharpness < self.blur_threshold:
            return False, "blurred"
        if elapsed_s >= self.max_interval_s:
            return True, "interval"
        if shift_px is None:
            return True, "no_flow"
        if shift_px >= self.min_shift_px:
            return True, "parallax"
        return False, "no_parallax"


class SceneUplink:
    """Best-effort TCP uplink to the Mac server. Reconnects with a retry
    window; on every (re)connect the session-start message is resent (the
    server's handling is idempotent) so a mid-walk Wi-Fi drop just resumes."""

    def __init__(
        self,
        host: str,
        port: int,
        device_id: str,
        scene_session: str,
        intrinsics: dict[str, Any] | None,
        settings: dict[str, Any],
        retry_s: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.device_id = device_id
        self.scene_session = scene_session
        self.intrinsics = intrinsics
        self.settings = settings
        self.retry_s = retry_s
        self.healthy = False
        self.sent_keyframes = 0
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
            self._request(
                make_scene_session_start_message(
                    self.device_id, self.scene_session, self.intrinsics, self.settings
                ),
                b"",
            )
            self.healthy = True
            logger.info("Uplink connected to %s:%s", self.host, self.port)
            return True
        except (ClientConnectionError, OSError) as exc:
            logger.warning("Uplink unavailable (%s) — falling back to local disk", exc)
            self._drop()
            return False

    def _request(self, message: Any, payload: bytes) -> None:
        assert self._client is not None
        response, _ = self._client.request(message, payload)
        if response.type == "error":
            raise ClientConnectionError(f"Server error: {response.payload.get('error')}")

    def _drop(self) -> None:
        self.healthy = False
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def send_keyframe(
        self, frame_idx: int, meta: dict[str, Any], rgb_jpg: bytes, masks_png: bytes
    ) -> bool:
        if not self._ensure():
            return False
        try:
            self._request(
                make_scene_keyframe_message(
                    self.device_id, self.scene_session, frame_idx, meta,
                    rgb_bytes=len(rgb_jpg), masks_bytes=len(masks_png),
                ),
                rgb_jpg + masks_png,
            )
            self.sent_keyframes += 1
            return True
        except (ClientConnectionError, OSError) as exc:
            logger.warning("Uplink lost mid-keyframe (%s) — saving locally", exc)
            self._drop()
            return False

    def send_preview(self, jpeg: bytes, width: int, height: int, fps: float) -> None:
        if not self.healthy or self._client is None:
            return
        try:
            frame = CameraFrame(
                frame_id=str(time.monotonic_ns()), width=width, height=height,
                image_format="jpeg", content_type="image/jpeg", data=jpeg,
            )
            self._request(
                make_camera_stream_frame_message(
                    device_id=self.device_id, frame=frame,
                    session_id=self.scene_session, frame_index=0, fps=fps,
                    jpeg_quality=70, save_frame=False, view="camera", mode="scene_rec",
                ),
                jpeg,
            )
        except (ClientConnectionError, OSError) as exc:
            logger.debug("Preview send failed: %s", exc)
            self._drop()

    def send_end(self, session_meta: dict[str, Any]) -> bool:
        if not self._ensure():
            return False
        try:
            self._request(
                make_scene_session_end_message(self.device_id, self.scene_session, session_meta),
                b"",
            )
            return True
        except (ClientConnectionError, OSError):
            self._drop()
            return False

    def close(self) -> None:
        self._drop()


class SceneRecorder:
    def __init__(self, args: argparse.Namespace) -> None:
        import cv2

        self._cv2 = cv2
        self.args = args
        self.session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = args.output_dir / self.session_name  # fallback only
        self.intrinsics = load_intrinsics(args.intrinsics)
        if self.intrinsics is None:
            logger.warning(
                "No intrinsics at %s — record anyway, but run scene_calibrate.py "
                "for metric-quality reconstruction", args.intrinsics
            )

        config = load_config(REPO_ROOT / "config/default.json", REPO_ROOT / ".env")
        host = args.host or config.get("client", {}).get("server_host", "127.0.0.1")
        port = args.port or int(config.get("server", {}).get("port", 8765))
        device_id = str(config.get("client", {}).get("device_id", "raspberry_pi_01"))

        self.uplink: SceneUplink | None = None
        if not args.offline:
            self.uplink = SceneUplink(
                host, port, device_id, self.session_name, self.intrinsics,
                settings={
                    "record_size": [args.width, args.height],
                    "fps_target": args.fps,
                    "seg_model": args.seg_hef.name,
                    "confidence": args.confidence,
                },
            )

        lens_position = args.lens_position
        if lens_position is None and self.intrinsics is not None:
            lens_position = self.intrinsics.get("lens_position")

        focus = CameraFocusOptions(
            autofocus_mode="manual" if lens_position is not None else "continuous",
            autofocus_range="normal",
            autofocus_speed="normal",
            lens_position=lens_position,
        )
        if args.source == "replay" and args.replay_session is None:
            raise SystemExit("--source replay requires --replay-session <dir>")
        self.source = make_capture_source(
            args.source,
            CaptureSettings(
                width=args.width, height=args.height, fps=args.fps,
                jpeg_quality=95, focus_options=focus,
            ),
            args.replay_session if args.source == "replay" else args.synthetic_image,
        )

        self.hailo: HailoMultiModel | None = None
        self.seg = None
        self.seg_arch = SEG_ARCHS[args.seg_arch]
        self.clip = None
        if args.source == "camera" or args.seg_hef.exists():
            self.hailo = HailoMultiModel()
            self.seg = self.hailo.load("seg", str(args.seg_hef))
            # Take the input shape from the hef rather than the arch default.
            # The decoder's grid geometry is derived from it, so a hef that is
            # not 640x640 would otherwise decode into the wrong coordinate
            # space silently — no exception, just wrong boxes and masks.
            self.seg_arch = dataclasses.replace(
                self.seg_arch, input_shape=tuple(self.seg.input_shape[:2])
            )
            logger.info("Seg arch: %s, input %s, %d classes",
                        self.seg_arch.name, self.seg_arch.input_shape,
                        self.seg_arch.num_classes)
        # Fine-tuned archs carry their own vocabulary; COCO archs use the
        # built-in list. Loaded once, at startup, so a mismatch stops the run
        # before it records a whole session of wrong labels.
        self.class_names: list[str] | None = None
        if self.seg_arch.num_classes not in (1, len(COCO80_CLASS_NAMES)):
            self.class_names = load_class_names(
                args.seg_classes, self.seg_arch.num_classes)
            logger.info("Class names from %s: %s ...",
                        args.seg_classes, ", ".join(self.class_names[:6]))
            if not args.no_clip and args.clip_hef.exists():
                self.clip = self.hailo.load("clip", str(args.clip_hef))
            elif not args.no_clip:
                logger.warning("CLIP hef missing (%s) — recording without embeddings", args.clip_hef)

        self.tracker = GreedyTracker(
            TrackerConfig(
                min_confidence=(args.track_min_confidence
                                if args.track_min_confidence is not None
                                else args.confidence),
                confirm_hits=2,
                lost_after_s=2.0,
                end_after_s=6.0,
                frame_width=float(args.width),
                frame_height=float(args.height),
            ),
            excluded_classes=set(),
        )
        self.selector = KeyframeSelector(
            min_interval_s=args.keyframe_min_interval,
            max_interval_s=args.keyframe_max_interval,
            min_shift_px=args.keyframe_shift_px,
            blur_threshold=args.blur_threshold,
        )
        # IMU. Optional by design: no sensor, no calibration, or an unplugged
        # cable all degrade to the Mac's floor-plane gravity fit rather than
        # stopping a recording. `read()` returns gravity (down) in the camera
        # frame — see imu_rvc.RvcReader/imu_shtp_motion.ShtpMotionReader and
        # _emit_keyframe. Driver choice matters: RVC and SHTP-UART are
        # mutually exclusive on the physical sensor (mode pins) — see
        # --imu-driver's help text.
        self.gravity_source = None
        if not args.no_imu:
            imu_calibration = args.imu_calibration
            if imu_calibration is None:
                imu_calibration = REPO_ROOT / (
                    "config/imu_calibration.json" if args.imu_driver == "rvc"
                    else "config/imu_calibration_shtp.json"
                )

            if args.imu_driver == "rvc":
                from pi_client.imu_rvc import RvcReader as ImuReaderCls
                default_port, default_baud = "/dev/ttyUSB0", 115200
            else:
                from pi_client.imu_shtp_motion import ShtpMotionReader as ImuReaderCls
                default_port, default_baud = "/dev/ttyUSB0", 3_000_000

            reader = ImuReaderCls(
                port=args.imu_port or default_port,
                baud=default_baud,
                calibration_path=imu_calibration,
            )
            if not imu_calibration.exists():
                logger.warning(
                    "IMU calibration missing (%s) — gravity will NOT be recorded. "
                    "Run ./scripts/run_imu_calibrate.sh --driver %s; without it the "
                    "sensor's orientation relative to the camera is unknown and a "
                    "guessed gravity vector would be worse than none.",
                    imu_calibration, args.imu_driver,
                )
            elif reader.start():
                self.gravity_source = reader
            else:
                logger.warning("IMU not available on %s — continuing without it.",
                               args.imu_port)
        self._keyframe_index = 0
        self._local_keyframes = 0
        self._last_keyframe_at = 0.0
        self._last_keyframe_gray: np.ndarray | None = None
        self._select_reasons: dict[str, int] = {}
        self._gate_reasons: dict[str, int] = {}
        self._frame_index = 0
        self._started_at = time.time()
        self._stop = False

    # ------------------------------------------------------------- loop

    def run(self) -> int:
        self.source.start()
        interval = 1.0 / max(self.args.fps, 0.2)
        target = (
            f"streaming to {self.uplink.host}:{self.uplink.port}"
            if self.uplink is not None
            else f"local dir {self.session_dir}"
        )
        logger.info("Recording session %s — %s (Ctrl+C to stop)", self.session_name, target)
        try:
            while not self._stop:
                tick = time.monotonic()
                try:
                    self._process_frame()
                except Exception as exc:
                    logger.error("Frame failed: %s", exc)
                if self._keyframe_index >= self.args.max_keyframes:
                    logger.info("Max keyframes reached")
                    break
                if getattr(self.source, "exhausted", False):
                    logger.info("Replay finished: %d source frames consumed",
                                self._frame_index)
                    break
                elapsed = time.monotonic() - tick
                if elapsed < interval:
                    # A replay is not rate-limited by a sensor; sleeping to the
                    # capture fps would make a 53-frame session take 18 s for
                    # no reason.
                    if self.args.source != "replay":
                        time.sleep(interval - elapsed)
        finally:
            self._finalize()
        return 0

    def stop(self) -> None:
        self._stop = True

    def _apply_gates(
        self, detections: list[dict[str, Any]], masks: np.ndarray | None,
        frame_shape: tuple[int, int],
    ) -> tuple[list[dict[str, Any]], np.ndarray | None]:
        keep, reasons = gate_proposals(
            detections, masks, frame_shape,
            min_area_frac=self.args.min_mask_area_frac,
            max_area_frac=self.args.max_mask_area_frac,
            reject_edge_count=self.args.reject_edge_count,
        )
        for reason, count in reasons.items():
            self._gate_reasons[reason] = self._gate_reasons.get(reason, 0) + count
        if len(keep) == len(detections):
            return detections, masks
        kept_masks = masks[keep] if masks is not None and len(masks) else masks
        return [detections[i] for i in keep], kept_masks

    def _class_name(self, class_id: int) -> str:
        """The name for a class index, per the arch actually in use.

        A class-agnostic arch (FastSAM) emits class 0 for everything, and
        indexing COCO80 with it would label every proposal "person" — a wrong
        label that reads as a real detection all the way through the graph.
        The honest name is "object", and downstream code that keys off COCO
        names then sees a word that is not in the list and can react.
        """
        if self.seg_arch.num_classes == 1:
            return "object"
        names = self.class_names if self.class_names is not None else COCO80_CLASS_NAMES
        if 0 <= class_id < len(names):
            return names[class_id]
        return str(class_id)

    def _process_frame(self) -> None:
        cv2 = self._cv2
        bgr = self.source.capture_bgr()
        self._frame_index += 1
        now = time.time()
        h, w = bgr.shape[:2]

        detections: list[dict[str, Any]] = []
        masks_640 = None
        if self.seg is not None:
            in_h, in_w = self.seg.input_shape[:2]
            # Letterbox, NOT a plain resize: the model was trained on aspect-
            # preserved input, and squashing 16:9 into a square measurably
            # costs detections (see seg_postprocess.Letterbox).
            padded, lb = letterbox(bgr, (in_h, in_w))
            rgb_in = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            raw = self.seg.infer(rgb_in)
            decoded = yolov8_seg_postprocess(
                raw,
                score_threshold=self.args.confidence,
                max_det=self.args.max_detections,
                arch=self.seg_arch,
            )
            boxes_frame = lb.box_to_frame(decoded["boxes_xyxy"])
            for i in range(len(decoded["scores"])):
                x0, y0, x1, y1 = boxes_frame[i]
                cls = int(decoded["classes"][i])
                detections.append(
                    {
                        "class": self._class_name(cls),
                        "class_id": cls,
                        "confidence": float(decoded["scores"][i]),
                        "bbox_xyxy": [
                            float(np.clip(x0, 0, w)), float(np.clip(y0, 0, h)),
                            float(np.clip(x1, 0, w)), float(np.clip(y1, 0, h)),
                        ],
                    }
                )
            # Strip the padding band so the stored mask grid stays proportional
            # to the frame; every reader can then scale it without knowing
            # letterboxing ever happened.
            masks_640 = lb.crop_masks(decoded["masks"])
            # Gate BEFORE tracking and before CLIP: a rejected proposal should
            # cost neither an NPU call nor a track id.
            detections, masks_640 = self._apply_gates(detections, masks_640, (h, w))

        # `assignments` is positionally aligned to `detections`. It replaces a
        # reverse lookup keyed on the rounded bbox, which silently merged two
        # detections whose boxes rounded the same — harmless at 14 detections
        # per session, wrong as soon as proposals are dense and nested.
        updates = self.tracker.update(detections, now)
        track_ids = updates.assignments

        if self._is_keyframe(bgr, now):
            self._emit_keyframe(bgr, detections, masks_640, track_ids, now)

        if self.uplink is not None and not self.args.no_preview:
            self._send_preview(bgr, detections)

    def _image_shift_px(self, gray: np.ndarray) -> float | None:
        """Median feature displacement since the last keyframe, in pixels of
        the downscaled gray image — a pose-free stand-in for parallax.

        Sparse LK on a 480x270 image costs a couple of milliseconds, which
        is nothing next to segmentation, and it is the only signal available
        on the Pi for "has the viewpoint actually changed".
        """
        cv2 = self._cv2
        previous = self._last_keyframe_gray
        if previous is None or previous.shape != gray.shape:
            return None
        points = cv2.goodFeaturesToTrack(
            previous, maxCorners=200, qualityLevel=0.01, minDistance=8
        )
        if points is None or len(points) < 12:
            return None
        moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
        if moved is None or status is None:
            return None
        ok = status.reshape(-1) == 1
        if ok.sum() < 12:
            return None
        deltas = moved.reshape(-1, 2)[ok] - points.reshape(-1, 2)[ok]
        return float(np.median(np.linalg.norm(deltas, axis=1)))

    def _imu_blur_risk(self) -> str | None:
        """Причина отбраковки кадра по IMU, или None если кадр годен.

        Пороги из руководства §7.5: 15°/с — предел, за которым выдержка 1/30 с
        начинает мазать; вибрация и удар портят кадр иначе, но так же
        необратимо. Молчит, когда данных мало: отбраковывать по одному отсчёту
        значит выбрасывать кадры по шуму.
        """
        try:
            window = self.gravity_source.window()
        except Exception:  # noqa: BLE001 — датчик не должен ронять запись
            return None
        if len(window) < 10:
            return None
        accel = [(s.accel_mg[0] / 1000.0 * 9.80665,
                  s.accel_mg[1] / 1000.0 * 9.80665,
                  s.accel_mg[2] / 1000.0 * 9.80665) for s in window]
        span_rad = math.radians(window[-1].yaw_deg - window[0].yaw_deg)
        dt = max(1e-3, window[-1].monotonic - window[0].monotonic)
        state = motion_state(accel, span_rad, dt)
        limit = self.args.max_keyframe_omega_dps
        if limit > 0 and state["omega_dps"] > limit:
            return "imu_turning"
        if state["impact"]:
            return "imu_impact"
        if state["vibration"]:
            return "imu_vibration"
        return None

    def _is_keyframe(self, bgr: np.ndarray, now: float) -> bool:
        cv2 = self._cv2
        gray = cv2.cvtColor(cv2.resize(bgr, (480, 270)), cv2.COLOR_BGR2GRAY)
        sharpness = variance_of_laplacian(gray)
        shift = self._image_shift_px(gray)
        # Zero-velocity update. The accelerometer cannot tell "standing
        # still" from "rolling at constant speed" — both read zero linear
        # acceleration — so the evidence has to come from the camera, and we
        # already have it: an image that did not shift means a rig that did
        # not move. Without this, velocity error integrates into position
        # without bound and the metric scale estimate degrades over a run.
        if (self.gravity_source is not None and shift is not None
                and shift < self.args.zupt_shift_px):
            self.gravity_source.integrator.zero_velocity()

        emit, reason = self.selector.decide(
            elapsed_s=now - self._last_keyframe_at,
            shift_px=shift,
            sharpness=sharpness,
            is_first=self._last_keyframe_gray is None,
        )
        # Отбраковка по IMU (руководство §7.5). Резкость по Лапласиану ловит
        # уже СЛУЧИВШИЙСЯ смаз, а курс говорит, что кадр смазан ПРЯМО СЕЙЧАС —
        # при выдержке 1/30 с поворот быстрее 15°/с растягивает точку на
        # полградуса кадра. Такой кадр портит и признаки DINOv2, и глубину, а
        # выглядит достаточно резким, чтобы порог по Лапласиану пройти.
        #
        # Гасится только ВЫДАЧА ключевого кадра: съёмка и трекинг продолжаются,
        # иначе робот, который просто разворачивается, переставал бы видеть.
        if emit and self.gravity_source is not None:
            blur = self._imu_blur_risk()
            if blur is not None:
                reason = blur
                emit = False
                self._select_reasons[blur] = self._select_reasons.get(blur, 0) + 1

        self._select_reasons[reason] = self._select_reasons.get(reason, 0) + 1
        if emit:
            self._last_keyframe_gray = gray
            logger.debug(
                "keyframe accepted (%s): shift %s px, VoL %.0f",
                reason, "n/a" if shift is None else f"{shift:.1f}", sharpness,
            )
        return emit

    # ------------------------------------------------------------- output

    def _emit_keyframe(
        self,
        bgr: np.ndarray,
        detections: list[dict[str, Any]],
        masks_640: np.ndarray | None,
        track_ids: list[int | None],
        now: float,
    ) -> None:
        cv2 = self._cv2
        self._keyframe_index += 1
        self._last_keyframe_at = now
        h, w = bgr.shape[:2]

        # Masks go out as `stack_v1`: the per-instance binary masks stacked
        # vertically, at INFERENCE resolution, still named masks.png.
        #
        # The old format was a single uint16 label image, which cannot
        # represent overlap — `instance_map[mask] = id` means the last
        # detection written wins every shared pixel. That was tolerable while
        # detections were a handful of disjoint COCO objects. It is wrong for
        # class-agnostic proposals, which produce NESTED masks by design
        # (a cabinet and its drawer, a table and the objects on it): the inner
        # mask punches a hole through the outer one, and nothing downstream can
        # tell that from a real occlusion.
        #
        # Inference resolution, not frame resolution, because that is the
        # resolution the mask was actually computed at — upsampling to
        # 1536x864 before storage would multiply the bytes by ~3 while adding
        # no information.
        mask_shape: list[int] | None = None
        if masks_640 is not None and masks_640.shape[0] > 0:
            binary = (masks_640[:len(detections)] > 0.5).astype(np.uint8) * 255
            mask_shape = [int(binary.shape[1]), int(binary.shape[2])]
            mask_image = binary.reshape(-1, binary.shape[2])
            mask_count = int(binary.shape[0])
        else:
            # cv2 cannot encode a zero-row image, and a keyframe with no
            # detections is a normal outcome worth recording as such.
            mask_image = np.zeros((1, 1), dtype=np.uint8)
            mask_count = 0

        meta_dets: list[dict[str, Any]] = []
        for idx, det in enumerate(detections):
            instance_id = idx + 1
            entry: dict[str, Any] = {
                "instance_id": instance_id,
                # -1 = the tracker declined this detection (below its
                # confidence gate, or an excluded anchor class), not "unknown".
                "track_id": (track_ids[idx] if idx < len(track_ids)
                             and track_ids[idx] is not None else -1),
                "class_coco": det["class"],
                "confidence": round(det["confidence"], 3),
                "bbox_xyxy": [round(v, 1) for v in det["bbox_xyxy"]],
            }
            if self.clip is not None:
                emb = self._clip_embedding(bgr, det["bbox_xyxy"])
                if emb is not None:
                    entry["clip_emb"] = emb
            meta_dets.append(entry)

        ok_rgb, rgb_encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        ok_masks, masks_encoded = cv2.imencode(".png", mask_image)
        if not ok_rgb or not ok_masks:
            logger.error("Keyframe %06d encode failed — dropped", self._keyframe_index)
            return
        meta = {
            "frame_idx": self._keyframe_index,
            "source_frame": self._frame_index,
            "timestamp_ns": int(now * 1e9),
            # Per keyframe, not only in session_meta.json: what "class" means
            # depends entirely on this, and a session must stay self-describing
            # after the config that produced it has moved on. Absent field =
            # yolov8_seg, which is what every existing session is.
            "seg_arch": self.seg_arch.name,
            # Absent = the legacy uint16 label image; readers must keep
            # supporting it, since every already-recorded session is that.
            "masks_format": "stack_v1",
            "mask_shape": mask_shape,       # [h, w] of ONE mask
            "mask_count": mask_count,
            "frame_shape": [h, w],
            "detections": meta_dets,
        }
        # Everything the IMU has to say about this keyframe. Two consumers:
        #
        #   `gravity` — down in the camera frame. SceneSession.imu_up_vector()
        #   rotates it into world space and occupancy.estimate_gravity()
        #   prefers it over the floor-plane fit.
        #
        #   `imu.segment` — motion preintegrated since the previous keyframe.
        #   Its displacement magnitude is a METRIC measurement of how far the
        #   rig moved, which is the one thing vision cannot supply on its own:
        #   COLMAP recovers motion only up to a scale factor, and the pipeline
        #   currently pins that factor with monocular depth that carries its
        #   own scale error. Comparing the two over many intervals grounds the
        #   scale in m/s^2 instead. Magnitudes only, so the rotation about
        #   gravity — which a robot that drives in-plane can never observe —
        #   never enters the answer.
        #
        # Sessions recorded without an IMU simply lack both fields.
        if self.gravity_source is not None:
            motion = self.gravity_source.read_motion()
            if motion is not None:
                gravity = motion.pop("gravity_camera", None)
                if gravity is not None:
                    meta["gravity"] = gravity
                # Ориентация на момент СЪЁМКИ кадра, а не на момент, когда до
                # неё дошли руки. Метка берётся с сенсора камеры, а сама
                # ориентация интерполируется по восстановленным часам IMU.
                # Время прихода IMU-кадров на хост врёт до 20 мс — адаптер
                # отдаёт их парами; время после инференса врёт ещё больше.
                # При 30 град/с каждые 20 мс это 0.6 град курса, приписанного
                # не тому мгновению.
                frame_time = getattr(self.source, "last_frame_time", None)
                if frame_time is not None:
                    at_frame = self.gravity_source.at_time(frame_time)
                    if at_frame is not None:
                        motion["at_frame"] = at_frame
                meta["imu"] = motion

        sent = False
        if self.uplink is not None:
            sent = self.uplink.send_keyframe(
                self._keyframe_index, meta, rgb_encoded.tobytes(), masks_encoded.tobytes()
            )
        if not sent:
            self._save_keyframe_locally(meta, rgb_encoded.tobytes(), masks_encoded.tobytes())
        logger.info(
            "keyframe %06d: %d detections%s -> %s",
            self._keyframe_index, len(meta_dets),
            " +clip" if self.clip is not None else "",
            "mac" if sent else "LOCAL",
        )

    def _save_keyframe_locally(self, meta: dict[str, Any], rgb: bytes, masks: bytes) -> None:
        kf_dir = self.session_dir / "keyframes" / f"{self._keyframe_index:06d}"
        kf_dir.mkdir(parents=True, exist_ok=True)
        if self.intrinsics is not None and not (self.session_dir / "intrinsics.json").exists():
            (self.session_dir / "intrinsics.json").write_text(
                json.dumps(self.intrinsics, indent=2), encoding="utf-8"
            )
        (kf_dir / "rgb.jpg").write_bytes(rgb)
        (kf_dir / "masks.png").write_bytes(masks)
        (kf_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        self._local_keyframes += 1

    def _send_preview(self, bgr: np.ndarray, detections: list[dict[str, Any]]) -> None:
        cv2 = self._cv2
        try:
            scale = 960 / bgr.shape[1]
            preview = cv2.resize(bgr, (960, int(bgr.shape[0] * scale)))
            for det in detections:
                x0, y0, x1, y1 = (int(v * scale) for v in det["bbox_xyxy"])
                cv2.rectangle(preview, (x0, y0), (x1, y1), (30, 220, 30), 2)
                cv2.putText(
                    preview, f"{det['class']} {det['confidence']:.2f}",
                    (x0, max(16, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (30, 220, 30), 1, cv2.LINE_AA,
                )
            cv2.putText(
                preview, f"REC {self.session_name}  kf {self._keyframe_index:06d}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2, cv2.LINE_AA,
            )
            ok, encoded = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                self.uplink.send_preview(
                    encoded.tobytes(), preview.shape[1], preview.shape[0], self.args.fps
                )
        except Exception as exc:
            logger.debug("Preview failed: %s", exc)

    def _clip_embedding(self, bgr: np.ndarray, bbox: list[float]) -> list[float] | None:
        cv2 = self._cv2
        try:
            h, w = bgr.shape[:2]
            x0, y0, x1, y1 = (int(v) for v in bbox)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                return None
            crop = bgr[y0:y1, x0:x1]
            in_h, in_w = self.clip.input_shape[:2]
            rgb = cv2.cvtColor(cv2.resize(crop, (in_w, in_h)), cv2.COLOR_BGR2RGB)
            out = self.clip.infer(rgb)
            vec = np.asarray(next(iter(out.values())), dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vec))
            if norm <= 0:
                return None
            return [round(float(v), 5) for v in (vec / norm)]
        except Exception as exc:
            logger.debug("CLIP embedding failed: %s", exc)
            return None

    def _finalize(self) -> None:
        try:
            self.source.stop()
        except Exception:
            pass
        if self.hailo is not None:
            self.hailo.close()
        if self.gravity_source is not None:
            try:
                logger.info("IMU: %s", self.gravity_source.stats())
                self.gravity_source.stop()
            except Exception:
                pass
        session_meta = {
            "created_at": self._started_at,
            "duration_s": round(time.time() - self._started_at, 1),
            "keyframes": self._keyframe_index,
            "frames_seen": self._frame_index,
            "record_size": [self.args.width, self.args.height],
            "seg_model": str(self.args.seg_hef.name),
            "seg_arch": self.seg_arch.name,
            "seg_input_shape": list(self.seg_arch.input_shape),
            "seg_num_classes": self.seg_arch.num_classes,
            "clip_model": str(self.args.clip_hef.name) if self.clip is not None else None,
            "confidence": self.args.confidence,
            "fps_target": self.args.fps,
            "keyframes_local_fallback": self._local_keyframes,
            "imu": self.gravity_source is not None,
            "keyframe_selection": dict(self._select_reasons),
            # Empty when the gates are at their inert defaults. Recorded
            # either way, so "the gates dropped it" is never confused with
            # "the model never proposed it".
            "proposal_gates": dict(self._gate_reasons),
            "track_min_confidence": (self.args.track_min_confidence
                                     if self.args.track_min_confidence is not None
                                     else self.args.confidence),
            "masks_format": "stack_v1",
        }
        sent_end = False
        if self.uplink is not None:
            sent_end = self.uplink.send_end(session_meta)
            self.uplink.close()
        if self._local_keyframes > 0 or (self.uplink is None):
            self.session_dir.mkdir(parents=True, exist_ok=True)
            (self.session_dir / "session_meta.json").write_text(
                json.dumps(session_meta, indent=2), encoding="utf-8"
            )

        logger.info(
            "Session done: %s — %d keyframes (%d streamed, %d local), %.0fs",
            self.session_name, self._keyframe_index,
            self.uplink.sent_keyframes if self.uplink else 0,
            self._local_keyframes, session_meta["duration_s"],
        )
        print(f"\nSession: {self.session_name}")
        if self.uplink is not None and sent_end and self._local_keyframes == 0:
            print("All keyframes are already on the Mac — open the Scene tab and run the pipeline.")
        elif self._local_keyframes > 0:
            print(f"{self._local_keyframes} keyframes stayed LOCAL in {self.session_dir}")
            print("Merge them into the Mac copy (run from the Mac):")
            print(f"  rsync -avP cv-pi.local:{self.session_dir}/ <repo>/data/scene_sessions/{self.session_name}/")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    recorder = SceneRecorder(args)

    def handle_signal(signum: int, _frame: Any) -> None:
        logger.info("Signal %s — stopping", signum)
        recorder.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    return recorder.run()


if __name__ == "__main__":
    raise SystemExit(main())
