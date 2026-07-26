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
import json
import logging
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
from pi_client.seg_postprocess import yolov8_seg_postprocess
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
    parser.add_argument("--seg-hef", type=Path, default=REPO_ROOT / "models/yolov8m_seg_h8.hef",
                        help="yolov8m_seg (40.1 mask mAP); pass models/yolov8s_seg_h8.hef for speed")
    parser.add_argument("--clip-hef", type=Path, default=REPO_ROOT / "models/clip_resnet_50x4_h8.hef")
    parser.add_argument("--no-clip", action="store_true", help="Skip CLIP embeddings")
    parser.add_argument("--source", choices=["camera", "synthetic"], default="camera")
    parser.add_argument("--synthetic-image", type=Path, default=REPO_ROOT / "data/cat.jpg")
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
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--max-detections", type=int, default=30)
    parser.add_argument("--lens-position", type=float, default=None,
                        help="Manual focus (1/m); default: from intrinsics.json")
    parser.add_argument("--max-keyframes", type=int, default=2000)
    parser.add_argument("--host", default=None, help="Mac server host (default: from .env/config)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-imu", action="store_true",
                        help="Ignore the IMU even if it is connected and calibrated")
    parser.add_argument("--imu-port", default="/dev/ttyUSB0")
    parser.add_argument("--imu-calibration", type=Path,
                        default=REPO_ROOT / "config/imu_calibration.json")
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
        self.source = make_capture_source(
            args.source,
            CaptureSettings(
                width=args.width, height=args.height, fps=args.fps,
                jpeg_quality=95, focus_options=focus,
            ),
            args.synthetic_image,
        )

        self.hailo: HailoMultiModel | None = None
        self.seg = None
        self.clip = None
        if args.source == "camera" or args.seg_hef.exists():
            self.hailo = HailoMultiModel()
            self.seg = self.hailo.load("seg", str(args.seg_hef))
            if not args.no_clip and args.clip_hef.exists():
                self.clip = self.hailo.load("clip", str(args.clip_hef))
            elif not args.no_clip:
                logger.warning("CLIP hef missing (%s) — recording without embeddings", args.clip_hef)

        self.tracker = GreedyTracker(
            TrackerConfig(
                min_confidence=args.confidence,
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
        # frame — see imu_rvc.RvcReader and _emit_keyframe.
        self.gravity_source = None
        if not args.no_imu:
            from pi_client.imu_rvc import RvcReader

            reader = RvcReader(
                port=args.imu_port, calibration_path=args.imu_calibration
            )
            if not args.imu_calibration.exists():
                logger.warning(
                    "IMU calibration missing (%s) — gravity will NOT be recorded. "
                    "Run ./scripts/run_imu_calibrate.sh; without it the sensor's "
                    "orientation relative to the camera is unknown and a guessed "
                    "gravity vector would be worse than none.",
                    args.imu_calibration,
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
                elapsed = time.monotonic() - tick
                if elapsed < interval:
                    time.sleep(interval - elapsed)
        finally:
            self._finalize()
        return 0

    def stop(self) -> None:
        self._stop = True

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
            rgb_in = cv2.cvtColor(cv2.resize(bgr, (in_w, in_h)), cv2.COLOR_BGR2RGB)
            raw = self.seg.infer(rgb_in)
            decoded = yolov8_seg_postprocess(
                raw,
                score_threshold=self.args.confidence,
                max_det=self.args.max_detections,
            )
            sx, sy = w / in_w, h / in_h
            for i in range(len(decoded["scores"])):
                x0, y0, x1, y1 = decoded["boxes_xyxy"][i]
                cls = int(decoded["classes"][i])
                detections.append(
                    {
                        "class": COCO80_CLASS_NAMES[cls] if cls < 80 else str(cls),
                        "class_id": cls,
                        "confidence": float(decoded["scores"][i]),
                        "bbox_xyxy": [
                            float(np.clip(x0 * sx, 0, w)), float(np.clip(y0 * sy, 0, h)),
                            float(np.clip(x1 * sx, 0, w)), float(np.clip(y1 * sy, 0, h)),
                        ],
                    }
                )
            masks_640 = decoded["masks"]

        self.tracker.update(detections, now)
        track_by_bbox = {
            tuple(round(v, 1) for v in t.bbox): t.track_id
            for t in self.tracker.tracks.values()
        }

        if self._is_keyframe(bgr, now):
            self._emit_keyframe(bgr, detections, masks_640, track_by_bbox, now)

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

    def _is_keyframe(self, bgr: np.ndarray, now: float) -> bool:
        cv2 = self._cv2
        gray = cv2.cvtColor(cv2.resize(bgr, (480, 270)), cv2.COLOR_BGR2GRAY)
        sharpness = variance_of_laplacian(gray)
        shift = self._image_shift_px(gray)
        emit, reason = self.selector.decide(
            elapsed_s=now - self._last_keyframe_at,
            shift_px=shift,
            sharpness=sharpness,
            is_first=self._last_keyframe_gray is None,
        )
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
        track_by_bbox: dict[tuple, int],
        now: float,
    ) -> None:
        cv2 = self._cv2
        self._keyframe_index += 1
        self._last_keyframe_at = now
        h, w = bgr.shape[:2]

        instance_map = np.zeros((h, w), dtype=np.uint16)
        meta_dets: list[dict[str, Any]] = []
        for idx, det in enumerate(detections):
            instance_id = idx + 1
            if masks_640 is not None and idx < masks_640.shape[0]:
                mask_full = cv2.resize(
                    (masks_640[idx] > 0.5).astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
                instance_map[mask_full > 0] = instance_id

            entry: dict[str, Any] = {
                "instance_id": instance_id,
                "track_id": track_by_bbox.get(
                    tuple(round(v, 1) for v in det["bbox_xyxy"]), -1
                ),
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
        ok_masks, masks_encoded = cv2.imencode(".png", instance_map)
        if not ok_rgb or not ok_masks:
            logger.error("Keyframe %06d encode failed — dropped", self._keyframe_index)
            return
        meta = {
            "frame_idx": self._keyframe_index,
            "source_frame": self._frame_index,
            "timestamp_ns": int(now * 1e9),
            "detections": meta_dets,
        }
        # IMU seam. `self.gravity_source` is None until the accelerometer is
        # fitted; when it exists it returns the gravity vector in the CAMERA
        # frame (pointing down), and the Mac side is already waiting for it —
        # SceneSession.imu_up_vector() rotates it into world space and
        # occupancy.estimate_gravity() prefers it over the floor fit. Adding
        # the sensor is therefore a change to this one attribute, not to the
        # pipeline, and old sessions without the field keep working.
        if self.gravity_source is not None:
            gravity = self.gravity_source.read()
            if gravity is not None:
                meta["gravity"] = [round(float(v), 5) for v in gravity]

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
            "clip_model": str(self.args.clip_hef.name) if self.clip is not None else None,
            "confidence": self.args.confidence,
            "fps_target": self.args.fps,
            "keyframes_local_fallback": self._local_keyframes,
            "imu": self.gravity_source is not None,
            "keyframe_selection": dict(self._select_reasons),
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
