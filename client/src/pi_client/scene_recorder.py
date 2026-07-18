"""Scene recording client for the Pi: walk the camera around a room and
write a session directory the Mac reconstruction pipeline consumes.

Per frame (~2-3 fps target, CPU stays light — NPU does the work):
  camera -> yolov8-seg hef (Hailo-8, host decode) -> greedy tracker
         -> keyframe selector (min interval + blur gate)
         -> keyframe saved: rgb.jpg (full res) + masks.png (uint16
            instance map) + meta.json (detections with track_id and,
            optionally, a CLIP embedding per detection from the CLIP hef
            running on the same NPU via the model scheduler).

Session layout (the Pi<->Mac data contract, docs/scene3d.md):

    session_YYYYMMDD_HHMMSS/
      intrinsics.json      # copied from calibration (scene_calibrate.py)
      session_meta.json    # written on stop
      keyframes/000001/{rgb.jpg, masks.png, meta.json}

Camera controls are FIXED for the whole session (manual focus at the
calibrated LensPosition, AE/AWB locked after warmup) — drifting intrinsics
or color would hurt both COLMAP and the embedding association.
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

from pi_client.camera import CameraFocusOptions
from pi_client.camera_session import CaptureSettings, make_capture_source
from pi_client.hailo_infer import HailoMultiModel
from pi_client.network import ClientConnectionError, PiClient
from pi_client.protocol import make_camera_stream_frame_message
from pi_client.seg_postprocess import yolov8_seg_postprocess
from pi_client.cv_models import COCO80_CLASS_NAMES
from pi_client.camera import CameraFrame
from shared.config import load_config

# Pure-python tracker shared with the Mac monitoring stack (no server
# runtime involved — just tested association code).
from mac_server.monitoring.tracker import GreedyTracker, TrackerConfig


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a scene session (Pi + Hailo)")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/scene_sessions")
    parser.add_argument("--intrinsics", type=Path, default=REPO_ROOT / "config/scene_intrinsics.json")
    parser.add_argument("--seg-hef", type=Path, default=REPO_ROOT / "models/yolov8s_seg_h8.hef")
    parser.add_argument("--clip-hef", type=Path, default=REPO_ROOT / "models/clip_resnet_50x4_h8.hef")
    parser.add_argument("--no-clip", action="store_true", help="Skip CLIP embeddings")
    parser.add_argument("--source", choices=["camera", "synthetic"], default="camera")
    parser.add_argument("--synthetic-image", type=Path, default=REPO_ROOT / "data/cat.jpg")
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=864)
    parser.add_argument("--fps", type=float, default=3.0, help="Capture/inference loop rate")
    parser.add_argument("--keyframe-interval", type=float, default=0.5, help="Min seconds between keyframes")
    parser.add_argument("--blur-threshold", type=float, default=100.0, help="Variance-of-Laplacian gate")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--max-detections", type=int, default=30)
    parser.add_argument("--lens-position", type=float, default=None,
                        help="Manual focus (1/m); default: from intrinsics.json")
    parser.add_argument("--max-keyframes", type=int, default=2000)
    parser.add_argument("--no-preview", action="store_true",
                        help="Don't stream a live preview to the Mac panel")
    parser.add_argument("--host", default=None,
                        help="Mac server host for the preview (default: from .env/config)")
    parser.add_argument("--port", type=int, default=None)
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


class SceneRecorder:
    def __init__(self, args: argparse.Namespace) -> None:
        import cv2

        self._cv2 = cv2
        self.args = args
        self.session_dir = args.output_dir / datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.keyframes_dir = self.session_dir / "keyframes"
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)

        self.intrinsics = load_intrinsics(args.intrinsics)
        if self.intrinsics is not None:
            (self.session_dir / "intrinsics.json").write_text(
                json.dumps(self.intrinsics, indent=2), encoding="utf-8"
            )
        else:
            logger.warning(
                "No intrinsics at %s — record anyway, but run scene_calibrate.py "
                "for metric-quality reconstruction", args.intrinsics
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
        self._keyframe_index = 0
        self._last_keyframe_at = 0.0
        self._frame_index = 0
        self._started_at = time.time()
        self._stop = False

    # ------------------------------------------------------------- loop

    def run(self) -> int:
        self.source.start()
        interval = 1.0 / max(self.args.fps, 0.2)
        logger.info("Recording to %s (Ctrl+C to stop)", self.session_dir)
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

        if not self._is_keyframe(bgr, now):
            return
        self._save_keyframe(bgr, detections, masks_640, track_by_bbox, now)

    def _is_keyframe(self, bgr: np.ndarray, now: float) -> bool:
        if now - self._last_keyframe_at < self.args.keyframe_interval:
            return False
        cv2 = self._cv2
        gray = cv2.cvtColor(cv2.resize(bgr, (480, 270)), cv2.COLOR_BGR2GRAY)
        blur = variance_of_laplacian(gray)
        if blur < self.args.blur_threshold:
            logger.debug("Skipping blurred frame (VoL %.1f)", blur)
            return False
        return True

    # ------------------------------------------------------------- output

    def _save_keyframe(
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
        kf_dir = self.keyframes_dir / f"{self._keyframe_index:06d}"
        kf_dir.mkdir(parents=True, exist_ok=True)
        h, w = bgr.shape[:2]

        cv2.imwrite(str(kf_dir / "rgb.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

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

        cv2.imwrite(str(kf_dir / "masks.png"), instance_map)
        meta = {
            "frame_idx": self._keyframe_index,
            "source_frame": self._frame_index,
            "timestamp_ns": int(now * 1e9),
            "detections": meta_dets,
        }
        (kf_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        logger.info(
            "keyframe %06d: %d detections%s",
            self._keyframe_index, len(meta_dets),
            " +clip" if self.clip is not None else "",
        )

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
        meta = {
            "created_at": self._started_at,
            "duration_s": round(time.time() - self._started_at, 1),
            "keyframes": self._keyframe_index,
            "frames_seen": self._frame_index,
            "record_size": [self.args.width, self.args.height],
            "seg_model": str(self.args.seg_hef.name),
            "clip_model": str(self.args.clip_hef.name) if self.clip is not None else None,
            "confidence": self.args.confidence,
            "fps_target": self.args.fps,
        }
        (self.session_dir / "session_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info(
            "Session done: %s (%d keyframes, %.0fs)",
            self.session_dir, self._keyframe_index, meta["duration_s"],
        )
        print(f"\nSession: {self.session_dir}")
        print("Transfer to the Mac with:")
        print(f"  rsync -avP {self.session_dir} <mac>:pi_cv/data/scene_sessions/")


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
