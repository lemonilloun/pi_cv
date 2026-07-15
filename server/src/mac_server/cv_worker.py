"""Server-side CV worker: metric depth on the Mac from the Pi camera stream.

Consumes the latest camera frame from the 'pi' frame store, runs Depth
Anything V2 (metric, indoor) on MPS/CPU, and publishes:
- a depth heatmap JPEG into the 'depth' view store for the web panel,
- the raw metric depth array (`DepthFrame`) into `depth_store` for the mapper.

Heavy imports (torch/cv2/numpy) happen lazily inside the worker thread so the
TCP server core stays stdlib-only when CV is disabled or unavailable.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mac_server.preview import FrameStoreHub, LatestFrameStore


logger = logging.getLogger(__name__)

MPS_CACHE_CLEAR_EVERY_FRAMES = 500


@dataclass(frozen=True)
class DepthFrame:
    sequence: int
    depth_m: Any  # float32 (H, W) metric depth in meters
    width: int
    height: int
    frame_metadata: dict[str, Any] = field(default_factory=dict)
    inference_ms: float = 0.0
    computed_at: float = 0.0


class LatestItemStore:
    """Single-slot, Condition-notified store for arbitrary items (generic
    sibling of preview.LatestFrameStore)."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._item: Any = None
        self._sequence = 0

    def update(self, item: Any) -> None:
        with self._condition:
            self._sequence += 1
            self._item = item
            self._condition.notify_all()

    def wait_for_next(self, last_sequence: int, timeout: float = 10.0) -> tuple[int, Any]:
        with self._condition:
            self._condition.wait_for(lambda: self._sequence != last_sequence, timeout=timeout)
            return self._sequence, self._item

    def latest(self) -> tuple[int, Any]:
        with self._condition:
            return self._sequence, self._item


class WarmMetricDepthAnythingV2:
    """Metric Depth Anything V2 loaded once; infer_bgr returns meters."""

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    def __init__(
        self,
        model_path: Path,
        encoder: str,
        input_size: int,
        max_depth_m: float,
        device: str,
        metric_repo_dir: Path,
    ) -> None:
        if not model_path.exists():
            raise RuntimeError(
                f"Metric depth checkpoint not found: {model_path}. "
                "Run: ./scripts/setup_server_cv.sh"
            )
        if not metric_repo_dir.exists():
            raise RuntimeError(
                f"Depth-Anything-V2 metric_depth dir not found: {metric_repo_dir}. "
                "Run: ./scripts/setup_server_cv.sh"
            )
        if encoder not in self.MODEL_CONFIGS:
            raise RuntimeError(f"Unsupported encoder: {encoder}")

        # The metric variant lives in the repo's metric_depth/ subdir, which
        # defines its own `depth_anything_v2` package whose DepthAnythingV2
        # accepts max_depth (head = sigmoid * max_depth -> meters).
        if str(metric_repo_dir) not in sys.path:
            sys.path.insert(0, str(metric_repo_dir))

        import torch
        from depth_anything_v2.dpt import DepthAnythingV2

        self._torch = torch
        self.input_size = input_size
        self.model_path = str(model_path)

        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but unavailable; falling back to CPU")
            device = "cpu"
        self.device = device

        model = DepthAnythingV2(**self.MODEL_CONFIGS[encoder], max_depth=max_depth_m)
        model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        self._model = model.to(torch.device(device)).eval()

    def infer_bgr(self, image_bgr: Any) -> tuple[Any, float]:
        import numpy as np

        started = time.perf_counter()
        with self._torch.no_grad():
            depth = self._model.infer_image(image_bgr, self.input_size)
        depth = np.asarray(depth, dtype="float32")
        return depth, (time.perf_counter() - started) * 1000

    def clear_device_cache(self) -> None:
        if self.device == "mps":
            try:
                self._torch.mps.empty_cache()
            except Exception:
                pass


class ServerCvWorker:
    def __init__(
        self,
        pi_store: LatestFrameStore,
        frame_hub: FrameStoreHub,
        config: dict[str, Any],
        repo_root: Path,
    ) -> None:
        self._pi_store = pi_store
        self._depth_view_store = frame_hub.get("depth")
        self.depth_store = LatestItemStore()
        self._config = config
        self._repo_root = repo_root
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "disabled"
        self._error: str | None = None
        self._inference_ms: float | None = None
        self._frame_times: deque[float] = deque(maxlen=60)
        self._device = str(config.get("device", "mps"))
        self._latest_objects: dict[str, Any] | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._state = "loading"
        self._thread = threading.Thread(target=self._run, daemon=True, name="server-cv")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        recent = [ts for ts in self._frame_times if now - ts <= 5.0]
        fps = None
        if len(recent) >= 2 and recent[-1] > recent[0]:
            fps = round((len(recent) - 1) / (recent[-1] - recent[0]), 1)
        return {
            "state": self._state,
            "device": self._device,
            "model": str(self._config.get("depth_model_path", "")),
            "input_size": int(self._config.get("depth_input_size", 392)),
            "fps": fps,
            "inference_ms": round(self._inference_ms, 1) if self._inference_ms else None,
            "error": self._error,
            "objects": self._latest_objects,
        }

    # ------------------------------------------------------------- worker

    def _run(self) -> None:
        try:
            model = self._load_model()
        except Exception as exc:
            logger.error("Server CV model load failed: %s", exc)
            self._state = "error"
            self._error = str(exc)
            return

        self._state = "running"
        self._device = model.device
        logger.info(
            "Server CV worker running (device=%s, input_size=%s)",
            model.device,
            model.input_size,
        )

        import cv2
        import numpy as np

        display_max_m = float(self._config.get("display_max_m", 8.0))
        target_fps = float(self._config.get("target_fps", 15.0))
        min_interval = 1.0 / target_fps if target_fps > 0 else 0.0
        sequence = -1
        frames_done = 0
        last_started = 0.0

        while not self._stop_event.is_set():
            sequence, frame = self._pi_store.wait_for_next(sequence, timeout=1.0)
            if frame is None or self._stop_event.is_set():
                continue
            # Never run depth on Pi-rendered depth heatmaps. Annotated
            # yolo/combined frames are allowed: the thin box overlays barely
            # affect depth, and their `objects` metadata is what we attach
            # metric distances to.
            view = frame.metadata.get("view", "camera")
            if view not in {"camera", "yolo", "combined"}:
                continue
            now = time.monotonic()
            if now - last_started < min_interval:
                continue
            last_started = now

            try:
                data = np.frombuffer(frame.data, dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                depth_m, inference_ms = model.infer_bgr(image)
            except Exception as exc:
                logger.error("Server depth inference failed: %s (backing off 5s)", exc)
                self._error = str(exc)
                self._stop_event.wait(timeout=5.0)
                continue

            self._error = None
            self._inference_ms = inference_ms
            self._frame_times.append(time.monotonic())
            frames_done += 1
            if frames_done % MPS_CACHE_CLEAR_EVERY_FRAMES == 0:
                model.clear_device_cache()

            height, width = depth_m.shape[:2]
            depth_frame = DepthFrame(
                sequence=sequence,
                depth_m=depth_m,
                width=width,
                height=height,
                frame_metadata=dict(frame.metadata),
                inference_ms=inference_ms,
                computed_at=time.time(),
            )
            self.depth_store.update(depth_frame)

            # Attach metric distances to Pi-detected objects (data channel:
            # the Pi runs YOLO on the AI HAT, the Mac contributes meters).
            objects = frame.metadata.get("objects")
            if objects:
                try:
                    self._latest_objects = {
                        "pi_frame_index": frame.metadata.get("frame_index"),
                        "view": view,
                        "computed_at": time.time(),
                        "objects": _attach_depth_to_objects(objects, depth_m, np),
                    }
                except Exception as exc:
                    logger.debug("Object depth attachment failed: %s", exc)

            heatmap = _depth_to_heatmap_jpeg_fixed(depth_m, display_max_m)
            self._depth_view_store.update(
                heatmap,
                {
                    "view": "depth",
                    "source": "server",
                    "width": width,
                    "height": height,
                    "inference_ms": round(inference_ms, 1),
                    "depth_min_m": round(float(depth_m.min()), 2),
                    "depth_max_m": round(float(depth_m.max()), 2),
                    "pi_frame_index": frame.metadata.get("frame_index"),
                },
            )

        self._state = "disabled"

    def _load_model(self) -> WarmMetricDepthAnythingV2:
        model_path = Path(str(self._config.get("depth_model_path", "")))
        if not model_path.is_absolute():
            model_path = self._repo_root / model_path
        return WarmMetricDepthAnythingV2(
            model_path=model_path,
            encoder=str(self._config.get("depth_encoder", "vits")),
            input_size=int(self._config.get("depth_input_size", 392)),
            max_depth_m=float(self._config.get("depth_max_depth_m", 20.0)),
            device=str(self._config.get("device", "mps")),
            metric_repo_dir=self._repo_root / "external/Depth-Anything-V2/metric_depth",
        )


def _attach_depth_to_objects(objects: list[dict[str, Any]], depth_m: Any, np: Any) -> list[dict[str, Any]]:
    """Median metric depth of each bbox's inner 50% region (robust against
    background pixels near the box edges)."""
    height, width = depth_m.shape[:2]
    enriched = []
    for obj in objects:
        entry = dict(obj)
        bbox = obj.get("bbox_xyxy")
        if isinstance(bbox, list) and len(bbox) == 4:
            x0, y0, x1, y1 = (float(v) for v in bbox)
            dx, dy = (x1 - x0) * 0.25, (y1 - y0) * 0.25
            ix0 = max(0, min(width - 1, int(x0 + dx)))
            ix1 = max(ix0 + 1, min(width, int(x1 - dx)))
            iy0 = max(0, min(height - 1, int(y0 + dy)))
            iy1 = max(iy0 + 1, min(height, int(y1 - dy)))
            region = depth_m[iy0:iy1, ix0:ix1]
            if region.size:
                entry["depth_median_m"] = round(float(np.median(region)), 3)
        enriched.append(entry)
    return enriched


def _depth_to_heatmap_jpeg_fixed(depth_m: Any, display_max_m: float) -> bytes:
    """Heatmap with a fixed 0..display_max_m range so colors stay stable
    across frames (unlike per-frame min/max normalization)."""
    import cv2
    import numpy as np

    clipped = np.clip(depth_m / max(display_max_m, 0.1), 0.0, 1.0)
    normalized = (clipped * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    ok, encoded = cv2.imencode(".jpg", heatmap, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Failed to encode depth heatmap")
    return encoded.tobytes()
