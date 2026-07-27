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
import math
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
        mapping_config: dict[str, Any] | None = None,
    ) -> None:
        self._pi_store = pi_store
        self._depth_view_store = frame_hub.get("depth")
        self.depth_store = LatestItemStore()
        self.objects_store = LatestItemStore()
        self._config = config
        self._repo_root = repo_root
        # Physical constants (camera height, wall height band) come from the
        # shared `mapping` config section — same room, same camera, whether
        # the consumer is the paused wall-scanner or this live wall/obstacle
        # scatter (docs/scene3d.md IMU section; the new panel tab).
        self._mapping_config = mapping_config or {}
        # Live map (IMU tab): frame-to-frame 2D ICP scan matching
        # (mapping/scan_match.py) accumulates a running pose, a trail, and
        # a deduped set of explored wall/obstacle cells — a dead-reckoning
        # "fog of war" map, NOT a survey. Only the worker thread (_run)
        # mutates this; reset_live_map() (called from an HTTP handler on a
        # different thread) only sets a flag _run checks, so there is no
        # lock needed and no risk of tearing state mid-update.
        self._pose = {"x": 0.0, "y": 0.0, "yaw_rad": 0.0}
        self._prev_scan_points: Any = None
        self._last_icp_estimate = {"dx": 0.0, "dy": 0.0, "dyaw_rad": 0.0}
        self._trail: deque = deque(maxlen=3000)
        self._explored_cells: set = set()
        self._reset_live_map = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "disabled"
        self._error: str | None = None
        self._inference_ms: float | None = None
        self._frame_times: deque[float] = deque(maxlen=60)
        self._device = str(config.get("device", "mps"))
        self._latest_objects: dict[str, Any] | None = None

    def reset_live_map(self) -> None:
        """Requested from the panel's "Reset map" button — clears the
        accumulated pose/trail/explored cells so the next frame starts a
        fresh map, same idea as starting a new game level."""
        self._reset_live_map.set()

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
            "hfov_deg": float(self._config.get("hfov_deg", 102.0)),
            "display_max_m": float(self._config.get("display_max_m", 8.0)),
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
        hfov_deg = float(self._config.get("hfov_deg", 102.0))
        vfov_deg = float(self._config.get("vfov_deg", 67.0))
        target_fps = float(self._config.get("target_fps", 15.0))
        min_interval = 1.0 / target_fps if target_fps > 0 else 0.0
        sequence = -1
        frames_done = 0
        last_started = 0.0

        while not self._stop_event.is_set():
            if self._reset_live_map.is_set():
                self._pose = {"x": 0.0, "y": 0.0, "yaw_rad": 0.0}
                self._prev_scan_points = None
                self._last_icp_estimate = {"dx": 0.0, "dy": 0.0, "dyaw_rad": 0.0}
                self._trail.clear()
                self._explored_cells.clear()
                self._reset_live_map.clear()

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

            # Attach metric distances + bearing to Pi-detected objects (data
            # channel: the Pi runs YOLO on the AI HAT, the Mac contributes
            # meters and camera-frame position). Empty lists are published
            # too: consumers (radar, monitoring tracker) need "nothing in
            # view" ticks to clear points and to age out tracks. Computed
            # every depth frame regardless of mode (not just yolo/pipeline)
            # so wall_points below — the live obstacle/wall scatter for the
            # IMU debug tab's map — updates during plain stream/depth too.
            objects = frame.metadata.get("objects") or []
            try:
                from mac_server.mapping.geometry import CameraIntrinsics

                intrinsics = CameraIntrinsics.from_fov(width, height, hfov_deg, vfov_deg)
                scan_points = _wall_points_array(depth_m, intrinsics, np, self._mapping_config)
                self._latest_objects = {
                    "pi_frame_index": frame.metadata.get("frame_index"),
                    "view": view,
                    "computed_at": time.time(),
                    "objects": _attach_depth_to_objects(objects, depth_m, np, intrinsics),
                    "wall_points": [[round(float(x), 2), round(float(z), 2)] for x, z in scan_points],
                    "live_map": self._update_live_map(scan_points, np),
                }
                self.objects_store.update(self._latest_objects)
            except Exception as exc:
                logger.debug("Object depth attachment / wall scatter failed: %s", exc)

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

    # ------------------------------------------------------------ live map

    def _update_live_map(self, scan_points: Any, np: Any) -> dict[str, Any]:
        """Frame-to-frame 2D ICP (mapping/scan_match.py) against the
        previous scan gives this tick's own motion; composed onto a
        running pose it's a dead-reckoning "fog of war" map — trail plus
        every wall/obstacle cell seen so far, in world frame. Explicitly
        NOT the offline scene3d reconstruction: no bundle adjustment, no
        loop closure, so it drifts on a long walk. It also does NOT use
        the IMU for translation — that's exactly the tilt-leak problem
        documented on `ImuIntegrator` — only ICP's own (independent,
        already-metric) depth-camera estimate ever moves the pose.
        """
        from mac_server.mapping.scan_match import icp_2d

        min_points = int(self._mapping_config.get("live_map_min_points", 15))
        result = None
        if (
            self._prev_scan_points is not None
            and len(self._prev_scan_points) >= min_points
            and len(scan_points) >= min_points
        ):
            result = icp_2d(
                self._prev_scan_points,
                scan_points,
                initial_yaw_rad=self._last_icp_estimate["dyaw_rad"],
                initial_translation=(
                    self._last_icp_estimate["dx"], self._last_icp_estimate["dy"],
                ),
                min_points=min_points,
            )
        self._prev_scan_points = scan_points

        if result is not None:
            self._last_icp_estimate = {
                "dx": result["dx"], "dy": result["dy"], "dyaw_rad": result["dyaw_rad"],
            }
            local_dx, local_dy, dyaw = result["dx"], result["dy"], result["dyaw_rad"]
            source, fitness = "icp", result["fitness"]
        else:
            # No trustworthy visual match this tick (too few points, first
            # frame ever, or the scan barely overlaps the previous one —
            # e.g. right after a big jump, or a textureless wall). Holding
            # the pose still is the honest choice: there is no independent
            # fallback translation source that doesn't have the same
            # tilt-leak problem ICP exists to avoid in the first place.
            local_dx, local_dy, dyaw = 0.0, 0.0, 0.0
            source, fitness = "none", None
            self._last_icp_estimate = {"dx": 0.0, "dy": 0.0, "dyaw_rad": 0.0}

        yaw_before = self._pose["yaw_rad"]
        cos_b, sin_b = math.cos(yaw_before), math.sin(yaw_before)
        self._pose["x"] += cos_b * local_dx - sin_b * local_dy
        self._pose["y"] += sin_b * local_dx + cos_b * local_dy
        self._pose["yaw_rad"] = yaw_before + dyaw

        px, py, yaw_now = self._pose["x"], self._pose["y"], self._pose["yaw_rad"]
        self._trail.append({"x": round(px, 3), "y": round(py, 3)})

        grid_m = max(0.01, float(self._mapping_config.get("grid_resolution_m", 0.05)))
        max_cells = int(self._mapping_config.get("live_map_max_cells", 8000))
        cos_n, sin_n = math.cos(yaw_now), math.sin(yaw_now)
        for lateral, forward in scan_points:
            cell = (
                round((px + cos_n * lateral - sin_n * forward) / grid_m),
                round((py + sin_n * lateral + cos_n * forward) / grid_m),
            )
            if cell not in self._explored_cells and len(self._explored_cells) >= max_cells:
                continue
            self._explored_cells.add(cell)

        return {
            "pose": {"x": round(px, 3), "y": round(py, 3), "yaw_rad": round(yaw_now, 4)},
            "source": source,
            "fitness": fitness,
            "trail": list(self._trail),
            "explored": [[round(cx * grid_m, 2), round(cy * grid_m, 2)] for cx, cy in self._explored_cells],
        }

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


def _attach_depth_to_objects(
    objects: list[dict[str, Any]],
    depth_m: Any,
    np: Any,
    intrinsics: Any = None,
) -> list[dict[str, Any]]:
    """Median metric depth of each bbox's inner 50% region (robust against
    background pixels near the box edges), plus camera-frame bearing and
    Cartesian (lateral_m, forward_m) for the radar view.

    lateral_m is NOT range*sin(bearing) — that's a different point on a wide
    lens. It's tan(bearing)*forward_m, the same relationship depth_to_points
    uses for the full-frame room-mapping projection (see mapping/geometry.py).
    """
    from mac_server.mapping.geometry import bbox_bearing_deg, bbox_camera_xy

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
                forward_m = float(np.median(region))
                entry["depth_median_m"] = round(forward_m, 3)
                if intrinsics is not None:
                    entry["bearing_deg"] = round(bbox_bearing_deg(bbox, intrinsics), 1)
                    lateral_m, forward_m = bbox_camera_xy(bbox, forward_m, intrinsics)
                    entry["lateral_m"] = round(lateral_m, 3)
                    entry["forward_m"] = round(forward_m, 3)
        enriched.append(entry)
    return enriched


def _wall_points_array(
    depth_m: Any,
    intrinsics: Any,
    np: Any,
    mapping_config: dict[str, Any],
) -> Any:
    """Single-frame top-down scatter of walls/obstacles at roughly the
    robot's own height, as a raw (N,2) [lateral_m, forward_m] array —
    the live-map layer of the IMU debug tab's radar (docs/scene3d.md IMU
    section). Reuses the same pure-numpy, unit-tested projection the
    (paused) multi-frame room scanner uses (mapping/geometry.py).

    This one frame's scatter has no ego-motion and therefore no drift to
    get wrong (same principle as the object Radar view) — it's the RAW
    material both `_wall_points` (JSON-rounded, for the wire) and the live
    map's frame-to-frame ICP (mapping/scan_match.py, wants unrounded
    points) are built from.

    Empty (0, 2) array (not None) when mapping isn't configured, so
    callers don't need a None-check to iterate or len() it.
    """
    if not mapping_config:
        return np.zeros((0, 2), dtype=np.float64)
    from mac_server.mapping.geometry import depth_to_points, filter_height_band

    points = depth_to_points(
        depth_m,
        intrinsics,
        stride=int(mapping_config.get("wall_points_stride", 16)),
        edge_crop_frac=float(mapping_config.get("edge_crop_frac", 0.1)),
        min_z=0.2,
        max_z=float(mapping_config.get("max_depth_use_m", 8.0)),
    )
    if points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    xz, _heights = filter_height_band(
        points,
        camera_height_m=float(mapping_config.get("camera_height_m", 0.3)),
        band=tuple(mapping_config.get("height_band_m", [0.1, 2.0])),
    )
    return xz


def _wall_points(
    depth_m: Any,
    intrinsics: Any,
    np: Any,
    mapping_config: dict[str, Any],
) -> list[list[float]]:
    """`_wall_points_array`, JSON-rounded — kept for callers (and the unit
    tests) that just want the wire format without touching the array."""
    xz = _wall_points_array(depth_m, intrinsics, np, mapping_config)
    return [[round(float(x), 2), round(float(z), 2)] for x, z in xz]


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
