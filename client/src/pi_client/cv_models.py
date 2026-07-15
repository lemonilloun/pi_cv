"""Optional CV model adapters for Raspberry Pi experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import tempfile
import sys
from io import BytesIO
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEPTH_ANYTHING_V2_SMALL_PATH = REPO_ROOT / "models/depth_anything_v2_vits.pth"
DEPTH_ANYTHING_V2_REPO_PATH = REPO_ROOT / "external/Depth-Anything-V2"


class CvModelError(RuntimeError):
    """Raised when an optional CV model dependency or inference step fails."""


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    mask_polygon: list[list[float]] | None = None
    depth: dict[str, float | str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class YoloResult:
    detections: list[Detection]
    annotated_image: bytes
    inference_ms: float
    model_path: str
    task: str


@dataclass(frozen=True)
class DepthResult:
    depth_map: Any
    heatmap_image: bytes
    inference_ms: float
    backend: str
    model_id: str
    input_size: int
    is_metric: bool
    min_depth: float
    max_depth: float


def check_depth_anything_v2_environment(
    model_path: str | None,
    encoder: str,
) -> dict[str, object]:
    resolved_model_path = _resolve_depth_anything_v2_model_path(model_path)
    _ensure_depth_anything_v2_import_path()

    status: dict[str, object] = {
        "repo_path": str(DEPTH_ANYTHING_V2_REPO_PATH),
        "repo_exists": DEPTH_ANYTHING_V2_REPO_PATH.exists(),
        "model_path": str(resolved_model_path),
        "model_exists": resolved_model_path.exists(),
        "encoder": encoder,
    }

    try:
        import cv2
        status["opencv_version"] = cv2.__version__
    except ImportError as exc:
        status["opencv_error"] = str(exc)

    try:
        import torch
        status["torch_version"] = torch.__version__
    except ImportError as exc:
        status["torch_error"] = str(exc)

    try:
        from depth_anything_v2.dpt import DepthAnythingV2  # noqa: F401
        status["depth_anything_v2_import"] = "ok"
    except ImportError as exc:
        status["depth_anything_v2_import_error"] = str(exc)

    if resolved_model_path.exists():
        status["model_size_mb"] = round(resolved_model_path.stat().st_size / 1024 / 1024, 2)

    return status


def depth_to_npz_bytes(depth_map: Any) -> bytes:
    try:
        import numpy as np
    except ImportError as exc:
        raise CvModelError("NumPy is required to save depth_raw.npz") from exc

    buffer = BytesIO()
    np.savez_compressed(buffer, depth=depth_map.astype("float32"))
    return buffer.getvalue()


class WarmYolo:
    """YOLO model loaded once and reused across frames."""

    def __init__(self, model_path: str, task: str = "detect") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise CvModelError(
                "YOLO dependencies are missing. Install on Raspberry Pi with: "
                "pip install ultralytics ncnn"
            ) from exc

        self.model_path = model_path
        self.task = task
        self._model = YOLO(model_path, task=task)

    def infer_bgr(self, image_array: Any, confidence: float) -> YoloResult:
        try:
            import cv2
        except ImportError as exc:
            raise CvModelError("OpenCV is required for YOLO inference") from exc

        started_at = time.perf_counter()
        results = self._model(image_array, conf=confidence, verbose=False)
        result = results[0]

        names = result.names
        detections: list[Detection] = []
        boxes = result.boxes
        masks = result.masks

        for index in range(len(boxes)):
            xyxy = boxes[index].xyxy.cpu().numpy().reshape(-1).astype(float).tolist()
            class_id = int(boxes[index].cls.item())
            class_name = str(names.get(class_id, class_id))
            conf = float(boxes[index].conf.item())
            polygon = None
            if masks is not None and masks.xy is not None and index < len(masks.xy):
                polygon = masks.xy[index].astype(float).tolist()

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox_xyxy=xyxy,
                    mask_polygon=polygon,
                )
            )

        plotted = result.plot()
        ok, encoded = cv2.imencode(".jpg", plotted)
        if not ok:
            raise CvModelError("Failed to encode YOLO annotated image")

        return YoloResult(
            detections=detections,
            annotated_image=encoded.tobytes(),
            inference_ms=(time.perf_counter() - started_at) * 1000,
            model_path=self.model_path,
            task=self.task,
        )

    def infer(self, image_bytes: bytes, confidence: float) -> YoloResult:
        return self.infer_bgr(_decode_image_bgr(image_bytes), confidence)


def run_yolo(image_bytes: bytes, model_path: str, confidence: float, task: str = "detect") -> YoloResult:
    return WarmYolo(model_path=model_path, task=task).infer(image_bytes, confidence)


COCO80_CLASS_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


class WarmHailoYolo:
    """YOLO detection on the Raspberry Pi AI HAT (Hailo) via picamera2's
    Hailo device helper. Expects a Hailo Model Zoo detection .hef with
    on-chip NMS postprocess (e.g. yolov8s): the output is a per-class list
    of (N, 5) arrays [ymin, xmin, ymax, xmax, score], normalized 0..1.
    """

    def __init__(self, hef_path: str, labels_path: str | None = None) -> None:
        resolved_hef = Path(hef_path)
        if not resolved_hef.is_absolute():
            resolved_hef = REPO_ROOT / resolved_hef
        if not resolved_hef.exists():
            raise CvModelError(
                f"Hailo model not found: {resolved_hef}. Install the AI HAT stack with "
                "'sudo apt install hailo-all' (models land in /usr/share/hailo-models) "
                "or point --hailo-hef at a Model Zoo .hef"
            )

        try:
            from picamera2.devices import Hailo
        except ImportError as exc:
            raise CvModelError(
                "Hailo support is missing. On Raspberry Pi OS install it with: "
                "sudo apt install hailo-all (then reboot). Requires the AI HAT/HAT+."
            ) from exc

        self.model_path = str(resolved_hef)
        self.task = "detect"
        self.class_names = self._load_labels(labels_path)
        try:
            self._hailo = Hailo(self.model_path)
        except Exception as exc:
            raise CvModelError(f"Failed to open Hailo device/model: {exc}") from exc
        input_shape = self._hailo.get_input_shape()
        self.input_h, self.input_w = int(input_shape[0]), int(input_shape[1])

    @staticmethod
    def _load_labels(labels_path: str | None) -> tuple[str, ...]:
        if not labels_path:
            return COCO80_CLASS_NAMES
        path = Path(labels_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        try:
            names = tuple(
                line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
            return names or COCO80_CLASS_NAMES
        except OSError:
            return COCO80_CLASS_NAMES

    def infer_bgr(self, image_array: Any, confidence: float) -> YoloResult:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise CvModelError("OpenCV/NumPy are required for Hailo inference") from exc

        frame_h, frame_w = image_array.shape[:2]
        # Zoo models are trained on RGB; plain stretch-resize keeps the
        # normalized output boxes directly mappable to the original frame.
        rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        model_input = cv2.resize(rgb, (self.input_w, self.input_h))

        started_at = time.perf_counter()
        raw_output = self._hailo.run(model_input)
        inference_ms = (time.perf_counter() - started_at) * 1000

        detections = self._extract_detections(raw_output, frame_w, frame_h, confidence, np)
        annotated = self._draw(image_array.copy(), detections, cv2)
        ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise CvModelError("Failed to encode Hailo annotated image")

        return YoloResult(
            detections=detections,
            annotated_image=encoded.tobytes(),
            inference_ms=inference_ms,
            model_path=self.model_path,
            task=self.task,
        )

    def _extract_detections(
        self,
        raw_output: Any,
        frame_w: int,
        frame_h: int,
        confidence: float,
        np: Any,
    ) -> list[Detection]:
        # The Hailo helper returns a single output for NMS-postprocessed
        # detection hefs; unwrap one nesting level if present.
        output = raw_output
        if isinstance(output, list) and len(output) == 1 and isinstance(output[0], (list, tuple)):
            output = output[0]

        detections: list[Detection] = []
        for class_id, class_dets in enumerate(output):
            class_dets = np.asarray(class_dets)
            if class_dets.size == 0:
                continue
            class_dets = class_dets.reshape(-1, class_dets.shape[-1])
            for det in class_dets:
                score = float(det[4])
                if score < confidence:
                    continue
                y0, x0, y1, x1 = (float(v) for v in det[:4])
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=(
                            self.class_names[class_id]
                            if class_id < len(self.class_names)
                            else str(class_id)
                        ),
                        confidence=score,
                        bbox_xyxy=[
                            x0 * frame_w,
                            y0 * frame_h,
                            x1 * frame_w,
                            y1 * frame_h,
                        ],
                    )
                )
        return detections

    @staticmethod
    def _draw(image: Any, detections: list[Detection], cv2: Any) -> Any:
        for det in detections:
            xmin, ymin, xmax, ymax = (int(v) for v in det.bbox_xyxy)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (30, 220, 30), 2)
            cv2.putText(
                image,
                label,
                (xmin, max(20, ymin - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (30, 220, 30),
                2,
            )
        return image

    def close(self) -> None:
        try:
            self._hailo.close()
        except Exception:
            pass


def run_depth(
    image_bytes: bytes,
    backend: str,
    model_path_or_id: str | None,
    encoder: str,
    input_size: int,
    is_metric: bool,
) -> DepthResult:
    if backend == "synthetic":
        return _run_synthetic_depth(image_bytes, input_size=input_size, is_metric=is_metric)
    if backend == "depth-anything-v2":
        return _run_depth_anything_v2(
            image_bytes=image_bytes,
            model_path=model_path_or_id,
            encoder=encoder,
            input_size=input_size,
            is_metric=is_metric,
        )
    if backend == "depth-anything-v3":
        return _run_depth_anything_v3(
            image_bytes=image_bytes,
            model_id=model_path_or_id or "depth-anything/DA3-BASE",
            input_size=input_size,
            is_metric=is_metric,
        )

    raise CvModelError(f"Unsupported depth backend: {backend}")


def render_combined_image(image_bytes: bytes, detections: list[Detection]) -> bytes:
    try:
        import cv2
    except ImportError as exc:
        raise CvModelError("OpenCV is required to render combined CV image") from exc

    image = _decode_image_bgr(image_bytes)
    for detection in detections:
        xmin, ymin, xmax, ymax = [int(value) for value in detection.bbox_xyxy]
        label = f"{detection.class_name} {detection.confidence:.2f}"
        median_depth = detection.depth.get("median") if detection.depth else None
        depth_unit = detection.depth.get("unit") if detection.depth else None
        if median_depth is not None:
            label += f" depth={float(median_depth):.3f} {depth_unit}"

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (30, 220, 30), 2)
        cv2.putText(
            image,
            label,
            (xmin, max(20, ymin - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (30, 220, 30),
            2,
        )

    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise CvModelError("Failed to encode combined CV image")
    return encoded.tobytes()


def attach_depth_to_detections(
    detections: list[Detection],
    depth_map: Any,
    image_width: int,
    image_height: int,
    is_metric: bool,
) -> list[Detection]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise CvModelError("NumPy and OpenCV are required to combine depth and detections") from exc

    depth_resized = cv2.resize(depth_map.astype("float32"), (image_width, image_height))
    updated: list[Detection] = []
    unit = "meters" if is_metric else "relative"

    for detection in detections:
        xmin, ymin, xmax, ymax = [int(value) for value in detection.bbox_xyxy]
        xmin = max(0, min(image_width - 1, xmin))
        xmax = max(0, min(image_width, xmax))
        ymin = max(0, min(image_height - 1, ymin))
        ymax = max(0, min(image_height, ymax))

        roi = depth_resized[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            depth_stats = {"median": None, "mean": None, "min": None, "max": None, "unit": unit}
        else:
            depth_stats = {
                "median": float(np.median(roi)),
                "mean": float(np.mean(roi)),
                "min": float(np.min(roi)),
                "max": float(np.max(roi)),
                "unit": unit,
            }

        updated.append(
            Detection(
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                bbox_xyxy=detection.bbox_xyxy,
                mask_polygon=detection.mask_polygon,
                depth=depth_stats,
            )
        )

    return updated


def image_size(image_bytes: bytes) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise CvModelError("Pillow is required to read image dimensions") from exc

    from io import BytesIO

    with Image.open(BytesIO(image_bytes)) as image:
        return image.size


def _run_synthetic_depth(image_bytes: bytes, input_size: int, is_metric: bool) -> DepthResult:
    started_at = time.perf_counter()
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise CvModelError("Synthetic depth backend requires NumPy and OpenCV") from exc

    image = _decode_image_bgr(image_bytes)
    height, width = image.shape[:2]
    y_gradient = np.linspace(0.0, 1.0, height, dtype="float32").reshape(height, 1)
    depth = np.repeat(y_gradient, width, axis=1)
    heatmap = _depth_to_heatmap_jpeg(depth)

    return DepthResult(
        depth_map=depth,
        heatmap_image=heatmap,
        inference_ms=(time.perf_counter() - started_at) * 1000,
        backend="synthetic",
        model_id="synthetic-gradient",
        input_size=input_size,
        is_metric=is_metric,
        min_depth=float(np.min(depth)),
        max_depth=float(np.max(depth)),
    )


class WarmDepthAnythingV2:
    """Depth Anything V2 model loaded once and reused across frames."""

    def __init__(
        self,
        model_path: str | None,
        encoder: str,
        input_size: int,
        is_metric: bool,
        torch_threads: int | None = None,
    ) -> None:
        resolved_model_path = _resolve_depth_anything_v2_model_path(model_path)
        if not resolved_model_path.exists():
            raise CvModelError(
                f"Depth Anything V2 checkpoint not found: {resolved_model_path}. "
                "Run: ./scripts/setup_depth_anything_v2.sh small"
            )

        _ensure_depth_anything_v2_import_path()

        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as exc:
            raise CvModelError(
                "Depth Anything V2 dependencies are missing. Run: ./scripts/setup_depth_anything_v2.sh small"
            ) from exc

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        if encoder not in model_configs:
            raise CvModelError(f"Unsupported Depth Anything V2 encoder: {encoder}")

        if torch_threads is not None and torch_threads > 0:
            torch.set_num_threads(torch_threads)

        self.model_id = str(resolved_model_path)
        self.input_size = input_size
        self.is_metric = is_metric
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DepthAnythingV2(**model_configs[encoder])
        checkpoint = torch.load(str(resolved_model_path), map_location="cpu")
        model.load_state_dict(checkpoint)
        self._model = model.to(self._device).eval()

    def infer_bgr(self, image_array: Any) -> DepthResult:
        try:
            import numpy as np
        except ImportError as exc:
            raise CvModelError("NumPy is required for depth inference") from exc

        started_at = time.perf_counter()
        with self._torch.no_grad():
            depth = self._model.infer_image(image_array, self.input_size)
        depth = np.asarray(depth, dtype="float32")

        return DepthResult(
            depth_map=depth,
            heatmap_image=_depth_to_heatmap_jpeg(depth),
            inference_ms=(time.perf_counter() - started_at) * 1000,
            backend="depth-anything-v2",
            model_id=self.model_id,
            input_size=self.input_size,
            is_metric=self.is_metric,
            min_depth=float(np.min(depth)),
            max_depth=float(np.max(depth)),
        )

    def infer(self, image_bytes: bytes) -> DepthResult:
        return self.infer_bgr(_decode_image_bgr(image_bytes))


class WarmSyntheticDepth:
    """Synthetic gradient depth used for loopback testing without torch."""

    def __init__(self, input_size: int, is_metric: bool) -> None:
        self.input_size = input_size
        self.is_metric = is_metric

    def infer_bgr(self, image_array: Any) -> DepthResult:
        try:
            import numpy as np
        except ImportError as exc:
            raise CvModelError("Synthetic depth backend requires NumPy") from exc

        started_at = time.perf_counter()
        height, width = image_array.shape[:2]
        y_gradient = np.linspace(0.0, 1.0, height, dtype="float32").reshape(height, 1)
        depth = np.repeat(y_gradient, width, axis=1)

        return DepthResult(
            depth_map=depth,
            heatmap_image=_depth_to_heatmap_jpeg(depth),
            inference_ms=(time.perf_counter() - started_at) * 1000,
            backend="synthetic",
            model_id="synthetic-gradient",
            input_size=self.input_size,
            is_metric=self.is_metric,
            min_depth=float(np.min(depth)),
            max_depth=float(np.max(depth)),
        )

    def infer(self, image_bytes: bytes) -> DepthResult:
        return self.infer_bgr(_decode_image_bgr(image_bytes))


def make_warm_depth_model(
    backend: str,
    model_path: str | None,
    encoder: str,
    input_size: int,
    is_metric: bool,
    torch_threads: int | None = None,
) -> WarmDepthAnythingV2 | WarmSyntheticDepth:
    if backend == "synthetic":
        return WarmSyntheticDepth(input_size=input_size, is_metric=is_metric)
    if backend == "depth-anything-v2":
        return WarmDepthAnythingV2(
            model_path=model_path,
            encoder=encoder,
            input_size=input_size,
            is_metric=is_metric,
            torch_threads=torch_threads,
        )
    raise CvModelError(f"Unsupported warm depth backend: {backend}")


def _run_depth_anything_v2(
    image_bytes: bytes,
    model_path: str | None,
    encoder: str,
    input_size: int,
    is_metric: bool,
) -> DepthResult:
    return WarmDepthAnythingV2(
        model_path=model_path,
        encoder=encoder,
        input_size=input_size,
        is_metric=is_metric,
    ).infer(image_bytes)


def _run_depth_anything_v3(
    image_bytes: bytes,
    model_id: str,
    input_size: int,
    is_metric: bool,
) -> DepthResult:
    started_at = time.perf_counter()
    try:
        import numpy as np
        import torch
        from PIL import Image
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise CvModelError(
            "Depth Anything 3 dependencies are missing. Install the official depth-anything-3 package first."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)

    with Image.open(BytesIO(image_bytes)) as image:
        with tempfile.TemporaryDirectory() as export_dir:
            prediction = model.inference(
                [image.convert("RGB")],
                export_dir=export_dir,
                export_format="mini_npz",
            )

    depth = np.asarray(prediction.depth[0], dtype="float32")
    return DepthResult(
        depth_map=depth,
        heatmap_image=_depth_to_heatmap_jpeg(depth),
        inference_ms=(time.perf_counter() - started_at) * 1000,
        backend="depth-anything-v3",
        model_id=model_id,
        input_size=input_size,
        is_metric=is_metric,
        min_depth=float(np.min(depth)),
        max_depth=float(np.max(depth)),
    )


def _decode_image_bgr(image_bytes: bytes):
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise CvModelError("NumPy and OpenCV are required for image decoding") from exc

    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise CvModelError("Failed to decode image bytes")
    return image


def _ensure_depth_anything_v2_import_path() -> None:
    if str(DEPTH_ANYTHING_V2_REPO_PATH) not in sys.path and DEPTH_ANYTHING_V2_REPO_PATH.exists():
        sys.path.insert(0, str(DEPTH_ANYTHING_V2_REPO_PATH))


def _resolve_depth_anything_v2_model_path(model_path: str | None) -> Path:
    resolved_model_path = Path(model_path) if model_path else DEFAULT_DEPTH_ANYTHING_V2_SMALL_PATH
    if not resolved_model_path.is_absolute():
        resolved_model_path = REPO_ROOT / resolved_model_path
    return resolved_model_path


def _depth_to_heatmap_jpeg(depth_map: Any) -> bytes:
    import cv2
    import numpy as np

    depth = depth_map.astype("float32")
    depth_min = float(np.min(depth))
    depth_max = float(np.max(depth))
    if depth_max - depth_min < 1e-8:
        normalized = np.zeros_like(depth, dtype=np.uint8)
    else:
        normalized = ((depth - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)

    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    ok, encoded = cv2.imencode(".jpg", heatmap)
    if not ok:
        raise CvModelError("Failed to encode depth heatmap")
    return encoded.tobytes()
