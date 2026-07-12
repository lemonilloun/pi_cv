"""High-level CV experiment pipeline packaging."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4
import zipfile
from io import BytesIO

from pi_client.camera import capture_camera_frame
from pi_client.cv_models import (
    CvModelError,
    attach_depth_to_detections,
    image_size,
    render_combined_image,
    run_depth,
    run_yolo,
)


def load_source_image(source: str, image_path: Path | None, width: int, height: int) -> tuple[bytes, dict[str, Any]]:
    if source == "camera":
        frame = capture_camera_frame(width=width, height=height, image_format="jpeg")
        return frame.data, {
            "source": "camera",
            "frame_id": frame.frame_id,
            "width": frame.width,
            "height": frame.height,
            "format": frame.image_format,
        }

    if source == "image":
        if image_path is None:
            raise CvModelError("--image is required when --source image")
        return image_path.read_bytes(), {
            "source": "image",
            "path": str(image_path),
            "filename": image_path.name,
        }

    raise CvModelError(f"Unsupported source: {source}")


def build_cv_package(
    mode: str,
    device_id: str,
    image_bytes: bytes,
    source_metadata: dict[str, Any],
    yolo_model: str | None,
    yolo_task: str,
    yolo_confidence: float,
    depth_backend: str,
    depth_model_path: str | None,
    depth_encoder: str,
    depth_input_size: int,
    depth_is_metric: bool,
) -> tuple[str, bytes, dict[str, Any]]:
    run_id = str(uuid4())
    started_at = time.perf_counter()
    artifacts: dict[str, bytes] = {"original.jpg": image_bytes}
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "device_id": device_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source_metadata,
        "models": {},
        "objects": [],
        "artifacts": sorted(artifacts.keys()),
    }

    yolo_result = None
    depth_result = None

    if mode in {"yolo", "pipeline"}:
        if not yolo_model:
            raise CvModelError("--yolo-model is required for yolo and pipeline modes")
        yolo_result = run_yolo(
            image_bytes=image_bytes,
            model_path=yolo_model,
            confidence=yolo_confidence,
            task=yolo_task,
        )
        artifacts["yolo_annotated.jpg"] = yolo_result.annotated_image
        metadata["models"]["yolo"] = {
            "model_path": yolo_result.model_path,
            "task": yolo_result.task,
            "confidence_threshold": yolo_confidence,
            "inference_ms": yolo_result.inference_ms,
        }
        metadata["objects"] = [asdict(detection) for detection in yolo_result.detections]

    if mode in {"depth", "pipeline"}:
        depth_result = run_depth(
            image_bytes=image_bytes,
            backend=depth_backend,
            model_path_or_id=depth_model_path,
            encoder=depth_encoder,
            input_size=depth_input_size,
            is_metric=depth_is_metric,
        )
        artifacts["depth_heatmap.jpg"] = depth_result.heatmap_image
        metadata["models"]["depth"] = {
            "backend": depth_result.backend,
            "model_id": depth_result.model_id,
            "encoder": depth_encoder,
            "input_size": depth_result.input_size,
            "is_metric": depth_result.is_metric,
            "inference_ms": depth_result.inference_ms,
            "min_depth": depth_result.min_depth,
            "max_depth": depth_result.max_depth,
        }

    if mode == "pipeline":
        if yolo_result is None or depth_result is None:
            raise CvModelError("Pipeline mode requires both YOLO and depth results")

        width, height = image_size(image_bytes)
        detections_with_depth = attach_depth_to_detections(
            detections=yolo_result.detections,
            depth_map=depth_result.depth_map,
            image_width=width,
            image_height=height,
            is_metric=depth_is_metric,
        )
        metadata["objects"] = [asdict(detection) for detection in detections_with_depth]
        artifacts["combined.jpg"] = render_combined_image(image_bytes, detections_with_depth)

    metadata["elapsed_ms"] = (time.perf_counter() - started_at) * 1000
    metadata["artifacts"] = sorted([*artifacts.keys(), "metadata.json"])

    package_bytes = _zip_artifacts(artifacts, metadata)
    summary = {
        "mode": mode,
        "object_count": len(metadata["objects"]),
        "artifacts": metadata["artifacts"],
        "elapsed_ms": metadata["elapsed_ms"],
    }
    return run_id, package_bytes, summary


def _zip_artifacts(artifacts: dict[str, bytes], metadata: dict[str, Any]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        for name, data in artifacts.items():
            archive.writestr(name, data)
    return buffer.getvalue()
