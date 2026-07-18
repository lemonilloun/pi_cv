"""Step 1: metric depth per keyframe (Depth Anything V2 metric on MPS).

Reuses the already-set-up server CV stack (external/Depth-Anything-V2 +
models/depth_anything_v2_metric_hypersim_vits.pth from setup_server_cv.sh).
Writes derived/depth/<idx>.npy (float32 meters) + a small preview png.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from mac_server.scene3d.session_io import SceneSession


logger = logging.getLogger(__name__)


def run_depth_step(
    session: SceneSession,
    config: dict[str, Any],
    repo_root: Path,
    progress: Callable[[str], None] = lambda msg: None,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    from mac_server.cv_worker import WarmMetricDepthAnythingV2

    depth_cfg = config.get("depth", {})
    model_path = Path(depth_cfg.get("model_path", "models/depth_anything_v2_metric_hypersim_vits.pth"))
    if not model_path.is_absolute():
        model_path = repo_root / model_path

    model = WarmMetricDepthAnythingV2(
        model_path=model_path,
        encoder=str(depth_cfg.get("encoder", "vits")),
        input_size=int(depth_cfg.get("input_size", 518)),
        max_depth_m=float(depth_cfg.get("max_depth_m", 20.0)),
        device=str(depth_cfg.get("device", "mps")),
        metric_repo_dir=repo_root / "external/Depth-Anything-V2/metric_depth",
    )

    out_dir = session.depth_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = session.keyframes()
    done = 0
    total_ms = 0.0
    for kf in frames:
        out_path = session.depth_path(kf.index)
        if out_path.exists():
            done += 1
            continue
        depth, ms = model.infer_bgr(kf.rgb_bgr())
        total_ms += ms
        np.save(out_path, depth.astype(np.float32))
        # Small preview for the panel (fixed 0..8m scale, like the live view).
        vis = np.clip(depth / 8.0, 0, 1)
        vis = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(out_dir / f"{kf.index:06d}_vis.jpg"), cv2.resize(vis, (384, 216)))
        done += 1
        if done % 10 == 0 or done == len(frames):
            progress(f"depth {done}/{len(frames)}")
    model.clear_device_cache()

    report = {
        "frames": len(frames),
        "avg_ms": round(total_ms / max(1, done), 1) if total_ms else None,
    }
    logger.info("Depth step done: %s", report)
    return report
