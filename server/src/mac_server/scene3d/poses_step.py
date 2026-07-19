"""Step 2: camera poses via pycolmap (sequential SfM) + metric scale.

COLMAP poses are up-to-scale; Depth Anything V2 metric gives meters. The
scale factor between the two worlds is estimated per frame as
median(d_metric / d_colmap) over the frame's visible sparse points, then
globally as the median over frames (MAD-based outlier rejection). All
translations are multiplied by the global scale -> metric poses.

Outputs:
    derived/colmap/           database + sparse model (kept for debugging)
    derived/poses.json        {"world_from_cam": {kf_index: 4x4}, ...}
    derived/scale_report.json per-frame scales + spread diagnostics
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable

from mac_server.scene3d.session_io import SceneSession


logger = logging.getLogger(__name__)


def median_scale(per_frame_scales: dict[int, float]) -> tuple[float, float, list[int]]:
    """Global scale from per-frame scales with MAD outlier rejection.
    Returns (scale, iqr_over_median, rejected_frames)."""
    import numpy as np

    frames = sorted(per_frame_scales)
    values = np.asarray([per_frame_scales[f] for f in frames], dtype=np.float64)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) or 1e-9
    keep = np.abs(values - med) <= 3.0 * 1.4826 * mad
    rejected = [f for f, k in zip(frames, keep) if not k]
    kept = values[keep]
    scale = float(np.median(kept)) if kept.size else med
    q25, q75 = np.percentile(kept if kept.size else values, [25, 75])
    iqr_over_median = float((q75 - q25) / scale) if scale else float("inf")
    return scale, iqr_over_median, rejected


def frame_scale(
    depth_map,
    points_uv,
    points_colmap_depth,
    min_depth_m: float = 0.2,
    max_depth_m: float = 10.0,
) -> float | None:
    """One frame's metric/COLMAP scale: median over sparse points of
    (DAv2 depth at the point's pixel) / (COLMAP depth)."""
    import numpy as np

    h, w = depth_map.shape[:2]
    ratios = []
    for (u, v), dc in zip(points_uv, points_colmap_depth):
        if dc <= 1e-6:
            continue
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < w and 0 <= y < h):
            continue
        dm = float(depth_map[y, x])
        if not (min_depth_m <= dm <= max_depth_m):
            continue
        ratios.append(dm / dc)
    if len(ratios) < 10:
        return None
    return float(np.median(ratios))


def run_poses_step(
    session: SceneSession,
    config: dict[str, Any],
    repo_root: Path,
    progress: Callable[[str], None] = lambda msg: None,
) -> dict[str, Any]:
    import numpy as np
    import pycolmap

    poses_cfg = config.get("poses", {})
    colmap_dir = session.colmap_dir()
    if colmap_dir.exists():
        shutil.rmtree(colmap_dir)  # rerun = clean run; COLMAP dbs don't append well
    colmap_dir.mkdir(parents=True, exist_ok=True)
    image_dir = colmap_dir / "images"
    image_dir.mkdir()

    frames = session.keyframes()
    for kf in frames:
        # COLMAP wants a flat image dir; hardlink to avoid copying JPEGs.
        target = image_dir / f"{kf.index:06d}.jpg"
        try:
            target.hardlink_to(kf.rgb_path)
        except OSError:
            shutil.copy2(kf.rgb_path, target)

    intr = session.intrinsics(float(config.get("fallback_hfov_deg", 102.0)))
    # OPENCV model consumes the calibration distortion directly.
    camera_params = ",".join(
        str(float(v))
        for v in (
            intr["fx"], intr["fy"], intr["cx"], intr["cy"],
            *(intr.get("dist") or [0, 0, 0, 0])[:4],
        )
    )

    database = colmap_dir / "database.db"
    progress("colmap: extracting features")
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "OPENCV"
    reader_options.camera_params = camera_params
    pycolmap.extract_features(
        database, image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
    )

    progress("colmap: sequential matching")
    pairing_options = pycolmap.SequentialPairingOptions()
    pairing_options.overlap = int(poses_cfg.get("sequential_overlap", 10))
    pycolmap.match_sequential(database, pairing_options=pairing_options)

    progress("colmap: incremental mapping")
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    pipeline_options = pycolmap.IncrementalPipelineOptions()
    pipeline_options.ba_refine_focal_length = bool(intr.get("estimated", False))
    reconstructions = pycolmap.incremental_mapping(
        database, image_dir, sparse_dir, options=pipeline_options
    )
    if not reconstructions:
        raise RuntimeError(
            "COLMAP registered nothing — too little texture/overlap. "
            "Record slower with more overlap between views."
        )
    rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    logger.info("COLMAP: %d/%d images registered, %d points",
                rec.num_reg_images(), len(frames), rec.num_points3D())

    # ------------------------------------------------- scale alignment
    progress("aligning metric scale")
    per_frame_scales: dict[int, float] = {}
    world_from_cam_unscaled: dict[int, np.ndarray] = {}
    for image in rec.images.values():
        kf_index = int(Path(image.name).stem)
        # pycolmap API drift: cam_from_world is a property (Rigid3d) in some
        # versions and an instance method in others.
        pose = image.cam_from_world
        if callable(pose):
            pose = pose()
        cam_from_world = pose.matrix()  # 3x4
        R, t = cam_from_world[:, :3], cam_from_world[:, 3]

        wfc = np.eye(4)
        wfc[:3, :3] = R.T
        wfc[:3, 3] = -R.T @ t
        world_from_cam_unscaled[kf_index] = wfc

        depth_path = session.depth_path(kf_index)
        if not depth_path.exists():
            continue
        depth_map = np.load(depth_path)
        uv, dc = [], []
        for p2d in image.points2D:
            if not p2d.has_point3D():
                continue
            xyz = rec.points3D[p2d.point3D_id].xyz
            z = float((R @ xyz + t)[2])
            uv.append(tuple(p2d.xy))
            dc.append(z)
        scale = frame_scale(depth_map, uv, dc)
        if scale is not None:
            per_frame_scales[kf_index] = scale

    if len(per_frame_scales) < 3:
        raise RuntimeError(
            f"Scale alignment failed: only {len(per_frame_scales)} frames had "
            "usable sparse depth (run the depth step first?)"
        )
    scale, iqr_over_median, rejected = median_scale(per_frame_scales)
    logger.info("Metric scale %.4f (IQR/median %.3f, %d frames rejected)",
                scale, iqr_over_median, len(rejected))

    world_from_cam = {}
    for kf_index, wfc in world_from_cam_unscaled.items():
        scaled = wfc.copy()
        scaled[:3, 3] *= scale
        world_from_cam[kf_index] = scaled

    session.derived.mkdir(parents=True, exist_ok=True)
    session.poses_path().write_text(
        json.dumps(
            {
                "world_from_cam": {
                    str(k): v.tolist() for k, v in sorted(world_from_cam.items())
                },
                "registered": rec.num_reg_images(),
                "total": len(frames),
                "scale": scale,
            }
        ),
        encoding="utf-8",
    )
    session.scale_report_path().write_text(
        json.dumps(
            {
                "global_scale": scale,
                "iqr_over_median": round(iqr_over_median, 4),
                "rejected_frames": rejected,
                "per_frame": {str(k): round(v, 4) for k, v in sorted(per_frame_scales.items())},
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    registered_frac = rec.num_reg_images() / max(1, len(frames))
    report = {
        "registered": rec.num_reg_images(),
        "total": len(frames),
        "registered_frac": round(registered_frac, 3),
        "scale": round(scale, 4),
        "scale_iqr_over_median": round(iqr_over_median, 3),
    }
    min_frac = float(poses_cfg.get("min_registered_frac", 0.5))
    if registered_frac < min_frac:
        report["warning"] = (
            f"Only {registered_frac:.0%} of keyframes registered (target ≥ {min_frac:.0%}) "
            "— bare walls/fast motion; the mesh will have holes."
        )
    return report
