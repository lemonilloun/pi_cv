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
import math
import shutil
from pathlib import Path
from typing import Any, Callable

from mac_server.scene3d.session_io import SceneSession


logger = logging.getLogger(__name__)


def resolve_matching_mode(requested: str, num_frames: int, max_exhaustive: int) -> str:
    """Resolve 'auto' to a concrete mode. Exhaustive matching closes loops
    for free but is O(N^2), so 'auto' uses it up to max_exhaustive frames and
    falls back to sequential+loop above that. Any explicit mode passes through."""
    mode = str(requested or "auto").lower()
    if mode != "auto":
        return mode
    return "exhaustive" if num_frames <= max_exhaustive else "sequential+loop"


def _run_matching(
    database: Path,
    poses_cfg: dict[str, Any],
    num_frames: int,
    progress: Callable[[str], None],
) -> str:
    """Feature matching. Sequential-only matching never compares two
    temporally-distant frames, so a loop (walk around the room and come
    back) is never closed and the two ends of the trajectory drift apart
    into separate sub-reconstructions. Exhaustive matching compares every
    pair, so any revisit is found — affordable for a few hundred frames and
    the single biggest lever on reconstruction quality here.

    Returns the mode actually used (for the report)."""
    import pycolmap

    mode = resolve_matching_mode(
        poses_cfg.get("matching_mode", "auto"),
        num_frames,
        int(poses_cfg.get("max_exhaustive_frames", 600)),
    )

    if mode == "exhaustive":
        progress(f"colmap: exhaustive matching ({num_frames} frames, all pairs)")
        pycolmap.match_exhaustive(database)
        return "exhaustive"

    # sequential, optionally with vocab-tree loop detection
    progress("colmap: sequential matching")
    pairing_options = pycolmap.SequentialPairingOptions()
    pairing_options.overlap = int(poses_cfg.get("sequential_overlap", 10))

    used = "sequential"
    if mode == "sequential+loop":
        vocab_tree = poses_cfg.get("vocab_tree_path")
        have_tree = bool(vocab_tree) and Path(vocab_tree).expanduser().exists()
        # loop_detection / vocab_tree_path exist only on pycolmap builds that
        # support it; guard with hasattr so this degrades on older versions.
        if have_tree and hasattr(pairing_options, "loop_detection"):
            pairing_options.loop_detection = True
            if hasattr(pairing_options, "vocab_tree_path"):
                pairing_options.vocab_tree_path = str(Path(vocab_tree).expanduser())
            used = "sequential+loop"
            progress("colmap: sequential matching + vocab-tree loop detection")
        else:
            logger.warning(
                "sequential+loop requested but no usable vocab tree "
                "(poses.vocab_tree_path=%r) — falling back to plain sequential; "
                "consider matching_mode=exhaustive for loop closure.",
                vocab_tree,
            )

    pycolmap.match_sequential(database, pairing_options=pairing_options)
    return used


def camera_to_intrinsics(camera, dist: list[float] | None = None) -> dict[str, Any]:
    """A pycolmap Camera -> our intrinsics dict.

    Duck-typed on purpose so it can be unit-tested without pycolmap. Newer
    pycolmap exposes focal_length_x/principal_point_x accessors; older builds
    only have the raw `params` vector, whose layout depends on the model
    (OPENCV: fx, fy, cx, cy, k1, k2, p1, p2).
    """
    width = int(getattr(camera, "width", 0))
    height = int(getattr(camera, "height", 0))
    params = [float(v) for v in (getattr(camera, "params", None) or [])]

    def accessor(name: str):
        value = getattr(camera, name, None)
        if callable(value):
            value = value()
        return None if value is None else float(value)

    fx = accessor("focal_length_x")
    fy = accessor("focal_length_y")
    cx = accessor("principal_point_x")
    cy = accessor("principal_point_y")
    if fx is None and len(params) >= 4:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    if fx is None:
        raise ValueError(f"Cannot read intrinsics off camera: {camera!r}")
    if fy is None:
        fy = fx

    return {
        "width": width,
        "height": height,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx if cx is not None else width / 2.0),
        "cy": float(cy if cy is not None else height / 2.0),
        "dist": list(dist or params[4:8] or [0.0, 0.0, 0.0, 0.0]),
        "source": "colmap_refined",
    }


def hfov_deg(fx: float, width: float) -> float:
    """Horizontal field of view implied by a focal length, for reporting.
    Two intrinsics disagreeing is much easier to see in degrees than in
    pixels."""
    if fx <= 0 or width <= 0:
        return 0.0
    return 2.0 * math.degrees(math.atan((width / 2.0) / fx))


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

    intr = session.intrinsics(float(config.get("fallback_hfov_deg", 75.0)))
    if intr.get("estimated"):
        logger.warning(
            "Session %s has no intrinsics.json — falling back to a %.1f deg HFOV "
            "guess (fx=%.0f). Bundle adjustment will refine it, but a real "
            "calibration is worth more than any of the tuning below: run "
            "scripts/run_scene_calibrate.sh on the Pi (fixed LensPosition) so "
            "config/scene_intrinsics.json exists before the next recording.",
            session.session_id, intr.get("fallback_hfov_deg", 0.0), intr["fx"],
        )
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

    matching_used = _run_matching(database, poses_cfg, len(frames), progress)

    progress("colmap: incremental mapping")
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    pipeline_options = pycolmap.IncrementalPipelineOptions()
    pipeline_options.ba_refine_focal_length = bool(intr.get("estimated", False))
    # Force COLMAP single-threaded. torch and pycolmap wheels each bundle
    # their own libomp.dylib; the process runs with KMP_DUPLICATE_LIB_OK=TRUE
    # (see run_server.sh) so both load without abort(), but COLMAP's
    # multi-threaded bundle adjustment then DEADLOCKS against torch's OpenMP
    # runtime (~0% CPU, threads parked). Single-threaded BA never forks the
    # conflicting parallel team — slower but it actually finishes. Bump
    # poses.colmap_num_threads only in an environment with one shared libomp.
    n_threads = int(poses_cfg.get("colmap_num_threads", 1))
    if hasattr(pipeline_options, "num_threads"):
        pipeline_options.num_threads = n_threads
    mapper_opts = getattr(pipeline_options, "mapper", None)
    if mapper_opts is not None and hasattr(mapper_opts, "num_threads"):
        mapper_opts.num_threads = n_threads
    reconstructions = pycolmap.incremental_mapping(
        database, image_dir, sparse_dir, options=pipeline_options
    )
    if not reconstructions:
        raise RuntimeError(
            "COLMAP registered nothing — too little texture/overlap. "
            "Record slower with more overlap between views."
        )

    # COLMAP returns one sub-reconstruction per connected component of the
    # match graph. When a loop doesn't close, "before the turn" and "after
    # the turn" land in separate components — we keep the largest and must
    # NOT pretend the rest never existed (that's the "other side of the room
    # vanished" symptom). Surface fragmentation loudly so it's actionable.
    frags = sorted(
        (r.num_reg_images() for r in reconstructions.values()), reverse=True
    )
    rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    logger.info(
        "COLMAP: %d/%d images registered, %d points, %d sub-reconstruction(s) %s",
        rec.num_reg_images(), len(frames), rec.num_points3D(), len(frags), frags,
    )

    # ------------------------------------------- the camera BA settled on
    # Bundle adjustment refines the focal length whenever we admitted the
    # calibration was a guess, and from that moment the poses mean nothing
    # except relative to the refined camera. Persisting it is what keeps the
    # TSDF and object steps unprojecting into the same geometry the poses
    # describe; before this, they re-derived the *original* guess and quietly
    # reconstructed a different room.
    refined = None
    try:
        cameras = list(rec.cameras.values())
        if cameras:
            refined = camera_to_intrinsics(cameras[0], dist=intr.get("dist"))
            if not refined["width"]:
                refined["width"], refined["height"] = intr["width"], intr["height"]
            session.derived.mkdir(parents=True, exist_ok=True)
            session.refined_intrinsics_path().write_text(
                json.dumps(refined, indent=1), encoding="utf-8"
            )
            before = hfov_deg(float(intr["fx"]), float(intr["width"]))
            after = hfov_deg(refined["fx"], refined["width"] or intr["width"])
            logger.info(
                "COLMAP camera: fx %.1f -> %.1f (HFOV %.1f -> %.1f deg)",
                float(intr["fx"]), refined["fx"], before, after,
            )
    except Exception:
        logger.exception("Could not read the refined camera; downstream steps "
                         "will fall back to the as-recorded intrinsics")

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
                # Poses are only meaningful together with the camera that
                # produced them — travel as one artifact.
                "intrinsics": refined or intr,
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
        "matching": matching_used,
        "sub_reconstructions": frags,
    }
    warnings = []
    if refined:
        report["hfov_deg"] = round(hfov_deg(refined["fx"], refined["width"]), 1)
        report["hfov_deg_assumed"] = round(
            hfov_deg(float(intr["fx"]), float(intr["width"])), 1
        )
        drift = abs(refined["fx"] - float(intr["fx"])) / max(float(intr["fx"]), 1e-6)
        if drift > 0.10:
            warnings.append(
                f"Bundle adjustment moved the focal length by {drift:.0%} "
                f"(HFOV {report['hfov_deg_assumed']}° assumed → "
                f"{report['hfov_deg']}° measured). The refined camera is now used "
                "downstream, so the reconstruction is consistent — but a real "
                "calibration (scripts/run_scene_calibrate.sh on the Pi) would give "
                "SfM a correct starting point and a better-conditioned solve."
            )
    min_frac = float(poses_cfg.get("min_registered_frac", 0.5))
    if registered_frac < min_frac:
        warnings.append(
            f"Only {registered_frac:.0%} of keyframes registered (target ≥ {min_frac:.0%}) "
            "— bare walls/fast motion; the mesh will have holes."
        )
    if len(frags) > 1:
        dropped = sum(frags[1:])
        warnings.append(
            f"COLMAP split into {len(frags)} sub-reconstructions {frags}; kept the "
            f"largest ({frags[0]} frames), dropped {dropped} frames in other pieces. "
            "The loop didn't close — the dropped frames are a part of the room that "
            "never stitched. Try matching_mode=exhaustive and re-record with a clear "
            "return to the starting viewpoint."
        )
    if warnings:
        report["warning"] = " ".join(warnings)
    return report
