"""Step 3: TSDF fusion (Open3D) -> room mesh + carved occupancy floor plan.

Three things happen here that did not before, each aimed at a measured
failure of the previous version (session_20260724_144728: 37 097 connected
components, mesh bbox 12.08 x 4.85 x 15.08 m for a room the camera crossed
inside 2.23 x 3.87 m, floor plan a starburst of radial streaks):

1. **Depth is filtered before it is fused**, once, into
   `derived/depth_filtered/` — a single source of truth shared with the
   objects step, which used to lift objects out of the *raw* depth and so
   built them from different geometry than the mesh. See `depth_filter.py`
   for what the filters do and why.

2. **The truncation distance is derived from the data** instead of a fixed
   5 m. A 5 m cutoff around a 3.9 m camera path permits a 13.9 m mesh by
   construction, which is exactly the extent that was observed. After
   filtering, the surviving depth distribution says how far this room
   actually is.

3. **The mesh is cleaned.** TSDF emits every isolated blob it ever saw
   evidence for; without a connected-component pass a third of the
   triangles are debris.

The plan itself moved to `occupancy.py`: ray carving instead of a vertex
histogram, and a RANSAC floor fit instead of averaging the cameras' own
down axis (the IMU seam lives there too).
"""

from __future__ import annotations

import json
import logging
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from mac_server.scene3d import depth_filter, occupancy
from mac_server.scene3d.session_io import SceneSession


logger = logging.getLogger(__name__)


def adaptive_depth_trunc(
    samples, configured_max: float, percentile: float = 98.0, floor_m: float = 2.0
) -> float:
    """How far to trust depth in this particular room.

    `samples` are valid depth values that already survived filtering, so the
    tail is real surface rather than the model's guess about a far wall it
    cannot resolve. Never exceeds the configured maximum and never drops
    below `floor_m` (a genuinely small room must still fuse).
    """
    import numpy as np

    values = np.asarray(samples, dtype=np.float32)
    values = values[values > 0]
    if values.size < 100:
        return float(configured_max)
    p = float(np.percentile(values, percentile))
    return float(min(configured_max, max(floor_m, p)))


def largest_components_mask(cluster_ids, cluster_sizes, min_triangles: int):
    """Triangles to *remove*: those in components smaller than the threshold."""
    import numpy as np

    ids = np.asarray(cluster_ids)
    sizes = np.asarray(cluster_sizes)
    if sizes.size == 0:
        return np.zeros(ids.shape, dtype=bool)
    return sizes[ids] < min_triangles


class _DepthCache:
    """Small LRU over depth maps.

    The multi-view check needs a handful of neighbours per frame; holding
    every map of a 235-frame session would be ~1.3 GB on a machine with 8.6
    GB total, and reloading each one every time makes the step disk-bound.
    """

    def __init__(self, session: SceneSession, capacity: int = 16) -> None:
        self.session = session
        self.capacity = capacity
        self._items: OrderedDict[int, Any] = OrderedDict()

    def get(self, index: int):
        import numpy as np

        if index in self._items:
            self._items.move_to_end(index)
            return self._items[index]
        path = self.session.depth_path(index)
        if not path.exists():
            return None
        depth = np.load(path).astype(np.float32)
        self._items[index] = depth
        if len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return depth


def _filter_pass(
    session: SceneSession,
    poses: dict[int, Any],
    intr: dict[str, Any],
    filter_cfg: dict[str, Any],
    progress: Callable[[str], None],
) -> tuple[list[float], dict[str, Any]]:
    """Write `derived/depth_filtered/` and return depth samples + stats."""
    import numpy as np

    out_dir = session.depth_filtered_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = _DepthCache(session)
    neighbour_count = int(filter_cfg.get("neighbour_views", 6))
    min_views = int(filter_cfg.get("min_views", 2))
    rel_tol = float(filter_cfg.get("rel_tol", 0.06))
    max_angle = float(filter_cfg.get("max_angle_deg", 85.0))
    hard_max = float(filter_cfg.get("hard_max_depth_m", 8.0))

    samples: list[float] = []
    totals = {"valid_in": 0, "dropped_flying": 0, "dropped_inconsistent": 0,
              "valid_out": 0}
    rng = np.random.default_rng(0)
    processed = 0
    for kf_index in sorted(poses):
        depth = cache.get(kf_index)
        if depth is None:
            continue
        neighbour_ids = depth_filter.select_neighbours(
            poses, kf_index, count=neighbour_count
        )
        neighbours = []
        for nid in neighbour_ids:
            nd = cache.get(nid)
            if nd is not None:
                neighbours.append((nd, poses[nid]))

        filtered, stats = depth_filter.filter_depth(
            depth,
            intr,
            world_from_cam=poses[kf_index],
            neighbours=neighbours,
            max_angle_deg=max_angle,
            rel_tol=rel_tol,
            min_views=min_views,
            max_depth_m=hard_max,
        )
        # float16 halves the cache (a 235-frame session is ~600 MB instead of
        # 1.2 GB). Its ~0.4 cm resolution at 8 m is far below the 2 cm TSDF
        # voxel, so nothing downstream can tell the difference.
        np.save(session.depth_filtered_path(kf_index), filtered.astype(np.float16))
        for key in totals:
            totals[key] += int(stats.get(key, 0))

        valid = filtered[filtered > 0]
        if valid.size:
            take = min(valid.size, 2000)
            samples.extend(
                float(v) for v in rng.choice(valid, size=take, replace=False)
            )
        processed += 1
        if processed % 10 == 0:
            progress(f"filtering depth {processed}/{len(poses)}")

    totals["kept_frac"] = (
        round(totals["valid_out"] / totals["valid_in"], 4)
        if totals["valid_in"] else 0.0
    )
    totals["frames"] = processed
    return samples, totals


def _reuse_filtered(
    session: SceneSession, poses: dict[int, Any]
) -> tuple[list[float], dict[str, Any]]:
    """Depth samples from an existing filtered cache (skips the filter pass)."""
    import numpy as np

    rng = np.random.default_rng(0)
    samples: list[float] = []
    frames = 0
    for kf_index in sorted(poses):
        path = session.depth_filtered_path(kf_index)
        if not path.exists():
            continue
        valid = np.load(path).astype(np.float32)
        valid = valid[valid > 0]
        if valid.size:
            take = min(valid.size, 2000)
            samples.extend(float(v) for v in rng.choice(valid, size=take, replace=False))
        frames += 1
    return samples, {"frames": frames, "kept_frac": None, "cached": True}


def run_tsdf_step(
    session: SceneSession,
    config: dict[str, Any],
    repo_root: Path,
    progress: Callable[[str], None] = lambda msg: None,
) -> dict[str, Any]:
    import cv2
    import numpy as np
    import open3d as o3d

    tsdf_cfg = config.get("tsdf", {})
    filter_cfg = config.get("filter", {})
    # The camera bundle adjustment settled on — NOT the as-recorded guess.
    # Unprojecting with a different focal length than the poses were solved
    # for is the single largest source of geometric error here.
    intr = session.active_intrinsics(float(config.get("fallback_hfov_deg", 75.0)))
    poses = session.load_poses()
    frames = {kf.index: kf for kf in session.keyframes()}
    if not poses:
        raise RuntimeError("No poses — run the poses step first")

    # The filter pass is the expensive part of this step, so its output is
    # cached — but only reused when it was produced by the same settings and
    # the same camera. Tuning `filter.*` or re-running `poses` (which can
    # change the refined intrinsics) must invalidate it, otherwise the tuning
    # appears to do nothing.
    signature = {
        "filter": {k: filter_cfg.get(k) for k in sorted(filter_cfg)},
        "fx": round(float(intr["fx"]), 3),
        "frames": len(poses),
    }
    signature_path = session.depth_filtered_dir() / "filter_meta.json"
    previous = None
    if signature_path.exists():
        try:
            previous = json.loads(signature_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
    reuse = (
        previous == signature
        and not config.get("force")
        and all(session.depth_filtered_path(i).exists() for i in poses)
    )

    if reuse:
        progress("reusing cached filtered depth")
        samples, filter_stats = _reuse_filtered(session, poses)
    else:
        progress("filtering depth (flying pixels + multi-view consistency)")
        samples, filter_stats = _filter_pass(session, poses, intr, filter_cfg, progress)
        session.depth_filtered_dir().mkdir(parents=True, exist_ok=True)
        signature_path.write_text(json.dumps(signature, indent=1), encoding="utf-8")
    if not filter_stats.get("frames"):
        raise RuntimeError("No frames with both pose and depth — run earlier steps first")
    logger.info("Depth filter: %s", filter_stats)

    depth_trunc = adaptive_depth_trunc(
        samples,
        configured_max=float(tsdf_cfg.get("depth_trunc_m", 5.0)),
        percentile=float(tsdf_cfg.get("trunc_percentile", 98.0)),
    )
    logger.info("Adaptive depth truncation: %.2f m", depth_trunc)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(tsdf_cfg.get("voxel_m", 0.02)),
        sdf_trunc=float(tsdf_cfg.get("sdf_trunc_m", 0.08)),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    integrated = 0
    plan_hits: list[np.ndarray] = []
    plan_origins: list[np.ndarray] = []
    for kf_index, world_from_cam in sorted(poses.items()):
        kf = frames.get(kf_index)
        path = session.depth_filtered_path(kf_index)
        if kf is None or not path.exists():
            continue
        depth = np.load(path).astype(np.float32)
        depth[depth > depth_trunc] = 0.0
        if not depth.any():
            continue
        bgr = kf.rgb_bgr()
        if bgr.shape[:2] != depth.shape[:2]:
            bgr = cv2.resize(bgr, (depth.shape[1], depth.shape[0]))

        # Rescale the calibration to the resolution actually in hand; the
        # depth map is not required to be at capture resolution.
        k = depth_filter.intrinsics_for_shape(intr, depth.shape)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(k["width"]), int(k["height"]),
            float(k["fx"]), float(k["fy"]), float(k["cx"]), float(k["cy"]),
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(world_from_cam))

        # Keep a subsampled world point set per frame for the carved plan;
        # doing it here avoids a third pass over the depth maps. Note the
        # intrinsics are rescaled to the *subsampled* shape — striding an
        # image scales its focal length and principal point too.
        stride = int(tsdf_cfg.get("plan_point_stride", 6))
        small = depth[::stride, ::stride]
        pts_cam = depth_filter.unproject(
            small, depth_filter.intrinsics_for_shape(intr, small.shape)
        ).reshape(-1, 3)
        pts_cam = pts_cam[np.isfinite(pts_cam).all(axis=1)]
        if len(pts_cam):
            rot, trans = world_from_cam[:3, :3], world_from_cam[:3, 3]
            plan_hits.append((pts_cam @ rot.T + trans).astype(np.float32))
            plan_origins.append(trans.copy())

        integrated += 1
        if integrated % 10 == 0:
            progress(f"tsdf {integrated}/{len(poses)}")

    if integrated == 0:
        raise RuntimeError("No frames survived filtering — depth or poses are unusable")

    progress("extracting mesh")
    mesh = volume.extract_triangle_mesh()
    raw_triangles = len(np.asarray(mesh.triangles))

    # ------------------------------------------------------ mesh cleanup
    min_component = int(tsdf_cfg.get("min_component_triangles", 800))
    components_before = 0
    if min_component > 0 and raw_triangles:
        progress("cleaning mesh (connected components)")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            cluster_ids, cluster_sizes, _ = mesh.cluster_connected_triangles()
        components_before = len(np.asarray(cluster_sizes))
        remove = largest_components_mask(cluster_ids, cluster_sizes, min_component)
        if remove.any() and not remove.all():
            mesh.remove_triangles_by_mask(remove)
            mesh.remove_unreferenced_vertices()
        elif remove.all():
            logger.warning(
                "Every mesh component is smaller than min_component_triangles=%d; "
                "keeping the mesh uncleaned rather than deleting all of it.",
                min_component,
            )

    decimate_to = int(tsdf_cfg.get("decimate_triangles", 0))
    if decimate_to and len(np.asarray(mesh.triangles)) > decimate_to:
        progress(f"decimating mesh to ~{decimate_to} triangles")
        mesh = mesh.simplify_quadric_decimation(decimate_to)

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(session.mesh_path()), mesh)
    vertices = np.asarray(mesh.vertices)

    # -------------------------------------------------- gravity + plan
    progress("estimating gravity and carving the floor plan")
    imu_up = session.imu_up_vector(poses)
    up, gravity_source = occupancy.estimate_gravity(
        vertices if len(vertices) else None, poses, imu_up=imu_up
    )
    logger.info("Up axis from %s: %s", gravity_source, np.round(up, 4).tolist())
    axis_a, axis_b = occupancy.plan_axes(up)

    heights = vertices @ up if len(vertices) else np.zeros(0)
    floor_offset = float(np.percentile(heights, 2)) if heights.size else 0.0

    band_lo, band_hi = tuple(tsdf_cfg.get("plan_height_band_m", [0.15, 1.9]))
    all_hits = np.concatenate(plan_hits, axis=0) if plan_hits else np.zeros((0, 3))
    hit_heights = all_hits @ up - floor_offset if len(all_hits) else np.zeros(0)
    resolution = float(tsdf_cfg.get("plan_resolution_m", 0.04))
    spec = occupancy.make_grid_spec(
        np.stack([all_hits @ axis_a, all_hits @ axis_b], axis=1)
        if len(all_hits) else np.zeros((0, 2)),
        resolution_m=resolution,
    )
    free_counts = np.zeros((spec["height"], spec["width"]), dtype=np.int32)
    hit_counts = np.zeros_like(free_counts)
    cursor = 0
    for pts, origin in zip(plan_hits, plan_origins):
        band = slice(cursor, cursor + len(pts))
        in_band = (hit_heights[band] >= band_lo) & (hit_heights[band] <= band_hi)
        cursor += len(pts)
        selected = pts[in_band]
        if not len(selected):
            continue
        occupancy.carve_rays(
            np.array([origin @ axis_a, origin @ axis_b]),
            np.stack([selected @ axis_a, selected @ axis_b], axis=1),
            spec, free_counts, hit_counts,
        )
    grid = occupancy.occupancy_from_counts(
        free_counts, hit_counts,
        min_hits=int(tsdf_cfg.get("plan_min_hits", 3)),
        min_free=int(tsdf_cfg.get("plan_min_free", 2)),
    )
    walls = occupancy.extract_walls(grid, min_cluster=int(tsdf_cfg.get("wall_min_cells", 12)))

    # Манхэттенская привязка курса (руководство §13.2). Комнаты почти всегда
    # прямоугольны, поэтому направления стен кучкуются кратно 90 градусам, и
    # отклонение этой кучки от осей плана говорит, насколько курс разъехался с
    # геометрией здания.
    #
    # Это единственная поправка курса, которая НЕ накапливает ошибку: она
    # наблюдает геометрию, а не интегрирует показания. Дрейф yaw у BNO085
    # 0.003 град/мин по приёмочному тесту B — мало, но за получасовую сессию
    # это уже градус, и его нечем было убрать.
    #
    # Считается здесь, а не в graph: сетка занятости строится именно тут, и
    # стены в ней уже выделены. Записывается как ФАКТ о сцене, применять его к
    # курсу или нет — решает навигация.
    from mac_server.scene3d.occupancy import wall_azimuths

    from shared.manhattan import manhattan_is_usable, manhattan_yaw_offset

    segments = wall_azimuths(walls)
    yaw_offset_rad, yaw_strength = manhattan_yaw_offset([a for a, _ in segments])
    manhattan = {
        "wall_segments": len(segments),
        "yaw_offset_deg": round(math.degrees(yaw_offset_rad), 3),
        "strength": round(float(yaw_strength), 3),
        "usable": manhattan_is_usable(yaw_strength, len(segments)),
    }
    logger.info("Manhattan: %s", manhattan)
    cv2.imwrite(str(session.plan_path()), occupancy.render_plan(grid, walls))
    np.savez_compressed(
        session.occupancy_path(), grid=grid, walls=walls,
        free_counts=free_counts, hit_counts=hit_counts,
    )

    # Same fields the three.js viewer and the graph step already consume,
    # plus where the up vector came from.
    (session.derived / "plan_frame.json").write_text(
        json.dumps(
            {
                "up": up.tolist(),
                "axis_a": axis_a.tolist(),
                "axis_b": axis_b.tolist(),
                "origin": list(spec["origin"]),
                "resolution_m": spec["resolution_m"],
                "grid_w": int(spec["width"]),
                "grid_h": int(spec["height"]),
                "floor_offset": floor_offset,
                "gravity_source": gravity_source,
                # Привязка курса к стенам комнаты. Навигация читает её отсюда
                # вместе с системой координат плана — иначе поправку пришлось
                # бы искать в отчёте шага, который к тому моменту уже забыт.
                "manhattan": manhattan,
            }
        ),
        encoding="utf-8",
    )

    extent = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    cam_positions = np.stack([m[:3, 3] for m in poses.values()])
    cam_extent = cam_positions.max(axis=0) - cam_positions.min(axis=0)
    report = {
        "frames_integrated": integrated,
        "vertices": len(vertices),
        "triangles": int(len(np.asarray(mesh.triangles))),
        "triangles_before_cleanup": raw_triangles,
        "components_before_cleanup": components_before,
        "bbox_extent_m": [round(float(v), 2) for v in extent],
        "camera_extent_m": [round(float(v), 2) for v in cam_extent],
        "depth_trunc_m": round(depth_trunc, 2),
        "depth_kept_frac": filter_stats.get("kept_frac"),
        "gravity_source": gravity_source,
        "manhattan": manhattan,
        "free_cells": int((grid == occupancy.FREE).sum()),
        "occupied_cells": int((grid == occupancy.OCCUPIED).sum()),
    }
    room_span = float(max(extent))
    path_span = float(max(cam_extent))
    if room_span > 4.0 * max(path_span, 0.5):
        report["warning"] = (
            f"Mesh spans {room_span:.1f} m while the camera only moved {path_span:.1f} m. "
            "That ratio usually means depth is still being trusted too far out "
            "(lower tsdf.depth_trunc_m) or the poses are wrong (check the poses "
            "step's sub_reconstructions and hfov_deg)."
        )
    logger.info("TSDF done: %s", report)
    return report
