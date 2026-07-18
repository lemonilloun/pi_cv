"""Step 3: TSDF fusion (Open3D) -> room mesh + top-down floor plan.

Depth hygiene before integration (monocular depth is soft at object
borders): median blur, truncation beyond depth_trunc_m, and erosion of the
depth support near instance-mask edges so boundary "smearing" doesn't fuse
phantom geometry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from mac_server.scene3d.session_io import SceneSession


logger = logging.getLogger(__name__)


def make_floor_plan(
    points_xz,
    heights_y,
    resolution_m: float = 0.03,
    height_band: tuple[float, float] = (0.15, 1.9),
):
    """Top-down occupancy image from mesh vertices. The world frame comes
    from COLMAP (gravity unknown) — callers pass Y-up-corrected heights."""
    import numpy as np

    band = (heights_y >= height_band[0]) & (heights_y <= height_band[1])
    pts = points_xz[band]
    if len(pts) < 100:
        pts = points_xz  # degenerate orientation — show everything
    x0, z0 = pts.min(axis=0) - 0.3
    x1, z1 = pts.max(axis=0) + 0.3
    w = max(64, int((x1 - x0) / resolution_m))
    h = max(64, int((z1 - z0) / resolution_m))
    grid = np.zeros((h, w), dtype=np.float32)
    ix = np.clip(((pts[:, 0] - x0) / resolution_m).astype(int), 0, w - 1)
    iz = np.clip(((pts[:, 1] - z0) / resolution_m).astype(int), 0, h - 1)
    np.add.at(grid, (iz, ix), 1.0)
    grid = np.log1p(grid)
    grid = (255 * grid / max(1e-6, grid.max())).astype(np.uint8)
    origin = (float(x0), float(z0))
    return grid, origin, resolution_m


def estimate_up_axis(world_from_cam: dict[int, Any]):
    """Rooms are scanned with a roughly level camera, so the average camera
    'down' (+Y of the camera frame) approximates gravity. Returns the world
    up unit vector."""
    import numpy as np

    downs = [wfc[:3, 1] for wfc in world_from_cam.values()]  # cam +Y is down
    down = np.mean(downs, axis=0)
    norm = np.linalg.norm(down)
    if norm < 1e-6:
        return np.array([0.0, -1.0, 0.0])
    return -(down / norm)


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
    intr = session.intrinsics(float(config.get("fallback_hfov_deg", 102.0)))
    poses = session.load_poses()
    frames = {kf.index: kf for kf in session.keyframes()}

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        int(intr["width"]), int(intr["height"]),
        float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"]),
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(tsdf_cfg.get("voxel_m", 0.02)),
        sdf_trunc=float(tsdf_cfg.get("sdf_trunc_m", 0.08)),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    depth_trunc = float(tsdf_cfg.get("depth_trunc_m", 5.0))
    erode_px = int(tsdf_cfg.get("mask_edge_erode_px", 6))
    integrated = 0
    for kf_index, world_from_cam in sorted(poses.items()):
        kf = frames.get(kf_index)
        depth_path = session.depth_path(kf_index)
        if kf is None or not depth_path.exists():
            continue
        bgr = kf.rgb_bgr()
        depth = np.load(depth_path).astype(np.float32)
        depth = cv2.medianBlur(depth, 5)
        depth[depth > depth_trunc] = 0.0

        # Kill depth near instance-mask borders (monocular halo artifacts).
        if erode_px > 0 and kf.masks_path.exists():
            masks = kf.masks()
            edges = cv2.morphologyEx(
                (masks > 0).astype(np.uint8),
                cv2.MORPH_GRADIENT,
                np.ones((3, 3), np.uint8),
            )
            edges = cv2.dilate(edges, np.ones((erode_px, erode_px), np.uint8))
            depth[edges > 0] = 0.0

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(world_from_cam))
        integrated += 1
        if integrated % 10 == 0:
            progress(f"tsdf {integrated}/{len(poses)}")

    if integrated == 0:
        raise RuntimeError("No frames with both pose and depth — run earlier steps first")

    progress("extracting mesh")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(session.mesh_path()), mesh)

    # Floor plan in an up-aligned frame.
    up = estimate_up_axis(poses)
    vertices = np.asarray(mesh.vertices)
    heights = vertices @ up
    heights -= np.percentile(heights, 2)  # floor ≈ 0
    # Two horizontal axes orthogonal to `up`.
    a = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 0.1:
        a = np.cross(up, [0.0, 0.0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    plan_xy = np.stack([vertices @ a, vertices @ b], axis=1)

    grid, origin, resolution = make_floor_plan(
        plan_xy, heights,
        resolution_m=float(tsdf_cfg.get("plan_resolution_m", 0.03)),
        height_band=tuple(tsdf_cfg.get("plan_height_band_m", [0.15, 1.9])),
    )
    plan_bgr = cv2.applyColorMap(grid, cv2.COLORMAP_BONE)
    cv2.imwrite(str(session.plan_path()), plan_bgr)
    # Persist the plan frame so the graph step can draw objects onto it.
    frame_path = session.derived / "plan_frame.json"
    import json

    frame_path.write_text(
        json.dumps(
            {
                "up": up.tolist(),
                "axis_a": a.tolist(),
                "axis_b": b.tolist(),
                "origin": list(origin),
                "resolution_m": resolution,
                "floor_offset": float(np.percentile(np.asarray(mesh.vertices) @ up, 2)),
            }
        ),
        encoding="utf-8",
    )

    extent = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    report = {
        "frames_integrated": integrated,
        "vertices": len(vertices),
        "triangles": len(np.asarray(mesh.triangles)),
        "bbox_extent_m": [round(float(v), 2) for v in extent],
    }
    logger.info("TSDF done: %s", report)
    return report
