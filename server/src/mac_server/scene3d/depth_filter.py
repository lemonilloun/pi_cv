"""Depth hygiene shared by the fusion and object steps.

Monocular metric depth is dense but not trustworthy everywhere, and until
now nothing checked it: every valid pixel went into the TSDF with equal
weight. On a real session that produced a mesh of 37k connected components
whose bounding box spanned 12 x 15 m while the camera itself only ever
moved inside 2.2 x 3.9 m — geometry projected straight through the walls,
and a floor plan that looked like a starburst of radial streaks.

Two independent failure modes produce that, and each gets a filter here:

1. **Flying pixels.** At a depth discontinuity (object edge against a far
   wall) the network interpolates across the jump, creating a thin surface
   that is nearly parallel to the viewing ray. Those pixels smear along the
   ray — the radial streaks. `flying_pixel_mask` rejects them by the angle
   between the local surface normal and the view direction, which is what
   "edge-on" means geometrically. This replaces the old blanket erosion of
   a 6 px band around every instance mask, which also destroyed the *real*
   object rims that the objects step needs for its bounding boxes.

2. **Plain wrong depth.** A pixel can be smooth and confident and still be
   metrically wrong. The check that catches it is multi-view: reproject the
   3D point into other keyframes and ask whether *their* depth maps agree.
   Uncorrelated errors do not survive it. `multiview_support` counts
   agreeing views; requiring >= 2 is the standard MVS fusion criterion.

Support is counted, never contradiction: a point legitimately occluded in
another view simply fails to gain that view's vote instead of being
rejected outright.

Everything here is pure numpy so it can be unit-tested without a session.
"""

from __future__ import annotations

import math

import numpy as np


def intrinsics_for_shape(intr: dict, shape: tuple[int, ...]) -> dict:
    """Rescale calibrated intrinsics to the resolution actually being used.

    The calibration is stored at capture resolution, but depth maps and
    masks may be produced at another one. Projecting with unscaled
    intrinsics silently warps the whole reconstruction, so every entry
    point here goes through this.
    """
    h, w = int(shape[0]), int(shape[1])
    src_w = float(intr.get("width") or w)
    src_h = float(intr.get("height") or h)
    sw = w / src_w if src_w else 1.0
    sh = h / src_h if src_h else 1.0
    return {
        "fx": float(intr["fx"]) * sw,
        "fy": float(intr["fy"]) * sh,
        "cx": float(intr["cx"]) * sw,
        "cy": float(intr["cy"]) * sh,
        "width": w,
        "height": h,
    }


def unproject(depth: np.ndarray, intr: dict) -> np.ndarray:
    """Per-pixel camera-frame XYZ, with invalid depth as NaN (so it
    propagates through the geometry instead of pretending to be the origin)."""
    k = intrinsics_for_shape(intr, depth.shape)
    h, w = depth.shape[:2]
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    z = np.where(depth > 0, depth, np.nan).astype(np.float64)
    x = (uu - k["cx"]) / k["fx"] * z
    y = (vv - k["cy"]) / k["fy"] * z
    return np.stack([x, y, z], axis=-1)


def surface_normal_cosine(depth: np.ndarray, intr: dict) -> np.ndarray:
    """|cos| between the local surface normal and the viewing ray.

    ~1 means the surface faces the camera (well conditioned); ~0 means it is
    seen edge-on, which for a depth map almost always means the pixel is
    interpolated across a discontinuity rather than a real grazing surface.
    """
    points = unproject(depth, intr)
    d_du = np.gradient(points, axis=1)
    d_dv = np.gradient(points, axis=0)
    normal = np.cross(d_du, d_dv)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_len = np.linalg.norm(normal, axis=-1)
        p_len = np.linalg.norm(points, axis=-1)
        cos = np.abs(np.sum(normal * points, axis=-1)) / (n_len * p_len)
    return cos


def flying_pixel_mask(
    depth: np.ndarray, intr: dict, max_angle_deg: float = 85.0
) -> np.ndarray:
    """True where a valid pixel lies on a surface too edge-on to trust."""
    cos = surface_normal_cosine(depth, intr)
    threshold = math.cos(math.radians(max_angle_deg))
    # NaN comparisons are False, so undefined geometry counts as bad.
    return (depth > 0) & ~(cos >= threshold)


def multiview_support(
    depth: np.ndarray,
    world_from_cam: np.ndarray,
    neighbours: list[tuple[np.ndarray, np.ndarray]],
    intr: dict,
    rel_tol: float = 0.06,
) -> np.ndarray:
    """How many neighbour views confirm each pixel's 3D point.

    `neighbours` is a list of (depth_map, world_from_cam) for other
    keyframes. A view confirms the point when the point reprojects inside
    that view and lands within `rel_tol` (relative) of that view's own
    measured depth.
    """
    k = intrinsics_for_shape(intr, depth.shape)
    h, w = depth.shape[:2]
    points_cam = unproject(depth, intr)
    rot = np.asarray(world_from_cam)[:3, :3]
    trans = np.asarray(world_from_cam)[:3, 3]
    points_world = points_cam @ rot.T + trans

    support = np.zeros((h, w), dtype=np.int16)
    for depth_k, wfc_k in neighbours:
        if depth_k.shape[:2] != depth.shape[:2]:
            continue
        rot_k = np.asarray(wfc_k)[:3, :3]
        trans_k = np.asarray(wfc_k)[:3, 3]
        # cam_from_world: R_k^T @ (p - t_k), row-vector form.
        local = (points_world - trans_k) @ rot_k
        z = local[..., 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            u = k["fx"] * local[..., 0] / z + k["cx"]
            v = k["fy"] * local[..., 1] / z + k["cy"]
        in_front = np.isfinite(z) & (z > 0)
        with np.errstate(invalid="ignore"):
            inside = (
                in_front
                & np.isfinite(u) & np.isfinite(v)
                & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            )
        ui = np.clip(np.nan_to_num(u), 0, w - 1).astype(np.int32)
        vi = np.clip(np.nan_to_num(v), 0, h - 1).astype(np.int32)
        observed = depth_k[vi, ui]
        with np.errstate(invalid="ignore"):
            agrees = (
                inside
                & (observed > 0)
                & (np.abs(observed - z) <= rel_tol * np.maximum(z, 1e-6))
            )
        support += agrees.astype(np.int16)
    return support


def select_neighbours(
    poses: dict[int, np.ndarray], kf_index: int, count: int = 6
) -> list[int]:
    """Nearest other keyframes by camera position.

    Nearest-by-position is what the consistency check wants: enough baseline
    to be an independent measurement, close enough to actually share
    surfaces with the reference view.
    """
    if kf_index not in poses:
        return []
    origin = np.asarray(poses[kf_index])[:3, 3]
    others = [
        (float(np.linalg.norm(np.asarray(m)[:3, 3] - origin)), idx)
        for idx, m in poses.items()
        if idx != kf_index
    ]
    others.sort()
    return [idx for _, idx in others[: max(0, count)]]


def confidence_weight(
    conf: np.ndarray | None, depth: np.ndarray, far_m: float = 4.0
) -> np.ndarray:
    """Per-pixel trust in [0, 1] combining model confidence with range.

    Monocular depth error grows with distance, so a far pixel deserves less
    say in the fusion even when the model is sure about it.
    """
    range_term = np.clip(far_m / np.maximum(depth, 1e-6), 0.0, 1.0)
    range_term = np.where(depth > 0, range_term, 0.0)
    if conf is None:
        return range_term.astype(np.float32)
    c = np.asarray(conf, dtype=np.float64)
    lo, hi = float(np.nanmin(c)), float(np.nanmax(c))
    if hi > 1.0 or lo < 0.0:  # normalize an unbounded confidence channel
        c = (c - lo) / max(hi - lo, 1e-9)
    return (np.nan_to_num(c) * range_term).astype(np.float32)


def filter_depth(
    depth: np.ndarray,
    intr: dict,
    world_from_cam: np.ndarray | None = None,
    neighbours: list[tuple[np.ndarray, np.ndarray]] | None = None,
    conf: np.ndarray | None = None,
    max_angle_deg: float = 85.0,
    rel_tol: float = 0.06,
    min_views: int = 2,
    conf_min: float = 0.0,
    max_depth_m: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply the full hygiene chain. Returns (depth, stats); rejected
    pixels are set to 0, the project-wide "no measurement" value."""
    out = np.array(depth, dtype=np.float32, copy=True)
    total = int((out > 0).sum())
    stats = {"valid_in": total}

    if max_depth_m is not None:
        out[out > max_depth_m] = 0.0
        stats["dropped_far"] = total - int((out > 0).sum())

    before = int((out > 0).sum())
    out[flying_pixel_mask(out, intr, max_angle_deg)] = 0.0
    stats["dropped_flying"] = before - int((out > 0).sum())

    if conf is not None and conf_min > 0:
        before = int((out > 0).sum())
        out[np.asarray(conf) < conf_min] = 0.0
        stats["dropped_lowconf"] = before - int((out > 0).sum())

    if world_from_cam is not None and neighbours and min_views > 0:
        before = int((out > 0).sum())
        support = multiview_support(out, world_from_cam, neighbours, intr, rel_tol)
        out[support < min_views] = 0.0
        stats["dropped_inconsistent"] = before - int((out > 0).sum())

    stats["valid_out"] = int((out > 0).sum())
    stats["kept_frac"] = round(stats["valid_out"] / total, 4) if total else 0.0
    return out, stats
