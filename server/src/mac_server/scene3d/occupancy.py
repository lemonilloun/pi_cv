"""Gravity estimation and a carved occupancy floor plan.

Two things the pipeline got wrong before this module existed.

**Gravity.** `estimate_up_axis` averaged the cameras' own "down" axis. That
is only correct if the camera is level, and this rig is a low robot whose
camera can be tilted — so the up vector was systematically off, which tilts
the height band, which tilts the floor plan and every `on` edge derived from
it. Here the up vector comes from the floor itself: a RANSAC plane fit on
the lowest slab of the reconstruction. `estimate_gravity` also takes an
optional measured vector, which is the seam an IMU plugs into — when the Pi
starts writing a gravity vector into each keyframe's meta.json, nothing else
in the pipeline has to change.

**The plan.** `make_floor_plan` was a histogram of mesh vertices: every
voxel that survived fusion added density, and nothing ever recorded that a
region was *observed to be empty*. A histogram cannot distinguish "wall"
from "noise that happened to pile up", and it renders open floor as blank —
identical to never-seen space. So the plan looked like a cloud.

Carving fixes that by using the ray, not just its endpoint. Everything
between the camera and a measured surface is space the camera looked
through, so it is free; the endpoint is occupied; past it is unknown. That
is the standard occupancy-grid formulation, and it is what turns a point
pile into a room with walls and traversable floor — which is also what a
navigation consumer actually needs.

Pure numpy; no open3d, no session I/O, so it unit-tests directly.
"""

from __future__ import annotations

import numpy as np


UNKNOWN, FREE, OCCUPIED = 0, 1, 2


# --------------------------------------------------------------- gravity


def fit_plane_ransac(
    points: np.ndarray,
    iterations: int = 200,
    threshold_m: float = 0.03,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Best-supported plane through `points`.

    Returns (unit normal, offset, inlier mask) for the plane
    ``normal . x + offset = 0``. Least squares would be dragged off the
    floor by every chair leg in the slab; RANSAC keeps the dominant surface.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        raise ValueError("need at least 3 points to fit a plane")
    rng = rng or np.random.default_rng(0)

    best_normal = np.array([0.0, 1.0, 0.0])
    best_offset = 0.0
    best_inliers = np.zeros(len(points), dtype=bool)

    for _ in range(iterations):
        idx = rng.choice(len(points), size=3, replace=False)
        a, b, c = points[idx]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:  # collinear sample
            continue
        normal = normal / norm
        offset = -float(normal @ a)
        inliers = np.abs(points @ normal + offset) <= threshold_m
        if inliers.sum() > best_inliers.sum():
            best_normal, best_offset, best_inliers = normal, offset, inliers

    if best_inliers.sum() >= 3:
        # Refit on the consensus set: RANSAC picks the right points, a
        # least-squares refit on those points picks the right plane.
        inlier_points = points[best_inliers]
        centroid = inlier_points.mean(axis=0)
        _, _, vt = np.linalg.svd(inlier_points - centroid)
        best_normal = vt[-1] / np.linalg.norm(vt[-1])
        best_offset = -float(best_normal @ centroid)
    return best_normal, best_offset, best_inliers


def camera_average_up(world_from_cam: dict[int, np.ndarray]) -> np.ndarray:
    """Fallback up vector: the mean of the cameras' own up axis (camera +Y
    points down). Correct only for a level camera — kept as the last resort
    when there is no floor to fit and no IMU."""
    downs = [np.asarray(m)[:3, 1] for m in world_from_cam.values()]
    if not downs:
        return np.array([0.0, -1.0, 0.0])
    down = np.mean(downs, axis=0)
    norm = np.linalg.norm(down)
    if norm < 1e-6:
        return np.array([0.0, -1.0, 0.0])
    return -(down / norm)


def estimate_gravity(
    points: np.ndarray | None,
    world_from_cam: dict[int, np.ndarray],
    imu_up: np.ndarray | None = None,
    floor_slab_frac: float = 0.25,
    max_tilt_deg: float = 35.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, str]:
    """World-space up vector and where it came from.

    Priority: measured (IMU) > floor plane fit > camera average.

    `imu_up` is the seam for S6: the Pi will put a gravity vector in each
    keyframe's meta.json, the caller averages it into world space, and this
    function starts returning `("imu")` without any other change.

    The floor fit only accepts a plane whose normal is within `max_tilt_deg`
    of the coarse prior — otherwise the "dominant plane in the lowest slab"
    could easily be a wall, and silently calling a wall the floor is worse
    than the naive estimate it replaced.
    """
    coarse = camera_average_up(world_from_cam)

    if imu_up is not None:
        up = np.asarray(imu_up, dtype=np.float64)
        norm = np.linalg.norm(up)
        if norm > 1e-6:
            return up / norm, "imu"

    if points is not None and len(points) >= 100:
        pts = np.asarray(points, dtype=np.float64)
        heights = pts @ coarse
        # Lowest slab by the coarse up: the floor is in here even if the
        # coarse vector is a few degrees off.
        cutoff = np.quantile(heights, floor_slab_frac)
        slab = pts[heights <= cutoff]
        if len(slab) >= 100:
            try:
                normal, _, inliers = fit_plane_ransac(slab, rng=rng)
            except ValueError:
                normal, inliers = None, np.zeros(0, dtype=bool)
            if normal is not None and inliers.sum() >= 0.3 * len(slab):
                if normal @ coarse < 0:  # orient upward
                    normal = -normal
                tilt = np.degrees(np.arccos(np.clip(normal @ coarse, -1.0, 1.0)))
                if tilt <= max_tilt_deg:
                    return normal, "floor_fit"

    return coarse, "camera_average"


def plan_axes(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal horizontal axes for a given up vector."""
    up = np.asarray(up, dtype=np.float64)
    up = up / (np.linalg.norm(up) or 1.0)
    a = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 0.1:
        a = np.cross(up, [0.0, 0.0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    return a, b


# ------------------------------------------------------------- occupancy


def make_grid_spec(
    points_xy: np.ndarray, resolution_m: float = 0.04, margin_m: float = 0.4
) -> dict:
    """Grid bounds covering `points_xy` with a margin."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return {"origin": (0.0, 0.0), "width": 1, "height": 1,
                "resolution_m": resolution_m}
    lo = pts.min(axis=0) - margin_m
    hi = pts.max(axis=0) + margin_m
    width = max(8, int(np.ceil((hi[0] - lo[0]) / resolution_m)))
    height = max(8, int(np.ceil((hi[1] - lo[1]) / resolution_m)))
    return {
        "origin": (float(lo[0]), float(lo[1])),
        "width": width,
        "height": height,
        "resolution_m": float(resolution_m),
    }


def world_to_grid(points_xy: np.ndarray, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Plan coordinates -> integer cell indices, clipped into the grid."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    ox, oy = spec["origin"]
    res = spec["resolution_m"]
    ix = np.clip(((pts[:, 0] - ox) / res).astype(np.int32), 0, spec["width"] - 1)
    iy = np.clip(((pts[:, 1] - oy) / res).astype(np.int32), 0, spec["height"] - 1)
    return ix, iy


def carve_rays(
    origin_xy: np.ndarray,
    hits_xy: np.ndarray,
    spec: dict,
    free_counts: np.ndarray,
    hit_counts: np.ndarray,
    stop_margin_m: float = 0.10,
) -> None:
    """Accumulate one camera's evidence into the count grids, in place.

    Samples each ray at grid resolution and marks those cells free, stopping
    `stop_margin_m` short of the surface so the last sample before a wall
    isn't carved out from under it (one voxel of depth error would otherwise
    punch a hole through every wall it touches).
    """
    hits = np.asarray(hits_xy, dtype=np.float64).reshape(-1, 2)
    if len(hits) == 0:
        return
    origin = np.asarray(origin_xy, dtype=np.float64).reshape(2)
    res = spec["resolution_m"]

    delta = hits - origin
    lengths = np.linalg.norm(delta, axis=1)
    usable = lengths > stop_margin_m
    if usable.any():
        d = delta[usable]
        L = lengths[usable]
        # One sample per cell along the longest ray; shorter rays simply
        # repeat their endpoint, which np.add.at collapses harmlessly.
        steps = int(np.ceil(L.max() / res)) + 1
        steps = max(1, min(steps, 4096))
        frac = np.linspace(0.0, 1.0, steps)[None, :]
        stop = np.clip(1.0 - stop_margin_m / L, 0.0, 1.0)[:, None]
        t = frac * stop
        xs = origin[0] + d[:, 0:1] * t
        ys = origin[1] + d[:, 1:2] * t
        ix, iy = world_to_grid(np.stack([xs.ravel(), ys.ravel()], axis=1), spec)
        np.add.at(free_counts, (iy, ix), 1)

    ix, iy = world_to_grid(hits, spec)
    np.add.at(hit_counts, (iy, ix), 1)


def occupancy_from_counts(
    free_counts: np.ndarray,
    hit_counts: np.ndarray,
    min_hits: int = 3,
    min_free: int = 2,
    hit_ratio: float = 0.05,
) -> np.ndarray:
    """Fuse the two count grids into UNKNOWN / FREE / OCCUPIED.

    A cell is occupied when enough rays *ended* there and they are not a
    negligible minority of the rays that passed through — a cell the camera
    saw through a hundred times and stopped in twice is glass or noise, not
    a wall.
    """
    grid = np.full(free_counts.shape, UNKNOWN, dtype=np.uint8)
    grid[free_counts >= min_free] = FREE
    total = free_counts + hit_counts
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(total > 0, hit_counts / np.maximum(total, 1), 0.0)
    grid[(hit_counts >= min_hits) & (ratio >= hit_ratio)] = OCCUPIED
    return grid


def extract_walls(grid: np.ndarray, min_cluster: int = 12) -> np.ndarray:
    """Occupied cells that border free space and belong to a run long enough
    to be structure rather than speckle."""
    import cv2

    occupied = (grid == OCCUPIED).astype(np.uint8)
    free = (grid == FREE).astype(np.uint8)
    free_dilated = cv2.dilate(free, np.ones((3, 3), np.uint8))
    boundary = occupied & (free_dilated > 0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(boundary, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_cluster
    return keep[labels]


def render_plan(grid: np.ndarray, walls: np.ndarray | None = None) -> np.ndarray:
    """Occupancy grid -> BGR image: dark unknown, light free, black walls."""
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[grid == UNKNOWN] = (40, 40, 40)
    img[grid == FREE] = (215, 215, 215)
    img[grid == OCCUPIED] = (90, 90, 150)
    if walls is not None:
        img[walls] = (20, 20, 20)
    return img
