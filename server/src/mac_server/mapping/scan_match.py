"""2D scan matching (point-to-point ICP) for live frame-to-frame ego-motion.

Consecutive `wall_points` scans (cv_worker._wall_points: camera-frame
lateral/forward metres, from the metric depth map) describe roughly the
same nearby geometry seen from two slightly different robot poses.
Registering one against the other recovers that pose change directly from
the depth camera — no gravity removal, no accelerometer, no scale
ambiguity (the depth model is already metric). This is the live map's
primary ego-motion source; integrating the IMU alone for this would hit
the same tilt-leak problem the offline metric scale has (see
ImuIntegrator's docstring and docs/scene3d.md's IMU section) — a wheeled
robot's chassis actually tilts a few degrees under acceleration/braking,
and that leaks into "distance travelled" without a calibrated tilt model.
IMU yaw is used only as a fallback when a scan match isn't trustworthy
(see cv_worker.py's pose update).

Pure numpy + scipy.spatial.cKDTree — no Open3D, so this stays on the
always-on server_cv path's lighter dependency set (open3d is reserved for
the heavier, offline scene3d pipeline).
"""

from __future__ import annotations

import math
from typing import Any


def best_rigid_transform_2d(source: Any, target: Any) -> tuple[Any, Any]:
    """2D rotation matrix R (2x2) and translation t (2,) minimizing
    sum ||R @ source_i + t - target_i||^2 — the 2D specialization of the
    same Kabsch/Wahba SVD trick `imu_rvc.kabsch_rotation` uses in 3D."""
    import numpy as np

    src = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    tgt = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    covariance = (src - src_mean).T @ (tgt - tgt_mean)
    u, _, vt = np.linalg.svd(covariance)
    # Guard against a reflection (det = -1): a mirror is not a rotation.
    correction = np.eye(2)
    correction[1, 1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    translation = tgt_mean - rotation @ src_mean
    return rotation, translation


def icp_2d(
    prev_points: Any,
    curr_points: Any,
    initial_yaw_rad: float = 0.0,
    initial_translation: tuple[float, float] = (0.0, 0.0),
    max_iterations: int = 25,
    tolerance_m: float = 1e-4,
    max_correspondence_m: float = 0.5,
    min_points: int = 15,
    min_fitness: float = 0.4,
) -> dict[str, Any] | None:
    """Register `curr_points` onto `prev_points` (both (N,2) camera-frame
    metres: [lateral, forward]).

    Returns the camera's own motion FROM the prev-scan pose TO the
    curr-scan pose, expressed in the prev scan's own local axes —
    `{dx, dy, dyaw_rad, fitness, rmse_m}` — or None when there isn't
    enough overlap to trust the result (too few points, or too few of them
    still have a close match after convergence). `dx`/`dy` follow the same
    [lateral, forward] axes as the input points; `dyaw_rad` is the
    rotation from the prev scan's heading to the curr scan's.

    `initial_yaw_rad`/`initial_translation` are a warm start — pass the
    PREVIOUS tick's result (constant-motion extrapolation) when calling
    this every frame; nearest-neighbor correspondence is only as good as
    the initial guess, and identity is a poor one whenever the rig is
    mid-turn or a meaningful fraction of the scan is newly-revealed
    geometry the previous scan never saw.

    Callers must treat None as "no trustworthy visual estimate this tick"
    and fall back to something else (IMU yaw, or simply not moving the
    pose) rather than integrating a fit this method doesn't trust itself.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    prev = np.asarray(prev_points, dtype=np.float64).reshape(-1, 2)
    curr = np.asarray(curr_points, dtype=np.float64).reshape(-1, 2)
    if len(prev) < min_points or len(curr) < min_points:
        return None

    tree = cKDTree(prev)

    cos_y, sin_y = math.cos(initial_yaw_rad), math.sin(initial_yaw_rad)
    rotation = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    translation = np.asarray(initial_translation, dtype=np.float64)

    fitness = 0.0
    rmse = float("inf")
    for _ in range(max_iterations):
        transformed = curr @ rotation.T + translation
        distances, indices = tree.query(transformed)
        keep = distances <= max_correspondence_m
        matched_count = int(keep.sum())
        if matched_count < min_points:
            return None

        new_rotation, new_translation = best_rigid_transform_2d(
            curr[keep], prev[indices[keep]]
        )
        angle_now = math.atan2(rotation[1, 0], rotation[0, 0])
        angle_new = math.atan2(new_rotation[1, 0], new_rotation[0, 0])
        delta = float(np.linalg.norm(new_translation - translation)) + abs(angle_new - angle_now)
        rotation, translation = new_rotation, new_translation
        fitness = matched_count / len(curr)
        rmse = float(np.sqrt(np.mean(distances[keep] ** 2)))
        if delta < tolerance_m:
            break

    if fitness < min_fitness:
        return None

    return {
        "dx": float(translation[0]),
        "dy": float(translation[1]),
        "dyaw_rad": float(math.atan2(rotation[1, 0], rotation[0, 0])),
        "fitness": round(fitness, 3),
        "rmse_m": round(rmse, 4),
    }
