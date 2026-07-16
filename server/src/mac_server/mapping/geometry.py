"""Camera geometry for room mapping: pinhole projection of metric depth maps
into 3D points and into the axis-aligned room frame.

Coordinate conventions:
- Camera frame: +X right, +Y down, +Z forward (optical axis).
- Room frame: x runs west->east in [0, W], y runs north->south in [0, L].
  "Facing north" means the optical axis points toward the north wall (y=0).

Pure numpy, no torch — unit-testable in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


DIRECTIONS = ("north", "east", "south", "west")


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float, vfov_deg: float) -> "CameraIntrinsics":
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
        return cls(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0, width=width, height=height)

    def scaled(self, width: int, height: int) -> "CameraIntrinsics":
        """Intrinsics for a resized image (e.g. the depth map resolution)."""
        sx = width / self.width
        sy = height / self.height
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=width,
            height=height,
        )


def depth_to_points(
    depth_m: Any,
    intrinsics: CameraIntrinsics,
    stride: int = 4,
    edge_crop_frac: float = 0.10,
    min_z: float = 0.2,
    max_z: float = 8.0,
) -> Any:
    """Project a metric depth map to (N, 3) camera-frame points [X, Y, Z].

    Edge columns/rows are dropped (wide-lens distortion), the map is sampled
    with `stride`, and points outside [min_z, max_z] are discarded.
    """
    import numpy as np

    depth = np.asarray(depth_m, dtype=np.float32)
    height, width = depth.shape[:2]
    intr = intrinsics
    if (width, height) != (intr.width, intr.height):
        intr = intrinsics.scaled(width, height)

    edge_x = int(width * edge_crop_frac)
    edge_y = int(height * edge_crop_frac)
    us = np.arange(edge_x, width - edge_x, stride)
    vs = np.arange(edge_y, height - edge_y, stride)
    uu, vv = np.meshgrid(us, vs)
    zz = depth[vv, uu]

    valid = (zz >= min_z) & (zz <= max_z) & np.isfinite(zz)
    uu, vv, zz = uu[valid], vv[valid], zz[valid]

    xs = (uu - intr.cx) / intr.fx * zz
    ys = (vv - intr.cy) / intr.fy * zz
    return np.stack([xs, ys, zz], axis=1)


def bbox_bearing_rad(bbox_xyxy: Sequence[float], intrinsics: CameraIntrinsics) -> float:
    """Ray angle (radians, +right) for a bbox's horizontal center column.

    This is the pixel column's true ray angle regardless of measured depth —
    the same (u-cx)/fx relationship depth_to_points uses, just as an angle.
    """
    cx_box = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
    return math.atan2(cx_box - intrinsics.cx, intrinsics.fx)


def bbox_bearing_deg(bbox_xyxy: Sequence[float], intrinsics: CameraIntrinsics) -> float:
    return math.degrees(bbox_bearing_rad(bbox_xyxy, intrinsics))


def bbox_camera_xy(
    bbox_xyxy: Sequence[float],
    forward_m: float,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    """Camera-frame (lateral_m, forward_m) for one bbox at a known Z-depth.

    Single-point specialization of depth_to_points's X = (u-cx)/fx * z:
    lateral = tan(bearing) * forward_m. NOT sin/cos of bearing — a polar
    (bearing, depth) plot is a different point except at bearing=0 (the
    lateral offset for a given Z-depth grows with tan, not sin, of the ray
    angle; the two diverge fast on a wide lens: at 51 deg, sin/tan ≈ 0.63).
    """
    bearing = bbox_bearing_rad(bbox_xyxy, intrinsics)
    return math.tan(bearing) * forward_m, forward_m


def filter_height_band(
    points: Any,
    camera_height_m: float,
    band: tuple[float, float] = (0.1, 2.0),
) -> tuple[Any, Any]:
    """Keep points whose height above the floor falls inside `band`.

    Returns ((N, 2) [X_lateral, Z_forward], (N,) heights above floor).
    Camera +Y is down, so height = camera_height − Y.
    """
    import numpy as np

    points = np.asarray(points, dtype=np.float32)
    heights = camera_height_m - points[:, 1]
    keep = (heights >= band[0]) & (heights <= band[1])
    kept = points[keep]
    return np.stack([kept[:, 0], kept[:, 2]], axis=1), heights[keep]


def estimate_wall_distance(
    depth_m: Any,
    patch_frac: float = 0.2,
) -> float | None:
    """Median metric depth of the central patch — the distance to whatever
    the camera is squarely facing (the anchor wall under the scan convention).
    """
    import numpy as np

    depth = np.asarray(depth_m, dtype=np.float32)
    height, width = depth.shape[:2]
    half_w = max(1, int(width * patch_frac / 2))
    half_h = max(1, int(height * patch_frac / 2))
    cx, cy = width // 2, height // 2
    patch = depth[cy - half_h : cy + half_h, cx - half_w : cx + half_w]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def camera_to_room(
    points_xz: Any,
    direction: str,
    wall_distance: float,
    lateral_offset: float,
    room_w: float,
    room_l: float,
) -> Any:
    """Map camera-frame [X_lateral, Z_forward] points into room-frame (x, y).

    The camera faces `direction`; the facing wall anchors the transform at
    distance `wall_distance`; `lateral_offset` is the camera line's offset
    along the perpendicular axis (defaults to the room midline upstream).
    """
    import numpy as np

    points_xz = np.asarray(points_xz, dtype=np.float32)
    x_lat = points_xz[:, 0]
    z_fwd = points_xz[:, 1]
    d = wall_distance

    if direction == "north":
        room_x = lateral_offset + x_lat
        room_y = d - z_fwd
    elif direction == "south":
        room_x = lateral_offset - x_lat
        room_y = room_l - (d - z_fwd)
    elif direction == "east":
        room_x = room_w - (d - z_fwd)
        room_y = lateral_offset + x_lat
    elif direction == "west":
        room_x = d - z_fwd
        room_y = lateral_offset - x_lat
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return np.stack([room_x, room_y], axis=1)
