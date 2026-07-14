"""Top-down occupancy grid for room mapping and its PNG rendering."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class OccupancyGrid:
    """Hit-count + max-height grid over the room rectangle.

    Rows index y (north->south), columns index x (west->east), so
    `hits[row, col]` covers room cell (x=col*res, y=row*res).
    """

    def __init__(self, width_m: float, length_m: float, resolution_m: float = 0.05) -> None:
        import numpy as np

        self.resolution_m = resolution_m
        self.width_m = width_m
        self.length_m = length_m
        self.cols = max(1, int(round(width_m / resolution_m)))
        self.rows = max(1, int(round(length_m / resolution_m)))
        self.hits = np.zeros((self.rows, self.cols), dtype=np.uint32)
        self.height_max = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.frames = 0

    def accumulate(self, room_xy: Any, heights: Any) -> int:
        """Add one frame's points. Returns how many landed inside the grid."""
        import numpy as np

        room_xy = np.asarray(room_xy, dtype=np.float32)
        heights = np.asarray(heights, dtype=np.float32)
        if room_xy.size == 0:
            self.frames += 1
            return 0

        cols = np.floor(room_xy[:, 0] / self.resolution_m).astype(np.int64)
        rows = np.floor(room_xy[:, 1] / self.resolution_m).astype(np.int64)
        inside = (cols >= 0) & (cols < self.cols) & (rows >= 0) & (rows < self.rows)
        cols, rows, heights = cols[inside], rows[inside], heights[inside]

        np.add.at(self.hits, (rows, cols), 1)
        np.maximum.at(self.height_max, (rows, cols), heights)
        self.frames += 1
        return int(cols.size)

    def occupied_mask(self, min_hits: int = 3) -> Any:
        return self.hits >= min_hits

    def wall_mask(self, min_wall_height: float = 1.6, min_hits: int = 3) -> Any:
        return self.occupied_mask(min_hits) & (self.height_max >= min_wall_height)

    def merge(self, other: "OccupancyGrid") -> None:
        import numpy as np

        if (other.rows, other.cols) != (self.rows, self.cols):
            raise ValueError("Cannot merge grids of different shapes")
        self.hits = self.hits + other.hits
        self.height_max = np.maximum(self.height_max, other.height_max)
        self.frames += other.frames

    def save_npz(self, path: Path) -> None:
        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            hits=self.hits,
            height_max=self.height_max,
            resolution_m=self.resolution_m,
            width_m=self.width_m,
            length_m=self.length_m,
            frames=self.frames,
        )

    @classmethod
    def load_npz(cls, path: Path) -> "OccupancyGrid":
        import numpy as np

        data = np.load(path)
        grid = cls(
            width_m=float(data["width_m"]),
            length_m=float(data["length_m"]),
            resolution_m=float(data["resolution_m"]),
        )
        grid.hits = data["hits"].astype(np.uint32)
        grid.height_max = data["height_max"].astype(np.float32)
        grid.frames = int(data["frames"])
        return grid


def render_map_png(
    grid: OccupancyGrid,
    crop_w_m: float | None = None,
    crop_l_m: float | None = None,
    min_hits: int = 3,
    min_wall_height: float = 1.6,
    upscale: int = 4,
    label: str | None = None,
) -> bytes:
    """Render the grid to a PNG: light floor, colored obstacles by height,
    dark walls, meter gridlines, and an optional label."""
    import cv2
    import numpy as np

    rows, cols = grid.rows, grid.cols
    if crop_w_m:
        cols = min(cols, max(1, int(round(crop_w_m / grid.resolution_m))))
    if crop_l_m:
        rows = min(rows, max(1, int(round(crop_l_m / grid.resolution_m))))

    hits = grid.hits[:rows, :cols]
    height_max = grid.height_max[:rows, :cols]
    occupied = hits >= min_hits
    walls = occupied & (height_max >= min_wall_height)
    obstacles = occupied & ~walls

    # Base: dark floor matching the panel theme.
    image = np.full((rows, cols, 3), (26, 26, 25), dtype=np.uint8)  # BGR of #191a1a

    # Obstacles colored by height: low = green, high = orange (BGR).
    if np.any(obstacles):
        h_norm = np.clip(height_max[obstacles] / max(min_wall_height, 0.1), 0.0, 1.0)
        colors = np.zeros((int(np.count_nonzero(obstacles)), 3), dtype=np.uint8)
        colors[:, 0] = (40 + 30 * (1 - h_norm)).astype(np.uint8)          # B
        colors[:, 1] = (160 * (1 - h_norm) + 100 * h_norm).astype(np.uint8)  # G
        colors[:, 2] = (60 * (1 - h_norm) + 230 * h_norm).astype(np.uint8)  # R
        image[obstacles] = colors

    # Walls: bright to stand out on the dark floor.
    image[walls] = (200, 200, 210)

    image = cv2.resize(image, (cols * upscale, rows * upscale), interpolation=cv2.INTER_NEAREST)

    # 1-meter gridlines.
    step_px = int(round(1.0 / grid.resolution_m)) * upscale
    for gx in range(0, cols * upscale, step_px):
        cv2.line(image, (gx, 0), (gx, rows * upscale), (55, 55, 52), 1)
    for gy in range(0, rows * upscale, step_px):
        cv2.line(image, (0, gy), (cols * upscale, gy), (55, 55, 52), 1)

    # Compass + label.
    cv2.putText(image, "N", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (195, 194, 183), 2)
    if label:
        cv2.putText(
            image,
            label,
            (8, rows * upscale - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (195, 194, 183),
            1,
        )

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode map PNG")
    return encoded.tobytes()
