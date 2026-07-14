"""Scan controller: consumes server-computed metric depth frames during a
directional room scan and accumulates them into the room occupancy grid.

Scan convention (no IMU yet): the camera faces one of north/east/south/west
and moves only forward/backward along the viewing axis. Ego-position is
anchored per frame by the metric distance to the facing wall, so drift never
accumulates; a stationary camera degenerates to a single-viewpoint scan.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from mac_server.mapping.geometry import (
    DIRECTIONS,
    CameraIntrinsics,
    camera_to_room,
    depth_to_points,
    estimate_wall_distance,
    filter_height_band,
)
from mac_server.mapping.grid import OccupancyGrid, render_map_png
from mac_server.mapping.rooms import RoomStore


logger = logging.getLogger(__name__)

MAP_RENDER_INTERVAL_S = 0.5


class ScanError(RuntimeError):
    """Raised when a scan cannot start or stop."""


class ScanController:
    def __init__(
        self,
        room_store: RoomStore,
        cv_worker: Any,
        frame_hub: Any,
        registry: Any,
        config: dict[str, Any],
    ) -> None:
        self._room_store = room_store
        self._cv_worker = cv_worker
        self._map_store = frame_hub.get("map")
        self._registry = registry
        self._config = config
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._scan: dict[str, Any] | None = None

    # ------------------------------------------------------------- API

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._scan is None:
                return {"active": False}
            scan = self._scan
            return {
                "active": scan.get("active", False),
                "room_id": scan["room_id"],
                "direction": scan["direction"],
                "frames_used": scan["frames_used"],
                "frames_rejected": scan["frames_rejected"],
                "wall_distance_m": scan["wall_distance_m"],
                "d_max_m": scan["d_max_m"],
                "started_at": scan["started_at"],
                "room_dims_used": scan["room_dims_used"],
                "error": scan.get("error"),
            }

    def start_scan(
        self,
        room_id: str,
        direction: str,
        lateral_offset_m: Any = None,
    ) -> dict[str, Any]:
        direction = direction.strip().lower()
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")

        room = self._room_store.load_room(room_id)
        if room is None:
            raise ValueError(f"Unknown room: {room_id}")

        cv_status = self._cv_worker.status()
        if cv_status.get("state") != "running":
            raise ScanError(f"Server CV worker is not running (state: {cv_status.get('state')})")

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ScanError("A scan is already active")

            room_w, room_l = self._working_dims(room)
            if lateral_offset_m is not None:
                lateral = float(lateral_offset_m)
            elif direction in ("north", "south"):
                lateral = room_w / 2.0
            else:
                lateral = room_l / 2.0

            max_room = float(self._config.get("max_room_m", 10.0))
            resolution = float(self._config.get("grid_resolution_m", 0.05))
            self._scan = {
                "active": True,
                "room_id": room.room_id,
                "room": room,
                "direction": direction,
                "lateral_offset": lateral,
                "room_dims_used": {"width_m": room_w, "length_m": room_l},
                "grid": OccupancyGrid(max_room, max_room, resolution),
                "frames_used": 0,
                "frames_rejected": 0,
                "wall_distance_m": None,
                "d_max_m": None,
                "started_at": time.time(),
                "error": None,
            }
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="room-scan")
            self._thread.start()

        # Best effort: the scan consumes camera frames, so switch the Pi to
        # stream mode (ignore failures — the user may already be streaming).
        try:
            self._registry.send_command(device_id=None, action="set_mode", mode="stream")
        except Exception as exc:
            logger.info("Could not auto-switch Pi to stream mode: %s", exc)

        return {
            "room_id": room.room_id,
            "direction": direction,
            "lateral_offset_m": lateral,
            "room_dims_used": self._scan["room_dims_used"],
        }

    def stop_scan(self) -> dict[str, Any]:
        with self._lock:
            if self._scan is None or self._thread is None:
                raise ScanError("No scan is active")
            scan = self._scan
        self._stop_event.set()
        self._thread.join(timeout=5)
        with self._lock:
            self._thread = None
            scan["active"] = False

        room_id = scan["room_id"]
        direction = scan["direction"]
        room_dir = self._room_store.room_dir(room_id)
        grid: OccupancyGrid = scan["grid"]

        saved: dict[str, str] = {}
        if scan["frames_used"] > 0:
            scan_path = room_dir / f"scan_{direction}.npz"
            grid.save_npz(scan_path)
            saved["scan_grid"] = str(scan_path)

            self._room_store.update_scan_state(
                room_id,
                direction,
                {
                    "frames_used": scan["frames_used"],
                    "d_max_m": scan["d_max_m"],
                    "finished_at": time.time(),
                },
                body_offset_m=float(self._config.get("body_offset_m", 0.4)),
            )

            fused = self._fuse_room_grids(room_id)
            if fused is not None:
                map_png = self._render(fused, scan["room"], room_id)
                (room_dir / "map.png").write_bytes(map_png)
                fused.save_npz(room_dir / "map_grid.npz")
                saved["map_png"] = str(room_dir / "map.png")
                saved["map_grid"] = str(room_dir / "map_grid.npz")
                self._publish_map(map_png, room_id)

        return {
            "room_id": room_id,
            "direction": direction,
            "frames_used": scan["frames_used"],
            "frames_rejected": scan["frames_rejected"],
            "wall_distance_m": scan["wall_distance_m"],
            "d_max_m": scan["d_max_m"],
            "saved": saved,
        }

    def shutdown(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)

    # ----------------------------------------------------------- worker

    def _run(self) -> None:
        scan = self._scan
        assert scan is not None
        room = scan["room"]
        direction = scan["direction"]

        edge_crop = float(self._config.get("edge_crop_frac", 0.10))
        stride = int(self._config.get("point_stride", 4))
        band = tuple(self._config.get("height_band_m", [0.1, 2.0]))
        max_depth_use = float(self._config.get("max_depth_use_m", 8.0))
        jump_reject = float(self._config.get("wall_jump_reject_m", 0.5))
        ema_alpha = float(self._config.get("wall_ema_alpha", 0.4))
        scale_correction = float(self._config.get("depth_scale_correction", 1.0))

        intrinsics = CameraIntrinsics.from_fov(
            width=1280, height=720, hfov_deg=room.hfov_deg, vfov_deg=room.vfov_deg
        )
        dims = scan["room_dims_used"]
        grid: OccupancyGrid = scan["grid"]

        anchor: float | None = None
        sequence = -1
        last_render = 0.0

        logger.info(
            "Scan started: room=%s direction=%s dims=%s", scan["room_id"], direction, dims
        )

        while not self._stop_event.is_set():
            sequence, depth_frame = self._cv_worker.depth_store.wait_for_next(sequence, timeout=1.0)
            if depth_frame is None or self._stop_event.is_set():
                continue

            try:
                depth_m = depth_frame.depth_m * scale_correction

                d_raw = estimate_wall_distance(depth_m)
                if d_raw is None:
                    scan["frames_rejected"] += 1
                    continue
                if anchor is None:
                    anchor = d_raw
                elif abs(d_raw - anchor) > jump_reject:
                    scan["frames_rejected"] += 1
                    continue
                else:
                    anchor = ema_alpha * d_raw + (1 - ema_alpha) * anchor

                scan["wall_distance_m"] = round(anchor, 2)
                scan["d_max_m"] = round(
                    max(scan["d_max_m"] or 0.0, anchor), 2
                )

                points = depth_to_points(
                    depth_m,
                    intrinsics,
                    stride=stride,
                    edge_crop_frac=edge_crop,
                    max_z=max_depth_use,
                )
                xz, heights = filter_height_band(points, room.camera_height_m, band=band)
                room_xy = camera_to_room(
                    xz,
                    direction=direction,
                    wall_distance=anchor,
                    lateral_offset=scan["lateral_offset"],
                    room_w=dims["width_m"],
                    room_l=dims["length_m"],
                )
                grid.accumulate(room_xy, heights)
                scan["frames_used"] += 1
            except Exception as exc:
                logger.error("Scan frame processing failed: %s", exc)
                scan["error"] = str(exc)
                scan["frames_rejected"] += 1
                continue

            now = time.monotonic()
            if now - last_render >= MAP_RENDER_INTERVAL_S:
                last_render = now
                try:
                    fused = self._fuse_room_grids(scan["room_id"], live_grid=grid)
                    if fused is not None:
                        self._publish_map(
                            self._render(fused, room, scan["room_id"]), scan["room_id"]
                        )
                except Exception as exc:
                    logger.error("Map render failed: %s", exc)

        logger.info(
            "Scan finished: room=%s direction=%s frames=%s rejected=%s",
            scan["room_id"],
            direction,
            scan["frames_used"],
            scan["frames_rejected"],
        )

    # ---------------------------------------------------------- helpers

    def _working_dims(self, room: Any) -> tuple[float, float]:
        max_room = float(self._config.get("max_room_m", 10.0))
        state = self._room_store.load_state(room.room_id)
        estimated = state.get("estimated_dims") or {}
        width = room.width_m or estimated.get("width_m") or max_room
        length = room.length_m or estimated.get("length_m") or max_room
        return float(width), float(length)

    def _fuse_room_grids(
        self, room_id: str, live_grid: OccupancyGrid | None = None
    ) -> OccupancyGrid | None:
        """Merge persisted directional grids with the live one."""
        max_room = float(self._config.get("max_room_m", 10.0))
        resolution = float(self._config.get("grid_resolution_m", 0.05))
        fused = OccupancyGrid(max_room, max_room, resolution)
        merged_any = False

        room_dir = self._room_store.room_dir(room_id)
        current_direction = self._scan["direction"] if self._scan else None
        for direction in DIRECTIONS:
            if live_grid is not None and direction == current_direction:
                continue  # live grid supersedes its persisted predecessor
            path = room_dir / f"scan_{direction}.npz"
            if not path.exists():
                continue
            try:
                fused.merge(OccupancyGrid.load_npz(path))
                merged_any = True
            except (OSError, ValueError) as exc:
                logger.warning("Skipping unreadable scan grid %s: %s", path, exc)

        if live_grid is not None:
            fused.merge(live_grid)
            merged_any = True

        return fused if merged_any else None

    def _render(self, fused: OccupancyGrid, room: Any, room_id: str) -> bytes:
        dims = self._working_dims(room)
        return render_map_png(
            fused,
            crop_w_m=dims[0],
            crop_l_m=dims[1],
            min_hits=int(self._config.get("min_hits", 3)),
            min_wall_height=float(self._config.get("min_wall_height_m", 1.6)),
            label=f"{room_id}  {dims[0]:.1f}x{dims[1]:.1f} m",
        )

    def _publish_map(self, map_png: bytes, room_id: str) -> None:
        self._map_store.update(
            map_png,
            {
                "view": "map",
                "room_id": room_id,
                "content_type": "image/png",
                "rendered_at": time.time(),
            },
        )
