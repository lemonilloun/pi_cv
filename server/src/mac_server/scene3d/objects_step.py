"""Step 4: 3D objects — per-keyframe mask segments lifted to world space,
associated across frames into an object bank.

Association priority per candidate (plan §M5):
  1. live Pi track_id match (same track seen before -> same object)
  2. DINOv2 cosine > threshold AND centroid distance < threshold

DINOv2 (dinov2_vits14, MPS) embeds the bbox crop with the background
outside the mask filled with the crop mean — appearance without context.
The Pi's per-detection CLIP embeddings ride along (averaged per object)
for the open-vocabulary labeling step.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from mac_server.scene3d import depth_filter, occupancy
from mac_server.scene3d.session_io import SceneSession


logger = logging.getLogger(__name__)


class DinoEmbedder:
    def __init__(self, device: str = "mps") -> None:
        import os

        # Framework Python on macOS ships without urllib CA certs; point
        # torch.hub's urlopen at certifi's bundle or the download fails
        # with CERTIFICATE_VERIFY_FAILED.
        try:
            import certifi

            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        except ImportError:
            pass
        import torch

        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self.device = device
        self._torch = torch
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.eval().to(device)

    def embed(self, crop_rgb) -> list[float]:
        import numpy as np

        torch = self._torch
        import cv2

        image = cv2.resize(crop_rgb, (224, 224)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        vec = features[0].float().cpu().numpy()
        norm = float((vec ** 2).sum() ** 0.5) or 1.0
        return [float(v) for v in vec / norm]


def cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    va, vb = np.asarray(a), np.asarray(b)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-9
    return float(va @ vb / denom)


class ObjectBank:
    """Incremental cross-frame association (pure logic, unit-testable).

    Three association bugs produced the duplicates measured on real
    sessions (four separate "sofa"s in one room; two "tv"s 0.61 m apart):

    * **No class gate at all.** Nothing stopped a chair from merging into a
      sofa when their crops happened to embed similarly.
    * **A fixed 0.5 m centroid gate.** That is smaller than the furniture
      itself. Two views of one sofa see different halves of it, so their
      centroids sit ~0.6 m apart and the gate rejects a correct merge — the
      gate has to grow with the object.
    * **track_id merged unconditionally.** The Pi's tracker reuses ids after
      a track dies, so a stale id could weld two unrelated objects together
      no matter where they were in the room.
    """

    def __init__(
        self,
        dino_cos_min: float = 0.6,
        centroid_max_m: float = 0.5,
        class_gate: bool = True,
        size_gate_factor: float = 0.5,
    ) -> None:
        self.dino_cos_min = dino_cos_min
        self.centroid_max_m = centroid_max_m
        self.class_gate = class_gate
        self.size_gate_factor = size_gate_factor
        self.objects: list[dict[str, Any]] = []
        self._by_track: dict[int, int] = {}  # pi track_id -> object idx

    def _gate_m(self, obj: dict[str, Any]) -> float:
        """Centroid gate scaled by how big the object has turned out to be."""
        import numpy as np

        lo, hi = obj.get("aabb_min"), obj.get("aabb_max")
        if lo is None or hi is None:
            return self.centroid_max_m
        diag = float(np.linalg.norm(np.asarray(hi) - np.asarray(lo)))
        return self.centroid_max_m + self.size_gate_factor * diag

    def _compatible(self, obj: dict[str, Any], candidate: dict[str, Any]) -> bool:
        if not self.class_gate:
            return True
        return candidate["class"] in obj["classes"]

    def add(self, candidate: dict[str, Any]) -> int:
        """candidate: {track_id, class, centroid(3), dino_emb, clip_emb?,
        points?, keyframe}. Returns the object index it merged into."""
        import numpy as np

        track_id = candidate.get("track_id", -1)
        centroid = np.asarray(candidate["centroid"], dtype=np.float64)

        if track_id is not None and track_id >= 0 and track_id in self._by_track:
            idx = self._by_track[track_id]
            obj = self.objects[idx]
            dist = float(np.linalg.norm(np.asarray(obj["centroid"]) - centroid))
            # A live track is strong evidence, so it gets a generous gate —
            # but not an unconditional one.
            if dist <= 2.0 * self._gate_m(obj) and self._compatible(obj, candidate):
                self._merge(idx, candidate)
                return idx
            del self._by_track[track_id]  # id was recycled; re-associate below

        best_idx, best_cos = None, 0.0
        for idx, obj in enumerate(self.objects):
            if not self._compatible(obj, candidate):
                continue
            dist = float(np.linalg.norm(np.asarray(obj["centroid"]) - centroid))
            if dist > self._gate_m(obj):
                continue
            cos = cosine(obj["dino_emb"], candidate["dino_emb"])
            if cos > self.dino_cos_min and cos > best_cos:
                best_idx, best_cos = idx, cos

        if best_idx is not None:
            self._merge(best_idx, candidate)
            if track_id is not None and track_id >= 0:
                self._by_track[track_id] = best_idx
            return best_idx

        points = candidate.get("points")
        obj = {
            "classes": {candidate["class"]: 1},
            "centroid": list(candidate["centroid"]),
            "dino_emb": list(candidate["dino_emb"]),
            "clip_embs": [candidate["clip_emb"]] if candidate.get("clip_emb") else [],
            "n_observations": 1,
            "keyframes": [candidate.get("keyframe")],
            "points": [points] if points is not None else [],
            "aabb_min": None,
            "aabb_max": None,
        }
        self.objects.append(obj)
        idx = len(self.objects) - 1
        self._grow_aabb(obj, points)
        if track_id is not None and track_id >= 0:
            self._by_track[track_id] = idx
        return idx

    @staticmethod
    def _grow_aabb(obj: dict[str, Any], points) -> None:
        import numpy as np

        if points is None or len(points) == 0:
            return
        pts = np.asarray(points)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        obj["aabb_min"] = lo.tolist() if obj["aabb_min"] is None else np.minimum(
            np.asarray(obj["aabb_min"]), lo).tolist()
        obj["aabb_max"] = hi.tolist() if obj["aabb_max"] is None else np.maximum(
            np.asarray(obj["aabb_max"]), hi).tolist()

    def _merge(self, idx: int, candidate: dict[str, Any]) -> None:
        import numpy as np

        obj = self.objects[idx]
        n = obj["n_observations"]
        obj["classes"][candidate["class"]] = obj["classes"].get(candidate["class"], 0) + 1
        obj["centroid"] = list(
            (np.asarray(obj["centroid"]) * n + np.asarray(candidate["centroid"])) / (n + 1)
        )
        # Averaging unit vectors shortens them; without re-normalising, the
        # running embedding drifts toward the origin and every subsequent
        # cosine comparison is measured against a vector of the wrong length.
        emb = (np.asarray(obj["dino_emb"]) * n + np.asarray(candidate["dino_emb"])) / (n + 1)
        obj["dino_emb"] = list(emb / (float(np.linalg.norm(emb)) or 1.0))
        if candidate.get("clip_emb"):
            obj["clip_embs"].append(candidate["clip_emb"])
        if candidate.get("points") is not None:
            obj["points"].append(candidate["points"])
            self._grow_aabb(obj, candidate["points"])
        obj["n_observations"] = n + 1
        obj["keyframes"].append(candidate.get("keyframe"))

    def finalize(self, min_observations: int = 3) -> list[dict[str, Any]]:
        import numpy as np

        final = []
        for obj in self.objects:
            if obj["n_observations"] < min_observations:
                continue
            classes = sorted(obj["classes"].items(), key=lambda kv: -kv[1])
            clip_emb = None
            if obj["clip_embs"]:
                mean = np.mean(np.asarray(obj["clip_embs"]), axis=0)
                norm = float(np.linalg.norm(mean)) or 1.0
                clip_emb = [float(v) for v in mean / norm]
            final.append(
                {
                    "class_top": classes[0][0],
                    "class_votes": dict(classes),
                    "centroid": [round(float(v), 3) for v in obj["centroid"]],
                    "n_observations": obj["n_observations"],
                    "keyframes": [k for k in obj["keyframes"] if k is not None],
                    "dino_emb": [round(float(v), 5) for v in obj["dino_emb"]],
                    "clip_emb": clip_emb,
                    "points": obj["points"],
                }
            )
        return final


# Plausible largest dimension per COCO class, in meters. Not a ground truth
# table — a coarse physical sanity check, because monocular depth on a
# partial view produces extents that are simply impossible ("bottle" 0.58 m,
# "person" 4.95 m, "bed" 0.09 m thick were all measured on real sessions).
# Objects outside the range are kept but flagged, so the graph step and the
# panel can treat them as unreliable instead of the pipeline silently
# deciding what the user is allowed to see.
CLASS_MAX_DIM_M = {
    "bottle": 0.40, "cup": 0.25, "wine glass": 0.30, "bowl": 0.40,
    "book": 0.40, "cell phone": 0.25, "remote": 0.30, "mouse": 0.20,
    "keyboard": 0.60, "laptop": 0.60, "vase": 0.60, "clock": 0.60,
    "potted plant": 1.50, "backpack": 0.80, "handbag": 0.60,
    "chair": 1.30, "tv": 1.80, "microwave": 0.80, "oven": 1.00,
    "sink": 1.20, "toilet": 0.90, "refrigerator": 2.10, "person": 2.20,
    "couch": 3.00, "bed": 2.40, "dining table": 2.50,
}
DEFAULT_MAX_DIM_M = 3.0
MIN_DIM_M = 0.03


def size_verdict(class_name: str, extent) -> str:
    """"ok" | "too_large" | "too_small" | "degenerate" for one OBB extent."""
    dims = [float(v) for v in extent]
    if not dims or min(dims) <= 0:
        return "degenerate"
    limit = CLASS_MAX_DIM_M.get(class_name, DEFAULT_MAX_DIM_M)
    if max(dims) > limit:
        return "too_large"
    if max(dims) < MIN_DIM_M:
        return "too_small"
    return "ok"


def gravity_aligned_obb(points, up) -> dict[str, Any] | None:
    """Yaw-only oriented bounding box around `up`.

    Open3D's `get_oriented_bounding_box` is free to rotate in all three
    axes, and PCA on a partial point cloud (one visible face of a sofa)
    happily tilts the box to hug that face — which is where extents like a
    0.09 m-thick "bed" (the box had aligned itself with the floor) came
    from. Furniture stands on the floor, so constraining the box to a
    rotation about gravity removes two of the three ways to be wrong.
    """
    import cv2
    import numpy as np

    from mac_server.scene3d.occupancy import plan_axes

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 10:
        return None
    up = np.asarray(up, dtype=np.float64)
    up = up / (np.linalg.norm(up) or 1.0)
    axis_a, axis_b = plan_axes(up)

    flat = np.stack([pts @ axis_a, pts @ axis_b], axis=1).astype(np.float32)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(flat)
    heights = pts @ up
    h_lo, h_hi = float(heights.min()), float(heights.max())

    theta = np.radians(angle_deg)
    u1 = np.cos(theta) * axis_a + np.sin(theta) * axis_b
    u2 = -np.sin(theta) * axis_a + np.cos(theta) * axis_b
    center = float(cx) * axis_a + float(cy) * axis_b + ((h_lo + h_hi) / 2.0) * up

    return {
        "center": [round(float(v), 3) for v in center],
        "extent": [round(float(w), 3), round(float(h), 3), round(h_hi - h_lo, 3)],
        "rotation": np.stack([u1, u2, up], axis=1).tolist(),
        "yaw_deg": round(float(angle_deg), 1),
        "aligned": "gravity",
    }


def mask_to_world_points(
    depth,
    mask,
    intr: dict[str, Any],
    world_from_cam,
    stride: int = 4,
    max_depth_m: float = 6.0,
):
    """Pixels of one instance mask -> world-space 3D points (N,3)."""
    import numpy as np

    ys, xs = np.nonzero(mask)
    ys, xs = ys[::stride], xs[::stride]
    z = depth[ys, xs]
    valid = (z > 0.15) & (z < max_depth_m)
    ys, xs, z = ys[valid], xs[valid], z[valid]
    if len(z) == 0:
        return np.zeros((0, 3))
    x = (xs - intr["cx"]) / intr["fx"] * z
    y = (ys - intr["cy"]) / intr["fy"] * z
    pts_cam = np.stack([x, y, z], axis=1)
    R, t = world_from_cam[:3, :3], world_from_cam[:3, 3]
    return pts_cam @ R.T + t


def is_oversized_mask(mask_area: int, frame_area: int, max_frac: float) -> bool:
    """A mask covering more than `max_frac` of the frame is almost
    certainly a mis-segmented wall/floor/ceiling, not a bounded object."""
    return frame_area > 0 and mask_area > max_frac * frame_area


def largest_dbscan_cluster(
    points, eps: float = 0.05, min_points: int = 20, max_points: int = 8000, rng=None
):
    """Keep the biggest DBSCAN cluster (drops depth-bleed outliers).

    open3d's DBSCAN is CPU-only and its cost grows badly with point count —
    a single mis-segmented mask (a wall/floor mistaken for one giant
    "object") can produce tens of thousands of points and turn one
    detection into a multi-minute stall that looks indistinguishable from a
    hang from the outside. `max_points` bounds worst-case cost unconditionally:
    a random subsample is enough to find the dominant cluster and estimate
    its extent, we don't need every point."""
    import numpy as np
    import open3d as o3d

    if len(points) < min_points:
        return points
    if len(points) > max_points:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.max() < 0:
        return points
    best = np.bincount(labels[labels >= 0]).argmax()
    return points[labels == best]


def run_objects_step(
    session: SceneSession,
    config: dict[str, Any],
    repo_root: Path,
    progress: Callable[[str], None] = lambda msg: None,
) -> dict[str, Any]:
    import time

    import cv2
    import numpy as np
    import open3d as o3d

    obj_cfg = config.get("objects", {})
    intr = session.active_intrinsics(float(config.get("fallback_hfov_deg", 75.0)))
    poses = session.load_poses()
    progress("loading DINOv2 (first run downloads it, cached after)")
    embedder = DinoEmbedder(device=str(obj_cfg.get("device", "mps")))
    bank = ObjectBank(
        dino_cos_min=float(obj_cfg.get("dino_cos_min", 0.6)),
        centroid_max_m=float(obj_cfg.get("centroid_max_m", 0.5)),
        class_gate=bool(obj_cfg.get("class_gate", True)),
        size_gate_factor=float(obj_cfg.get("size_gate_factor", 0.5)),
    )
    stride = int(obj_cfg.get("point_stride", 4))
    max_depth_m = float(obj_cfg.get("max_depth_m", 6.0))
    mask_fill = str(obj_cfg.get("mask_fill", "none"))
    # The mesh and the objects must come from the same geometry, so prefer
    # the filtered depth the TSDF step cached. Falling back to raw depth
    # keeps the step runnable on its own.
    use_filtered = session.depth_filtered_dir().is_dir()
    if not use_filtered:
        logger.warning(
            "No derived/depth_filtered — lifting objects out of RAW depth, which "
            "is the geometry the mesh no longer uses. Run the tsdf step first for "
            "objects consistent with the mesh."
        )
    eps = float(obj_cfg.get("dbscan_eps_m", 0.05))
    min_points = int(obj_cfg.get("dbscan_min_points", 20))
    dbscan_max_points = int(obj_cfg.get("dbscan_max_points", 8000))
    max_mask_area_frac = float(obj_cfg.get("max_mask_area_frac", 0.35))
    rng = np.random.default_rng(0)

    frames = [kf for kf in session.keyframes() if kf.index in poses]
    started = time.monotonic()
    oversized_skipped = 0
    detections_processed = 0
    for done, kf in enumerate(frames, 1):
        depth_path = (
            session.depth_filtered_path(kf.index) if use_filtered
            else session.depth_path(kf.index)
        )
        if not depth_path.exists():
            depth_path = session.depth_path(kf.index)
            if not depth_path.exists():
                continue
        depth = np.load(depth_path).astype(np.float32)
        masks = kf.masks()
        bgr = kf.rgb_bgr()
        meta = kf.meta()
        wfc = poses[kf.index]
        if masks.shape[:2] != depth.shape[:2]:
            # Instance ids are labels, not intensities — nearest only.
            masks = cv2.resize(
                masks, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        depth_intr = depth_filter.intrinsics_for_shape(intr, depth.shape)
        frame_area = masks.shape[0] * masks.shape[1]

        for det in meta.get("detections", []):
            instance_id = det.get("instance_id")
            mask = masks == instance_id
            mask_area = int(mask.sum())
            if mask_area < 200:
                continue
            if is_oversized_mask(mask_area, frame_area, max_mask_area_frac):
                # A mask covering a huge chunk of the frame is almost always
                # a mis-segmented wall/floor/ceiling, not a bounded object —
                # skip it outright rather than feeding a massive point cloud
                # into DBSCAN (see largest_dbscan_cluster's max_points cap
                # for the belt-and-suspenders version of this guard).
                oversized_skipped += 1
                continue
            points = mask_to_world_points(
                depth, mask, depth_intr, wfc, stride=stride, max_depth_m=max_depth_m
            )
            points = largest_dbscan_cluster(
                points, eps=eps, min_points=min_points, max_points=dbscan_max_points, rng=rng
            )
            if len(points) < min_points:
                continue

            x0, y0, x1, y1 = (int(v) for v in det["bbox_xyxy"])
            crop = bgr[max(0, y0):y1, max(0, x0):x1]
            if crop.size == 0:
                continue
            crop = crop.copy()
            if mask_fill == "mean":
                # Flattening the background to one colour is a large
                # distribution shift for a ViT — it was measurably costing
                # cosine similarity between two views of the same sofa
                # (< 0.6, i.e. no merge). Off by default; kept because it
                # does help for objects on very busy backgrounds.
                full_mask = mask
                if full_mask.shape[:2] != bgr.shape[:2]:
                    full_mask = cv2.resize(
                        mask.astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                crop_mask = full_mask[max(0, y0):y1, max(0, x0):x1]
                if crop_mask.shape[:2] == crop.shape[:2] and crop_mask.any():
                    crop[~crop_mask] = crop[crop_mask].mean(axis=0)
            dino = embedder.embed(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            bank.add(
                {
                    "track_id": det.get("track_id", -1),
                    "class": det.get("class_coco", "?"),
                    "centroid": [float(v) for v in points.mean(axis=0)],
                    "dino_emb": dino,
                    "clip_emb": det.get("clip_emb"),
                    "points": points,
                    "keyframe": kf.index,
                }
            )
            detections_processed += 1
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 else 0
        eta_s = (len(frames) - done) / rate if rate > 0 else None
        progress(
            f"objects {done}/{len(frames)} keyframes "
            f"({detections_processed} objects, {oversized_skipped} oversized masks skipped)"
            + (f", ~{eta_s:.0f}s left" if eta_s is not None else "")
        )

    objects = bank.finalize(min_observations=int(obj_cfg.get("min_observations", 3)))

    # The same up vector the mesh and floor plan used, so an object's box and
    # its footprint on the plan cannot disagree about which way is down.
    up = None
    plan_frame_path = session.derived / "plan_frame.json"
    if plan_frame_path.exists():
        try:
            up = np.asarray(
                json.loads(plan_frame_path.read_text(encoding="utf-8"))["up"],
                dtype=np.float64,
            )
        except (OSError, ValueError, KeyError):
            up = None
    if up is None:
        up, _ = occupancy.estimate_gravity(
            None, poses, imu_up=session.imu_up_vector(poses)
        )

    pcd_dir = session.objects_pcd_dir()
    pcd_dir.mkdir(parents=True, exist_ok=True)
    export = []
    size_flags: dict[str, int] = {}
    for obj_id, obj in enumerate(objects):
        merged = np.concatenate(obj.pop("points"), axis=0) if obj.get("points") else np.zeros((0, 3))
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged))
        pcd = pcd.voxel_down_sample(0.01)
        o3d.io.write_point_cloud(str(pcd_dir / f"object_{obj_id:03d}.ply"), pcd)
        obb, verdict = None, "no_box"
        if len(pcd.points) >= 10:
            try:
                obb = gravity_aligned_obb(np.asarray(pcd.points), up)
            except Exception:
                logger.exception("OBB failed for object %d", obj_id)
                obb = None
            if obb is not None:
                verdict = size_verdict(obj["class_top"], obb["extent"])
        size_flags[verdict] = size_flags.get(verdict, 0) + 1
        obj.pop("aabb_min", None)
        obj.pop("aabb_max", None)
        export.append(
            {
                "object_id": obj_id,
                "obb": obb,
                "n_points": len(pcd.points),
                "size_verdict": verdict,
                **obj,
            }
        )

    session.objects_path().write_text(json.dumps({"objects": export}), encoding="utf-8")
    report = {
        "objects": len(export),
        "classes": sorted({o["class_top"] for o in export}),
        "detections_processed": detections_processed,
        "oversized_masks_skipped": oversized_skipped,
        "depth_source": "filtered" if use_filtered else "raw",
        "size_verdicts": size_flags,
    }
    implausible = sum(v for k, v in size_flags.items() if k not in ("ok", "no_box"))
    if implausible:
        report["warning"] = (
            f"{implausible} of {len(export)} objects have physically implausible "
            "dimensions (flagged in size_verdict, not removed). That usually means "
            "the object was only ever seen from one side, or depth bled from the "
            "background into the mask."
        )
    logger.info("Objects step done: %s", report)
    return report
