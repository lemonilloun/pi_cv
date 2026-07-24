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
    """Incremental cross-frame association (pure logic, unit-testable)."""

    def __init__(self, dino_cos_min: float = 0.6, centroid_max_m: float = 0.5) -> None:
        self.dino_cos_min = dino_cos_min
        self.centroid_max_m = centroid_max_m
        self.objects: list[dict[str, Any]] = []
        self._by_track: dict[int, int] = {}  # pi track_id -> object idx

    def add(self, candidate: dict[str, Any]) -> int:
        """candidate: {track_id, class, centroid(3), dino_emb, clip_emb?,
        points?, keyframe}. Returns the object index it merged into."""
        track_id = candidate.get("track_id", -1)
        if track_id is not None and track_id >= 0 and track_id in self._by_track:
            idx = self._by_track[track_id]
            self._merge(idx, candidate)
            return idx

        best_idx, best_cos = None, 0.0
        import numpy as np

        for idx, obj in enumerate(self.objects):
            dist = float(
                np.linalg.norm(
                    np.asarray(obj["centroid"]) - np.asarray(candidate["centroid"])
                )
            )
            if dist > self.centroid_max_m:
                continue
            cos = cosine(obj["dino_emb"], candidate["dino_emb"])
            if cos > self.dino_cos_min and cos > best_cos:
                best_idx, best_cos = idx, cos

        if best_idx is not None:
            self._merge(best_idx, candidate)
            if track_id is not None and track_id >= 0:
                self._by_track[track_id] = best_idx
            return best_idx

        obj = {
            "classes": {candidate["class"]: 1},
            "centroid": list(candidate["centroid"]),
            "dino_emb": list(candidate["dino_emb"]),
            "clip_embs": [candidate["clip_emb"]] if candidate.get("clip_emb") else [],
            "n_observations": 1,
            "keyframes": [candidate.get("keyframe")],
            "points": [candidate.get("points")] if candidate.get("points") is not None else [],
        }
        self.objects.append(obj)
        idx = len(self.objects) - 1
        if track_id is not None and track_id >= 0:
            self._by_track[track_id] = idx
        return idx

    def _merge(self, idx: int, candidate: dict[str, Any]) -> None:
        import numpy as np

        obj = self.objects[idx]
        n = obj["n_observations"]
        obj["classes"][candidate["class"]] = obj["classes"].get(candidate["class"], 0) + 1
        obj["centroid"] = list(
            (np.asarray(obj["centroid"]) * n + np.asarray(candidate["centroid"])) / (n + 1)
        )
        obj["dino_emb"] = list(
            (np.asarray(obj["dino_emb"]) * n + np.asarray(candidate["dino_emb"])) / (n + 1)
        )
        if candidate.get("clip_emb"):
            obj["clip_embs"].append(candidate["clip_emb"])
        if candidate.get("points") is not None:
            obj["points"].append(candidate["points"])
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
    intr = session.intrinsics(float(config.get("fallback_hfov_deg", 102.0)))
    poses = session.load_poses()
    progress("loading DINOv2 (first run downloads it, cached after)")
    embedder = DinoEmbedder(device=str(obj_cfg.get("device", "mps")))
    bank = ObjectBank(
        dino_cos_min=float(obj_cfg.get("dino_cos_min", 0.6)),
        centroid_max_m=float(obj_cfg.get("centroid_max_m", 0.5)),
    )
    stride = int(obj_cfg.get("point_stride", 4))
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
        depth_path = session.depth_path(kf.index)
        if not depth_path.exists():
            continue
        depth = np.load(depth_path)
        masks = kf.masks()
        bgr = kf.rgb_bgr()
        meta = kf.meta()
        wfc = poses[kf.index]
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
            points = mask_to_world_points(depth, mask, intr, wfc, stride=stride)
            points = largest_dbscan_cluster(
                points, eps=eps, min_points=min_points, max_points=dbscan_max_points, rng=rng
            )
            if len(points) < min_points:
                continue

            x0, y0, x1, y1 = (int(v) for v in det["bbox_xyxy"])
            crop = bgr[max(0, y0):y1, max(0, x0):x1]
            if crop.size == 0:
                continue
            crop_mask = mask[max(0, y0):y1, max(0, x0):x1]
            crop = crop.copy()
            if crop_mask.shape[:2] == crop.shape[:2] and crop_mask.any():
                fill = crop[crop_mask].mean(axis=0)
                crop[~crop_mask] = fill
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

    pcd_dir = session.objects_pcd_dir()
    pcd_dir.mkdir(parents=True, exist_ok=True)
    export = []
    for obj_id, obj in enumerate(objects):
        merged = np.concatenate(obj.pop("points"), axis=0) if obj.get("points") else np.zeros((0, 3))
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged))
        pcd = pcd.voxel_down_sample(0.01)
        o3d.io.write_point_cloud(str(pcd_dir / f"object_{obj_id:03d}.ply"), pcd)
        obb = None
        if len(pcd.points) >= 10:
            try:
                box = pcd.get_oriented_bounding_box()
                obb = {
                    "center": [round(float(v), 3) for v in box.center],
                    "extent": [round(float(v), 3) for v in box.extent],
                    "rotation": np.asarray(box.R).tolist(),
                }
            except Exception:
                obb = None
        export.append({"object_id": obj_id, "obb": obb, "n_points": len(pcd.points), **obj})

    session.objects_path().write_text(json.dumps({"objects": export}), encoding="utf-8")
    report = {
        "objects": len(export),
        "classes": sorted({o["class_top"] for o in export}),
        "detections_processed": detections_processed,
        "oversized_masks_skipped": oversized_skipped,
    }
    logger.info("Objects step done: %s", report)
    return report
