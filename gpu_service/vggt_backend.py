"""VGGT-1B feed-forward reconstruction: joint camera pose + depth estimation
over a whole session's keyframes in one forward pass (CVPR 2025, Wang et
al.). Replaces the old COLMAP-SfM + Depth-Anything-V2 split - one network
solves both consistently instead of two independently-scaled estimates
that had to be reconciled after the fact.

Loaded once and kept warm (app.py's LOADED_MODELS tracks this so /health
can report it) - the ~5.6GB of weights and CUDA context init are the
expensive part, not a single inference call.

Convention note: VGGT's own `extrinsic` is camera-from-world (OpenCV
convention: X_cam = R @ X_world + t, with frame 0 as the identity/world
origin). This project's poses.json contract (session_io.py) is
world-from-cam, so `reconstruct()` inverts every extrinsic before
returning.

VGGT also predicts its own intrinsics per frame (it doesn't need our
calibrated K at all - unlike the old COLMAP path, which started from our
calibration and let bundle adjustment refine it). Frame-to-frame spread in
that estimate is reported back to the caller as a diagnostic, mirroring
how poses_step.py used to report COLMAP's focal-length drift from the
as-recorded calibration.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "facebook/VGGT-1B"


class VggtOutOfMemoryError(RuntimeError):
    """The requested frame batch didn't fit in GPU memory even after
    clearing the cache. Callers (reconstruct_step.py) fall back to
    overlapping windows on this, not on any other exception."""


class VGGTBackend:
    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = device
        self._model = None
        self._dtype = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        from vggt.models.vggt import VGGT

        logger.info("Loading VGGT checkpoint %s onto %s ...", self.checkpoint, self.device)
        t0 = time.time()
        model = VGGT.from_pretrained(self.checkpoint).to(self.device).eval()
        self._model = model
        self._dtype = (
            torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        )
        logger.info("VGGT loaded in %.1fs.", time.time() - t0)

    def reconstruct(self, image_paths: list[Path]) -> dict:
        """One joint forward pass over `image_paths`, in the order the
        session was walked - VGGT anchors its world frame at the first
        image, so the input order is what defines "world" here (matches
        the old pipeline's convention: frame order = recording order).

        Returns:
            world_from_cam: (N, 4, 4) float64
            intrinsic: (N, 3, 3) float64 - VGGT's own per-frame estimate
            depth: (N, H, W) float32, up-to-scale (no metric meaning yet -
                reconstruct_step.py applies the camera-height scale)
            depth_conf: (N, H, W) float32
            input_hw: (H, W) the resolution VGGT actually ran at
        """
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        self.load()
        str_paths = [str(p) for p in image_paths]
        images = load_and_preprocess_images(str_paths, mode="pad").to(self.device)

        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=self._dtype):
                    predictions = self._model(images)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise VggtOutOfMemoryError(
                f"VGGT OOM at {len(image_paths)} frames"
            ) from exc

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )
        extrinsic = extrinsic.squeeze(0).float().cpu().numpy()  # (N, 3, 4) cam_from_world
        intrinsic = intrinsic.squeeze(0).float().cpu().numpy()  # (N, 3, 3)
        depth = predictions["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()  # (N, H, W)
        depth_conf = predictions["depth_conf"].squeeze(0).float().cpu().numpy()  # (N, H, W)

        world_from_cam = np.stack([_invert_extrinsic(e) for e in extrinsic])

        return {
            "world_from_cam": world_from_cam,
            "intrinsic": intrinsic,
            "depth": depth,
            "depth_conf": depth_conf,
            "input_hw": tuple(int(x) for x in images.shape[-2:]),
        }


def _invert_extrinsic(extrinsic_3x4: np.ndarray) -> np.ndarray:
    """cam_from_world 3x4 [R|t] (OpenCV convention) -> world_from_cam 4x4."""
    r = extrinsic_3x4[:, :3]
    t = extrinsic_3x4[:, 3]
    world_from_cam = np.eye(4, dtype=np.float64)
    world_from_cam[:3, :3] = r.T
    world_from_cam[:3, 3] = -r.T @ t
    return world_from_cam
