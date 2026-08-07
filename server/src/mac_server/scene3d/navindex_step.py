"""Step 6: place-recognition index for localization (pi_navigation).

For every keyframe with a COLMAP pose, a FULL-FRAME CLIP embedding is
computed with the same model family the Pi runs on its NPU (RN50x4 /
openai — the hef is a quantized copy of exactly this model, so Pi-side
embeddings and this index live in one similarity space).

Output derived/place_index.npz:
    kf        (N,)   keyframe indices
    embs      (N,640) float16, L2-normalized
    positions (N,3)  camera centers, metric world
    forwards  (N,3)  camera +Z (view direction), world

At query time (mac_server.scene3d.navindex) the Pi's live embedding is
matched against every session's index — best match = which room and,
via the stored pose, where in it / which way the camera points.
"""

from __future__ import annotations

import json

import logging
from pathlib import Path
from typing import Any, Callable

from mac_server.scene3d.session_io import SceneSession


logger = logging.getLogger(__name__)


class ClipFrameEncoder:
    def __init__(self, model_name: str = "RN50x4", pretrained: str = "openai", device: str = "mps"):
        import open_clip
        import torch

        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self.device = device
        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval().to(device)

    def encode_bgr(self, bgr) -> "Any":
        import cv2
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].float().cpu().numpy()


def run_navindex_step(
    session: SceneSession,
    config: dict[str, Any],
    repo_root: Path,
    progress: Callable[[str], None] = lambda msg: None,
) -> dict[str, Any]:
    import numpy as np

    graph_cfg = config.get("graph", {})
    emb_cfg = config.get("embeddings", {})
    poses = session.load_poses()
    frames = [kf for kf in session.keyframes() if kf.index in poses]
    if not frames:
        raise RuntimeError("No posed keyframes — run the poses step first")

    # MUST stay RN50x4/openai regardless of backend — see graph_step.py's
    # module docstring on why (Pi-side Hailo hef embedding space parity).
    if str(emb_cfg.get("backend", "remote")).lower() == "remote":
        from mac_server.scene3d.gpu_client import RemoteClipEncoder

        encoder = RemoteClipEncoder(
            base_url=str(emb_cfg.get("gpu_service_url", "http://127.0.0.1:8700"))
        )
    else:
        encoder = ClipFrameEncoder(
            model_name=str(graph_cfg.get("clip_text_model", "RN50x4")),
            pretrained=str(graph_cfg.get("clip_text_pretrained", "openai")),
            device=str(config.get("depth", {}).get("device", "mps")),
        )

    kf_indices, embs, positions, forwards = [], [], [], []
    for done, kf in enumerate(frames, 1):
        wfc = poses[kf.index]
        emb = encoder.encode_bgr(kf.rgb_bgr())
        kf_indices.append(kf.index)
        embs.append(emb.astype(np.float16))
        positions.append(wfc[:3, 3])
        forwards.append(wfc[:3, 2])  # camera +Z in world
        if done % 10 == 0 or done == len(frames):
            progress(f"navindex {done}/{len(frames)}")

    # Stamp the metric scale these positions were built at. The index stores
    # world coordinates, so it is only meaningful against the poses.json that
    # produced it — re-running `reconstruct` with a different scale and NOT
    # re-running this step leaves the navigation reporting the robot at
    # coordinates from the previous geometry. Measured: after a scale fix from
    # 25.62 to 1.515 the stale index still held positions spanning 24 m while
    # the new floor plan was 4 m across, so the live marker landed off the plan
    # entirely and simply stopped being drawn — silent, and easy to blame on
    # navigation rather than on a step that was skipped.
    built_scale = None
    poses_path = session.derived / "poses.json"
    if poses_path.exists():
        try:
            built_scale = json.loads(poses_path.read_text(encoding="utf-8")).get("scale")
        except (OSError, ValueError):
            pass

    out = session.derived / "place_index.npz"
    np.savez_compressed(
        out,
        kf=np.asarray(kf_indices, dtype=np.int32),
        embs=np.asarray(embs, dtype=np.float16),
        positions=np.asarray(positions, dtype=np.float32),
        forwards=np.asarray(forwards, dtype=np.float32),
        built_scale=np.asarray([-1.0 if built_scale is None else float(built_scale)],
                               dtype=np.float64),
    )
    report = {"frames": len(kf_indices), "dim": int(np.asarray(embs).shape[1]),
              "built_scale": built_scale}
    logger.info("Nav index done: %s", report)
    return report
