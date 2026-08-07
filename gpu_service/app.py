"""FastAPI service for GPU-backed scene reconstruction, run on pdfserver.

Binds to 127.0.0.1 only - reached from the new laptop through the SSH
tunnel set up by ~/Library/LaunchAgents/com.pi_cv.gpu-tunnel.plist there
(see scripts/deploy_gpu_service.sh for how this process itself is
launched). The launcher sets CUDA_VISIBLE_DEVICES=1 before this process
starts, so torch.cuda's device 0 here is always the physical GPU1 -
pdfserver's GPU0 (someone else's long-running vLLM engine) is never even
visible to this process, let alone touched.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embeddings import ClipBackend, DinoV2Backend
from vggt_backend import DEFAULT_CHECKPOINT, VGGTBackend, VggtOutOfMemoryError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="pi_cv gpu_service")

# Session image data arrives here by rsync (reconstruct_step.py, laptop
# side), NOT by HTTP upload - re-uploading hundreds of JPEGs through this
# API would be unnecessary engineering when rsync already does
# incremental, resumable transfer well.
SESSIONS_DIR = Path.home() / "cv_research" / "sessions"

# Populated as backends load their models (M2: VGGT, M3: DINOv2/CLIP), so
# /health can report what's actually warm without each backend needing its
# own bookkeeping.
LOADED_MODELS: dict[str, str] = {}

_vggt = VGGTBackend(checkpoint=DEFAULT_CHECKPOINT)
_dinov2 = DinoV2Backend()
_clip = ClipBackend()


def _decode_image_b64(image_b64: str) -> np.ndarray:
    data = base64.b64decode(image_b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Could not decode image_b64 as JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@app.get("/health")
def health() -> dict:
    import torch

    cuda_available = torch.cuda.is_available()
    info: dict = {
        "cuda_available": cuda_available,
        "loaded_models": sorted(LOADED_MODELS),
    }
    if cuda_available:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        info["device_name"] = torch.cuda.get_device_name(0)
        info["free_gb"] = round(free_bytes / 1e9, 2)
        info["total_gb"] = round(total_bytes / 1e9, 2)
    return info


class ReconstructRequest(BaseModel):
    session_id: str
    frame_indices: list[int] | None = None


@app.post("/reconstruct")
def reconstruct(req: ReconstructRequest) -> dict:
    """Run VGGT over one session's keyframes (already rsynced to
    SESSIONS_DIR by the caller) and write raw (unscaled) poses/depth to
    that same session's derived_gpu/ directory. Returns small JSON only -
    the caller rsyncs derived_gpu/ back to pick up the depth arrays.

    Deliberately returns raw, unscaled geometry: metric scale (camera-
    height method) and picking between scale sources is session-specific
    business logic that belongs in reconstruct_step.py on the laptop, not
    in this "run the heavy model" service.
    """
    session_dir = SESSIONS_DIR / req.session_id
    keyframes_dir = session_dir / "keyframes"
    if not keyframes_dir.is_dir():
        raise HTTPException(404, f"No such session synced here: {keyframes_dir}")

    if req.frame_indices is not None:
        indices = sorted(req.frame_indices)
    else:
        indices = sorted(
            int(p.name) for p in keyframes_dir.iterdir()
            if p.is_dir() and p.name.isdigit() and (p / "rgb.jpg").exists()
        )
    if not indices:
        raise HTTPException(400, "No keyframes found for this session")

    image_paths = [keyframes_dir / f"{idx:06d}" / "rgb.jpg" for idx in indices]
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        raise HTTPException(400, f"{len(missing)} keyframe(s) missing rgb.jpg, e.g. {missing[0]}")

    try:
        result = _vggt.reconstruct(image_paths)
    except VggtOutOfMemoryError as exc:
        raise HTTPException(507, str(exc)) from exc
    LOADED_MODELS["vggt"] = _vggt.checkpoint

    out_dir = session_dir / "derived_gpu"
    depth_dir = out_dir / "depth_raw"
    depth_dir.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(indices):
        np.save(depth_dir / f"{idx:06d}.npy", result["depth"][i].astype(np.float32))
        np.save(depth_dir / f"{idx:06d}_conf.npy", result["depth_conf"][i].astype(np.float32))

    return {
        "frame_indices": indices,
        "world_from_cam": {
            str(idx): result["world_from_cam"][i].tolist() for i, idx in enumerate(indices)
        },
        "intrinsic": {
            str(idx): result["intrinsic"][i].tolist() for i, idx in enumerate(indices)
        },
        "input_hw": list(result["input_hw"]),
        "checkpoint": _vggt.checkpoint,
    }


class ImageEmbedRequest(BaseModel):
    image_b64: str  # JPEG bytes, base64-encoded


@app.post("/embed/dinov2")
def embed_dinov2(req: ImageEmbedRequest) -> dict:
    rgb = _decode_image_b64(req.image_b64)
    embedding = _dinov2.embed(rgb)
    LOADED_MODELS["dinov2"] = _dinov2.model_name
    return {"embedding": embedding}


@app.post("/embed/clip_image")
def embed_clip_image(req: ImageEmbedRequest) -> dict:
    rgb = _decode_image_b64(req.image_b64)
    embedding = _clip.embed_image(rgb)
    LOADED_MODELS["clip"] = f"{_clip.model_name}/{_clip.pretrained}"
    return {"embedding": embedding}


class TextEmbedRequest(BaseModel):
    texts: list[str]


@app.post("/embed/clip_text")
def embed_clip_text(req: TextEmbedRequest) -> dict:
    if not req.texts:
        raise HTTPException(400, "texts must be non-empty")
    embeddings = _clip.embed_text(req.texts)
    LOADED_MODELS["clip"] = f"{_clip.model_name}/{_clip.pretrained}"
    return {"embeddings": embeddings}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8700)
