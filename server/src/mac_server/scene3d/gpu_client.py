"""Thin HTTP client for gpu_service's /embed/* endpoints (DINOv2 object
re-id, CLIP RN50x4/openai object labeling + navindex place recognition),
reached over the same SSH tunnel reconstruct_step.py uses.

RemoteDinoEmbedder/RemoteClipEncoder mirror the local DinoEmbedder
(objects_step.py) / ClipFrameEncoder (navindex_step.py) classes' exact
interface, so callers only change which embedder they instantiate, not
their calling code — see scene3d.embeddings.backend in config/default.json
("remote" default, "local" keeps the old on-laptop torch/open_clip path
importable).
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

import numpy as np


class GpuServiceError(RuntimeError):
    pass


def _post(base_url: str, path: str, payload: dict, timeout_s: float = 60.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GpuServiceError(f"gpu_service {path} failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise GpuServiceError(
            f"Could not reach gpu_service at {base_url}{path} — is the SSH tunnel "
            f"(com.pi_cv.gpu-tunnel launchd agent) up? ({exc})"
        ) from exc


def _encode_rgb_jpeg(rgb: np.ndarray) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class RemoteDinoEmbedder:
    """Drop-in for objects_step.DinoEmbedder — same .embed(crop_rgb) interface."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def embed(self, crop_rgb: np.ndarray) -> list[float]:
        result = _post(self.base_url, "/embed/dinov2", {"image_b64": _encode_rgb_jpeg(crop_rgb)})
        return result["embedding"]


class RemoteClipEncoder:
    """Drop-in for navindex_step.ClipFrameEncoder — same .encode_bgr(bgr) interface."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def encode_bgr(self, bgr: np.ndarray) -> np.ndarray:
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = _post(self.base_url, "/embed/clip_image", {"image_b64": _encode_rgb_jpeg(rgb)})
        return np.asarray(result["embedding"], dtype=np.float32)


def remote_clip_text_embed(base_url: str, texts: list[str]) -> list[list[float]]:
    result = _post(base_url, "/embed/clip_text", {"texts": texts}, timeout_s=120.0)
    return result["embeddings"]
