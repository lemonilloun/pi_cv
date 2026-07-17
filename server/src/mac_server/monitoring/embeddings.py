"""CLIP image embeddings as appearance signatures (SignatureProvider seam).

Replaces the HSV histogram provider as the default: a small CLIP model
(MobileCLIP via open_clip, ~50 MB, resident on MPS next to the depth model)
embeds a bbox crop once per track confirmation. Cosine similarity between
embeddings drives presence re-binding ("which occluded laptop is this?")
and gives graph nodes a semantic vector — similar-looking moments can be
found by cosine search with no separate vector DB.

Mac-only dependency (`open_clip_torch` in requirements-server-cv.txt); when
it is missing the controller falls back to the HSV provider automatically.
"""

from __future__ import annotations

import logging
import math
from typing import Any


logger = logging.getLogger(__name__)


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity mapped to 0..1 (CLIP vectors are ~unit-norm, but
    normalize defensively). 0.0 on any shape mismatch."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return max(0.0, min(1.0, (dot / (norm_a * norm_b) + 1.0) / 2.0))


class ClipSignatureProvider:
    kind = "clip_v1"

    def __init__(self, model_name: str = "MobileCLIP-S1", device: str = "mps") -> None:
        import open_clip
        import torch

        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self.device = device
        # open_clip resolves MobileCLIP weights via its pretrained registry;
        # the first tag listed for the model is the sane default.
        pretrained = open_clip.list_pretrained_tags_by_model(model_name)
        if not pretrained:
            raise RuntimeError(f"No pretrained weights known for {model_name}")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained[0]
        )
        self._model.eval().to(device)
        self._torch = torch
        logger.info("CLIP signature provider ready: %s (%s) on %s", model_name, pretrained[0], device)

    def compute_from_jpeg(self, jpeg_bytes: bytes, bbox_xyxy: tuple) -> dict[str, Any] | None:
        import cv2
        import numpy as np
        from PIL import Image

        image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        h, w = image.shape[:2]
        x0, y0, x1, y1 = bbox_xyxy
        ix0 = max(0, min(w - 1, int(x0)))
        ix1 = max(ix0 + 1, min(w, int(x1)))
        iy0 = max(0, min(h - 1, int(y0)))
        iy1 = max(iy0 + 1, min(h, int(y1)))
        crop = image[iy0:iy1, ix0:ix1]
        if crop.size == 0:
            return None

        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = self._preprocess(pil).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        vec = features[0].float().cpu().tolist()
        return {
            "kind": self.kind,
            "vec": [round(float(v), 5) for v in vec],
            "bbox_h": float(y1 - y0),
        }

    def similarity(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        if a.get("kind") != self.kind or b.get("kind") != self.kind:
            return 0.0
        return cosine(a.get("vec") or [], b.get("vec") or [])
