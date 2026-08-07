"""DINOv2 (object re-id) and CLIP (object labeling + navindex place
recognition) backends for gpu_service, run on GPU1.

DINOv2 is upgraded from the old `dinov2_vits14` (small) to `dinov2_vitg14`
(giant): unlike CLIP below, DINOv2 embeddings are computed entirely on
this side (crops taken from keyframes already on disk, nothing about it
is tied to the Pi's hardware), so there's no reason to stay small now
that GPU1 is fully dedicated - see objects_step.py's DinoEmbedder, which
this mirrors.

CLIP is deliberately kept at `RN50x4`/`openai` - graph_step.py and
navindex_step.py compare embeddings computed here against `clip_emb`
vectors the Pi already computed **on-device** via its Hailo
`clip_resnet_50x4_h8.hef`. A bigger CLIP model here would put Mac/GPU and
Pi embeddings in different vector spaces and silently break every
cosine-similarity comparison. This module only moves that computation
onto GPU1 for speed - not to a different model.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_DINOV2_MODEL = "dinov2_vitg14"
DEFAULT_CLIP_MODEL = "RN50x4"
DEFAULT_CLIP_PRETRAINED = "openai"


class DinoV2Backend:
    def __init__(self, model_name: str = DEFAULT_DINOV2_MODEL, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading DINOv2 %s onto %s ...", self.model_name, self.device)
        self._model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self._model.eval().to(self.device)

    def embed(self, crop_rgb: np.ndarray) -> list[float]:
        import cv2

        self.load()
        image = cv2.resize(crop_rgb, (224, 224)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._model(tensor)
        vec = features[0].float().cpu().numpy()
        norm = float((vec ** 2).sum() ** 0.5) or 1.0
        return [float(v) for v in vec / norm]


class ClipBackend:
    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        pretrained: str = DEFAULT_CLIP_PRETRAINED,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        import open_clip

        logger.info("Loading CLIP %s/%s onto %s ...", self.model_name, self.pretrained, self.device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self._model = model.eval().to(self.device)
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    def embed_image(self, rgb: np.ndarray) -> list[float]:
        from PIL import Image

        self.load()
        pil = Image.fromarray(rgb)
        tensor = self._preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return [float(v) for v in features[0].float().cpu().numpy()]

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        self.load()
        with torch.no_grad():
            tokens = self._tokenizer(texts)
            tokens = tokens.to(self.device)
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return [[float(v) for v in vec] for vec in features.float().cpu().numpy()]
