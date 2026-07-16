"""Appearance signatures and persistent entity resolution.

v1 signature: HSV color histogram of the bbox's inner 80% + depth-normalized
size + typical position. Honest limitations (documented in the panel too):
clothing change between sessions creates a new entity; similarly dressed
people can merge. Thresholds are split-biased — on doubt, a NEW entity is
created (safer for summaries than merging strangers).

The `SignatureProvider` protocol is the seam for a future Hailo-osnet
embedding provider; `signature["kind"]` keeps generations apart (cross-kind
similarity is 0 → clean cutover).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Protocol

from mac_server.monitoring.store import MonitoringStore


logger = logging.getLogger(__name__)


class SignatureProvider(Protocol):
    kind: str

    def compute_from_jpeg(self, jpeg_bytes: bytes, bbox_xyxy: tuple) -> dict[str, Any] | None: ...

    def similarity(self, a: dict[str, Any], b: dict[str, Any]) -> float: ...


class HistogramSignatureProvider:
    kind = "hsv_hist_v1"

    def __init__(self, bins: tuple[int, int, int] = (8, 8, 4)) -> None:
        self.bins = bins

    def compute_from_jpeg(self, jpeg_bytes: bytes, bbox_xyxy: tuple) -> dict[str, Any] | None:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        return self.compute(image, bbox_xyxy)

    def compute(self, image_bgr: Any, bbox_xyxy: tuple) -> dict[str, Any] | None:
        import cv2
        import numpy as np

        h, w = image_bgr.shape[:2]
        x0, y0, x1, y1 = bbox_xyxy
        # Inner 80% of the box drops background at the edges.
        dx, dy = (x1 - x0) * 0.1, (y1 - y0) * 0.1
        ix0 = max(0, min(w - 1, int(x0 + dx)))
        ix1 = max(ix0 + 1, min(w, int(x1 - dx)))
        iy0 = max(0, min(h - 1, int(y0 + dy)))
        iy1 = max(iy0 + 1, min(h, int(y1 - dy)))
        crop = image_bgr[iy0:iy1, ix0:ix1]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(self.bins), [0, 180, 0, 256, 0, 256])
        hist = hist.flatten()
        total = float(hist.sum())
        if total <= 0:
            return None
        hist = (hist / total).astype(float)

        return {
            "kind": self.kind,
            "hist": [round(float(v), 5) for v in hist],
            "bbox_h": float(y1 - y0),
        }

    def similarity(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        if a.get("kind") != self.kind or b.get("kind") != self.kind:
            return 0.0
        hist_a, hist_b = a.get("hist"), b.get("hist")
        if not hist_a or not hist_b or len(hist_a) != len(hist_b):
            return 0.0
        # Histogram intersection: 1.0 for identical L1-normalized histograms.
        return float(sum(min(x, y) for x, y in zip(hist_a, hist_b)))


def merge_signatures(old: dict[str, Any], new: dict[str, Any], alpha: float = 0.25) -> dict[str, Any]:
    """EMA-merge so an entity's signature tracks gradual appearance drift."""
    if old.get("kind") != new.get("kind") or not old.get("hist") or not new.get("hist"):
        return new
    merged_hist = [
        (1 - alpha) * o + alpha * n for o, n in zip(old["hist"], new["hist"])
    ]
    total = sum(merged_hist) or 1.0
    return {
        "kind": new["kind"],
        "hist": [round(v / total, 5) for v in merged_hist],
        "bbox_h": (1 - alpha) * float(old.get("bbox_h", 0)) + alpha * float(new.get("bbox_h", 0)),
    }


def match_entity(
    signature: dict[str, Any],
    candidates: list[dict[str, Any]],
    provider: SignatureProvider,
    now: float,
    threshold: float = 0.6,
    margin: float = 0.1,
    size_scale: float = 120.0,
    recency_scale_s: float = 4 * 3600,
) -> int | None:
    """Score candidates; accept the best only when it clears the threshold
    AND beats the runner-up by `margin` (split-biased)."""
    scored: list[tuple[float, int]] = []
    for cand in candidates:
        cand_sig = cand.get("signature") or {}
        hist_sim = provider.similarity(signature, cand_sig)

        size_a = float(signature.get("bbox_h", 0.0))
        size_b = float(cand_sig.get("bbox_h", 0.0))
        size_sim = math.exp(-abs(size_a - size_b) / size_scale) if size_a and size_b else 0.5

        age_s = max(0.0, now - float(cand.get("last_seen", now)))
        recency = math.exp(-age_s / recency_scale_s)

        score = 0.65 * hist_sim + 0.20 * size_sim + 0.15 * recency
        scored.append((score, int(cand["id"])))

    if not scored:
        return None
    scored.sort(reverse=True)
    best_score, best_id = scored[0]
    if best_score < threshold:
        return None
    if len(scored) > 1 and best_score - scored[1][0] < margin:
        return None
    return best_id


class EntityResolver:
    """Maps a freshly confirmed track to a persistent entity (or creates one)."""

    def __init__(
        self,
        store: MonitoringStore,
        provider: SignatureProvider,
        reacquire_window_h: float = 24.0,
        match_threshold: float = 0.6,
        match_margin: float = 0.1,
    ) -> None:
        self.store = store
        self.provider = provider
        self.reacquire_window_h = reacquire_window_h
        self.match_threshold = match_threshold
        self.match_margin = match_margin

    def resolve(
        self, scene_id: str, class_name: str, signature: dict[str, Any], now: float
    ) -> tuple[int, str]:
        since = now - self.reacquire_window_h * 3600
        candidates = self.store.candidate_entities(scene_id, class_name, since)
        matched_id = match_entity(
            signature,
            candidates,
            self.provider,
            now,
            threshold=self.match_threshold,
            margin=self.match_margin,
        )
        if matched_id is not None:
            for cand in candidates:
                if cand["id"] == matched_id:
                    merged = merge_signatures(cand["signature"], signature)
                    self.store.update_entity(matched_id, merged, last_seen=now, visible_s=0.0)
                    logger.info("Re-acquired entity %s (%s)", cand["label"], class_name)
                    return matched_id, cand["label"]

        count = self.store.entity_count(scene_id, class_name)
        label = f"{class_name.title().replace(' ', '')}#{count + 1}"
        entity_id = self.store.insert_entity(scene_id, class_name, label, signature, now)
        logger.info("New entity %s (%s)", label, class_name)
        return entity_id, label
