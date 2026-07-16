"""Unit tests for appearance signatures and entity matching."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

try:
    import numpy as np

    HAS_CV = True
    try:
        import cv2  # noqa: F401
    except ImportError:
        HAS_CV = False
except ImportError:
    HAS_CV = False

from mac_server.monitoring.signatures import (  # noqa: E402
    HistogramSignatureProvider,
    match_entity,
    merge_signatures,
)


def sig(hist: list[float], bbox_h: float = 200.0) -> dict:
    total = sum(hist) or 1.0
    return {"kind": "hsv_hist_v1", "hist": [v / total for v in hist], "bbox_h": bbox_h}


def cand(entity_id: int, signature: dict, last_seen: float = 0.0, label: str = "X#1") -> dict:
    return {"id": entity_id, "label": label, "signature": signature, "last_seen": last_seen}


PROVIDER = HistogramSignatureProvider()

RED = sig([1.0, 0.0, 0.0, 0.0])
BLUE = sig([0.0, 0.0, 0.0, 1.0])
REDDISH = sig([0.9, 0.1, 0.0, 0.0])


class SimilarityTest(unittest.TestCase):
    def test_identical_is_one(self) -> None:
        self.assertAlmostEqual(PROVIDER.similarity(RED, RED), 1.0)

    def test_disjoint_is_zero(self) -> None:
        self.assertAlmostEqual(PROVIDER.similarity(RED, BLUE), 0.0)

    def test_cross_kind_is_zero(self) -> None:
        other = dict(RED)
        other["kind"] = "osnet_v1"
        self.assertEqual(PROVIDER.similarity(RED, other), 0.0)


class MatchEntityTest(unittest.TestCase):
    def test_matches_same_appearance(self) -> None:
        result = match_entity(RED, [cand(1, REDDISH, last_seen=0.0)], PROVIDER, now=60.0)
        self.assertEqual(result, 1)

    def test_rejects_different_appearance(self) -> None:
        result = match_entity(RED, [cand(1, BLUE, last_seen=0.0)], PROVIDER, now=60.0)
        self.assertIsNone(result)

    def test_margin_rejects_ambiguous(self) -> None:
        # Two nearly identical candidates: ambiguous -> None (split-biased).
        result = match_entity(
            RED,
            [cand(1, REDDISH, last_seen=0.0), cand(2, REDDISH, last_seen=0.0, label="X#2")],
            PROVIDER,
            now=60.0,
        )
        self.assertIsNone(result)

    def test_recency_decay(self) -> None:
        fresh = match_entity(RED, [cand(1, REDDISH, last_seen=0.0)], PROVIDER, now=60.0)
        self.assertEqual(fresh, 1)
        # Same appearance but last seen 3 days ago -> recency term ~0;
        # 0.65*0.9 + 0.20*1.0 + 0 = 0.785 still passes... use weaker hist too.
        weak = sig([0.55, 0.45, 0.0, 0.0])
        stale = match_entity(weak, [cand(1, RED, last_seen=-3 * 86400)], PROVIDER, now=60.0)
        self.assertIsNone(stale)

    def test_empty_candidates(self) -> None:
        self.assertIsNone(match_entity(RED, [], PROVIDER, now=0.0))


class MergeTest(unittest.TestCase):
    def test_merge_moves_toward_new(self) -> None:
        merged = merge_signatures(RED, BLUE, alpha=0.25)
        self.assertAlmostEqual(sum(merged["hist"]), 1.0, places=5)
        self.assertGreater(merged["hist"][0], merged["hist"][3])  # still mostly red
        self.assertGreater(merged["hist"][3], 0.2)  # but drifted toward blue

    def test_cross_kind_returns_new(self) -> None:
        other = dict(BLUE)
        other["kind"] = "osnet_v1"
        merged = merge_signatures(RED, other)
        self.assertEqual(merged["kind"], "osnet_v1")


@unittest.skipUnless(HAS_CV, "numpy+cv2 required")
class ProviderComputeTest(unittest.TestCase):
    def test_compute_on_synthetic_image(self) -> None:
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        image[:, :, 2] = 200  # red-ish in BGR
        signature = PROVIDER.compute(image, (100, 100, 300, 350))
        self.assertIsNotNone(signature)
        self.assertEqual(signature["kind"], "hsv_hist_v1")
        self.assertAlmostEqual(sum(signature["hist"]), 1.0, places=3)
        self.assertEqual(signature["bbox_h"], 250.0)

    def test_same_color_matches_itself(self) -> None:
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        image[:, :, 0] = 180  # blue-ish
        a = PROVIDER.compute(image, (50, 50, 250, 300))
        b = PROVIDER.compute(image, (300, 60, 500, 320))
        self.assertGreater(PROVIDER.similarity(a, b), 0.95)


if __name__ == "__main__":
    unittest.main()
