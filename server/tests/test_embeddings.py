"""Unit tests for the CLIP signature provider (pure parts always run; the
model-backed path only when open_clip is installed)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mac_server.monitoring.embeddings import cosine  # noqa: E402


class CosineTest(unittest.TestCase):
    def test_identical_is_one(self) -> None:
        self.assertAlmostEqual(cosine([1.0, 0.0, 0.5], [1.0, 0.0, 0.5]), 1.0, places=5)

    def test_orthogonal_is_half(self) -> None:
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.5, places=5)

    def test_opposite_is_zero(self) -> None:
        self.assertAlmostEqual(cosine([1.0, 0.0], [-1.0, 0.0]), 0.0, places=5)

    def test_mismatch_is_zero(self) -> None:
        self.assertEqual(cosine([1.0], [1.0, 2.0]), 0.0)
        self.assertEqual(cosine([], []), 0.0)


class ProviderTest(unittest.TestCase):
    def test_provider_roundtrip_if_available(self) -> None:
        try:
            import cv2  # noqa: F401
            import numpy as np
            import open_clip  # noqa: F401

            from mac_server.monitoring.embeddings import ClipSignatureProvider
        except Exception:
            self.skipTest("open_clip/cv2 not installed")

        provider = ClipSignatureProvider(device="cpu")
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        sig = provider.compute_from_jpeg(jpeg.tobytes(), (10, 10, 190, 190))
        self.assertIsNotNone(sig)
        self.assertEqual(sig["kind"], "clip_v1")
        self.assertGreater(len(sig["vec"]), 100)
        self.assertAlmostEqual(provider.similarity(sig, sig), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
