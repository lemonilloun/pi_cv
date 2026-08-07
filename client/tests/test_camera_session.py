"""Tests for the replay capture source.

The replay source is the tool that makes every later perception change
measurable without a robot, so its ordering and exhaustion behaviour are worth
pinning: a source that silently reorders frames or loops forever would make the
comparisons it exists to enable meaningless.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "client/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from pi_client.camera_session import (  # noqa: E402
    CaptureSettings,
    FrameSourceError,
    ReplaySource,
    make_capture_source,
)


def _make_session(root: Path, count: int) -> Path:
    import cv2
    import numpy as np

    session = root / "session_test"
    for i in range(1, count + 1):
        kf = session / "keyframes" / f"{i:06d}"
        kf.mkdir(parents=True)
        # A distinct constant colour per frame, so ordering is checkable from
        # the pixels rather than from the filename we already sorted on.
        img = np.full((16, 16, 3), i * 10, dtype=np.uint8)
        cv2.imwrite(str(kf / "rgb.jpg"), img)
    return session


class ReplaySourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_serves_keyframes_in_recording_order(self) -> None:
        session = _make_session(self.root, 12)
        source = ReplaySource(CaptureSettings(width=16, height=16, fps=1.0), session)
        source.start()
        self.assertEqual(len(source), 12)
        seen = [int(source.capture_bgr()[0, 0, 0]) for _ in range(12)]
        self.assertEqual(seen, [i * 10 for i in range(1, 13)])

    def test_exhausts_rather_than_looping(self) -> None:
        session = _make_session(self.root, 3)
        source = ReplaySource(CaptureSettings(width=16, height=16, fps=1.0), session)
        source.start()
        for _ in range(3):
            self.assertFalse(source.exhausted)
            source.capture_bgr()
        self.assertTrue(source.exhausted)

    def test_current_path_traces_output_back_to_a_keyframe(self) -> None:
        session = _make_session(self.root, 3)
        source = ReplaySource(CaptureSettings(width=16, height=16, fps=1.0), session)
        source.start()
        self.assertIsNone(source.current_path)
        source.capture_bgr()
        self.assertEqual(source.current_path.parent.name, "000001")
        source.capture_bgr()
        self.assertEqual(source.current_path.parent.name, "000002")

    def test_empty_or_missing_session_is_an_error_not_an_empty_run(self) -> None:
        settings = CaptureSettings(width=16, height=16, fps=1.0)
        with self.assertRaises(FrameSourceError):
            ReplaySource(settings, self.root / "nope").start()
        empty = self.root / "empty" / "keyframes"
        empty.mkdir(parents=True)
        with self.assertRaises(FrameSourceError):
            ReplaySource(settings, empty.parent).start()

    def test_factory_builds_it(self) -> None:
        session = _make_session(self.root, 2)
        source = make_capture_source(
            "replay", CaptureSettings(width=16, height=16, fps=1.0), session
        )
        self.assertIsInstance(source, ReplaySource)


if __name__ == "__main__":
    unittest.main()
