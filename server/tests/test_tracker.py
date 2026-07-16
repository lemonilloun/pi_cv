"""Unit tests for the monitoring tracker (pure logic, no camera)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mac_server.monitoring.tracker import (  # noqa: E402
    GreedyTracker,
    TrackerConfig,
    centroid,
    iou,
    is_stationary,
)


def det(cls: str, bbox: list[float], conf: float = 0.9) -> dict:
    return {"class": cls, "confidence": conf, "bbox_xyxy": bbox, "depth_median_m": 2.0}


def make_tracker(**overrides) -> GreedyTracker:
    config = TrackerConfig(**overrides)
    return GreedyTracker(config, mobile_classes={"person", "cat", "dog"})


class IouTest(unittest.TestCase):
    def test_identical(self) -> None:
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint(self) -> None:
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_half_overlap(self) -> None:
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3, places=5)


class TrackerLifecycleTest(unittest.TestCase):
    def test_confirmation_after_hits(self) -> None:
        tracker = make_tracker(confirm_hits=3)
        bbox = [100, 100, 200, 300]
        t = 0.0
        updates = tracker.update([det("person", bbox)], t)
        self.assertEqual(updates.confirmed_new, [])
        updates = tracker.update([det("person", bbox)], t + 0.1)
        self.assertEqual(updates.confirmed_new, [])
        updates = tracker.update([det("person", bbox)], t + 0.2)
        self.assertEqual(len(updates.confirmed_new), 1)
        self.assertEqual(updates.confirmed_new[0].class_name, "person")
        self.assertEqual(updates.confirmed_new[0].state, "confirmed")

    def test_single_frame_ghost_never_confirms(self) -> None:
        tracker = make_tracker(confirm_hits=3)
        tracker.update([det("person", [0, 0, 50, 100])], 0.0)
        # Ghost absent on the next tick -> dropped silently.
        updates = tracker.update([], 0.1)
        self.assertEqual(updates.confirmed_new, [])
        self.assertEqual(updates.ended, [])
        self.assertEqual(len(tracker.tracks), 0)

    def test_association_under_jitter(self) -> None:
        tracker = make_tracker(confirm_hits=2)
        t = 0.0
        tracker.update([det("person", [100, 100, 200, 300])], t)
        updates = tracker.update([det("person", [104, 98, 206, 305])], t + 0.1)
        self.assertEqual(len(updates.confirmed_new), 1)
        track = updates.confirmed_new[0]
        # Third frame, still one track.
        tracker.update([det("person", [110, 95, 210, 310])], t + 0.2)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.tracks[track.track_id].hits, 3)

    def test_lost_then_reacquired(self) -> None:
        tracker = make_tracker(confirm_hits=2, lost_after_s=0.5, end_after_s=5.0)
        bbox = [100, 100, 200, 300]
        tracker.update([det("person", bbox)], 0.0)
        updates = tracker.update([det("person", bbox)], 0.1)
        track_id = updates.confirmed_new[0].track_id
        # Occlusion: nothing for 1s -> lost, but not ended.
        tracker.update([], 0.6)
        tracker.update([], 1.1)
        self.assertEqual(tracker.tracks[track_id].state, "lost")
        # Reappears near the old spot -> same track, confirmed again.
        updates = tracker.update([det("person", [105, 102, 205, 302])], 1.3)
        self.assertEqual(tracker.tracks[track_id].state, "confirmed")
        self.assertEqual(len(tracker.tracks), 1)

    def test_exit_via_empty_ticks(self) -> None:
        tracker = make_tracker(confirm_hits=2, lost_after_s=0.5, end_after_s=1.0)
        bbox = [100, 100, 200, 300]
        tracker.update([det("person", bbox)], 0.0)
        tracker.update([det("person", bbox)], 0.1)
        ended = []
        t = 0.2
        while t < 3.0 and not ended:
            updates = tracker.update([], t)
            ended.extend(updates.ended)
            t += 0.2
        self.assertEqual(len(ended), 1)
        self.assertEqual(ended[0].state, "ended")
        self.assertEqual(len(tracker.tracks), 0)

    def test_two_people_stay_distinct(self) -> None:
        tracker = make_tracker(confirm_hits=2)
        left = [100, 100, 200, 300]
        right = [600, 100, 700, 300]
        tracker.update([det("person", left), det("person", right)], 0.0)
        tracker.update([det("person", left), det("person", right)], 0.1)
        self.assertEqual(len(tracker.tracks), 2)
        states = {t.state for t in tracker.tracks.values()}
        self.assertEqual(states, {"confirmed"})

    def test_class_mismatch_never_associates(self) -> None:
        tracker = make_tracker(confirm_hits=2)
        bbox = [100, 100, 200, 300]
        tracker.update([det("person", bbox)], 0.0)
        tracker.update([det("cat", bbox)], 0.1)  # same box, different class
        # person tentative dropped (missed a tick), cat is a new tentative.
        classes = {t.class_name for t in tracker.tracks.values()}
        self.assertEqual(classes, {"cat"})

    def test_anchor_classes_ignored(self) -> None:
        tracker = make_tracker()
        tracker.update([det("couch", [0, 0, 500, 400])], 0.0)
        self.assertEqual(len(tracker.tracks), 0)

    def test_low_confidence_filtered(self) -> None:
        tracker = make_tracker(min_confidence=0.5)
        tracker.update([det("person", [0, 0, 100, 200], conf=0.4)], 0.0)
        self.assertEqual(len(tracker.tracks), 0)

    def test_force_end_all(self) -> None:
        tracker = make_tracker(confirm_hits=2)
        bbox = [100, 100, 200, 300]
        tracker.update([det("person", bbox)], 0.0)
        tracker.update([det("person", bbox)], 0.1)
        ended = tracker.force_end_all(0.2)
        self.assertEqual(len(ended), 1)
        self.assertEqual(len(tracker.tracks), 0)


class StationaryTest(unittest.TestCase):
    def test_stationary_detection(self) -> None:
        history = [(t * 0.1, 100.0 + (t % 2), 200.0) for t in range(60)]
        result = is_stationary(history, now=6.0, window_s=5.0, px_frac=0.02, frame_diag=1468.6)
        self.assertTrue(result)

    def test_moving_detection(self) -> None:
        history = [(t * 0.1, 100.0 + t * 10.0, 200.0) for t in range(60)]
        result = is_stationary(history, now=6.0, window_s=5.0, px_frac=0.02, frame_diag=1468.6)
        self.assertFalse(result)

    def test_insufficient_window_returns_none(self) -> None:
        history = [(0.0, 100.0, 200.0), (0.1, 101.0, 200.0)]
        result = is_stationary(history, now=0.2, window_s=5.0, px_frac=0.02, frame_diag=1468.6)
        self.assertIsNone(result)


class EmaTest(unittest.TestCase):
    def test_depth_ema_smooths(self) -> None:
        tracker = make_tracker(confirm_hits=2, ema_alpha=0.5)
        bbox = [100, 100, 200, 300]
        d1 = dict(det("person", bbox))
        d1["depth_median_m"] = 2.0
        d2 = dict(det("person", bbox))
        d2["depth_median_m"] = 4.0
        tracker.update([d1], 0.0)
        tracker.update([d2], 0.1)
        track = next(iter(tracker.tracks.values()))
        self.assertAlmostEqual(track.depth_m, 3.0)  # 0.5*4 + 0.5*2


if __name__ == "__main__":
    unittest.main()
