"""Unit tests for the monitoring event engine (pure logic)."""

from __future__ import annotations

import sys
import unittest
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mac_server.monitoring.events import (  # noqa: E402
    EventEngine,
    EventRuleConfig,
    bbox_posture,
)
from mac_server.monitoring.tracker import Track  # noqa: E402


def make_track(track_id=1, cls="person", bbox=(100, 100, 200, 400), depth=2.0) -> Track:
    track = Track(
        track_id=track_id,
        class_name=cls,
        state="confirmed",
        first_seen=0.0,
        last_seen=0.0,
        hits=5,
        bbox=bbox,
        bbox_history=deque(maxlen=64),
        centroid_history=deque(maxlen=64),
        depth_m=depth,
    )
    return track


COUCH = {"anchor_id": "couch_1", "class": "couch", "bbox_xyxy": [80, 200, 400, 450], "depth_m": 2.2}


class PostureTest(unittest.TestCase):
    def test_vertical_is_sitting(self) -> None:
        cfg = EventRuleConfig()
        self.assertEqual(bbox_posture((0, 0, 100, 300), cfg), "sitting")

    def test_horizontal_is_lying(self) -> None:
        cfg = EventRuleConfig()
        self.assertEqual(bbox_posture((0, 0, 300, 100), cfg), "lying")

    def test_square_is_unclear(self) -> None:
        cfg = EventRuleConfig()
        self.assertEqual(bbox_posture((0, 0, 100, 100), cfg), "unclear")


class HysteresisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = EventRuleConfig(open_hold_s=2.0, close_hold_s=3.0)
        self.engine = EventEngine(self.cfg)
        # person overlapping the couch at matching depth
        self.track = make_track(bbox=(150, 220, 250, 430), depth=2.3)

    def tick(self, t: float, on_couch: bool):
        track = self.track if on_couch else make_track(bbox=(700, 100, 800, 400), depth=4.0)
        track.track_id = self.track.track_id
        return self.engine.evaluate([track], [COUCH], t)

    def test_opens_only_after_hold(self) -> None:
        transitions = self.tick(0.0, True)
        self.assertEqual(transitions, [])
        transitions = self.tick(1.0, True)
        self.assertEqual(transitions, [])
        transitions = self.tick(2.1, True)
        opened = [tr for tr in transitions if tr.action == "open"]
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0].event_type, "on_furniture")
        self.assertEqual(opened[0].object_label, "couch_1")
        self.assertAlmostEqual(opened[0].t_start, 0.0)

    def test_flapping_no_event(self) -> None:
        # Condition true for 1s, false, true for 1s, false — never opens.
        for t, on in [(0.0, True), (1.0, True), (1.5, False), (2.0, True), (3.0, True), (3.5, False)]:
            transitions = self.tick(t, on)
            self.assertEqual([tr for tr in transitions if tr.action == "open"], [])

    def test_close_bridges_brief_absence(self) -> None:
        # Open the event.
        self.tick(0.0, True)
        self.tick(2.1, True)
        # Brief absence shorter than close_hold — must NOT close.
        transitions = self.tick(3.0, False)
        self.assertEqual([tr for tr in transitions if tr.action == "close"], [])
        transitions = self.tick(4.0, True)  # back on the couch
        self.assertEqual([tr for tr in transitions if tr.action == "close"], [])
        self.assertEqual(len(self.engine.open_events), 1)

    def test_close_after_hold(self) -> None:
        self.tick(0.0, True)
        self.tick(2.1, True)
        self.tick(5.0, False)
        transitions = self.tick(8.1, False)
        closed = [tr for tr in transitions if tr.action == "close"]
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].event_type, "on_furniture")
        self.assertIsNotNone(closed[0].t_end)
        self.assertIn("duration_s", closed[0].details)

    def test_depth_mismatch_blocks_on_furniture(self) -> None:
        engine = EventEngine(self.cfg)
        # Overlapping in image, but person is 2m in front of the couch.
        track = make_track(bbox=(150, 220, 250, 430), depth=0.2)
        engine.evaluate([track], [COUCH], 0.0)
        transitions = engine.evaluate([track], [COUCH], 2.5)
        self.assertEqual([tr for tr in transitions if tr.action == "open"], [])

    def test_posture_majority_vote_at_close(self) -> None:
        engine = EventEngine(EventRuleConfig(open_hold_s=0.5, close_hold_s=0.5))
        sitting = make_track(bbox=(150, 220, 250, 430), depth=2.3)  # h/w = 2.1
        lying = make_track(bbox=(100, 300, 400, 420), depth=2.3)  # h/w = 0.4
        engine.evaluate([sitting], [COUCH], 0.0)
        engine.evaluate([sitting], [COUCH], 0.6)  # open, vote sitting
        engine.evaluate([sitting], [COUCH], 1.0)  # vote sitting
        engine.evaluate([lying], [COUCH], 1.4)  # vote lying (same track id)
        away = make_track(bbox=(700, 0, 750, 50), depth=5.0)
        engine.evaluate([away], [COUCH], 2.5)  # condition breaks (starts close hold)
        transitions = engine.evaluate([away], [COUCH], 3.1)  # close hold elapsed
        closed = [tr for tr in transitions if tr.action == "close"]
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].details["posture"], "sitting")


class PointEventsTest(unittest.TestCase):
    def test_entered_and_exited(self) -> None:
        engine = EventEngine(EventRuleConfig())
        track = make_track()
        entered = engine.track_confirmed(track, 10.0)
        self.assertEqual(entered.event_type, "entered")
        self.assertEqual(entered.action, "point")

        track.first_seen = 10.0
        track.last_seen = 40.0
        transitions = engine.track_ended(track, 40.0)
        exited = [tr for tr in transitions if tr.event_type == "exited"]
        self.assertEqual(len(exited), 1)
        self.assertAlmostEqual(exited[0].details["visible_s"], 30.0)

    def test_track_end_closes_open_events(self) -> None:
        cfg = EventRuleConfig(open_hold_s=0.5, close_hold_s=3.0)
        engine = EventEngine(cfg)
        track = make_track(bbox=(150, 220, 250, 430), depth=2.3)
        engine.evaluate([track], [COUCH], 0.0)
        engine.evaluate([track], [COUCH], 0.6)
        self.assertEqual(len(engine.open_events), 1)
        track.last_seen = 5.0
        transitions = engine.track_ended(track, 5.0)
        closed = [tr for tr in transitions if tr.action == "close"]
        self.assertEqual(len(closed), 1)
        self.assertEqual(len(engine.open_events), 0)

    def test_force_close_all(self) -> None:
        cfg = EventRuleConfig(open_hold_s=0.5, close_hold_s=3.0)
        engine = EventEngine(cfg)
        track = make_track(bbox=(150, 220, 250, 430), depth=2.3)
        engine.evaluate([track], [COUCH], 0.0)
        engine.evaluate([track], [COUCH], 0.6)
        transitions = engine.force_close_all(2.0)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].action, "close")


if __name__ == "__main__":
    unittest.main()
