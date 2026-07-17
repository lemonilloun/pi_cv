"""Unit tests for the presence layer (class-keyed entities, edge-exit rule)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mac_server.monitoring.presence import (  # noqa: E402
    PresenceConfig,
    PresenceManager,
    bbox_near_edge,
)
from mac_server.monitoring.tracker import Track  # noqa: E402


def make_track(track_id: int, cls: str, bbox: tuple, now: float = 0.0) -> Track:
    return Track(
        track_id=track_id,
        class_name=cls,
        state="confirmed",
        first_seen=now,
        last_seen=now,
        hits=3,
        bbox=bbox,
    )


MID_BBOX = (500.0, 300.0, 700.0, 500.0)  # far from every border of 1280x720
EDGE_BBOX = (10.0, 300.0, 150.0, 500.0)  # touches the left border


def manager(**overrides) -> PresenceManager:
    return PresenceManager(PresenceConfig(**overrides))


class NearEdgeTest(unittest.TestCase):
    def test_mid_frame_not_near_edge(self) -> None:
        self.assertFalse(bbox_near_edge(MID_BBOX, 1280, 720, 0.10))

    def test_left_border_near_edge(self) -> None:
        self.assertTrue(bbox_near_edge(EDGE_BBOX, 1280, 720, 0.10))

    def test_bottom_border_near_edge(self) -> None:
        self.assertTrue(bbox_near_edge((500, 300, 700, 715), 1280, 720, 0.10))


class PresenceLifecycleTest(unittest.TestCase):
    def test_entered_once(self) -> None:
        mgr = manager()
        track = make_track(1, "person", MID_BBOX)
        event = mgr.on_track_confirmed(track, 0.0)
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, "entered")
        self.assertEqual(event.entity.class_name, "person")
        self.assertEqual(mgr.label_of(event.entity), "person")

    def test_flicker_no_events(self) -> None:
        """Track dies mid-frame and a new one confirms: no exited, no
        entered — the same entity re-binds."""
        mgr = manager(exit_absent_s=25.0)
        track = make_track(1, "laptop", MID_BBOX)
        first = mgr.on_track_confirmed(track, 0.0)
        self.assertEqual(first.event_type, "entered")

        mgr.on_track_ended(track, 5.0)
        # A few empty seconds pass: still no exit (mid-frame = occluded).
        self.assertEqual(mgr.update([], 10.0), [])

        track2 = make_track(2, "laptop", (510, 310, 710, 510))
        rebind = mgr.on_track_confirmed(track2, 12.0)
        self.assertIsNone(rebind)  # same laptop, no new entered
        self.assertEqual(len(mgr.alive_entities()), 1)
        self.assertEqual(mgr.alive_entities()[0].state, "present")
        self.assertIs(mgr.alive_entities()[0], first.entity)

    def test_exit_requires_edge_and_absence(self) -> None:
        mgr = manager(exit_absent_s=25.0, presence_timeout_s=300.0)
        track = make_track(1, "person", EDGE_BBOX)
        mgr.on_track_confirmed(track, 0.0)
        mgr.update([track], 1.0)
        mgr.on_track_ended(track, 2.0)

        # Absent 10s near the edge: not yet gone.
        self.assertEqual(mgr.update([], 12.0), [])
        # Absent 26s near the edge: exited.
        events = mgr.update([], 27.5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "exited")
        self.assertEqual(events[0].details["reason"], "left_frame")

    def test_mid_frame_disappearance_waits_for_timeout(self) -> None:
        mgr = manager(exit_absent_s=25.0, presence_timeout_s=300.0)
        track = make_track(1, "person", MID_BBOX)
        mgr.on_track_confirmed(track, 0.0)
        mgr.on_track_ended(track, 1.0)

        # Way past exit_absent_s but mid-frame: still occluded.
        self.assertEqual(mgr.update([], 100.0), [])
        self.assertEqual(mgr.alive_entities()[0].state, "occluded")
        # Fallback timeout finally releases it.
        events = mgr.update([], 302.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].details["reason"], "timeout")

    def test_labels_suffix_only_during_concurrency(self) -> None:
        mgr = manager()
        e1 = mgr.on_track_confirmed(make_track(1, "person", MID_BBOX), 0.0).entity
        self.assertEqual(mgr.label_of(e1), "person")

        e2 = mgr.on_track_confirmed(make_track(2, "person", (100, 100, 200, 300)), 1.0).entity
        self.assertEqual(mgr.label_of(e1), "person_1")
        self.assertEqual(mgr.label_of(e2), "person_2")

        # Second person leaves -> the first goes back to plain "person".
        e2.state = "left"
        self.assertEqual(mgr.label_of(e1), "person")

    def test_rebind_prefers_nearest_position(self) -> None:
        mgr = manager()
        left = mgr.on_track_confirmed(make_track(1, "person", (100, 100, 200, 300)), 0.0).entity
        right = mgr.on_track_confirmed(make_track(2, "person", (900, 100, 1000, 300)), 0.0).entity
        mgr.on_track_ended(make_track(1, "person", (100, 100, 200, 300)), 1.0)
        mgr.on_track_ended(make_track(2, "person", (900, 100, 1000, 300)), 1.0)

        rebind = mgr.on_track_confirmed(make_track(3, "person", (880, 110, 990, 310)), 2.0)
        self.assertIsNone(rebind)
        self.assertEqual(right.state, "present")
        self.assertEqual(left.state, "occluded")

    def test_rebind_uses_signature_similarity(self) -> None:
        def sim(a: dict, b: dict) -> float:
            return 1.0 if a.get("tag") == b.get("tag") else 0.0

        mgr = PresenceManager(PresenceConfig(rebind_signature_weight=0.9), similarity=sim)
        t_red = make_track(1, "person", (100, 100, 200, 300))
        t_red.signature = {"tag": "red"}
        t_blue = make_track(2, "person", (900, 100, 1000, 300))
        t_blue.signature = {"tag": "blue"}
        red = mgr.on_track_confirmed(t_red, 0.0).entity
        blue = mgr.on_track_confirmed(t_blue, 0.0).entity
        mgr.update([t_red, t_blue], 0.5)
        mgr.on_track_ended(t_red, 1.0)
        mgr.on_track_ended(t_blue, 1.0)

        # Red reappears at BLUE's old position — signature should win.
        t_back = make_track(3, "person", (900, 100, 1000, 300))
        t_back.signature = {"tag": "red"}
        mgr.on_track_confirmed(t_back, 2.0)
        self.assertEqual(red.state, "present")
        self.assertEqual(blue.state, "occluded")

    def test_force_leave_all(self) -> None:
        mgr = manager()
        mgr.on_track_confirmed(make_track(1, "person", MID_BBOX), 0.0)
        mgr.on_track_confirmed(make_track(2, "cat", (100, 100, 200, 200)), 0.0)
        events = mgr.force_leave_all(5.0)
        self.assertEqual(len(events), 2)
        self.assertEqual({e.event_type for e in events}, {"exited"})
        mgr.prune_left()
        self.assertEqual(mgr.alive_entities(), [])
        self.assertEqual(mgr.entities, {})

    def test_visible_time_accumulates(self) -> None:
        mgr = manager(exit_absent_s=5.0)
        track = make_track(1, "person", EDGE_BBOX)
        mgr.on_track_confirmed(track, 0.0)
        mgr.update([track], 10.0)
        mgr.on_track_ended(track, 10.0)
        events = mgr.update([], 16.0)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].details["visible_s"], 10.0, places=1)


if __name__ == "__main__":
    unittest.main()
