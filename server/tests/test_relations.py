"""Unit tests for the relation engine (graph edges with hysteresis)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mac_server.monitoring.events import (  # noqa: E402
    RelationConfig,
    RelationEngine,
    bbox_gap_frac,
    bbox_relation,
)
from mac_server.monitoring.presence import PresentEntity  # noqa: E402


COUCH = {"anchor_id": "couch_1", "bbox_xyxy": [400.0, 300.0, 900.0, 650.0], "depth_m": 3.0}


def entity(key: int, bbox: tuple, depth: float | None = 3.0, **kw) -> PresentEntity:
    return PresentEntity(
        entity_key=key,
        class_name=kw.get("class_name", "person"),
        state="present",
        first_seen=0.0,
        last_seen=0.0,
        last_bbox=bbox,
        depth_m=depth,
        lateral_m=kw.get("lateral_m"),
        forward_m=kw.get("forward_m"),
    )


def engine(**overrides) -> RelationEngine:
    return RelationEngine(RelationConfig(**overrides))


def run_ticks(eng, entities, labels, anchors, t0, t1, step=0.5, occluded=None):
    transitions = []
    t = t0
    while t <= t1:
        transitions.extend(eng.evaluate(entities, labels, anchors, t, occluded))
        t += step
    return transitions


class GeometryTest(unittest.TestCase):
    def test_gap_zero_when_overlapping(self) -> None:
        self.assertEqual(bbox_gap_frac((0, 0, 100, 100), [50, 50, 150, 150], 1468.6), 0.0)

    def test_gap_positive_when_apart(self) -> None:
        gap = bbox_gap_frac((0, 0, 100, 100), [200, 0, 300, 100], 1468.6)
        self.assertAlmostEqual(gap, 100 / 1468.6, places=4)

    def test_relation_left_of(self) -> None:
        self.assertEqual(bbox_relation((0, 0, 100, 100), [300, 0, 400, 100]), "left-of")

    def test_relation_overlapping(self) -> None:
        self.assertEqual(bbox_relation((0, 0, 100, 100), [50, 50, 150, 150]), "overlapping")

    def test_relation_above(self) -> None:
        self.assertEqual(bbox_relation((0, 0, 100, 100), [0, 300, 100, 400]), "above")


class OnRelationTest(unittest.TestCase):
    def test_on_opens_after_hold(self) -> None:
        eng = engine(open_hold_s=2.0)
        person = entity(1, (500.0, 250.0, 700.0, 640.0), depth=3.1)
        labels = {1: "person"}
        got = run_ticks(eng, [person], labels, [COUCH], 0.0, 3.0)
        opens = [t for t in got if t.action == "open" and t.relation == "on"]
        self.assertEqual(len(opens), 1)
        self.assertEqual(opens[0].subject_label, "person")
        self.assertEqual(opens[0].object_label, "couch_1")
        self.assertIn("arrangement", opens[0].details)

    def test_on_depth_mismatch_never_opens(self) -> None:
        eng = engine()
        person = entity(1, (500.0, 250.0, 700.0, 640.0), depth=6.0)  # 3m behind couch
        got = run_ticks(eng, [person], {1: "person"}, [COUCH], 0.0, 5.0)
        self.assertEqual([t for t in got if t.relation == "on"], [])

    def test_on_subsumes_near(self) -> None:
        eng = engine()
        person = entity(1, (500.0, 250.0, 700.0, 640.0), depth=3.0)
        got = run_ticks(eng, [person], {1: "person"}, [COUCH], 0.0, 5.0)
        self.assertEqual([t for t in got if t.relation == "near"], [])
        self.assertEqual(len([t for t in got if t.relation == "on" and t.action == "open"]), 1)

    def test_posture_majority_vote_at_close(self) -> None:
        eng = engine(open_hold_s=1.0, close_hold_s=1.0)
        lying = entity(1, (450.0, 400.0, 850.0, 560.0), depth=3.0)  # wide box
        eng.evaluate([lying], {1: "person"}, [COUCH], 0.0)
        for t in (1.5, 2.0, 2.5, 3.0):
            eng.evaluate([lying], {1: "person"}, [COUCH], t)
        away = entity(1, (100.0, 100.0, 200.0, 400.0), depth=1.0)
        closes = run_ticks(eng, [away], {1: "person"}, [COUCH], 4.0, 6.0)
        close = next(t for t in closes if t.action == "close" and t.relation == "on")
        self.assertEqual(close.details["posture"], "lying")
        self.assertGreater(close.details["duration_s"], 0)


class NearRelationTest(unittest.TestCase):
    def test_entity_pair_near(self) -> None:
        eng = engine(open_hold_s=1.0)
        person = entity(1, (400.0, 200.0, 600.0, 600.0), depth=3.0, lateral_m=0.0, forward_m=3.0)
        laptop = entity(
            2, (610.0, 400.0, 760.0, 520.0), depth=3.2,
            class_name="laptop", lateral_m=0.5, forward_m=3.2,
        )
        got = run_ticks(eng, [person, laptop], {1: "person", 2: "laptop"}, [], 0.0, 2.0)
        opens = [t for t in got if t.action == "open" and t.relation == "near"]
        self.assertEqual(len(opens), 1)
        self.assertEqual(opens[0].subject_label, "person")
        self.assertEqual(opens[0].object_key, 2)
        self.assertIsNotNone(opens[0].details["distance_m"])

    def test_far_pair_never_near(self) -> None:
        eng = engine()
        a = entity(1, (0.0, 0.0, 100.0, 200.0))
        b = entity(2, (1000.0, 0.0, 1100.0, 200.0), class_name="laptop")
        got = run_ticks(eng, [a, b], {1: "person", 2: "laptop"}, [], 0.0, 5.0)
        self.assertEqual([t for t in got if t.relation == "near"], [])

    def test_occlusion_pauses_close(self) -> None:
        """Subject occluded for far longer than close_hold_s: the edge stays
        open and survives to its reappearance — no close/reopen churn."""
        eng = engine(open_hold_s=1.0, close_hold_s=3.0)
        person = entity(1, (500.0, 250.0, 700.0, 640.0), depth=3.0)
        run_ticks(eng, [person], {1: "person"}, [COUCH], 0.0, 2.0)
        self.assertEqual(len(eng.open_relations), 1)

        # 30 seconds of occlusion (entity not in the present list).
        got = run_ticks(
            eng, [], {1: "person"}, [COUCH], 3.0, 33.0, occluded={1}
        )
        self.assertEqual([t for t in got if t.action == "close"], [])
        self.assertEqual(len(eng.open_relations), 1)

        # Reappears on the couch: still one continuous edge, no new open.
        got = run_ticks(eng, [person], {1: "person"}, [COUCH], 34.0, 36.0)
        self.assertEqual([t for t in got if t.action == "open"], [])

    def test_close_after_hold_when_visible(self) -> None:
        eng = engine(open_hold_s=1.0, close_hold_s=3.0)
        person = entity(1, (500.0, 250.0, 700.0, 640.0), depth=3.0)
        run_ticks(eng, [person], {1: "person"}, [COUCH], 0.0, 2.0)
        away = entity(1, (100.0, 100.0, 200.0, 400.0), depth=1.0)
        got = run_ticks(eng, [away], {1: "person"}, [COUCH], 3.0, 7.0)
        self.assertEqual(len([t for t in got if t.action == "close"]), 1)

    def test_entity_left_force_closes(self) -> None:
        eng = engine(open_hold_s=1.0)
        person = entity(1, (500.0, 250.0, 700.0, 640.0), depth=3.0)
        run_ticks(eng, [person], {1: "person"}, [COUCH], 0.0, 2.0)
        self.assertEqual(len(eng.open_relations), 1)
        closes = eng.entity_left(1, 10.0)
        self.assertEqual(len(closes), 1)
        self.assertEqual(closes[0].action, "close")
        self.assertEqual(eng.open_relations, [])


class MovedTest(unittest.TestCase):
    def test_micro_jitter_never_moves(self) -> None:
        eng = engine()
        got = []
        for i in range(20):
            jitter = entity(1, (500.0 + (i % 3) * 4, 300.0, 700.0 + (i % 3) * 4, 500.0))
            got.extend(eng.evaluate([jitter], {1: "person"}, [], i * 0.5))
        self.assertEqual([t for t in got if t.relation == "moved"], [])

    def test_real_displacement_fires_once(self) -> None:
        eng = engine(open_hold_s=1.0, move_min_frac=0.15)
        got = []
        # Rest at the left, then jump to the right side and stay there.
        for t in (0.0, 0.5, 1.0):
            got.extend(eng.evaluate([entity(1, (100.0, 300.0, 250.0, 500.0))], {1: "person"}, [], t))
        for t in (1.5, 2.0, 2.5, 3.0, 3.5):
            got.extend(eng.evaluate([entity(1, (900.0, 300.0, 1050.0, 500.0))], {1: "person"}, [], t))
        moves = [t for t in got if t.relation == "moved"]
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].details["from_zone"], "left")
        self.assertEqual(moves[0].details["to_zone"], "right")

    def test_brief_spike_does_not_move(self) -> None:
        """A single-tick bbox teleport (association hiccup) that reverts
        before open_hold_s never fires `moved`."""
        eng = engine(open_hold_s=1.0)
        home = entity(1, (100.0, 300.0, 250.0, 500.0))
        spike = entity(1, (900.0, 300.0, 1050.0, 500.0))
        got = []
        got.extend(eng.evaluate([home], {1: "person"}, [], 0.0))
        got.extend(eng.evaluate([spike], {1: "person"}, [], 0.5))
        for t in (1.0, 1.5, 2.0, 5.0):
            got.extend(eng.evaluate([home], {1: "person"}, [], t))
        self.assertEqual([t for t in got if t.relation == "moved"], [])

    def test_move_reports_nearest_anchor(self) -> None:
        eng = engine(open_hold_s=1.0)
        for t in (0.0, 0.5, 1.0):
            eng.evaluate([entity(1, (100.0, 300.0, 250.0, 500.0))], {1: "person"}, [COUCH], t)
        got = []
        for t in (1.5, 2.0, 2.5, 3.0):
            got.extend(
                eng.evaluate([entity(1, (500.0, 300.0, 700.0, 640.0))], {1: "person"}, [COUCH], t)
            )
        moves = [t for t in got if t.relation == "moved"]
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].details["nearest_anchor"], "couch_1")


if __name__ == "__main__":
    unittest.main()
