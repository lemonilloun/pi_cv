"""Unit tests for the scene knowledge graph (nodes + episodic edges)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mac_server.monitoring.events import (  # noqa: E402
    ActiveRelation,
    RelationTransition,
)
from mac_server.monitoring.presence import PresenceEvent, PresentEntity  # noqa: E402
from mac_server.monitoring.graph import SceneGraph  # noqa: E402
from mac_server.monitoring.store import MonitoringStore  # noqa: E402


def entity(key: int, cls: str = "person") -> PresentEntity:
    return PresentEntity(
        entity_key=key,
        class_name=cls,
        state="present",
        first_seen=0.0,
        last_seen=0.0,
        last_bbox=(100.0, 100.0, 200.0, 300.0),
    )


class SceneGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MonitoringStore(Path(self._tmp.name))
        self.graph = SceneGraph(self.store, "living_room")

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def test_singleton_node_label_is_plain_class(self) -> None:
        node_id = self.graph.bind_entity(entity(1, "laptop"), 10.0)
        nodes = self.store.nodes_for_scene("living_room", "entity")
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["label"], "laptop")
        self.assertEqual(nodes[0]["id"], node_id)

    def test_concurrent_same_class_gets_suffixed_node(self) -> None:
        self.graph.bind_entity(entity(1), 0.0)
        self.graph.bind_entity(entity(2), 1.0)
        labels = {n["label"] for n in self.store.nodes_for_scene("living_room", "entity")}
        self.assertEqual(labels, {"person", "person_2"})

    def test_node_reused_across_sessions(self) -> None:
        """The laptop that leaves and returns binds to the SAME node —
        its edge history accumulates on one identity."""
        first = entity(1, "laptop")
        node_a = self.graph.bind_entity(first, 0.0)
        self.graph.release_entity(first, 100.0, visible_s=100.0)

        second = entity(2, "laptop")
        node_b = self.graph.bind_entity(second, 200.0)
        self.assertEqual(node_a, node_b)
        nodes = self.store.nodes_for_scene("living_room", "entity")
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["total_visible_s"], 100.0)

    def test_plain_label_preferred_on_rebind(self) -> None:
        a, b = entity(1), entity(2)
        self.graph.bind_entity(a, 0.0)
        self.graph.bind_entity(b, 1.0)
        self.graph.release_entity(a, 10.0, visible_s=10.0)
        self.graph.release_entity(b, 10.0, visible_s=9.0)
        # A single person returns: binds to the plain "person" node.
        back = entity(3)
        self.graph.bind_entity(back, 20.0)
        nodes = {n["id"]: n["label"] for n in self.store.nodes_for_scene("living_room", "entity")}
        self.assertEqual(nodes[back.node_db_id], "person")

    def test_presence_events_become_unary_edges(self) -> None:
        person = entity(1)
        node_id = self.graph.bind_entity(person, 5.0)
        self.graph.record_presence_event(
            PresenceEvent(event_type="entered", entity=person, t=5.0), node_id
        )
        edges = self.store.edges_between("living_room", 0.0, 10.0)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["relation"], "entered")
        self.assertEqual(edges[0]["subject_label"], "person")
        self.assertIsNone(edges[0]["object_label"])

    def test_durational_edge_open_close(self) -> None:
        person = entity(1)
        self.graph.bind_entity(person, 0.0)
        self.graph.ensure_anchor_nodes(
            [{"anchor_id": "couch_1", "class": "couch", "bbox_xyxy": [0, 0, 1, 1]}], 0.0
        )
        active = ActiveRelation(
            key=("on", 1, "couch_1"), relation="on", subject_key=1,
            subject_label="person", object_key=None, object_label="couch_1",
            t_start=10.0,
        )
        edge_id = self.graph.record_transition(
            RelationTransition(
                action="open", relation="on", subject_key=1, subject_label="person",
                object_key=None, object_label="couch_1", t_start=10.0, t_end=None,
                details={"overlap": 0.5}, active=active,
            )
        )
        self.assertEqual(active.db_id, edge_id)
        open_edges = self.store.edges_between("living_room", 0.0, 100.0)
        self.assertIsNone(open_edges[0]["t_end"])

        self.graph.record_transition(
            RelationTransition(
                action="close", relation="on", subject_key=1, subject_label="person",
                object_key=None, object_label="couch_1", t_start=10.0, t_end=40.0,
                details={"overlap": 0.5, "duration_s": 30.0, "posture": "sitting"},
                active=active,
            )
        )
        edges = self.store.edges_between("living_room", 0.0, 100.0)
        self.assertEqual(edges[0]["t_end"], 40.0)
        self.assertEqual(edges[0]["details"]["posture"], "sitting")
        self.assertEqual(edges[0]["object_label"], "couch_1")

    def test_entity_pair_edge_uses_node_labels(self) -> None:
        person, laptop = entity(1), entity(2, "laptop")
        self.graph.bind_entity(person, 0.0)
        self.graph.bind_entity(laptop, 0.0)
        self.graph.record_transition(
            RelationTransition(
                action="point", relation="near", subject_key=1, subject_label="person",
                object_key=2, object_label=None, t_start=5.0, t_end=5.0,
                details={"gap_frac": 0.01},
            )
        )
        edges = self.store.edges_between("living_room", 0.0, 10.0)
        self.assertEqual(edges[0]["subject_label"], "person")
        self.assertEqual(edges[0]["object_label"], "laptop")

    def test_key_moment_marking(self) -> None:
        person = entity(1)
        node_id = self.graph.bind_entity(person, 0.0)
        edge_id = self.graph.record_presence_event(
            PresenceEvent(event_type="entered", entity=person, t=1.0), node_id
        )
        self.store.mark_edge_key_moment(edge_id, "snap/1.jpg")
        edges = self.store.edges_between("living_room", 0.0, 10.0, key_only=True)
        self.assertEqual(len(edges), 1)
        self.assertTrue(edges[0]["key_moment"])
        self.assertEqual(edges[0]["snapshot_path"], "snap/1.jpg")

    def test_stale_open_edges_closed_on_startup(self) -> None:
        person = entity(1)
        self.graph.bind_entity(person, 0.0)
        self.store.insert_edge("living_room", person.node_db_id, None, "on", 10.0, None, {})
        # Simulate a crash + new session for the same scene.
        graph2 = SceneGraph(self.store, "living_room")
        edges = self.store.edges_between("living_room", 0.0, 100.0)
        self.assertTrue(all(e["t_end"] is not None for e in edges))
        self.assertIsNotNone(graph2)

    def test_window_query_includes_open_edges(self) -> None:
        person = entity(1)
        self.graph.bind_entity(person, 0.0)
        self.store.insert_edge("living_room", person.node_db_id, None, "on", 10.0, None, {})
        edges = self.store.edges_between("living_room", 50.0, 100.0)
        self.assertEqual(len(edges), 1)  # open edge overlaps every later window

    def test_now_state_symmetric_near(self) -> None:
        person, laptop = entity(1), entity(2, "laptop")
        labels = {1: "person", 2: "laptop"}
        rel = ActiveRelation(
            key=("near", 1, ("entity", 2)), relation="near", subject_key=1,
            subject_label="person", object_key=2, object_label=None, t_start=0.0,
        )
        state = self.graph.now_state([person, laptop], labels, [rel])
        by_label = {s["label"]: s for s in state}
        self.assertIn("near laptop", by_label["person"]["relations"])
        self.assertIn("near person", by_label["laptop"]["relations"])


if __name__ == "__main__":
    unittest.main()
