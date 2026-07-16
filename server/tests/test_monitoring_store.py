"""Unit tests for MonitoringStore (SQLite) and SceneStore."""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mac_server.monitoring.scenes import SceneStore  # noqa: E402
from mac_server.monitoring.store import MonitoringStore  # noqa: E402


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.store = MonitoringStore(self.data_dir)

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def test_track_roundtrip(self) -> None:
        db_id = self.store.insert_track("living", "person", 100.0)
        self.store.finish_track(db_id, 200.0, frames=500, entity_id=None, meta={"end": "exited"})
        conn = self.store.read_connection()
        row = conn.execute("SELECT class, t_end, frames FROM tracks WHERE id = ?", (db_id,)).fetchone()
        conn.close()
        self.assertEqual(row["class"], "person")
        self.assertEqual(row["t_end"], 200.0)
        self.assertEqual(row["frames"], 500)

    def test_event_open_close(self) -> None:
        event_id = self.store.insert_event(
            "living", "on_furniture", "person track#1", 100.0, None,
            object_label="couch_1", details={"posture": "sitting"},
        )
        events = self.store.events_between("living", 0, 1000)
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0]["t_end"])

        self.store.close_event(event_id, 150.0, {"posture": "sitting", "duration_s": 50.0})
        events = self.store.events_between("living", 0, 1000)
        self.assertEqual(events[0]["t_end"], 150.0)
        self.assertEqual(events[0]["details"]["duration_s"], 50.0)

    def test_events_between_filters(self) -> None:
        self.store.insert_event("living", "entered", "cat track#2", 100.0, 100.0)
        self.store.insert_event("living", "entered", "cat track#3", 300.0, 300.0)
        self.store.insert_event("bedroom", "entered", "cat track#4", 100.0, 100.0)
        events = self.store.events_between("living", 50, 200)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["subject_label"], "cat track#2")

    def test_entity_lifecycle(self) -> None:
        sig = {"kind": "hsv_hist_v1", "hist": [0.1] * 4, "size": 1.2, "pos": [0.0, 2.0]}
        entity_id = self.store.insert_entity("living", "person", "Person#1", sig, 100.0)
        self.assertEqual(self.store.entity_count("living", "person"), 1)
        candidates = self.store.candidate_entities("living", "person", since=0)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["label"], "Person#1")
        self.store.update_entity(entity_id, None, last_seen=200.0, visible_s=50.0)
        candidates = self.store.candidate_entities("living", "person", since=150.0)
        self.assertEqual(len(candidates), 1)

    def test_retention_sweep(self) -> None:
        old_t = time.time() - 30 * 86400
        snapshot = self.data_dir / "snap_old.jpg"
        snapshot.write_bytes(b"x" * 100)
        self.store.insert_event(
            "living", "entered", "cat track#1", old_t, old_t, snapshot_path=str(snapshot)
        )
        self.store.insert_event("living", "entered", "cat track#2", time.time(), time.time())
        self.store.sweep(retention_days=14, max_snapshot_mb=200, snapshots_root=self.data_dir)
        events = self.store.events_between("living", 0, time.time() + 10)
        self.assertEqual(len(events), 1)
        self.assertFalse(snapshot.exists())

    def test_digests(self) -> None:
        self.store.insert_digest("living", 100.0, 400.0, "Person#1 sat on the couch.", "apfel")
        digests = self.store.digests_between("living", 0, 1000)
        self.assertEqual(len(digests), 1)
        self.assertIn("couch", digests[0]["text"])


class SceneStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.scenes = SceneStore(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_create_and_list(self) -> None:
        scene = self.scenes.create_scene({"name": "Living Room"})
        self.assertEqual(scene["scene_id"], "living_room")
        self.assertEqual(len(self.scenes.list_scenes()), 1)
        with self.assertRaises(ValueError):
            self.scenes.create_scene({"name": "Living Room"})

    def test_freeze_anchors(self) -> None:
        self.scenes.create_scene({"name": "Living Room"})
        objects = [
            {"class": "couch", "bbox_xyxy": [0, 200, 400, 450], "depth_median_m": 2.4,
             "lateral_m": 0.1, "forward_m": 2.4},
            {"class": "person", "bbox_xyxy": [100, 100, 200, 400], "depth_median_m": 2.0},
            {"class": "tv", "bbox_xyxy": [500, 50, 700, 250], "depth_median_m": 3.1},
        ]
        anchors = self.scenes.set_anchors(
            "living_room", objects, anchor_classes={"couch", "tv", "chair"}
        )
        self.assertEqual(len(anchors), 2)
        self.assertEqual(anchors[0]["anchor_id"], "couch_1")
        self.assertEqual(anchors[1]["anchor_id"], "tv_1")
        # person is mobile — never an anchor
        classes = {a["class"] for a in anchors}
        self.assertNotIn("person", classes)
        # persisted
        scene = self.scenes.load_scene("living_room")
        self.assertEqual(len(scene["anchors"]), 2)


if __name__ == "__main__":
    unittest.main()
