"""Integration test of the v2 monitoring pipeline: a stubbed cv_worker feeds
scripted detections through the REAL MonitorController — tracker → presence →
relations → graph → SQLite. No servers, no camera, no AI services."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mac_server.monitoring.controller import MonitorController  # noqa: E402
from mac_server.monitoring.scenes import SceneStore  # noqa: E402
from mac_server.monitoring.store import MonitoringStore  # noqa: E402


class StubObjectsStore:
    def __init__(self):
        self._cond = threading.Condition()
        self._seq = 0
        self._batch = None

    def publish(self, objects, frame_index):
        with self._cond:
            self._seq += 1
            self._batch = {
                "objects": objects,
                "pi_frame_index": frame_index,
                "computed_at": time.time(),
            }
            self._cond.notify_all()

    def wait_for_next(self, last_seq, timeout=1.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._seq, self._batch

    def latest(self):
        return self._seq, self._batch


class StubCvWorker:
    def __init__(self):
        self.objects_store = StubObjectsStore()

    def status(self):
        return {"state": "running"}


class StubFrameStore:
    def latest(self):
        return 0, None  # no frame -> snapshots/signatures skip gracefully


class StubFrameHub:
    def get(self, name):
        return StubFrameStore()


class StubRegistry:
    def send_command(self, **kw):
        pass


def det(cls, bbox, conf=0.9, depth=2.5):
    return {"class": cls, "confidence": conf, "bbox_xyxy": bbox,
            "depth_median_m": depth, "lateral_m": 0.1, "forward_m": depth}


COUCH = det("couch", [400, 300, 900, 650], depth=2.5)
PERSON = det("person", [500, 250, 700, 640], depth=2.6)


class ControllerIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        data_dir = Path(self._tmp.name)
        config = json.load(open(REPO_ROOT / "config/default.json"))["monitoring"]
        config["agent"] = {"enabled": False}
        config["vision"] = {"enabled": False}
        config["embeddings"] = {"enabled": False}  # no torch in the test

        self.cv = StubCvWorker()
        self.store = MonitoringStore(data_dir)
        scene_store = SceneStore(data_dir / "scenes")
        self.ctrl = MonitorController(
            cv_worker=self.cv, frame_hub=StubFrameHub(), registry=StubRegistry(),
            scene_store=scene_store, store=self.store, config=config,
        )
        scene = scene_store.create_scene({"name": "Test room"})
        self.scene_id = scene["scene_id"]
        scene_store.set_anchors(self.scene_id, [COUCH], anchor_classes={"couch"})

    def tearDown(self) -> None:
        try:
            self.ctrl.stop()
        except Exception:
            pass
        self.store.close()
        self._tmp.cleanup()

    def _feed(self, objects, ticks, fi_start, dt=0.12):
        fi = fi_start
        for _ in range(ticks):
            fi += 1
            self.cv.objects_store.publish(list(objects), fi)
            time.sleep(dt)
        return fi

    def test_full_pipeline(self) -> None:
        self.ctrl.start(self.scene_id)

        # Person sits mid-frame on the couch for ~5 simulated seconds.
        fi = self._feed([PERSON, COUCH], 40, 0)
        status = self.ctrl.status()
        self.assertTrue(status["active"])
        labels = [e["label"] for e in status["entities"]]
        self.assertEqual(labels, ["person"])  # plain class, no numbering
        relations = [r for e in status["entities"] for r in e["relations"]]
        self.assertTrue(
            any("on couch" in r for r in relations),
            f"no on-couch relation in {relations}",
        )

        # Detection flicker: person vanishes mid-frame for ~1s of ticks —
        # entity must stay (occluded), no exit, no duplicate entered later.
        fi = self._feed([COUCH], 8, fi)
        status = self.ctrl.status()
        self.assertIn("person", [e["label"] for e in status["entities"]])

        # Person re-appears: same entity re-binds.
        fi = self._feed([PERSON, COUCH], 10, fi)
        status = self.ctrl.status()
        self.assertEqual([e["label"] for e in status["entities"]], ["person"])

        self.ctrl.stop()

        edges = self.store.edges_between(self.scene_id, 0, time.time() + 10)
        relations = [e["relation"] for e in edges]
        self.assertEqual(relations.count("entered"), 1, f"flicker spam: {relations}")
        self.assertIn("on", relations)
        self.assertIn("exited", relations)  # forced by session stop
        self.assertTrue(all(e["t_end"] is not None for e in edges))

        node_labels = {n["label"] for n in self.store.nodes_for_scene(self.scene_id)}
        self.assertLessEqual({"person", "couch_1"}, node_labels)

    def test_restart_reuses_node(self) -> None:
        self.ctrl.start(self.scene_id)
        self._feed([PERSON, COUCH], 10, 0)
        self.ctrl.stop()

        self.ctrl.start(self.scene_id)
        self._feed([PERSON, COUCH], 10, 1000)
        self.ctrl.stop()

        entity_nodes = [
            n for n in self.store.nodes_for_scene(self.scene_id, "entity")
            if n["class"] == "person"
        ]
        self.assertEqual(len(entity_nodes), 1, "second session must reuse the person node")


if __name__ == "__main__":
    unittest.main()
