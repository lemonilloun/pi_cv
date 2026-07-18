"""Unit tests for Scene3D pure logic: scale alignment, object association,
labeling, graph edges, floor plan, session IO and the seg decode NMS."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src"), str(REPO_ROOT / "client/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np

from mac_server.scene3d.poses_step import frame_scale, median_scale
from mac_server.scene3d.objects_step import ObjectBank, cosine
from mac_server.scene3d.graph_step import build_edges, label_objects
from mac_server.scene3d.tsdf_step import estimate_up_axis, make_floor_plan
from mac_server.scene3d.session_io import SceneSession, list_sessions
from pi_client.seg_postprocess import numpy_nms, order_endnodes


class ScaleTest(unittest.TestCase):
    def test_median_scale_rejects_outliers(self) -> None:
        scales = {i: 2.0 + 0.02 * (i % 3) for i in range(20)}
        scales[5] = 40.0  # broken frame
        scale, iqr, rejected = median_scale(scales)
        self.assertAlmostEqual(scale, 2.02, delta=0.05)
        self.assertIn(5, rejected)
        self.assertLess(iqr, 0.15)

    def test_frame_scale_median_of_ratios(self) -> None:
        depth = np.full((100, 100), 3.0, dtype=np.float32)
        uv = [(10.0, 10.0), (50.0, 50.0), (90.0, 90.0)] * 5
        d_colmap = [1.5] * 15  # metric/colmap = 2.0
        self.assertAlmostEqual(frame_scale(depth, uv, d_colmap), 2.0, places=5)

    def test_frame_scale_needs_enough_points(self) -> None:
        depth = np.full((10, 10), 3.0, dtype=np.float32)
        self.assertIsNone(frame_scale(depth, [(1.0, 1.0)], [1.0]))


class ObjectBankTest(unittest.TestCase):
    def _candidate(self, track=1, cls="chair", centroid=(0, 0, 2), emb=None, kf=1):
        return {
            "track_id": track,
            "class": cls,
            "centroid": list(centroid),
            "dino_emb": emb or [1.0, 0.0, 0.0],
            "clip_emb": None,
            "points": np.zeros((5, 3)),
            "keyframe": kf,
        }

    def test_track_id_prior_merges(self) -> None:
        bank = ObjectBank()
        a = bank.add(self._candidate(track=7, centroid=(0, 0, 2)))
        # Far away, dissimilar embedding — but same live track wins.
        b = bank.add(self._candidate(track=7, centroid=(3, 0, 2), emb=[0.0, 1.0, 0.0]))
        self.assertEqual(a, b)
        self.assertEqual(len(bank.objects), 1)

    def test_embedding_association(self) -> None:
        bank = ObjectBank(dino_cos_min=0.6, centroid_max_m=0.5)
        bank.add(self._candidate(track=-1, centroid=(0, 0, 2), emb=[1.0, 0.0, 0.0]))
        idx = bank.add(self._candidate(track=-1, centroid=(0.1, 0, 2.1), emb=[0.95, 0.05, 0.0]))
        self.assertEqual(idx, 0)
        self.assertEqual(len(bank.objects), 1)

    def test_distance_gate_creates_new_object(self) -> None:
        bank = ObjectBank(centroid_max_m=0.5)
        bank.add(self._candidate(track=-1, centroid=(0, 0, 2)))
        idx = bank.add(self._candidate(track=-1, centroid=(2, 0, 2)))  # same look, 2m away
        self.assertEqual(idx, 1)

    def test_finalize_drops_rare(self) -> None:
        bank = ObjectBank()
        for i in range(4):
            bank.add(self._candidate(track=1, kf=i))
        bank.add(self._candidate(track=99, cls="cup", centroid=(5, 5, 5)))
        final = bank.finalize(min_observations=3)
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["class_top"], "chair")
        self.assertEqual(final[0]["n_observations"], 4)

    def test_cosine(self) -> None:
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0, places=6)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0, places=6)


class LabelTest(unittest.TestCase):
    def test_coco_prior_boost(self) -> None:
        text = {"chair": [1.0, 0.0], "table": [0.9, 0.1]}
        # Normalize
        for k in text:
            v = np.asarray(text[k]); text[k] = list(v / np.linalg.norm(v))
        objects = [
            {"object_id": 0, "class_top": "chair", "clip_emb": list(np.asarray([0.95, 0.05]) / np.linalg.norm([0.95, 0.05])), "centroid": [0, 0, 0], "n_observations": 3}
        ]
        labeled = label_objects(objects, text, coco_boost=0.5)
        self.assertEqual(labeled[0]["label_top"], "chair")
        self.assertEqual(len(labeled[0]["labels"]), 2)

    def test_no_emb_falls_back_to_coco(self) -> None:
        labeled = label_objects(
            [{"object_id": 0, "class_top": "tv", "clip_emb": None, "centroid": [0, 0, 0], "n_observations": 3}],
            {"chair": [1.0, 0.0]},
        )
        self.assertEqual(labeled[0]["label_top"], "tv")


class EdgesTest(unittest.TestCase):
    def test_near_edge(self) -> None:
        objects = [
            {"object_id": 0, "centroid": [0, 0, 0], "obb": None},
            {"object_id": 1, "centroid": [0.3, 0, 0], "obb": None},
            {"object_id": 2, "centroid": [5, 0, 0], "obb": None},
        ]
        edges = build_edges(objects, up=[0, 0, 1], near_max_m=0.5)
        rel = {(e["src"], e["dst"], e["relation"]) for e in edges}
        self.assertIn((0, 1, "near"), rel)
        self.assertNotIn((0, 2, "near"), rel)

    def test_on_edge(self) -> None:
        eye = np.eye(3).tolist()
        table = {"object_id": 1, "centroid": [0, 0, 0.4],
                 "obb": {"center": [0, 0, 0.4], "extent": [1.0, 1.0, 0.8], "rotation": eye}}
        monitor = {"object_id": 0, "centroid": [0.1, 0, 1.0],
                   "obb": {"center": [0.1, 0, 1.0], "extent": [0.4, 0.2, 0.4], "rotation": eye}}
        edges = build_edges([monitor, table], up=[0, 0, 1], near_max_m=0.1, on_gap_m=0.1)
        rel = {(e["src"], e["dst"], e["relation"]) for e in edges}
        self.assertIn((0, 1, "on"), rel)


class PlanTest(unittest.TestCase):
    def test_up_axis_level_camera(self) -> None:
        # Identity pose: camera +Y (down) is world +Y -> up is -Y.
        up = estimate_up_axis({1: np.eye(4)})
        np.testing.assert_allclose(up, [0, -1, 0], atol=1e-6)

    def test_floor_plan_bounds(self) -> None:
        rng = np.random.default_rng(1)
        pts = rng.uniform(-2, 2, size=(5000, 2))
        heights = rng.uniform(0.2, 1.5, size=5000)
        grid, origin, res = make_floor_plan(pts, heights, resolution_m=0.05)
        self.assertGreater(grid.max(), 0)
        self.assertLess(origin[0], -2)
        self.assertEqual(res, 0.05)


class SessionIoTest(unittest.TestCase):
    def test_intrinsics_fallback_and_listing(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "session_test"
            kf = root / "keyframes" / "000001"
            kf.mkdir(parents=True)
            cv2.imwrite(str(kf / "rgb.jpg"), np.zeros((864, 1536, 3), dtype=np.uint8))
            (kf / "meta.json").write_text(json.dumps({"frame_idx": 1, "detections": []}))

            session = SceneSession(root)
            intr = session.intrinsics(fallback_hfov_deg=102.0)
            self.assertTrue(intr["estimated"])
            self.assertAlmostEqual(intr["fx"], (1536 / 2) / np.tan(np.radians(51)), delta=1)

            sessions = list_sessions(Path(tmp))
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0]["keyframes"], 1)
            self.assertFalse(sessions[0]["calibrated"])


class SegDecodeTest(unittest.TestCase):
    def test_numpy_nms_suppresses_overlap(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
        scores = np.array([0.9, 0.8, 0.7])
        keep = numpy_nms(boxes, scores, iou_thres=0.5)
        self.assertEqual(list(keep), [0, 2])

    def test_order_endnodes_by_shape(self) -> None:
        outputs = {}
        for scale in (40, 80, 20):
            outputs[f"b{scale}"] = np.zeros((scale, scale, 64), dtype=np.float32)
            outputs[f"s{scale}"] = np.zeros((scale, scale, 80), dtype=np.float32)
            outputs[f"c{scale}"] = np.zeros((scale, scale, 32), dtype=np.float32)
        outputs["p"] = np.zeros((160, 160, 32), dtype=np.float32)
        endnodes = order_endnodes(outputs)
        self.assertEqual(len(endnodes), 10)
        self.assertEqual(endnodes[0].shape[1:3], (20, 20))
        self.assertEqual(endnodes[0].shape[3], 64)
        self.assertEqual(endnodes[9].shape[1:3], (160, 160))


if __name__ == "__main__":
    unittest.main()
