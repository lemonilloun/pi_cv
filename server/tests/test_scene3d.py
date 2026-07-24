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
from mac_server.scene3d.objects_step import (
    ObjectBank,
    cosine,
    is_oversized_mask,
    largest_dbscan_cluster,
)
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


class OversizedMaskGuardTest(unittest.TestCase):
    """A mis-segmented wall/floor/ceiling can span most of the frame and
    turn into a pathologically large point cloud — these guards bound
    that cost unconditionally instead of letting one bad detection stall
    the whole objects step for minutes (see server/src/mac_server/scene3d/
    objects_step.py's run loop)."""

    def test_normal_object_not_flagged(self) -> None:
        frame_area = 1536 * 864
        self.assertFalse(is_oversized_mask(5000, frame_area, 0.35))

    def test_wall_sized_mask_flagged(self) -> None:
        frame_area = 1536 * 864
        self.assertTrue(is_oversized_mask(int(frame_area * 0.6), frame_area, 0.35))

    def test_boundary_is_exclusive(self) -> None:
        frame_area = 10000
        self.assertFalse(is_oversized_mask(3500, frame_area, 0.35))
        self.assertTrue(is_oversized_mask(3501, frame_area, 0.35))

    def test_zero_frame_area_never_flags(self) -> None:
        self.assertFalse(is_oversized_mask(100, 0, 0.35))


class DbscanCapTest(unittest.TestCase):
    def test_small_cloud_passes_through_unmodified_shape(self) -> None:
        rng = np.random.default_rng(1)
        points = rng.normal(size=(50, 3)) * 0.02
        result = largest_dbscan_cluster(points, eps=0.05, min_points=5, max_points=8000, rng=rng)
        self.assertLessEqual(len(result), 50)
        self.assertGreater(len(result), 0)

    def test_huge_cloud_is_bounded_before_clustering(self) -> None:
        """The real regression test: a mask this large used to make DBSCAN
        itself the bottleneck. Subsampling to max_points before clustering
        keeps runtime bounded regardless of input size, and the result can
        never exceed what went into DBSCAN."""
        import time

        rng = np.random.default_rng(2)
        # A large, spatially spread cluster plus scattered noise, mimicking
        # an over-segmented wall — no natural dense-enough single cluster.
        points = rng.uniform(-5, 5, size=(60000, 3))
        started = time.monotonic()
        result = largest_dbscan_cluster(
            points, eps=0.05, min_points=20, max_points=3000, rng=rng
        )
        elapsed = time.monotonic() - started
        # The point-count cap is the real, deterministic guarantee; the
        # wall-clock check is a loose sanity bound only (this machine may be
        # under unrelated load), generous enough to never flake but tight
        # enough to catch the cap silently not applying (which would turn
        # this into the multi-minute stall it's meant to prevent).
        self.assertLessEqual(len(result), 3000)
        self.assertLess(elapsed, 30.0, "DBSCAN cap did not bound runtime as expected")


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


class SceneUplinkHandlerTest(unittest.TestCase):
    """The online-transfer server path materializes the exact session
    layout the pipeline reads (start -> keyframes -> end)."""

    def test_stream_roundtrip(self) -> None:
        from mac_server.handlers import handle_message
        from pi_client.protocol import (
            make_scene_keyframe_message,
            make_scene_session_end_message,
            make_scene_session_start_message,
        )

        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp) / "received"
            storage_dir.mkdir()

            start = make_scene_session_start_message(
                "pi", "session_test1", {"fx": 600.0}, {"fps_target": 3.0}
            )
            response = handle_message(start, b"", storage_dir)
            self.assertEqual(response.type, "ack")

            rgb, masks = b"\xff\xd8jpegdata", b"\x89PNGdata"
            kf = make_scene_keyframe_message(
                "pi", "session_test1", 1,
                {"frame_idx": 1, "detections": [{"instance_id": 1}]},
                rgb_bytes=len(rgb), masks_bytes=len(masks),
            )
            response = handle_message(kf, rgb + masks, storage_dir)
            self.assertEqual(response.type, "ack")

            end = make_scene_session_end_message("pi", "session_test1", {"keyframes": 1})
            handle_message(end, b"", storage_dir)

            root = Path(tmp) / "scene_sessions" / "session_test1"
            self.assertEqual((root / "keyframes/000001/rgb.jpg").read_bytes(), rgb)
            self.assertEqual((root / "keyframes/000001/masks.png").read_bytes(), masks)
            meta = json.loads((root / "keyframes/000001/meta.json").read_text())
            self.assertEqual(meta["detections"][0]["instance_id"], 1)
            self.assertEqual(json.loads((root / "intrinsics.json").read_text())["fx"], 600.0)
            self.assertEqual(
                json.loads((root / "session_meta.json").read_text())["keyframes"], 1
            )
            sessions = list_sessions(Path(tmp) / "scene_sessions")
            self.assertEqual(sessions[0]["session_id"], "session_test1")

    def test_payload_length_mismatch_rejected(self) -> None:
        from mac_server.handlers import handle_message
        from pi_client.protocol import make_scene_keyframe_message

        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp) / "received"
            storage_dir.mkdir()
            kf = make_scene_keyframe_message("pi", "s1", 1, {}, rgb_bytes=100, masks_bytes=5)
            with self.assertRaises(ValueError):
                handle_message(kf, b"short", storage_dir)

    def test_session_name_sanitized(self) -> None:
        from mac_server.handlers import handle_message
        from pi_client.protocol import make_scene_session_start_message

        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp) / "received"
            storage_dir.mkdir()
            evil = make_scene_session_start_message("pi", "../../etc", None, {})
            handle_message(evil, b"", storage_dir)
            self.assertTrue((Path(tmp) / "scene_sessions" / "etc").exists())
            self.assertFalse((Path(tmp) / "etc").exists())


class SceneUplinkTcpTest(unittest.TestCase):
    """End-to-end over real TCP: SceneUplink -> ephemeral in-process
    MacServer -> session directory on disk."""

    def test_uplink_streams_session(self) -> None:
        import threading

        from mac_server.server import MacServer
        from pi_client.scene_recorder import SceneUplink

        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp) / "received"
            storage_dir.mkdir()
            server = MacServer(
                host="127.0.0.1", port=0, storage_dir=storage_dir,
                preview_enabled=False,
            )
            server.start()
            port = server._socket.getsockname()[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                uplink = SceneUplink(
                    "127.0.0.1", port, "pi_test", "session_tcp1",
                    intrinsics={"fx": 622.0}, settings={"fps_target": 3.0},
                )
                rgb, masks = b"\xff\xd8jpeg", b"\x89PNGmask"
                self.assertTrue(uplink.send_keyframe(1, {"frame_idx": 1, "detections": []}, rgb, masks))
                self.assertTrue(uplink.send_keyframe(2, {"frame_idx": 2, "detections": []}, rgb, masks))
                self.assertTrue(uplink.send_end({"keyframes": 2}))
                uplink.close()

                root = Path(tmp) / "scene_sessions" / "session_tcp1"
                self.assertEqual((root / "keyframes/000002/rgb.jpg").read_bytes(), rgb)
                self.assertEqual((root / "keyframes/000001/masks.png").read_bytes(), masks)
                self.assertTrue((root / "intrinsics.json").exists())
                self.assertEqual(
                    json.loads((root / "session_meta.json").read_text())["keyframes"], 2
                )
            finally:
                server.stop()

    def test_uplink_unreachable_degrades(self) -> None:
        from pi_client.scene_recorder import SceneUplink

        uplink = SceneUplink("127.0.0.1", 1, "pi", "s", None, {}, retry_s=0.0)
        self.assertFalse(uplink.send_keyframe(1, {}, b"x", b"y"))
        self.assertFalse(uplink.healthy)


class NavIndexTest(unittest.TestCase):
    def _make_index(self, root: Path, session: str, embs, positions=None, forwards=None) -> None:
        import numpy as np

        derived = root / session / "derived"
        derived.mkdir(parents=True, exist_ok=True)
        n = len(embs)
        forwards_arr = (
            np.asarray(forwards, dtype=np.float32)
            if forwards is not None
            else np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
        )
        np.savez_compressed(
            derived / "place_index.npz",
            kf=np.arange(1, n + 1, dtype=np.int32),
            embs=np.asarray(embs, dtype=np.float16),
            positions=np.asarray(positions if positions is not None else np.zeros((n, 3)), dtype=np.float32),
            forwards=forwards_arr,
        )
        (derived / "plan_frame.json").write_text(json.dumps({
            "axis_a": [1, 0, 0], "up": [0, -1, 0], "axis_b": [0, 0, 1],
            "origin": [-2.0, -2.0], "resolution_m": 0.1,
            "grid_w": 40, "grid_h": 40,
        }))
        (root / session / "keyframes").mkdir(exist_ok=True)

    @staticmethod
    def _forward_for_heading(deg: float) -> list[float]:
        import math

        r = math.radians(deg)
        return [math.cos(r), 0.0, math.sin(r)]

    def test_query_picks_right_room(self) -> None:
        import numpy as np

        from mac_server.scene3d import navindex

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            e1 = np.eye(3, 8)[0]  # room A looks like [1,0,0,...]
            e2 = np.eye(3, 8)[1]
            self._make_index(root, "session_a", [e1], positions=[[0, 0, 0]])
            self._make_index(root, "session_b", [e2])

            result = navindex.query(root, list(e1 * 0.9 + e2 * 0.1))
            self.assertTrue(result["located"])
            self.assertEqual(result["best"]["session_id"], "session_a")
            self.assertGreater(result["best"]["similarity"], 0.9)
            self.assertIn("plan_frac", result["best"])
            self.assertIn("heading_deg", result["best"])
            # position (0,0,0) with origin -2, res 0.1, grid 40:
            # frac = (x + 2) / 0.1 / 40
            self.assertAlmostEqual(result["best"]["plan_frac"][0], 0.5, places=2)
            self.assertEqual(navindex.get_last()["best"]["session_id"], "session_a")

    def test_heading_aggregation_tight_cluster(self) -> None:
        """Several near-identical viewpoints (a real recorded walkthrough
        revisiting the same spot) should average to a steady heading with
        low spread — not just borrow whichever single frame happened to be
        the top match."""
        import numpy as np

        from mac_server.scene3d import navindex

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            e = np.eye(1, 8)[0]
            forwards = [self._forward_for_heading(h) for h in (0.0, 8.0, -8.0)]
            self._make_index(root, "session_c", [e, e, e], positions=[[0, 0, 0]] * 3, forwards=forwards)

            result = navindex.query(root, list(e), intra_k=5)
            best = result["best"]
            self.assertAlmostEqual(best["heading_deg"], 0.0, delta=1.0)
            self.assertLess(best["heading_spread_deg"], 10.0)
            self.assertEqual(best["neighbors_used"], 3)

    def test_heading_aggregation_wide_spread_is_flagged(self) -> None:
        """Neighbors that disagree wildly on facing direction (a genuinely
        ambiguous CLIP match) must show a large spread, not a confident but
        meaningless average."""
        import numpy as np

        from mac_server.scene3d import navindex

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            e = np.eye(1, 8)[0]
            forwards = [self._forward_for_heading(h) for h in (0.0, 120.0, -120.0)]
            self._make_index(root, "session_d", [e, e, e], positions=[[0, 0, 0]] * 3, forwards=forwards)

            result = navindex.query(root, list(e), intra_k=5)
            self.assertGreater(result["best"]["heading_spread_deg"], 60.0)

    def test_heading_wraps_correctly_across_zero(self) -> None:
        """Headings clustered near the 0/360 wrap (350, 0, 10) must average
        to ~0, not the ~120 a naive (non-circular) mean would wrongly give."""
        import numpy as np

        from mac_server.scene3d import navindex

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            e = np.eye(1, 8)[0]
            forwards = [self._forward_for_heading(h) for h in (350.0, 0.0, 10.0)]
            self._make_index(root, "session_e", [e, e, e], positions=[[0, 0, 0]] * 3, forwards=forwards)

            result = navindex.query(root, list(e), intra_k=5)
            heading = result["best"]["heading_deg"] % 360
            self.assertTrue(heading < 2.0 or heading > 358.0, f"unwrapped mean: {heading}")

    def test_nav_query_handler_roundtrip(self) -> None:
        import numpy as np

        from mac_server.handlers import handle_message
        from mac_server.scene3d import navindex
        from pi_client.protocol import make_nav_query_message

        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp) / "received"
            storage_dir.mkdir()
            sessions = Path(tmp) / "scene_sessions"
            emb = list(np.eye(1, 16)[0])
            self._make_index(sessions, "session_x", [emb])

            message = make_nav_query_message("pi", emb, depth_center_m=2.4)
            response = handle_message(message, b"", storage_dir)
            self.assertEqual(response.type, "nav_result")
            self.assertTrue(response.payload["located"])
            self.assertEqual(response.payload["best"]["session_id"], "session_x")
            self.assertEqual(response.payload["depth_center_m"], 2.4)

    def test_no_index_answer(self) -> None:
        from mac_server.scene3d import navindex

        with tempfile.TemporaryDirectory() as tmp:
            result = navindex.query(Path(tmp), [1.0] * 16)
            self.assertFalse(result["located"])


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
