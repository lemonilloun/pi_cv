"""Unit tests for Scene3D pure logic: scale alignment, object association,
labeling, graph edges, floor plan, session IO and the seg decode NMS."""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src"), str(REPO_ROOT / "client/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np

from mac_server.scene3d.poses_step import (
    frame_scale,
    median_scale,
    resolve_matching_mode,
)
from mac_server.scene3d.objects_step import (
    ObjectBank,
    cosine,
    is_oversized_mask,
    largest_dbscan_cluster,
)
from mac_server.scene3d.depth_filter import (
    confidence_weight,
    filter_depth,
    flying_pixel_mask,
    intrinsics_for_shape,
    multiview_support,
    select_neighbours,
    unproject,
)
from mac_server.scene3d.graph_step import build_edges, label_objects
from mac_server.scene3d import occupancy
from mac_server.scene3d.tsdf_step import adaptive_depth_trunc, largest_components_mask
from mac_server.scene3d.poses_step import camera_to_intrinsics, hfov_deg
from mac_server.scene3d.objects_step import (
    gravity_aligned_obb,
    size_verdict,
)
from mac_server.scene3d.graph_step import obb_gap_m, obb_support
from mac_server.scene3d.session_io import SceneSession, list_sessions
from pi_client.seg_postprocess import (
    FASTSAM_S,
    INDOOR_ADE20K,
    YOLOV8_SEG,
    SegArch,
    numpy_nms,
    order_endnodes,
    yolov8_seg_postprocess,
)


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


class MatchingModeTest(unittest.TestCase):
    def test_auto_small_session_uses_exhaustive(self) -> None:
        self.assertEqual(resolve_matching_mode("auto", 147, 600), "exhaustive")

    def test_auto_large_session_falls_back_to_sequential_loop(self) -> None:
        self.assertEqual(resolve_matching_mode("auto", 900, 600), "sequential+loop")

    def test_auto_at_threshold_is_exhaustive(self) -> None:
        self.assertEqual(resolve_matching_mode("auto", 600, 600), "exhaustive")

    def test_explicit_mode_passes_through(self) -> None:
        self.assertEqual(resolve_matching_mode("sequential", 100, 600), "sequential")
        self.assertEqual(resolve_matching_mode("exhaustive", 5000, 600), "exhaustive")


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
        # Dissimilar embedding, but the same live track and a plausible
        # position — the track prior is allowed to win.
        b = bank.add(self._candidate(track=7, centroid=(0.3, 0, 2), emb=[0.0, 1.0, 0.0]))
        self.assertEqual(a, b)
        self.assertEqual(len(bank.objects), 1)

    def test_recycled_track_id_does_not_weld_across_the_room(self) -> None:
        """The Pi's tracker reuses ids after a track dies. Merging on the id
        alone once welded two unrelated objects 3 m apart into one."""
        bank = ObjectBank()
        a = bank.add(self._candidate(track=7, centroid=(0, 0, 2)))
        b = bank.add(self._candidate(track=7, centroid=(3, 0, 2), emb=[0.0, 1.0, 0.0]))
        self.assertNotEqual(a, b)
        self.assertEqual(len(bank.objects), 2)

    def test_class_gate_keeps_a_chair_out_of_a_sofa(self) -> None:
        bank = ObjectBank(dino_cos_min=0.6, centroid_max_m=0.5)
        bank.add(self._candidate(track=-1, cls="couch", centroid=(0, 0, 2)))
        idx = bank.add(self._candidate(track=-1, cls="chair", centroid=(0.1, 0, 2)))
        self.assertEqual(idx, 1)

    def test_gate_grows_with_the_object(self) -> None:
        """Two views of one sofa see different halves, so their centroids sit
        further apart than the fixed 0.5 m gate ever allowed."""
        big = np.array([[-1.0, 0, 0], [1.0, 0, 0], [0, 0.4, 0], [0, -0.4, 0.6]])
        first = self._candidate(track=-1, cls="couch", centroid=(0, 0, 2))
        first["points"] = big
        bank = ObjectBank(centroid_max_m=0.5, size_gate_factor=0.5)
        bank.add(first)
        second = self._candidate(track=-1, cls="couch", centroid=(0.8, 0, 2))
        second["points"] = big
        self.assertEqual(bank.add(second), 0)

    def test_merged_embedding_stays_unit_length(self) -> None:
        bank = ObjectBank(dino_cos_min=0.5)
        bank.add(self._candidate(track=-1, centroid=(0, 0, 2), emb=[1.0, 0.0, 0.0]))
        bank.add(self._candidate(track=-1, centroid=(0.1, 0, 2), emb=[0.6, 0.8, 0.0]))
        norm = float(np.linalg.norm(bank.objects[0]["dino_emb"]))
        self.assertAlmostEqual(norm, 1.0, places=6)

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


class GravityTest(unittest.TestCase):
    def test_camera_average_level_camera(self) -> None:
        # Identity pose: camera +Y (down) is world +Y -> up is -Y.
        up = occupancy.camera_average_up({1: np.eye(4)})
        np.testing.assert_allclose(up, [0, -1, 0], atol=1e-6)

    def test_floor_fit_beats_a_tilted_camera(self) -> None:
        """The whole point of the RANSAC fit: a consistently tilted camera
        makes the camera-average estimate wrong, the floor does not move."""
        rng = np.random.default_rng(0)
        # True floor is the y=0 plane, so true up is -Y in this convention.
        floor = np.stack(
            [rng.uniform(-2, 2, 4000), np.zeros(4000), rng.uniform(-2, 2, 4000)], axis=1
        )
        floor[:, 1] += rng.normal(0, 0.005, 4000)
        stuff = np.stack(
            [rng.uniform(-2, 2, 1000), rng.uniform(-1.5, -0.5, 1000),
             rng.uniform(-2, 2, 1000)], axis=1
        )
        points = np.vstack([floor, stuff])

        tilt = np.radians(15.0)
        pose = np.eye(4)
        pose[:3, :3] = np.array(
            [[1, 0, 0],
             [0, np.cos(tilt), -np.sin(tilt)],
             [0, np.sin(tilt), np.cos(tilt)]]
        )
        poses = {i: pose for i in range(5)}

        coarse = occupancy.camera_average_up(poses)
        fitted, source = occupancy.estimate_gravity(points, poses, rng=rng)
        self.assertEqual(source, "floor_fit")
        truth = np.array([0.0, -1.0, 0.0])
        angle = lambda v: np.degrees(np.arccos(np.clip(v @ truth, -1, 1)))
        self.assertGreater(angle(coarse), 10.0)
        self.assertLess(angle(fitted), 2.0)

    def test_imu_wins_and_is_normalized(self) -> None:
        up, source = occupancy.estimate_gravity(None, {1: np.eye(4)}, imu_up=[0, 0, 5])
        self.assertEqual(source, "imu")
        np.testing.assert_allclose(up, [0, 0, 1], atol=1e-9)

    def test_falls_back_without_points(self) -> None:
        up, source = occupancy.estimate_gravity(None, {1: np.eye(4)})
        self.assertEqual(source, "camera_average")
        np.testing.assert_allclose(up, [0, -1, 0], atol=1e-6)

    def test_plan_axes_orthonormal(self) -> None:
        for up in ([0, 1, 0], [1, 0, 0], [0.3, 0.9, -0.2]):
            a, b = occupancy.plan_axes(np.asarray(up, dtype=float))
            u = np.asarray(up, dtype=float)
            u = u / np.linalg.norm(u)
            self.assertAlmostEqual(float(a @ b), 0.0, places=9)
            self.assertAlmostEqual(float(a @ u), 0.0, places=9)
            self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=9)
            self.assertAlmostEqual(float(np.linalg.norm(b)), 1.0, places=9)


class OccupancyTest(unittest.TestCase):
    def _carve_box_room(self):
        """A 4x4 m room seen from the middle: walls occupied, interior free,
        everything outside the walls never observed."""
        spec = occupancy.make_grid_spec(
            np.array([[-2.0, -2.0], [2.0, 2.0]]), resolution_m=0.05
        )
        free = np.zeros((spec["height"], spec["width"]), dtype=np.int32)
        hits = np.zeros_like(free)
        angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        # Ray/box intersection for a square room centred on the origin.
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.minimum(
                np.abs(2.0 / np.cos(angles)), np.abs(2.0 / np.sin(angles))
            )
        wall = np.stack([t * np.cos(angles), t * np.sin(angles)], axis=1)
        occupancy.carve_rays(np.array([0.0, 0.0]), wall, spec, free, hits)
        return spec, free, hits

    def test_carving_marks_interior_free_and_walls_occupied(self) -> None:
        spec, free, hits = self._carve_box_room()
        grid = occupancy.occupancy_from_counts(free, hits, min_hits=1, min_free=1)
        centre = occupancy.world_to_grid(np.array([[0.0, 0.0]]), spec)
        self.assertEqual(grid[centre[1][0], centre[0][0]], occupancy.FREE)
        wall = occupancy.world_to_grid(np.array([[0.0, 2.0]]), spec)
        self.assertEqual(grid[wall[1][0], wall[0][0]], occupancy.OCCUPIED)
        # Beyond the wall the camera never saw anything.
        self.assertEqual(grid[0, 0], occupancy.UNKNOWN)

    def test_free_space_dominates_the_interior(self) -> None:
        _, free, hits = self._carve_box_room()
        grid = occupancy.occupancy_from_counts(free, hits, min_hits=1, min_free=1)
        self.assertGreater((grid == occupancy.FREE).sum(),
                           5 * (grid == occupancy.OCCUPIED).sum())

    def test_a_cell_seen_through_far_more_often_than_hit_is_free(self) -> None:
        """Glass, or one bad depth pixel, must not become a wall."""
        free = np.full((4, 4), 100, dtype=np.int32)
        hits = np.full((4, 4), 3, dtype=np.int32)
        grid = occupancy.occupancy_from_counts(free, hits, min_hits=3, hit_ratio=0.05)
        self.assertTrue((grid == occupancy.FREE).all())

    def test_carve_rays_leaves_the_surface_intact(self) -> None:
        spec = occupancy.make_grid_spec(
            np.array([[0.0, 0.0], [3.0, 0.0]]), resolution_m=0.05
        )
        free = np.zeros((spec["height"], spec["width"]), dtype=np.int32)
        hits = np.zeros_like(free)
        occupancy.carve_rays(np.array([0.0, 0.0]), np.array([[3.0, 0.0]]), spec, free, hits)
        ix, iy = occupancy.world_to_grid(np.array([[3.0, 0.0]]), spec)
        self.assertEqual(int(hits[iy[0], ix[0]]), 1)
        self.assertEqual(int(free[iy[0], ix[0]]), 0)

    def test_grid_spec_covers_points_with_margin(self) -> None:
        spec = occupancy.make_grid_spec(
            np.array([[-1.0, -1.0], [1.0, 1.0]]), resolution_m=0.1, margin_m=0.5
        )
        self.assertAlmostEqual(spec["origin"][0], -1.5)
        ix, iy = occupancy.world_to_grid(np.array([[1.0, 1.0]]), spec)
        self.assertLess(int(ix[0]), spec["width"])


class MeshCleanupTest(unittest.TestCase):
    def test_small_components_are_the_ones_removed(self) -> None:
        # Two components: id 0 has 500 triangles, id 1 has 5.
        cluster_ids = np.array([0] * 500 + [1] * 5)
        sizes = np.array([500, 5])
        remove = largest_components_mask(cluster_ids, sizes, min_triangles=100)
        self.assertEqual(int(remove.sum()), 5)
        self.assertFalse(bool(remove[:500].any()))

    def test_adaptive_trunc_follows_the_data(self) -> None:
        near_room = np.concatenate([np.full(5000, 1.5), np.full(200, 2.4)])
        self.assertLess(adaptive_depth_trunc(near_room, configured_max=5.0), 3.0)

    def test_adaptive_trunc_never_exceeds_the_configured_max(self) -> None:
        self.assertEqual(
            adaptive_depth_trunc(np.full(5000, 40.0), configured_max=5.0), 5.0
        )

    def test_adaptive_trunc_without_enough_samples(self) -> None:
        self.assertEqual(adaptive_depth_trunc(np.array([1.0]), configured_max=5.0), 5.0)


class IntrinsicsTest(unittest.TestCase):
    class _Camera:
        def __init__(self, params, width=1536, height=864):
            self.params = params
            self.width = width
            self.height = height

    def test_reads_opencv_params(self) -> None:
        cam = self._Camera([988.9, 990.0, 768.0, 432.0, 0.01, -0.02, 0.0, 0.0])
        intr = camera_to_intrinsics(cam)
        self.assertAlmostEqual(intr["fx"], 988.9)
        self.assertAlmostEqual(intr["cy"], 432.0)
        self.assertEqual(intr["source"], "colmap_refined")

    def test_accessors_win_over_raw_params(self) -> None:
        cam = self._Camera([1.0, 2.0, 3.0, 4.0])
        cam.focal_length_x = 1011.0
        cam.focal_length_y = 1011.0
        cam.principal_point_x = 768.0
        cam.principal_point_y = 432.0
        self.assertAlmostEqual(camera_to_intrinsics(cam)["fx"], 1011.0)

    def test_hfov_matches_the_measured_camera(self) -> None:
        # The mismatch that motivated persisting the refined camera at all.
        self.assertAlmostEqual(hfov_deg(988.9, 1536), 75.7, places=1)
        self.assertAlmostEqual(hfov_deg(622.0, 1536), 102.0, places=0)

    def test_active_intrinsics_prefers_the_refined_camera(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "session_test"
            kf = root / "keyframes" / "000001"
            kf.mkdir(parents=True)
            cv2.imwrite(str(kf / "rgb.jpg"), np.zeros((864, 1536, 3), dtype=np.uint8))
            (kf / "meta.json").write_text(json.dumps({"frame_idx": 1, "detections": []}))

            session = SceneSession(root)
            self.assertTrue(session.active_intrinsics(75.0)["estimated"])
            session.derived.mkdir(parents=True, exist_ok=True)
            session.refined_intrinsics_path().write_text(
                json.dumps({"width": 1536, "height": 864, "fx": 988.9, "fy": 988.9,
                            "cx": 768.0, "cy": 432.0, "source": "colmap_refined"})
            )
            self.assertAlmostEqual(session.active_intrinsics(75.0)["fx"], 988.9)


class GravityObbTest(unittest.TestCase):
    def _slab(self, rng, size=(1.8, 0.8, 0.4), yaw_deg=30.0):
        """A box of known size, rotated about the up axis only."""
        half = np.asarray(size) / 2.0
        local = rng.uniform(-half, half, size=(4000, 3))
        theta = np.radians(yaw_deg)
        rot = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]]
        )
        return local @ rot.T

    def test_recovers_the_true_extent(self) -> None:
        rng = np.random.default_rng(0)
        points = self._slab(rng)
        obb = gravity_aligned_obb(points, up=[0, 0, 1])
        self.assertIsNotNone(obb)
        got = sorted(obb["extent"])
        np.testing.assert_allclose(got, [0.4, 0.8, 1.8], atol=0.1)

    def test_vertical_axis_is_gravity(self) -> None:
        rng = np.random.default_rng(1)
        obb = gravity_aligned_obb(self._slab(rng), up=[0, 0, 1])
        rotation = np.asarray(obb["rotation"])
        np.testing.assert_allclose(rotation[:, 2], [0, 0, 1], atol=1e-9)

    def test_partial_view_does_not_tilt_the_box(self) -> None:
        """One visible face used to let PCA tilt the box until a bed came
        out 0.09 m thick."""
        rng = np.random.default_rng(2)
        points = self._slab(rng, size=(1.8, 0.8, 0.4), yaw_deg=0.0)
        front = points[points[:, 1] > 0.3]  # only the near face
        obb = gravity_aligned_obb(front, up=[0, 0, 1])
        self.assertAlmostEqual(obb["extent"][2], 0.4, delta=0.06)

    def test_too_few_points(self) -> None:
        self.assertIsNone(gravity_aligned_obb(np.zeros((3, 3)), up=[0, 0, 1]))

    def test_size_verdicts(self) -> None:
        self.assertEqual(size_verdict("bottle", [0.08, 0.08, 0.25]), "ok")
        self.assertEqual(size_verdict("bottle", [0.58, 0.66, 0.52]), "too_large")
        self.assertEqual(size_verdict("person", [4.95, 2.22, 1.14]), "too_large")
        self.assertEqual(size_verdict("couch", [2.1, 0.9, 0.8]), "ok")
        self.assertEqual(size_verdict("chair", [0.0, 0.5, 0.5]), "degenerate")
        self.assertEqual(size_verdict("book", [0.01, 0.01, 0.01]), "too_small")


class ObbDistanceTest(unittest.TestCase):
    @staticmethod
    def _box(center, extent):
        return {
            "object_id": 0,
            "centroid": list(center),
            "obb": {"center": list(center), "extent": list(extent),
                    "rotation": np.eye(3).tolist()},
        }

    def test_support_of_an_axis_aligned_box(self) -> None:
        box = self._box([0, 0, 0], [2.0, 1.0, 0.5])["obb"]
        self.assertAlmostEqual(obb_support(box, [1, 0, 0]), 1.0)
        self.assertAlmostEqual(obb_support(box, [0, 1, 0]), 0.5)

    def test_touching_boxes_have_no_gap(self) -> None:
        a = self._box([0, 0, 0], [2.0, 1.0, 1.0])
        b = self._box([2.0, 0, 0], [2.0, 1.0, 1.0])
        self.assertAlmostEqual(obb_gap_m(a, b), 0.0, places=6)

    def test_gap_is_surface_to_surface_not_centre_to_centre(self) -> None:
        """A lamp beside a 2 m sofa: 1.1 m between centres, 0.05 m apart."""
        sofa = self._box([0, 0, 0], [2.0, 0.9, 0.8])
        lamp = self._box([1.1, 0, 0], [0.2, 0.2, 1.5])
        self.assertAlmostEqual(obb_gap_m(sofa, lamp), 0.0, places=6)
        centre_distance = 1.1
        self.assertLess(obb_gap_m(sofa, lamp), centre_distance)

    def test_far_apart_boxes_report_the_real_gap(self) -> None:
        a = self._box([0, 0, 0], [1.0, 1.0, 1.0])
        b = self._box([3.0, 0, 0], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(obb_gap_m(a, b), 2.0, places=6)


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


class InstanceMaskTest(unittest.TestCase):
    """`instance_mask` has to serve both storage formats, because every
    session recorded so far is the old one and re-recording is a drive."""

    def _keyframe(self, tmp: str, mask_image, meta: dict):
        import cv2

        root = Path(tmp) / "session_test"
        kf = root / "keyframes" / "000001"
        kf.mkdir(parents=True)
        cv2.imwrite(str(kf / "rgb.jpg"), np.zeros((40, 40, 3), dtype=np.uint8))
        cv2.imwrite(str(kf / "masks.png"), mask_image)
        (kf / "meta.json").write_text(json.dumps(meta))
        return SceneSession(root).keyframes()[0]

    def test_legacy_label_image_still_reads(self) -> None:
        labels = np.zeros((40, 40), dtype=np.uint16)
        labels[0:10, 0:10] = 1
        labels[20:30, 20:30] = 2
        with tempfile.TemporaryDirectory() as tmp:
            kf = self._keyframe(tmp, labels, {"frame_idx": 1, "detections": []})
            self.assertEqual(kf.masks_format(), "labels_v0")
            self.assertEqual(int(kf.instance_mask(1).sum()), 100)
            self.assertEqual(int(kf.instance_mask(2).sum()), 100)
            self.assertEqual(int(kf.instance_mask(3).sum()), 0)

    def test_stack_v1_preserves_overlap_that_labels_lose(self) -> None:
        # This is the whole reason for the format. Instance 1 is a 20x20 box;
        # instance 2 is a 10x10 box entirely inside it (a drawer in a cabinet).
        big = np.zeros((20, 20), dtype=np.uint8)
        big[:, :] = 255
        small = np.zeros((20, 20), dtype=np.uint8)
        small[5:15, 5:15] = 255
        stack = np.concatenate([big, small], axis=0)
        meta = {"frame_idx": 1, "masks_format": "stack_v1",
                "mask_shape": [20, 20], "mask_count": 2, "detections": []}
        with tempfile.TemporaryDirectory() as tmp:
            kf = self._keyframe(tmp, stack, meta)
            self.assertEqual(kf.masks_format(), "stack_v1")
            self.assertEqual(int(kf.instance_mask(1).sum()), 400)  # intact
            self.assertEqual(int(kf.instance_mask(2).sum()), 100)

        # The same pair stored the legacy way: the inner mask overwrote the
        # outer one's pixels at write time, so the cabinet comes back with a
        # 100 px hole. Asserted so the loss is documented, not just claimed.
        with tempfile.TemporaryDirectory() as tmp:
            labels = np.ones((20, 20), dtype=np.uint16)
            labels[5:15, 5:15] = 2
            kf = self._keyframe(tmp, labels, {"frame_idx": 1, "detections": []})
            self.assertEqual(int(kf.instance_mask(1).sum()), 300)  # 400 - 100

    def test_out_of_range_instance_is_empty_not_an_exception(self) -> None:
        # A meta/mask mismatch should not abort a whole reconstruction over
        # one detection; the object simply contributes no points.
        stack = np.full((20, 20), 255, dtype=np.uint8)
        meta = {"frame_idx": 1, "masks_format": "stack_v1",
                "mask_shape": [20, 20], "mask_count": 1, "detections": []}
        with tempfile.TemporaryDirectory() as tmp:
            kf = self._keyframe(tmp, stack, meta)
            self.assertEqual(int(kf.instance_mask(5).sum()), 0)

    def test_resizes_to_the_requested_shape(self) -> None:
        stack = np.full((20, 20), 255, dtype=np.uint8)
        meta = {"frame_idx": 1, "masks_format": "stack_v1",
                "mask_shape": [20, 20], "mask_count": 1, "detections": []}
        with tempfile.TemporaryDirectory() as tmp:
            kf = self._keyframe(tmp, stack, meta)
            mask = kf.instance_mask(1, (40, 60))
            self.assertEqual(mask.shape, (40, 60))
            self.assertTrue(mask.all())

    def test_unknown_format_refuses(self) -> None:
        meta = {"frame_idx": 1, "masks_format": "rle_v9", "detections": []}
        with tempfile.TemporaryDirectory() as tmp:
            kf = self._keyframe(tmp, np.zeros((4, 4), dtype=np.uint8), meta)
            with self.assertRaises(ValueError):
                kf.instance_mask(1)


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


class DepthFilterTest(unittest.TestCase):
    INTR = {"width": 160, "height": 120, "fx": 100.0, "fy": 100.0, "cx": 80.0, "cy": 60.0}

    def _flat(self, value: float = 2.0) -> np.ndarray:
        return np.full((120, 160), value, dtype=np.float32)

    def test_intrinsics_rescaled_to_actual_resolution(self) -> None:
        k = intrinsics_for_shape(self.INTR, (60, 80))
        self.assertAlmostEqual(k["fx"], 50.0)
        self.assertAlmostEqual(k["cx"], 40.0)
        self.assertAlmostEqual(k["cy"], 30.0)

    def test_unproject_centre_pixel_sits_on_optical_axis(self) -> None:
        pts = unproject(self._flat(2.0), self.INTR)
        self.assertAlmostEqual(pts[60, 80, 0], 0.0)
        self.assertAlmostEqual(pts[60, 80, 1], 0.0)
        self.assertAlmostEqual(pts[60, 80, 2], 2.0)

    def test_invalid_depth_unprojects_to_nan(self) -> None:
        depth = self._flat(2.0)
        depth[0, 0] = 0.0
        self.assertTrue(np.isnan(unproject(depth, self.INTR)[0, 0, 2]))

    def test_flat_plane_has_no_flying_pixels(self) -> None:
        # A fronto-parallel wall is the best-conditioned case there is; a
        # filter that flags it would erase real geometry at the frame edges.
        self.assertFalse(flying_pixel_mask(self._flat(2.0), self.INTR).any())

    def test_depth_discontinuity_is_flagged(self) -> None:
        depth = self._flat(1.0)
        depth[:, 80:] = 4.0  # near object against a far wall
        mask = flying_pixel_mask(depth, self.INTR)
        self.assertTrue(mask[:, 78:82].any(), "the jump itself must be rejected")
        self.assertFalse(mask[:, 10:60].any(), "near surface must survive")
        self.assertFalse(mask[:, 100:150].any(), "far surface must survive")

    def test_consistent_views_support_each_other(self) -> None:
        pose_i = np.eye(4)
        pose_k = np.eye(4)
        pose_k[0, 3] = 0.15  # 15 cm sideways baseline
        support = multiview_support(
            self._flat(2.0), pose_i, [(self._flat(2.0), pose_k)], self.INTR
        )
        # u reprojects to u - 7.5, so only the left edge falls outside.
        self.assertEqual(int(support[:, 20:].min()), 1)
        self.assertEqual(int(support.max()), 1)

    def test_disagreeing_depth_gets_no_support(self) -> None:
        pose_i = np.eye(4)
        pose_k = np.eye(4)
        pose_k[0, 3] = 0.15
        support = multiview_support(
            self._flat(2.0), pose_i, [(self._flat(3.0), pose_k)], self.INTR
        )
        self.assertEqual(int(support.max()), 0)

    def test_occluded_pixel_is_unsupported_not_contradicted(self) -> None:
        # Support is counted, never contradiction: a view that cannot see the
        # point simply withholds its vote, so one occluder cannot veto a
        # point that other views confirm.
        pose_i = np.eye(4)
        near = np.eye(4)
        near[0, 3] = 0.15
        far = np.eye(4)
        far[0, 3] = -0.15
        support = multiview_support(
            self._flat(2.0),
            pose_i,
            [(self._flat(0.5), near), (self._flat(2.0), far)],
            self.INTR,
        )
        self.assertEqual(int(support[:, 20:140].min()), 1)

    def test_select_neighbours_picks_nearest_cameras(self) -> None:
        poses = {}
        for i, x in enumerate([0.0, 0.1, 5.0, 0.3]):
            m = np.eye(4)
            m[0, 3] = x
            poses[i] = m
        self.assertEqual(select_neighbours(poses, 0, count=2), [1, 3])

    def test_select_neighbours_unknown_frame(self) -> None:
        self.assertEqual(select_neighbours({0: np.eye(4)}, 99), [])

    def test_confidence_weight_penalises_distance(self) -> None:
        depth = np.array([[1.0, 8.0]], dtype=np.float32)
        w = confidence_weight(None, depth, far_m=4.0)
        self.assertAlmostEqual(float(w[0, 0]), 1.0)
        self.assertAlmostEqual(float(w[0, 1]), 0.5)

    def test_confidence_weight_zero_where_no_measurement(self) -> None:
        depth = np.array([[0.0, 2.0]], dtype=np.float32)
        w = confidence_weight(None, depth, far_m=4.0)
        self.assertAlmostEqual(float(w[0, 0]), 0.0)

    def test_filter_depth_reports_what_it_removed(self) -> None:
        depth = self._flat(1.0)
        depth[:, 80:] = 4.0
        out, stats = filter_depth(depth, self.INTR, max_depth_m=3.0)
        self.assertEqual(stats["valid_in"], 120 * 160)
        self.assertGreater(stats["dropped_far"], 0)
        self.assertFalse((out > 3.0).any())
        self.assertLess(stats["valid_out"], stats["valid_in"])

    def test_filter_depth_keeps_a_clean_plane_intact(self) -> None:
        out, stats = filter_depth(self._flat(2.0), self.INTR)
        self.assertEqual(stats["kept_frac"], 1.0)
        self.assertTrue(np.allclose(out, 2.0))


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


class SegArchTest(unittest.TestCase):
    """`SegArch` replaced literals that used to be spread across the decoder.
    These tests pin the two things that made that worth doing: a wrong arch
    must fail loudly, and a non-640 hef must decode in its own coordinates."""

    @staticmethod
    def _outputs(num_classes: int, input_h: int = 640):
        out = {}
        for stride in (8, 16, 32):
            s = input_h // stride
            out[f"b{s}"] = np.zeros((s, s, 64), dtype=np.float32)
            out[f"s{s}"] = np.zeros((s, s, num_classes), dtype=np.float32)
            out[f"c{s}"] = np.zeros((s, s, 32), dtype=np.float32)
        out["p"] = np.zeros((input_h // 4, input_h // 4, 32), dtype=np.float32)
        return out

    def test_default_arch_is_unchanged_behaviour(self) -> None:
        # The whole parameterization is only safe if the default path is
        # byte-identical to what shipped, so assert the two agree.
        outputs = self._outputs(80)
        self.assertEqual(
            [e.shape for e in order_endnodes(outputs)],
            [e.shape for e in order_endnodes(outputs, YOLOV8_SEG)],
        )

    def test_fastsam_single_class_decodes(self) -> None:
        endnodes = order_endnodes(self._outputs(1), FASTSAM_S)
        self.assertEqual(len(endnodes), 10)
        self.assertEqual(endnodes[1].shape[3], 1)      # class head
        self.assertEqual(endnodes[9].shape[1:3], (160, 160))

    def test_wrong_arch_is_rejected_not_misread(self) -> None:
        # An 80-class hef decoded as FastSAM: the 80-channel blobs match no
        # role. This must raise — misreading them as something else would give
        # boxes and masks that look plausible and are wrong.
        with self.assertRaises(ValueError):
            order_endnodes(self._outputs(80), FASTSAM_S)
        with self.assertRaises(ValueError):
            order_endnodes(self._outputs(1), YOLOV8_SEG)

    def test_non_640_input_uses_its_own_grid(self) -> None:
        # The bug this prevents: `h == 160` and the [20,40,80] scale list were
        # literals, so a 1024-input hef decoded into 640-space silently.
        arch = dataclasses.replace(YOLOV8_SEG, input_shape=(1024, 1024))
        self.assertEqual(arch.scale_heights, (32, 64, 128))
        endnodes = order_endnodes(self._outputs(80, input_h=1024), arch)
        self.assertEqual(endnodes[0].shape[1:3], (32, 32))
        self.assertEqual(endnodes[9].shape[1:3], (256, 256))
        # ...and the 640 layout must NOT satisfy the 1024 arch.
        with self.assertRaises(ValueError):
            order_endnodes(self._outputs(80), arch)

    def test_undecodable_arch_refuses_at_construction(self) -> None:
        # 64 classes collides with the DFL box head's 4*(15+1) = 64 channels.
        # Nothing downstream could tell those blobs apart, so the refusal
        # belongs here rather than at the first inference.
        with self.assertRaises(ValueError):
            SegArch(name="collides", num_classes=64)

    def test_decode_end_to_end_on_a_single_class_arch(self) -> None:
        # Zeros in means no detection out — but it must reach that answer
        # without an index error on the 1-wide class head, which is the shape
        # the old fixed-80 reshape would have thrown on.
        result = yolov8_seg_postprocess(self._outputs(1), arch=FASTSAM_S)
        self.assertEqual(result["boxes_xyxy"].shape[1], 4)
        self.assertEqual(len(result["scores"]), 0)


if __name__ == "__main__":
    unittest.main()


class KeyframeSelectorTest(unittest.TestCase):
    """The selector decides what the whole reconstruction is built from, so
    its rules are worth pinning down. Sharp/blurred is a quality gate;
    parallax is the one that changed — a stationary robot used to emit a
    keyframe every 0.5 s forever."""

    def setUp(self) -> None:
        from pi_client.scene_recorder import KeyframeSelector

        self.sel = KeyframeSelector(
            min_interval_s=0.2, max_interval_s=3.0, min_shift_px=12.0,
            blur_threshold=100.0,
        )

    def test_first_sharp_frame_is_always_taken(self) -> None:
        self.assertEqual(self.sel.decide(0.0, None, 500.0, is_first=True), (True, "first"))

    def test_first_blurred_frame_is_not(self) -> None:
        emit, _ = self.sel.decide(0.0, None, 10.0, is_first=True)
        self.assertFalse(emit)

    def test_standing_still_emits_nothing_until_the_max_interval(self) -> None:
        self.assertEqual(self.sel.decide(1.0, 0.5, 500.0), (False, "no_parallax"))
        self.assertEqual(self.sel.decide(3.5, 0.5, 500.0), (True, "interval"))

    def test_real_motion_emits_immediately(self) -> None:
        self.assertEqual(self.sel.decide(0.3, 40.0, 500.0), (True, "parallax"))

    def test_blur_beats_parallax(self) -> None:
        self.assertEqual(self.sel.decide(1.0, 40.0, 20.0), (False, "blurred"))

    def test_min_interval_is_a_hard_floor(self) -> None:
        self.assertEqual(self.sel.decide(0.05, 999.0, 500.0), (False, "too_soon"))

    def test_missing_flow_does_not_stall_the_recording(self) -> None:
        # A featureless white wall gives no trackable points; falling back to
        # "emit" is the safe direction — a missing keyframe cannot be recovered.
        self.assertEqual(self.sel.decide(1.0, None, 500.0), (True, "no_flow"))


class ImuSeamTest(unittest.TestCase):
    def _session(self, tmp, gravity_by_index):
        import cv2

        root = Path(tmp) / "session_imu"
        for index, gravity in gravity_by_index.items():
            kf = root / "keyframes" / f"{index:06d}"
            kf.mkdir(parents=True)
            cv2.imwrite(str(kf / "rgb.jpg"), np.zeros((64, 64, 3), dtype=np.uint8))
            meta = {"frame_idx": index, "detections": []}
            if gravity is not None:
                meta["gravity"] = gravity
            (kf / "meta.json").write_text(json.dumps(meta))
        return SceneSession(root)

    def test_absent_gravity_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp, {1: None, 2: None})
            self.assertIsNone(session.imu_up_vector({1: np.eye(4), 2: np.eye(4)}))

    def test_gravity_is_rotated_into_world_space(self) -> None:
        """A tilted camera is the whole point: gravity read as +Z in the
        camera frame must be reported in WORLD coordinates, not passed
        through. This pose maps camera +Z onto world -Y, so down is -Y and
        up is +Y."""
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp, {1: [0.0, 0.0, 1.0]})
            pose = np.eye(4)
            pose[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            up = session.imu_up_vector({1: pose})
            np.testing.assert_allclose(up, [0, 1, 0], atol=1e-9)

    def test_identity_pose_passes_gravity_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp, {1: [0.0, 9.8, 0.0]})
            np.testing.assert_allclose(
                session.imu_up_vector({1: np.eye(4)}), [0, -1, 0], atol=1e-9
            )


def _rvc_frame(index=0, yaw=0.0, pitch=0.0, roll=0.0, accel=(0, 0, 1000)):
    """Build a well-formed RVC frame for tests."""
    import struct

    body = bytes([index]) + struct.pack(
        "<hhhhhh",
        int(round(yaw * 100)), int(round(pitch * 100)), int(round(roll * 100)),
        int(accel[0]), int(accel[1]), int(accel[2]),
    ) + bytes(3)
    return b"\xaa\xaa" + body + bytes([sum(body) & 0xFF])


class ImuRvcParseTest(unittest.TestCase):
    """The framing has to survive a lossy USB-serial bridge: a decoder that
    only advances whole frames stays misaligned forever after one dropped
    byte, which looks exactly like a broken sensor."""

    def setUp(self) -> None:
        from pi_client import imu_rvc

        self.imu = imu_rvc

    def test_frame_is_19_bytes(self) -> None:
        self.assertEqual(len(_rvc_frame()), self.imu.FRAME_LEN)

    def test_round_trip(self) -> None:
        sample = self.imu.parse_frame(
            _rvc_frame(index=42, yaw=5.14, pitch=0.23, roll=-1.29, accel=(20, 4, 997))
        )
        self.assertIsNotNone(sample)
        self.assertEqual(sample.index, 42)
        self.assertAlmostEqual(sample.yaw_deg, 5.14, places=2)
        self.assertAlmostEqual(sample.roll_deg, -1.29, places=2)
        self.assertEqual(sample.accel_mg, (20.0, 4.0, 997.0))

    def test_negative_values_are_signed(self) -> None:
        sample = self.imu.parse_frame(_rvc_frame(pitch=-30.0, accel=(-500, 0, -866)))
        self.assertAlmostEqual(sample.pitch_deg, -30.0, places=2)
        self.assertEqual(sample.accel_mg[0], -500.0)
        self.assertEqual(sample.accel_mg[2], -866.0)

    def test_bad_checksum_rejected(self) -> None:
        frame = bytearray(_rvc_frame())
        frame[18] ^= 0xFF
        self.assertIsNone(self.imu.parse_frame(bytes(frame)))

    def test_wrong_header_rejected(self) -> None:
        frame = bytearray(_rvc_frame())
        frame[0] = 0xAB
        self.assertIsNone(self.imu.parse_frame(bytes(frame)))

    def test_accelerometer_up_points_up_at_rest(self) -> None:
        # An accelerometer at rest reads the normal force, i.e. UP; gravity
        # is its negation. Getting this backwards silently inverts the mesh.
        sample = self.imu.parse_frame(_rvc_frame(accel=(0, 0, 1000)))
        self.assertEqual(sample.up_sensor(), (0.0, 0.0, 1.0))
        self.assertEqual(sample.down_sensor(), (-0.0, -0.0, -1.0))

    def test_zero_acceleration_has_no_direction(self) -> None:
        self.assertIsNone(self.imu.parse_frame(_rvc_frame(accel=(0, 0, 0))).up_sensor())

    def test_stream_of_frames(self) -> None:
        stream = b"".join(_rvc_frame(index=i) for i in range(5))
        samples, rest, _ = self.imu.iter_frames(stream)
        self.assertEqual([s.index for s in samples], [0, 1, 2, 3, 4])
        self.assertEqual(rest, b"")

    def test_resyncs_after_a_dropped_byte(self) -> None:
        good = _rvc_frame(index=7)
        stream = b"\x12\x34" + _rvc_frame(index=6)[3:] + good  # first frame truncated
        samples, _, _ = self.imu.iter_frames(stream)
        self.assertEqual([s.index for s in samples], [7])

    def test_partial_trailing_frame_is_kept_for_next_read(self) -> None:
        whole = _rvc_frame(index=1)
        samples, rest, _ = self.imu.iter_frames(whole + _rvc_frame(index=2)[:10])
        self.assertEqual([s.index for s in samples], [1])
        self.assertEqual(len(rest), 10)
        # The remainder plus the rest of the frame must decode next time.
        more, _, _ = self.imu.iter_frames(rest + _rvc_frame(index=2)[10:])
        self.assertEqual([s.index for s in more], [2])

    def test_header_bytes_inside_the_payload_do_not_derail_it(self) -> None:
        # 0xAAAA is a legal accel value (-21846 mg is not, but the byte
        # pattern can still occur across fields), so the decoder must rely on
        # the checksum rather than the header alone.
        stream = _rvc_frame(index=3, accel=(-21846, 0, 1000)) + _rvc_frame(index=4)
        samples, _, _ = self.imu.iter_frames(stream)
        self.assertEqual([s.index for s in samples], [3, 4])

    def test_tilt_angles_agree_between_the_two_sources(self) -> None:
        # Convention-free cross-check used by the calibration's rest stage.
        sample = self.imu.parse_frame(_rvc_frame(pitch=0.23, roll=-1.29, accel=(22, 4, 995)))
        self.assertAlmostEqual(sample.euler_tilt_deg(), sample.accel_tilt_deg(), delta=0.5)


class KabschTest(unittest.TestCase):
    def setUp(self) -> None:
        from pi_client.imu_rvc import angle_between_deg, kabsch_rotation

        self.kabsch = kabsch_rotation
        self.angle = angle_between_deg

    @staticmethod
    def _rotation(yaw, pitch, roll):
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return rz @ ry @ rx

    def test_recovers_a_known_rotation(self) -> None:
        truth = self._rotation(0.4, -0.25, 0.9)
        rng = np.random.default_rng(0)
        source = rng.normal(size=(8, 3))
        source /= np.linalg.norm(source, axis=1, keepdims=True)
        target = source @ truth.T
        np.testing.assert_allclose(self.kabsch(source, target), truth, atol=1e-9)

    def test_result_is_a_rotation_not_a_reflection(self) -> None:
        rng = np.random.default_rng(1)
        source = rng.normal(size=(6, 3))
        target = rng.normal(size=(6, 3))
        rotation = self.kabsch(source, target)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)

    def test_tolerates_noise(self) -> None:
        truth = self._rotation(-0.7, 0.3, 0.15)
        rng = np.random.default_rng(2)
        source = rng.normal(size=(20, 3))
        source /= np.linalg.norm(source, axis=1, keepdims=True)
        target = source @ truth.T + rng.normal(scale=0.02, size=(20, 3))
        target /= np.linalg.norm(target, axis=1, keepdims=True)
        recovered = self.kabsch(source, target)
        residual = max(self.angle(recovered @ s, t) for s, t in zip(source, target))
        self.assertLess(residual, 5.0)

    def test_rejects_too_few_pairs(self) -> None:
        with self.assertRaises(ValueError):
            self.kabsch([[0, 0, 1]], [[0, 0, 1]])


class BoardUpTest(unittest.TestCase):
    """Reading gravity off a wall board hinges on picking the right in-plane
    axis and the right sign. Corner ordering flips with viewing angle, so the
    choice is made physically (closest to the camera's own up) — these pin
    that down, including the case where the board is NOT on a wall."""

    def setUp(self) -> None:
        from pi_client.imu_calibrate import board_up_camera

        self.board_up_camera = board_up_camera

    @staticmethod
    def _rvec(matrix):
        import cv2

        rvec, _ = cv2.Rodrigues(np.asarray(matrix, dtype=np.float64))
        return rvec

    @staticmethod
    def _board(up, right):
        """Board pose whose in-plane axes are `right` and -`up` (the board's
        +Y runs downward, the usual raster order). Columns of R_cam_from_board
        are the board's axes in camera coords, so the third column must be
        the cross product of the first two — otherwise the matrix is
        left-handed and Rodrigues returns nonsense."""
        up = np.asarray(up, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        normal = np.cross(right, -up)
        return np.stack([right, -up, normal], axis=1)

    def test_upright_board_facing_the_camera(self) -> None:
        # Camera axes: X right, Y down, Z forward. Board on the wall ahead.
        up, _ = self.board_up_camera(self._rvec(
            self._board(up=[0, -1, 0], right=[1, 0, 0])
        ))
        self.assertIsNotNone(up)
        np.testing.assert_allclose(up, [0, -1, 0], atol=1e-9)

    def test_board_rotated_90_in_its_plane_still_reads_up(self) -> None:
        # The vertical direction is now the board's OTHER in-plane axis.
        up, _ = self.board_up_camera(self._rvec(
            self._board(up=[0, -1, 0], right=[0, 0, -1])
        ))
        np.testing.assert_allclose(up, [0, -1, 0], atol=1e-9)

    def test_sign_flip_is_corrected(self) -> None:
        # Board detected upside down: raw axis points at camera +Y (down).
        up, _ = self.board_up_camera(self._rvec(
            self._board(up=[0, 1, 0], right=[-1, 0, 0])
        ))
        np.testing.assert_allclose(up, [0, -1, 0], atol=1e-9)

    def test_camera_pitched_up_still_works(self) -> None:
        # Camera tilted up 20 deg: world up is no longer camera -Y exactly.
        theta = np.radians(20.0)
        world_up_in_cam = np.array([0.0, -np.cos(theta), -np.sin(theta)])
        right = np.array([1.0, 0.0, 0.0])
        up, reason = self.board_up_camera(self._rvec(
            self._board(up=world_up_in_cam, right=right)
        ))
        self.assertIsNotNone(up, reason)
        np.testing.assert_allclose(up, world_up_in_cam, atol=1e-9)

    def test_board_lying_flat_is_rejected(self) -> None:
        # A board on a table/floor has a vertical normal — this procedure
        # reads gravity off the in-plane axis, so it must refuse.
        flat = np.stack([
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, -1.0, 0.0]),   # normal points up
        ], axis=1)
        up, reason = self.board_up_camera(self._rvec(flat))
        self.assertIsNone(up)
        self.assertIn("vertical", reason)


class ImuReadModeTest(unittest.TestCase):
    """`read()` is what ends up in every keyframe, so its two modes and its
    refusal cases are worth locking down."""

    def _reader(self, calibration):
        from pi_client.imu_rvc import RvcReader

        reader = RvcReader.__new__(RvcReader)
        reader.calibration = calibration
        reader._suppressed = 0
        reader._lock = __import__("threading").Lock()
        reader._samples = []
        return reader

    @staticmethod
    def _sample(accel):
        from pi_client.imu_rvc import ImuSample

        return ImuSample(index=0, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
                         accel_mg=accel, monotonic=0.0)

    def test_no_calibration_means_no_gravity(self) -> None:
        reader = self._reader(None)
        reader._samples = [self._sample((0, 0, 1000))]
        self.assertIsNone(reader.read())

    def test_reference_mode_returns_the_measured_constant(self) -> None:
        reader = self._reader({
            "mode": "reference",
            "up_camera_reference": [0.0, -1.0, 0.0],
            "up_imu_reference": [0.0, 0.0, 1.0],
            "tilt_tolerance_deg": 4.0,
        })
        reader._samples = [self._sample((0, 0, 1000))]
        np.testing.assert_allclose(reader.read(), [0.0, 1.0, 0.0], atol=1e-9)

    def test_reference_mode_withholds_when_the_rig_has_tilted(self) -> None:
        reader = self._reader({
            "mode": "reference",
            "up_camera_reference": [0.0, -1.0, 0.0],
            "up_imu_reference": [0.0, 0.0, 1.0],
            "tilt_tolerance_deg": 4.0,
        })
        # ~17 deg away from the reference attitude.
        reader._samples = [self._sample((300, 0, 1000))]
        self.assertIsNone(reader.read())
        self.assertEqual(reader._suppressed, 1)

    def test_reference_mode_tolerates_small_drift(self) -> None:
        reader = self._reader({
            "mode": "reference",
            "up_camera_reference": [0.0, -1.0, 0.0],
            "up_imu_reference": [0.0, 0.0, 1.0],
            "tilt_tolerance_deg": 4.0,
        })
        reader._samples = [self._sample((30, 0, 1000))]  # ~1.7 deg
        self.assertIsNotNone(reader.read())

    def test_full_mode_tracks_the_current_attitude(self) -> None:
        # cam_from_imu maps IMU +Z (up) onto camera -Y (up).
        rotation = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        reader = self._reader({"mode": "full", "cam_from_imu": rotation})
        reader._samples = [self._sample((0, 0, 1000))]
        np.testing.assert_allclose(reader.read(), [0.0, 1.0, 0.0], atol=1e-9)
        # Tilted: the answer must move, unlike reference mode.
        reader._samples = [self._sample((500, 0, 866))]
        tilted = reader.read()
        self.assertGreater(abs(tilted[0]), 0.4)


class ImuIntegratorTest(unittest.TestCase):
    """Preintegration is the piece that makes the accelerometer useful: over
    a keyframe interval its error is millimetres, and the next camera frame
    resets it. These pin down the arithmetic and the guards."""

    def setUp(self) -> None:
        from pi_client.imu_rvc import G_MS2, ImuIntegrator, ImuSample

        self.G = G_MS2
        self.ImuIntegrator = ImuIntegrator
        self.ImuSample = ImuSample

    def _sample(self, t, accel_mg):
        return self.ImuSample(index=int(t * 100) % 256, yaw_deg=0.0, pitch_deg=0.0,
                              roll_deg=0.0, accel_mg=accel_mg, monotonic=t)

    def _feed(self, integrator, accel_mg, duration, rate=100.0):
        step = 1.0 / rate
        t = 0.0
        while t <= duration + 1e-9:
            integrator.add(self._sample(t, accel_mg))
            t += step

    def test_at_rest_nothing_moves(self) -> None:
        integrator = self.ImuIntegrator(up_sensor=(0.0, 0.0, 1.0))
        self._feed(integrator, (0.0, 0.0, 1000.0), 2.0)
        segment = integrator.cut()
        self.assertAlmostEqual(segment["distance_m"], 0.0, places=6)
        self.assertAlmostEqual(segment["speed_ms"], 0.0, places=6)
        self.assertAlmostEqual(segment["dt_s"], 2.0, places=2)

    def test_gravity_is_removed_using_the_measured_direction(self) -> None:
        # Sensor mounted so that "up" is its -X axis: gravity shows on X, and
        # a datasheet-derived assumption about which axis is vertical would
        # produce a metre per second of phantom motion here.
        integrator = self.ImuIntegrator(up_sensor=(-1.0, 0.0, 0.0))
        self._feed(integrator, (-1000.0, 0.0, 0.0), 1.0)
        self.assertAlmostEqual(integrator.cut()["distance_m"], 0.0, places=6)

    def test_constant_acceleration_matches_the_closed_form(self) -> None:
        # 0.5 m/s^2 for 1 s -> 0.25 m travelled, 0.5 m/s reached.
        integrator = self.ImuIntegrator(up_sensor=(0.0, 0.0, 1.0))
        extra_mg = 0.5 / self.G * 1000.0
        self._feed(integrator, (extra_mg, 0.0, 1000.0), 1.0)
        segment = integrator.cut()
        self.assertAlmostEqual(segment["distance_m"], 0.25, delta=0.01)
        self.assertAlmostEqual(segment["speed_ms"], 0.5, delta=0.01)

    def test_realistic_interval_error_is_millimetres(self) -> None:
        """The number the whole approach rests on: a 0.2 deg gravity error
        over one keyframe interval must stay far below the motion itself."""
        integrator = self.ImuIntegrator(up_sensor=(0.0, 0.0, 1.0))
        tilt = np.radians(0.2)
        leaked = (np.sin(tilt) * 1000.0, 0.0, np.cos(tilt) * 1000.0)
        self._feed(integrator, leaked, 0.4)
        drift = integrator.cut()["distance_m"]
        self.assertLess(drift, 0.005)          # millimetres, not centimetres
        self.assertLess(drift, 0.05 * 0.12)    # under 5% of a ~12 cm step

    def test_no_reference_means_no_integration(self) -> None:
        integrator = self.ImuIntegrator(up_sensor=None)
        self._feed(integrator, (0.0, 0.0, 1000.0), 1.0)
        segment = integrator.cut()
        self.assertEqual(segment["samples"], 0)
        self.assertEqual(segment["distance_m"], 0.0)

    def test_a_long_gap_is_not_integrated_across(self) -> None:
        # A dropped USB connection must not turn into a metre of phantom
        # motion when samples resume.
        integrator = self.ImuIntegrator(up_sensor=(0.0, 0.0, 1.0))
        extra_mg = 1.0 / self.G * 1000.0
        integrator.add(self._sample(0.0, (extra_mg, 0.0, 1000.0)))
        integrator.add(self._sample(30.0, (extra_mg, 0.0, 1000.0)))
        self.assertEqual(integrator.cut()["samples"], 0)

    def test_cut_resets_but_keeps_continuity(self) -> None:
        integrator = self.ImuIntegrator(up_sensor=(0.0, 0.0, 1.0))
        self._feed(integrator, (0.0, 0.0, 1000.0), 0.5)
        integrator.cut()
        second = integrator.cut()
        self.assertEqual(second["samples"], 0)
        self.assertEqual(second["dt_s"], 0.0)

    def test_peak_acceleration_is_reported_for_shake_rejection(self) -> None:
        integrator = self.ImuIntegrator(up_sensor=(0.0, 0.0, 1.0))
        integrator.add(self._sample(0.00, (0.0, 0.0, 1000.0)))
        integrator.add(self._sample(0.01, (0.0, 0.0, 1000.0)))
        integrator.add(self._sample(0.02, (500.0, 0.0, 1000.0)))  # a jolt
        integrator.add(self._sample(0.03, (0.0, 0.0, 1000.0)))
        self.assertGreater(integrator.cut()["peak_linear_accel_ms2"], 4.0)


class ImuScaleTest(unittest.TestCase):
    """The IMU scale is meant to replace a monocular estimate the project
    already knows it cannot trust, so its gates matter as much as its
    arithmetic: a bad interval that slips through moves the size of the whole
    reconstruction."""

    def setUp(self) -> None:
        from mac_server.scene3d.poses_step import imu_scale_samples

        self.samples = imu_scale_samples

    @staticmethod
    def _centers(step, count=6):
        # COLMAP positions, in COLMAP units, marching along +X.
        return {i: np.array([i * step, 0.0, 0.0]) for i in range(count)}

    @staticmethod
    def _segments(distance_m, count=6, yaw=0.0, speed=0.3):
        return {
            i: {"distance_m": distance_m, "speed_ms": speed, "yaw_deg": yaw * i}
            for i in range(count)
        }

    def test_recovers_a_known_scale(self) -> None:
        # COLMAP steps of 0.05 units, real steps of 0.15 m -> scale 3.0.
        got = self.samples(self._segments(0.15), self._centers(0.05))
        self.assertEqual(len(got), 5)  # interval 0 has no predecessor
        for value in got.values():
            self.assertAlmostEqual(value, 3.0, places=6)

    def test_tiny_motion_is_skipped(self) -> None:
        # One small number over another is noise, not a measurement.
        got = self.samples(self._segments(0.005), self._centers(0.05))
        self.assertEqual(got, {})

    def test_a_gap_in_registration_is_not_paired(self) -> None:
        centers = self._centers(0.05)
        del centers[3]  # COLMAP failed to register keyframe 3
        got = self.samples(self._segments(0.15), centers)
        self.assertNotIn(3, got)  # 3 has no predecessor pose either side
        self.assertNotIn(4, got)

    def test_turning_intervals_are_skipped(self) -> None:
        """Displacement is accumulated in the rotating sensor frame, so a
        big yaw step mid-interval shortens the vector and would bias the
        scale downward."""
        got = self.samples(self._segments(0.15, yaw=45.0), self._centers(0.05))
        self.assertEqual(got, {})

    def test_small_yaw_steps_are_kept(self) -> None:
        got = self.samples(self._segments(0.15, yaw=3.0), self._centers(0.05))
        self.assertEqual(len(got), 5)

    def test_yaw_wraparound_is_not_a_big_step(self) -> None:
        segments = {
            1: {"distance_m": 0.15, "speed_ms": 0.3, "yaw_deg": 359.0},
            2: {"distance_m": 0.15, "speed_ms": 0.3, "yaw_deg": 2.0},
        }
        segments[0] = {"distance_m": 0.15, "speed_ms": 0.3, "yaw_deg": 356.0}
        got = self.samples(segments, self._centers(0.05))
        self.assertIn(2, got)  # 359 -> 2 is a 3 deg step, not 357

    def test_implausible_speed_is_skipped(self) -> None:
        # Velocity had drifted; the displacement it produced is fiction.
        got = self.samples(self._segments(0.15, speed=9.0), self._centers(0.05))
        self.assertEqual(got, {})

    def test_missing_yaw_does_not_block_the_interval(self) -> None:
        segments = {i: {"distance_m": 0.15, "speed_ms": 0.3} for i in range(4)}
        self.assertEqual(len(self.samples(segments, self._centers(0.05))), 3)

    def test_scale_survives_a_few_bad_intervals(self) -> None:
        from mac_server.scene3d.poses_step import median_scale

        segments = self._segments(0.15, count=30)
        segments[7]["distance_m"] = 2.5    # a jolt
        segments[19]["distance_m"] = 0.9
        got = self.samples(segments, self._centers(0.05, count=30))
        scale, iqr, rejected = median_scale(got)
        self.assertAlmostEqual(scale, 3.0, places=6)
        self.assertIn(7, rejected)
        self.assertIn(19, rejected)


class CameraParamsNumpyTest(unittest.TestCase):
    """pycolmap hands `params` over as a numpy array, and `array or []`
    raises "truth value of an array is ambiguous" — which silently cost a
    whole run its refined intrinsics, the single largest quality fix in the
    pipeline. The type has to be exercised, not just the values."""

    class _Camera:
        width, height = 1536, 864

        def __init__(self, params):
            self.params = params

    def test_numpy_params_are_read(self) -> None:
        cam = self._Camera(np.array([1006.3, 1007.6, 789.2, 419.4,
                                     -0.003, 0.051, -0.004, 0.007]))
        intr = camera_to_intrinsics(cam)
        self.assertAlmostEqual(intr["fx"], 1006.3, places=3)
        self.assertAlmostEqual(intr["dist"][1], 0.051, places=4)

    def test_numpy_dist_argument_is_read(self) -> None:
        cam = self._Camera(np.array([1000.0, 1000.0, 768.0, 432.0]))
        intr = camera_to_intrinsics(cam, dist=np.array([0.1, 0.2, 0.0, 0.0, 0.0]))
        self.assertAlmostEqual(intr["dist"][0], 0.1, places=6)

    def test_empty_dist_falls_back_to_the_camera_params(self) -> None:
        cam = self._Camera(np.array([1000.0, 1000.0, 768.0, 432.0, 0.5, 0.0, 0.0, 0.0]))
        self.assertAlmostEqual(camera_to_intrinsics(cam, dist=[])["dist"][0], 0.5)

    def test_pinhole_params_without_distortion(self) -> None:
        cam = self._Camera(np.array([1000.0, 1000.0, 768.0, 432.0]))
        self.assertEqual(camera_to_intrinsics(cam, dist=None)["dist"], [0.0, 0.0, 0.0, 0.0])


class BaselineScaleTest(unittest.TestCase):
    """Metric scale from a tape-measured driven line.

    Replaces camera-height scale as the default: the camera now sits at
    10-12 cm tilted up, so the floor is a sliver at the frame edge and the
    plane fit that method depends on is ill-conditioned. The previous round
    also left camera_height_m at 0.3 against an actual 0.05-0.06, so every
    map was 5-6x the wrong size and nothing flagged it.
    """

    @staticmethod
    def _centers():
        import numpy as np
        return {
            10: np.array([0.0, 0.0, 0.0]),
            20: np.array([0.5, 0.0, 0.0]),   # half a unit apart
            30: np.array([0.0, 0.0, 0.0]),   # coincident with 10
        }

    def test_recovers_the_scale_factor(self):
        from mac_server.scene3d.reconstruct_step import baseline_scale
        scale, diag = baseline_scale(
            self._centers(), {"from_frame": 10, "to_frame": 20, "length_m": 1.0}
        )
        self.assertAlmostEqual(scale, 2.0, places=6)
        self.assertAlmostEqual(diag["reconstructed_chord"], 0.5, places=6)

    def test_none_without_a_baseline_file(self):
        from mac_server.scene3d.reconstruct_step import baseline_scale
        scale, diag = baseline_scale(self._centers(), None)
        self.assertIsNone(scale)
        self.assertIn("no baseline", diag["reason"])

    def test_reports_unregistered_frames_by_number(self):
        """A silent None here would fall through to whatever method is next."""
        from mac_server.scene3d.reconstruct_step import baseline_scale
        scale, diag = baseline_scale(
            self._centers(), {"from_frame": 10, "to_frame": 999, "length_m": 1.0}
        )
        self.assertIsNone(scale)
        self.assertIn("999", diag["reason"])

    def test_rejects_a_degenerate_chord(self):
        from mac_server.scene3d.reconstruct_step import baseline_scale
        scale, diag = baseline_scale(
            self._centers(), {"from_frame": 10, "to_frame": 30, "length_m": 1.0}
        )
        self.assertIsNone(scale)
        self.assertIn("same point", diag["reason"])

    def test_rejects_malformed_and_non_positive_input(self):
        from mac_server.scene3d.reconstruct_step import baseline_scale
        for bad in ({"from_frame": 10}, {"from_frame": 10, "to_frame": 20, "length_m": 0.0},
                    {"from_frame": "a", "to_frame": 20, "length_m": 1.0}):
            scale, diag = baseline_scale(self._centers(), bad)
            self.assertIsNone(scale, bad)
            self.assertIn("reason", diag)


class ScaleSourceFallbackTest(unittest.TestCase):
    """`auto` must never leave a session unbuildable, and must never let a
    weak estimate pass for a strong one.

    The first version defaulted to `known_baseline` with NO fallback, on the
    reasoning that a silent substitution is what let a 5-6x scale error
    survive a whole round. Right about the danger, wrong about the remedy:
    every session recorded without a baseline became a hard failure with no
    way forward (session_20260803_171339). The fallback is loud now, not fatal.
    """

    def _order(self, preference):
        """Mirror of the selection in reconstruct_step._run_vggt."""
        if preference == "auto":
            return ["known_baseline", "imu", "camera_height"]
        return [preference]

    def test_auto_prefers_a_measured_baseline(self):
        self.assertEqual(self._order("auto")[0], "known_baseline")

    def test_camera_height_is_the_last_resort_not_the_first(self):
        """At 13 cm looking up, the floor is a sliver at the frame edge and
        its plane fit is ill-conditioned — 15% inliers on the real session."""
        self.assertEqual(self._order("auto")[-1], "camera_height")

    def test_an_explicit_choice_is_honoured_strictly(self):
        """Silently substituting a method the operator rejected is exactly
        how the original error survived."""
        self.assertEqual(self._order("imu"), ["imu"])

    def test_config_default_is_auto(self):
        import json
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        config = json.loads((repo / "config/default.json").read_text())
        self.assertEqual(config["scene3d"]["reconstruction"]["scale_source"], "auto")


class ClassLogitsTest(unittest.TestCase):
    """Whether the class head needs a sigmoid depends on where the hef's graph
    was cut, not on the architecture — and being wrong about it produces zero
    detections forever with no error anywhere."""

    @staticmethod
    def _outputs(num_classes, class_fill):
        out = {}
        for stride in (8, 16, 32):
            s = 640 // stride
            out[f"b{s}"] = np.zeros((s, s, 64), dtype=np.float32)
            out[f"s{s}"] = np.full((s, s, num_classes), class_fill, dtype=np.float32)
            out[f"c{s}"] = np.zeros((s, s, 32), dtype=np.float32)
        out["p"] = np.zeros((160, 160, 32), dtype=np.float32)
        return out

    def test_logit_arch_recovers_a_detection_a_probability_arch_would_miss(self) -> None:
        # A logit of 2.0 is sigmoid 0.881 — comfortably over a 0.4 threshold.
        # Read as a probability it is 2.0, which also passes, but a logit of
        # -1.0 (sigmoid 0.27) vs raw -1.0 is the case that separates them.
        logits = self._outputs(31, 2.0)
        with_sigmoid = yolov8_seg_postprocess(logits, score_threshold=0.4,
                                              arch=INDOOR_ADE20K)
        self.assertGreater(len(with_sigmoid["scores"]), 0)
        self.assertLessEqual(float(with_sigmoid["scores"].max()), 1.0)

    def test_negative_logits_stay_below_threshold(self) -> None:
        # The real hef's class blob ranged -45.7..1.48 on a live frame. Under
        # the sigmoid those are 0.0..0.81; read raw they are nonsense.
        quiet = self._outputs(31, -34.0)
        result = yolov8_seg_postprocess(quiet, score_threshold=0.4, arch=INDOOR_ADE20K)
        self.assertEqual(len(result["scores"]), 0)

    def test_default_arch_is_untouched(self) -> None:
        self.assertFalse(YOLOV8_SEG.class_logits)
        probs = self._outputs(80, 0.9)
        result = yolov8_seg_postprocess(probs, score_threshold=0.4, arch=YOLOV8_SEG)
        # 0.9 is already a probability; a stray sigmoid would make it 0.71 and
        # still pass, so assert the value itself rather than just the count.
        self.assertAlmostEqual(float(result["scores"].max()), 0.9, places=4)


class ImuGravityGateTest(unittest.TestCase):
    """A segment whose gravity the driver refused to publish must not set the
    size of a reconstruction — measured cost of ignoring this: a 17x scale
    error and a floor plan with no walls (session_20260807_171938)."""

    @staticmethod
    def _centers():
        return {1: np.array([0.0, 0, 0]), 2: np.array([0.1, 0, 0]),
                3: np.array([0.2, 0, 0])}

    def test_untrusted_segments_are_dropped(self) -> None:
        from mac_server.scene3d.poses_step import imu_scale_samples

        segments = {
            2: {"distance_m": 0.5, "speed_ms": 0.4, "gravity_ok": False},
            3: {"distance_m": 0.5, "speed_ms": 0.4, "gravity_ok": True},
        }
        out = imu_scale_samples(segments, self._centers())
        self.assertNotIn(2, out)
        self.assertIn(3, out)

    def test_legacy_sessions_without_the_field_still_work(self) -> None:
        # Every session recorded before the flag existed omits it; treating
        # absent as untrusted would silently disable the IMU on all of them.
        from mac_server.scene3d.poses_step import imu_scale_samples

        segments = {2: {"distance_m": 0.5, "speed_ms": 0.4}}
        self.assertIn(2, imu_scale_samples(segments, self._centers()))


class WallAzimuthTest(unittest.TestCase):
    """Направления стен для манхэттенской привязки. Первая версия искала их
    связными компонентами и на замкнутом контуре комнаты давала ноль: четыре
    стены сходятся в углах в одну компоненту, у которой главная ось —
    диагональ. Хаф ищет прямые куски независимо от связности."""

    @staticmethod
    def _room(rotate_deg=0.0):
        import cv2

        g = np.zeros((80, 80), np.uint8)
        g[10, 5:75] = 1; g[70, 5:75] = 1
        g[5:75, 10] = 1; g[5:75, 70] = 1
        if rotate_deg:
            m = cv2.getRotationMatrix2D((40, 40), rotate_deg, 1.0)
            g = (cv2.warpAffine(g * 255, m, (80, 80),
                                flags=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        return g

    def test_a_closed_room_yields_its_four_walls(self) -> None:
        from mac_server.scene3d.occupancy import wall_azimuths

        segments = wall_azimuths(self._room())
        self.assertGreaterEqual(len(segments), 4)

    def test_axis_aligned_room_needs_no_correction(self) -> None:
        from pi_client.imu_rvc_math import manhattan_yaw_offset
        from mac_server.scene3d.occupancy import wall_azimuths

        offset, strength = manhattan_yaw_offset(
            [a for a, _ in wall_azimuths(self._room())])
        self.assertAlmostEqual(math.degrees(offset), 0.0, delta=1.0)
        self.assertGreater(strength, 0.9)

    def test_a_rotated_room_reports_its_rotation(self) -> None:
        from pi_client.imu_rvc_math import manhattan_yaw_offset
        from mac_server.scene3d.occupancy import wall_azimuths

        offset, strength = manhattan_yaw_offset(
            [a for a, _ in wall_azimuths(self._room(12.0))])
        self.assertAlmostEqual(abs(math.degrees(offset)), 12.0, delta=1.5)
        self.assertGreater(strength, 0.9)

    def test_empty_and_speckle_grids_report_nothing(self) -> None:
        from mac_server.scene3d.occupancy import wall_azimuths

        self.assertEqual(wall_azimuths(np.zeros((40, 40), np.uint8)), [])
        speckle = np.zeros((40, 40), np.uint8)
        speckle[5, 5] = speckle[20, 30] = speckle[33, 7] = 1
        self.assertEqual(wall_azimuths(speckle), [])


class HoughShapeTest(unittest.TestCase):
    """HoughLinesP отдаёт то (N,1,4), то (N,4). Предположение об одной форме
    уронило весь шаг tsdf на живой сессии пользователя, хотя на тех же данных
    у меня проходило."""

    def test_both_return_shapes_are_handled(self) -> None:
        from unittest.mock import patch

        from mac_server.scene3d.occupancy import wall_azimuths

        grid = np.zeros((40, 40), np.uint8)
        grid[10, 5:35] = 1
        lines = np.array([[5, 10, 34, 10]], dtype=np.int32)
        for shape in ((1, 1, 4), (1, 4)):
            with patch("cv2.HoughLinesP", return_value=lines.reshape(shape)):
                out = wall_azimuths(grid)
            self.assertEqual(len(out), 1, f"форма {shape} не обработана")
            self.assertAlmostEqual(math.degrees(out[0][0]), 0.0, delta=1.0)

    def test_no_lines_found_is_not_a_crash(self) -> None:
        from unittest.mock import patch

        from mac_server.scene3d.occupancy import wall_azimuths

        grid = np.zeros((40, 40), np.uint8)
        grid[10, 5:35] = 1
        with patch("cv2.HoughLinesP", return_value=None):
            self.assertEqual(wall_azimuths(grid), [])
