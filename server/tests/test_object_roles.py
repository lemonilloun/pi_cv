"""Tests for class roles and label-flip-tolerant association.

These rules decide whether a room ends up with one cabinet or eleven, so they
are tested against the failures actually observed on this robot's output — not
invented ones. Every case below cites the measurement it comes from.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mac_server.scene3d.object_roles import (  # noqa: E402
    AGGREGATE_CLASSES,
    ROLE_AREA,
    ROLE_OBJECT,
    classes_may_be_same,
    load_roles,
    partition_detections,
    scale_plausibility,
    vote_label,
)
from mac_server.scene3d.objects_step import ObjectBank  # noqa: E402


CONFIG = REPO_ROOT / "config/seg_classes_indoor.json"


class RolesTest(unittest.TestCase):
    def test_real_config_marks_only_the_regions_as_area(self) -> None:
        roles = load_roles(CONFIG)
        self.assertEqual(roles["wall"], ROLE_AREA)
        self.assertEqual(roles["floor"], ROLE_AREA)
        self.assertEqual(roles["ceiling"], ROLE_AREA)
        # Doors and windows are structure but they are LANDMARKS — "the white
        # door 15 degrees to the left" is exactly the phrase this whole
        # pipeline exists to produce, so they must stay instanced.
        self.assertEqual(roles["door"], ROLE_OBJECT)
        self.assertEqual(roles["window"], ROLE_OBJECT)
        self.assertEqual(roles["cabinet"], ROLE_OBJECT)

    def test_partition_splits_objects_floor_and_ignored(self) -> None:
        roles = load_roles(CONFIG)
        dets = [
            {"class": "cabinet"}, {"class": "wall"}, {"class": "floor"},
            {"class": "wall"}, {"class": "door"}, {"class": "ceiling"},
        ]
        objects, aggregate, ignored = partition_detections(dets, roles)
        self.assertEqual(objects, [0, 4])       # cabinet, door
        self.assertEqual(aggregate, [2])        # floor -> one record
        self.assertEqual(ignored, [1, 3, 5])    # walls + ceiling

    def test_partition_returns_indices_not_copies(self) -> None:
        # Callers keep masks, points and embeddings in parallel lists; indices
        # are what lets them stay aligned.
        roles = {"a": ROLE_OBJECT, "wall": ROLE_AREA}
        objects, _, ignored = partition_detections(
            [{"class": "wall"}, {"class": "a"}], roles)
        self.assertEqual(objects, [1])
        self.assertEqual(ignored, [0])

    def test_unknown_class_defaults_to_object(self) -> None:
        # A class someone adds without setting a role should appear on the plan
        # and get noticed, not vanish silently.
        objects, _, ignored = partition_detections([{"class": "newthing"}], {})
        self.assertEqual(objects, [0])
        self.assertEqual(ignored, [])

    def test_floor_is_the_only_aggregate(self) -> None:
        self.assertEqual(set(AGGREGATE_CLASSES), {"floor"})


class ConfusableTest(unittest.TestCase):
    def test_identical_names_always_match(self) -> None:
        self.assertTrue(classes_may_be_same("cabinet", "cabinet"))

    def test_the_measured_cabinet_door_flip_is_forgiven(self) -> None:
        # Measured on a real snapshot: one box came back as cabinet 0.339 AND
        # door 0.339. Refusing that merge creates two objects in one place.
        self.assertTrue(classes_may_be_same("cabinet", "door"))

    def test_the_measured_wall_curtain_flip_is_forgiven(self) -> None:
        # Also measured: `wall 0.32` landed on a curtain.
        self.assertTrue(classes_may_be_same("wall", "curtain"))

    def test_area_and_furniture_classes_do_not_merge(self) -> None:
        # wall is in the flat-vertical-surface group and cabinet in the
        # storage-furniture one, so they stay apart. In practice this never
        # comes up — wall is an area class and is dropped before the bank —
        # but the gate should not depend on that happening upstream.
        self.assertFalse(classes_may_be_same("wall", "cabinet"))

    def test_unrelated_classes_still_block(self) -> None:
        # The gate must still do its job, or appearance alone could weld a
        # chair to a refrigerator.
        self.assertFalse(classes_may_be_same("chair", "refrigerator"))
        self.assertFalse(classes_may_be_same("plant", "toilet"))


class VoteLabelTest(unittest.TestCase):
    def test_confidence_beats_raw_count(self) -> None:
        # Evidence mass, not popularity: five weak `door` guesses total 1.25
        # while two confident `cabinet` ones total 1.70, so cabinet wins on
        # confidence and loses on a plain tally. Note the rule really is a SUM,
        # so enough weak votes do outweigh a few strong ones — that is
        # deliberate (a thing seen consistently is evidence too), and the
        # numbers here are chosen to separate the two rules rather than to
        # flatter either.
        counts = {"door": 5, "cabinet": 2}
        conf = {"door": 5 * 0.25, "cabinet": 2 * 0.85}
        self.assertEqual(vote_label(counts, conf)[0], "cabinet")
        self.assertEqual(vote_label(counts)[0], "door")

    def test_agreement_reports_how_settled_the_label_is(self) -> None:
        sure = vote_label({"bed": 5}, {"bed": 4.0})
        torn = vote_label({"cabinet": 3, "door": 3}, {"cabinet": 1.0, "door": 1.0})
        self.assertAlmostEqual(sure[1], 1.0)
        self.assertAlmostEqual(torn[1], 0.5)

    def test_empty_votes_do_not_crash(self) -> None:
        self.assertEqual(vote_label({}), ("unknown", 0.0))


class BankDedupTest(unittest.TestCase):
    """The end-to-end property the user asked for: one real object stays one
    object even when the detector flickers and renames it between frames."""

    @staticmethod
    def _emb(seed: float):
        import numpy as np

        rng = np.random.default_rng(int(seed * 1000))
        v = rng.normal(size=64)
        return [float(x) for x in v / np.linalg.norm(v)]

    def test_label_flip_does_not_split_one_object(self) -> None:
        bank = ObjectBank(dino_cos_min=0.6, centroid_max_m=0.5)
        emb = self._emb(1.0)
        idx_a = bank.add({"track_id": -1, "class": "cabinet", "centroid": [1.0, 0.0, 0.5],
                          "dino_emb": emb, "confidence": 0.34, "keyframe": 1})
        idx_b = bank.add({"track_id": -1, "class": "door", "centroid": [1.05, 0.0, 0.5],
                          "dino_emb": emb, "confidence": 0.34, "keyframe": 2})
        self.assertEqual(idx_a, idx_b, "cabinet/door flip split one object in two")
        self.assertEqual(len(bank.objects), 1)

    def test_flicker_across_frames_still_yields_one_object(self) -> None:
        # "в рамках нескольких секунд один и тот же объект может появляться и
        # исчезать" — a gap in the track must not mint a new object, because
        # position and appearance still agree.
        bank = ObjectBank()
        emb = self._emb(2.0)
        for keyframe, track in ((1, 7), (2, -1), (5, 12), (6, 12)):
            bank.add({"track_id": track, "class": "sofa", "centroid": [2.0, 1.0, 0.4],
                      "dino_emb": emb, "confidence": 0.7, "keyframe": keyframe})
        self.assertEqual(len(bank.objects), 1)
        self.assertEqual(bank.objects[0]["n_observations"], 4)

    def test_genuinely_different_objects_stay_separate(self) -> None:
        # The guard against over-merging: same class, but metres apart.
        bank = ObjectBank()
        bank.add({"track_id": -1, "class": "chair", "centroid": [0.0, 0.0, 0.0],
                  "dino_emb": self._emb(3.0), "confidence": 0.8, "keyframe": 1})
        bank.add({"track_id": -1, "class": "chair", "centroid": [4.0, 3.0, 0.0],
                  "dino_emb": self._emb(4.0), "confidence": 0.8, "keyframe": 2})
        self.assertEqual(len(bank.objects), 2)

    def test_finalize_reports_the_voted_label_and_agreement(self) -> None:
        # door/cabinet, the pair actually measured flipping on one box. `wall`
        # deliberately cannot appear here: it is an area class and never
        # reaches the bank at all.
        bank = ObjectBank()
        emb = self._emb(5.0)
        for cls, conf in (("door", 0.30), ("door", 0.30), ("cabinet", 0.85)):
            bank.add({"track_id": -1, "class": cls, "centroid": [1.0, 0.0, 0.5],
                      "dino_emb": emb, "confidence": conf, "keyframe": 1})
        out = bank.finalize(min_observations=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["class_top"], "cabinet")
        self.assertLess(out[0]["label_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()


class ArbiterTest(unittest.TestCase):
    """The VLM is consulted only where the embeddings are undecided, and an
    unreachable VLM must never read as 'these are different objects'."""

    @staticmethod
    def _emb_pair(cos_target: float):
        """Two unit vectors with a chosen cosine between them."""
        import numpy as np

        a = np.zeros(64); a[0] = 1.0
        b = np.zeros(64); b[0] = cos_target; b[1] = (1 - cos_target ** 2) ** 0.5
        return [float(v) for v in a], [float(v) for v in b]

    def _bank(self, verdict, **kw):
        calls = []

        def arbiter(crop_a, crop_b):
            calls.append((crop_a, crop_b))
            return verdict

        return ObjectBank(dino_cos_min=0.6, arbiter=arbiter, arbiter_band=0.15, **kw), calls

    def _add_pair(self, bank, cos_target):
        a, b = self._emb_pair(cos_target)
        bank.add({"track_id": -1, "class": "cabinet", "centroid": [1.0, 0.0, 0.5],
                  "dino_emb": a, "confidence": 0.6, "keyframe": 1, "crop_jpeg": b"A"})
        bank.add({"track_id": -1, "class": "cabinet", "centroid": [1.1, 0.0, 0.5],
                  "dino_emb": b, "confidence": 0.6, "keyframe": 2, "crop_jpeg": b"B"})

    def test_confident_match_never_asks(self) -> None:
        bank, calls = self._bank(True)
        self._add_pair(bank, 0.9)          # well above the 0.6 threshold
        self.assertEqual(calls, [])
        self.assertEqual(len(bank.objects), 1)

    def test_far_below_threshold_never_asks(self) -> None:
        bank, calls = self._bank(True)
        self._add_pair(bank, 0.2)          # far outside the 0.45-0.6 band
        self.assertEqual(calls, [])
        self.assertEqual(len(bank.objects), 2)

    def test_borderline_pair_is_merged_when_the_vlm_says_same(self) -> None:
        bank, calls = self._bank(True)
        self._add_pair(bank, 0.55)         # inside the band
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(bank.objects), 1)
        self.assertEqual(bank.arbiter_merges, 1)

    def test_borderline_pair_stays_split_when_the_vlm_says_different(self) -> None:
        bank, calls = self._bank(False)
        self._add_pair(bank, 0.55)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(bank.objects), 2)

    def test_unreachable_vlm_falls_back_and_does_not_split_extra(self) -> None:
        # None means "no answer". It must behave exactly as if no arbiter were
        # configured — an outage that silently doubled every object would be
        # far worse than no arbitration.
        bank_none, calls = self._bank(None)
        self._add_pair(bank_none, 0.55)
        plain = ObjectBank(dino_cos_min=0.6)
        self._add_pair(plain, 0.55)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(bank_none.objects), len(plain.objects))

    def test_no_crops_means_no_call(self) -> None:
        # Old sessions carry no crop bytes; the arbiter cannot judge blind.
        bank, calls = self._bank(True)
        a, b = self._emb_pair(0.55)
        bank.add({"track_id": -1, "class": "cabinet", "centroid": [1.0, 0.0, 0.5],
                  "dino_emb": a, "confidence": 0.6, "keyframe": 1})
        bank.add({"track_id": -1, "class": "cabinet", "centroid": [1.1, 0.0, 0.5],
                  "dino_emb": b, "confidence": 0.6, "keyframe": 2})
        self.assertEqual(calls, [])


class ScalePlausibilityTest(unittest.TestCase):
    """A global scale error passes every internal consistency check and then
    ruins the floor plan. This is the only cheap outside opinion available."""

    @staticmethod
    def _obj(cls, largest):
        return {"class_top": cls, "extent": [0.3, largest, 0.4]}

    def test_correctly_scaled_room_passes(self) -> None:
        out = scale_plausibility([
            self._obj("bed", 2.0), self._obj("door", 2.0), self._obj("chair", 1.1),
        ])
        self.assertTrue(out["plausible"])

    def test_the_measured_two_times_error_is_caught(self) -> None:
        # The real numbers from session_20260807_165420.
        out = scale_plausibility([
            self._obj("bed", 4.08), self._obj("window", 3.12),
            self._obj("person", 2.50), self._obj("curtain", 1.73),
        ])
        self.assertFalse(out["plausible"])
        self.assertGreater(out["median_ratio"], 1.5)
        self.assertIn("baseline.json", out["warning"])

    def test_too_few_objects_declines_to_judge(self) -> None:
        # Two objects is noise; claiming a scale error from it would be worse
        # than staying quiet.
        out = scale_plausibility([self._obj("bed", 4.0), self._obj("door", 4.0)])
        self.assertFalse(out["checked"])

    def test_unknown_classes_are_ignored_not_counted(self) -> None:
        out = scale_plausibility([
            self._obj("bed", 2.0), self._obj("door", 2.0), self._obj("chair", 1.1),
            {"class_top": "spaceship", "extent": [90.0, 90.0, 90.0]},
        ])
        self.assertTrue(out["plausible"])
        self.assertEqual(out["n_objects"], 3)

    def test_a_too_small_reconstruction_is_caught_too(self) -> None:
        # Well under the partial-view floor: a 0.2 m bed is not a bed seen
        # from one side, it is a wrong scale.
        out = scale_plausibility([
            self._obj("bed", 0.2), self._obj("door", 0.2), self._obj("sofa", 0.2),
        ])
        self.assertFalse(out["plausible"])

    def test_partially_observed_objects_are_not_flagged(self) -> None:
        # The false alarm this asymmetry exists to prevent. Real numbers from
        # session_20260807_171938, whose scale was independently confirmed by
        # the room measuring 4.09 x 3.58 m: a median ratio of 0.61 with every
        # object individually plausible. You never see the whole of anything,
        # so measured extents run small.
        out = scale_plausibility([
            self._obj("door", 1.30), self._obj("table", 1.00),
            self._obj("window", 0.95), self._obj("curtain", 1.60),
            self._obj("painting", 0.70), self._obj("lamp", 1.10),
        ])
        self.assertLess(out["median_ratio"], 0.8)
        self.assertTrue(out["plausible"], out.get("warning"))


class ImuPoseCheckTest(unittest.TestCase):
    """IMU — независимый свидетель для поз: он не смотрит на картинку и не
    может ошибиться так же, как ошибается визуальное сопоставление."""

    def test_pure_yaw_is_measured_as_that_yaw(self) -> None:
        from mac_server.scene3d.object_roles import relative_rotation_deg

        a = {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
        b = {"yaw_deg": 30.0, "pitch_deg": 0.0, "roll_deg": 0.0}
        self.assertAlmostEqual(relative_rotation_deg(a, b), 30.0, places=4)

    def test_the_pm180_wrap_is_two_degrees_not_358(self) -> None:
        # Вычитание углов Эйлера здесь даёт 358 — метрикой на поворотах оно
        # не является, поэтому считается настоящий угол.
        from mac_server.scene3d.object_roles import relative_rotation_deg

        a = {"yaw_deg": 179.0, "pitch_deg": 0.0, "roll_deg": 0.0}
        b = {"yaw_deg": -179.0, "pitch_deg": 0.0, "roll_deg": 0.0}
        self.assertAlmostEqual(relative_rotation_deg(a, b), 2.0, places=4)

    def test_agreeing_pair_is_not_flagged(self) -> None:
        from mac_server.scene3d.object_roles import poses_disagreeing_with_imu

        imu = {1: {"yaw_deg": 0.0}, 2: {"yaw_deg": 20.0}}
        self.assertEqual(poses_disagreeing_with_imu(imu, {(1, 2): 21.0}), [])

    def test_a_mismatched_pair_is_reported_with_both_numbers(self) -> None:
        from mac_server.scene3d.object_roles import poses_disagreeing_with_imu

        imu = {1: {"yaw_deg": 0.0}, 2: {"yaw_deg": 5.0}}
        out = poses_disagreeing_with_imu(imu, {(1, 2): 90.0})
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["imu_deg"], 5.0, places=1)
        self.assertAlmostEqual(out[0]["pose_deg"], 90.0, places=1)

    def test_frames_without_imu_are_skipped_not_flagged(self) -> None:
        # Старые сессии без IMU не должны выглядеть как сплошная ошибка поз.
        from mac_server.scene3d.object_roles import poses_disagreeing_with_imu

        self.assertEqual(poses_disagreeing_with_imu({}, {(1, 2): 90.0}), [])
