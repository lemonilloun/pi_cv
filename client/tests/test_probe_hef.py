"""Tests for the hef probe's layout classifier.

`classify` is the part of the probe that turns shapes into a decision, and it
is the part that can be wrong while looking right — so it is tested against
shapes taken from the real models rather than invented ones. Everything else in
probe_hef.py is I/O against the NPU and can only be exercised on the Pi.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "probe_hef", REPO_ROOT / "scripts" / "probe_hef.py"
)
probe_hef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe_hef)


def _streams(shapes):
    return {"inputs": [{"name": "input", "shape": [640, 640, 3]}],
            "outputs": [{"name": f"out{i}", "shape": list(s)}
                        for i, s in enumerate(shapes)]}


# The blob shapes yolov8s_seg_h8 actually emits, in the order
# seg_postprocess.order_endnodes already copes with (it sorts by shape, so the
# listing order here is deliberately shuffled — a classifier that depends on
# order would be fragile against a HailoRT version bump).
YOLOV8_SEG_SHAPES = [
    (80, 80, 64), (80, 80, 80), (80, 80, 32),
    (40, 40, 64), (40, 40, 80), (40, 40, 32),
    (20, 20, 64), (20, 20, 80), (20, 20, 32),
    (160, 160, 32),
]

# FastSAM-s is the same architecture with a single class.
FASTSAM_SHAPES = [
    (80, 80, 64), (80, 80, 1), (80, 80, 32),
    (40, 40, 64), (40, 40, 1), (40, 40, 32),
    (20, 20, 64), (20, 20, 1), (20, 20, 32),
    (160, 160, 32),
]


class ClassifyTest(unittest.TestCase):
    def test_yolov8_seg_layout_recovers_the_architecture(self):
        result = probe_hef.classify(_streams(YOLOV8_SEG_SHAPES))
        self.assertEqual(result["layout"], "yolov8_seg_raw")
        self.assertEqual(result["num_classes"], 80)
        self.assertEqual(result["num_masks"], 32)
        self.assertEqual(result["reg_max"], 15)
        self.assertEqual(result["proto_h"], 160)
        self.assertEqual(result["scale_h"], [20, 40, 80])

    def test_fastsam_single_class_head_is_derived_not_assumed(self):
        # The whole point of M0: nothing in the classifier knows the number 1.
        result = probe_hef.classify(_streams(FASTSAM_SHAPES))
        self.assertEqual(result["layout"], "yolov8_seg_raw")
        self.assertEqual(result["num_classes"], 1)
        self.assertEqual(result["num_masks"], 32)
        self.assertEqual(result["reg_max"], 15)

    def test_on_chip_nms_single_flat_output(self):
        result = probe_hef.classify(_streams([(100, 6)]))
        self.assertEqual(result["layout"], "on_chip_nms")

    def test_embedding_head_is_named_not_left_unknown(self):
        # The shape clip_resnet_50x4_h8 really emits, measured on the Pi.
        result = probe_hef.classify(_streams([(1, 1, 640)]))
        self.assertEqual(result["layout"], "embedding")
        self.assertEqual(result["dim"], 640)

    def test_unknown_layout_refuses_rather_than_guessing(self):
        result = probe_hef.classify(_streams([(80, 80, 64), (40, 40, 64)]))
        self.assertEqual(result["layout"], "unknown")
        # The evidence must travel with the refusal, or the operator has to
        # re-run the probe to find out what it saw.
        self.assertIn((80, 80, 64), result["evidence"])

    def test_ambiguous_class_and_mask_channel_counts_are_flagged(self):
        # A 32-class segmentation model with 32 prototypes: shape alone cannot
        # tell the class head from the coefficients. The classifier must say so
        # rather than pick one, because picking wrong yields masks that look
        # plausible and are attached to the wrong objects.
        shapes = [
            (80, 80, 64), (80, 80, 32), (80, 80, 32),
            (40, 40, 64), (40, 40, 32), (40, 40, 32),
            (20, 20, 64), (20, 20, 32), (20, 20, 32),
            (160, 160, 32),
        ]
        result = probe_hef.classify(_streams(shapes))
        joined = " ".join(result["notes"])
        self.assertIn("AMBIGUOUS", joined)


class ClassNamesTest(unittest.TestCase):
    """The names file and the trained weights share an ordering that exists in
    no single place, so the only defence is refusing when the counts differ."""

    def setUp(self) -> None:
        import sys
        for entry in (str(REPO_ROOT / "client/src"), str(REPO_ROOT / "server/src")):
            if entry not in sys.path:
                sys.path.insert(0, entry)

    def test_real_config_matches_the_declared_arch(self) -> None:
        from pi_client.scene_recorder import load_class_names
        from pi_client.seg_postprocess import INDOOR_ADE20K

        names = load_class_names(
            REPO_ROOT / "config/seg_classes_indoor.json", INDOOR_ADE20K.num_classes)
        self.assertEqual(len(names), 31)
        # Order is load-bearing: id 0 is what the model emits for a wall.
        self.assertEqual(names[0], "wall")
        self.assertIn("cabinet", names)
        self.assertIn("door", names)

    def test_count_mismatch_refuses(self) -> None:
        import json
        import tempfile
        from pi_client.scene_recorder import load_class_names

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "classes.json"
            path.write_text(json.dumps({"classes": [{"name": "wall"}, {"name": "floor"}]}))
            with self.assertRaises(ValueError):
                load_class_names(path, 31)
            # ...and the matching count is accepted, so the guard is not just
            # rejecting everything.
            self.assertEqual(load_class_names(path, 2), ["wall", "floor"])


if __name__ == "__main__":
    unittest.main()
