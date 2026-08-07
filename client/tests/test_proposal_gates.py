"""Tests for the recorder's proposal gates.

The gates are inert by default and only matter for class-agnostic proposals,
which is exactly why they need tests now: their first real use will be on a
model whose output nobody has looked at yet, and a gate that is subtly wrong
there would present as "the new detector found nothing".
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "client/src"), str(REPO_ROOT / "server/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402

from pi_client.scene_recorder import gate_proposals  # noqa: E402

FRAME = (100, 200)  # h, w


def det(bbox):
    return {"class": "object", "confidence": 0.9, "bbox_xyxy": list(bbox)}


def mask_of_area(area_frac, size=20):
    """A mask covering `area_frac` of its own frame."""
    m = np.zeros((size, size), dtype=np.float32)
    cells = int(round(area_frac * size * size))
    m.reshape(-1)[:cells] = 1.0
    return m


class GateProposalsTest(unittest.TestCase):
    def test_defaults_keep_everything(self) -> None:
        dets = [det((0, 0, 200, 100)), det((10, 10, 20, 20))]
        masks = np.stack([mask_of_area(0.99), mask_of_area(0.0001)])
        keep, reasons = gate_proposals(dets, masks, FRAME)
        self.assertEqual(keep, [0, 1])
        self.assertEqual(reasons, {})

    def test_area_bounds_reject_specks_and_walls(self) -> None:
        dets = [det((10, 10, 20, 20)), det((10, 10, 90, 90)), det((10, 10, 30, 30))]
        masks = np.stack([mask_of_area(0.001), mask_of_area(0.30), mask_of_area(0.95)])
        keep, reasons = gate_proposals(
            dets, masks, FRAME, min_area_frac=0.01, max_area_frac=0.60
        )
        self.assertEqual(keep, [1])
        self.assertEqual(reasons, {"mask_too_small": 1, "mask_too_large": 1})

    def test_edge_rule_counts_edges_not_size(self) -> None:
        # A big central box touches nothing; a thin strip pinned left/right/top
        # touches three. Size alone must not decide this.
        big_central = det((20, 10, 180, 90))
        three_edges = det((0, 0, 200, 30))
        keep, reasons = gate_proposals(
            [big_central, three_edges], None, FRAME, reject_edge_count=3
        )
        self.assertEqual(keep, [0])
        self.assertEqual(reasons, {"touches_frame_edges": 1})

    def test_a_box_on_two_edges_survives_a_three_edge_rule(self) -> None:
        # A cabinet in a corner legitimately touches two edges.
        corner = det((0, 0, 60, 60))
        keep, _ = gate_proposals([corner], None, FRAME, reject_edge_count=3)
        self.assertEqual(keep, [0])

    def test_full_frame_box_is_rejected_at_four(self) -> None:
        keep, reasons = gate_proposals(
            [det((0, 0, 200, 100))], None, FRAME, reject_edge_count=4
        )
        self.assertEqual(keep, [])
        self.assertEqual(reasons, {"touches_frame_edges": 1})

    def test_zero_disables_the_edge_rule_entirely(self) -> None:
        # The default. A full-frame box survives, because turning a gate on
        # must be an explicit act — otherwise adding the feature silently
        # changes what every existing recording command does.
        keep, reasons = gate_proposals(
            [det((0, 0, 200, 100))], None, FRAME, reject_edge_count=0
        )
        self.assertEqual(keep, [0])
        self.assertEqual(reasons, {})

    def test_area_measured_on_the_mask_not_the_box(self) -> None:
        # A thin diagonal sliver's box is huge while its mask is tiny. The
        # mask is what gets reconstructed, so the mask is what must decide.
        sliver = det((0, 0, 199, 99))
        masks = np.stack([mask_of_area(0.002)])
        keep, reasons = gate_proposals(
            [sliver], masks, FRAME, min_area_frac=0.01, reject_edge_count=0
        )
        self.assertEqual(keep, [])
        self.assertEqual(reasons, {"mask_too_small": 1})

    def test_missing_masks_only_disables_the_area_gates(self) -> None:
        keep, reasons = gate_proposals(
            [det((0, 0, 200, 100))], None, FRAME,
            min_area_frac=0.5, reject_edge_count=4,
        )
        self.assertEqual(keep, [])
        self.assertEqual(reasons, {"touches_frame_edges": 1})

    def test_indices_refer_to_the_input_list(self) -> None:
        # The recorder slices both detections and masks by these indices, so
        # they must be positions in the list that was passed in.
        dets = [det((0, 0, 200, 100)), det((20, 20, 40, 40)), det((0, 0, 200, 100))]
        keep, _ = gate_proposals(dets, None, FRAME, reject_edge_count=4)
        self.assertEqual(keep, [1])

    def test_empty_input(self) -> None:
        self.assertEqual(gate_proposals([], None, FRAME), ([], {}))


class LetterboxTest(unittest.TestCase):
    """The letterbox and its inverse are a matched pair. A sign or an offset
    wrong in either puts every box and mask in the wrong place while still
    looking like a plausible detection, so the round trip is asserted."""

    def test_preserves_aspect_and_centres(self) -> None:
        from pi_client.seg_postprocess import letterbox

        frame = np.zeros((864, 1536, 3), dtype=np.uint8)
        padded, lb = letterbox(frame, (640, 640))
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertEqual(lb.content_w, 640)          # width is the limiting side
        self.assertEqual(lb.content_h, 360)          # 864 * 640/1536
        self.assertEqual(lb.pad_x, 0)
        self.assertEqual(lb.pad_y, 140)              # (640 - 360) // 2
        # The padding really is YOLO's grey, not black: a black band is a
        # plausible dark object, 114 grey is what the model was trained on.
        self.assertEqual(int(padded[0, 0, 0]), 114)

    def test_box_round_trip_is_exact(self) -> None:
        from pi_client.seg_postprocess import letterbox

        frame = np.zeros((864, 1536, 3), dtype=np.uint8)
        _, lb = letterbox(frame, (640, 640))
        # Take frame boxes, push them into model space by hand, map back.
        frame_boxes = np.array([[0.0, 0.0, 1536.0, 864.0],
                                [100.0, 200.0, 400.0, 500.0]])
        model = frame_boxes.copy()
        model[:, [0, 2]] = model[:, [0, 2]] * lb.scale + lb.pad_x
        model[:, [1, 3]] = model[:, [1, 3]] * lb.scale + lb.pad_y
        back = lb.box_to_frame(model)
        np.testing.assert_allclose(back, frame_boxes, atol=1e-6)

    def test_squash_would_have_moved_the_box(self) -> None:
        # Guards against someone "simplifying" back to a plain resize: under
        # squashing, the old sy = h/in_h mapping puts a box at a different y.
        from pi_client.seg_postprocess import letterbox

        frame = np.zeros((864, 1536, 3), dtype=np.uint8)
        _, lb = letterbox(frame, (640, 640))
        # Deliberately OFF-centre: the two mappings agree exactly at the image
        # centre by construction, so testing there proves nothing.
        model_box = np.array([[320.0, 500.0, 400.0, 560.0]])
        letterboxed_y = lb.box_to_frame(model_box)[0, 1]
        squashed_y = 500.0 * (864 / 640)
        self.assertGreater(abs(letterboxed_y - squashed_y), 100.0)

    def test_crop_masks_removes_the_padding_band(self) -> None:
        from pi_client.seg_postprocess import letterbox

        frame = np.zeros((864, 1536, 3), dtype=np.uint8)
        _, lb = letterbox(frame, (640, 640))
        masks = np.zeros((2, 640, 640), dtype=np.float32)
        masks[:, lb.pad_y:lb.pad_y + lb.content_h, :] = 1.0
        cropped = lb.crop_masks(masks)
        self.assertEqual(cropped.shape, (2, 360, 640))
        self.assertTrue((cropped == 1.0).all())
        # Cropped masks must be proportional to the frame, which is the whole
        # reason readers can keep resizing them naively.
        self.assertAlmostEqual(cropped.shape[2] / cropped.shape[1], 1536 / 864, places=2)

    def test_portrait_frame_pads_horizontally(self) -> None:
        from pi_client.seg_postprocess import letterbox

        padded, lb = letterbox(np.zeros((800, 400, 3), dtype=np.uint8), (640, 640))
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertEqual(lb.content_h, 640)
        self.assertEqual(lb.content_w, 320)
        self.assertEqual(lb.pad_y, 0)
        self.assertEqual(lb.pad_x, 160)

    def test_empty_mask_stack_survives(self) -> None:
        from pi_client.seg_postprocess import letterbox

        _, lb = letterbox(np.zeros((864, 1536, 3), dtype=np.uint8), (640, 640))
        self.assertEqual(lb.crop_masks(np.zeros((0, 640, 640))).shape[0], 0)


if __name__ == "__main__":
    unittest.main()
