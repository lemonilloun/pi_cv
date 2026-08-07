"""Tests for the object description layer.

The parts that can be tested without a GPU, a camera or a vLLM are exactly the
parts most likely to be wrong: which view gets shown to the model, and what
happens when the model does not reply the way it was asked to.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402

from mac_server.scene3d.describe_step import (  # noqa: E402
    crop_with_outline,
    parse_json_reply,
    rank_views,
    same_object_verdict,
)

FRAME = (864, 1536)  # h, w


class RankViewsTest(unittest.TestCase):
    def test_bigger_mask_wins_all_else_equal(self) -> None:
        small = {"mask_area": 1000, "bbox_xyxy": [400, 300, 500, 400]}
        big = {"mask_area": 9000, "bbox_xyxy": [400, 300, 700, 600]}
        self.assertEqual(rank_views([small, big], FRAME)[0], big)

    def test_a_clipped_view_loses_to_a_smaller_complete_one(self) -> None:
        # The failure this exists to prevent: an object entering the frame has
        # lots of visible pixels and shows half of itself. Describing that half
        # as the whole object is the error.
        clipped = {"mask_area": 9000, "bbox_xyxy": [0, 300, 400, 700]}   # touches x=0
        complete = {"mask_area": 6000, "bbox_xyxy": [500, 300, 800, 600]}
        self.assertEqual(rank_views([clipped, complete], FRAME)[0], complete)

    def test_clipping_only_halves_it_rather_than_disqualifying(self) -> None:
        # A big clipped view still beats a tiny complete one — half of a
        # wardrobe is more informative than a distant speck.
        clipped = {"mask_area": 90000, "bbox_xyxy": [0, 300, 400, 700]}
        tiny = {"mask_area": 500, "bbox_xyxy": [500, 300, 520, 320]}
        self.assertEqual(rank_views([clipped, tiny], FRAME)[0], clipped)

    def test_empty_input(self) -> None:
        self.assertEqual(rank_views([], FRAME), [])


class ParseReplyTest(unittest.TestCase):
    def test_plain_json(self) -> None:
        out = parse_json_reply('{"name": "white wardrobe", "label_ok": true}')
        self.assertEqual(out["name"], "white wardrobe")

    def test_json_wrapped_in_a_code_fence_and_chatter(self) -> None:
        # What models actually send, rather than what they were asked for.
        text = 'Sure! Here is the JSON:\n```json\n{"name": "oak table"}\n```\nHope that helps.'
        self.assertEqual(parse_json_reply(text)["name"], "oak table")

    def test_nested_braces_do_not_truncate_the_object(self) -> None:
        out = parse_json_reply('prefix {"a": {"b": 1}, "c": 2} suffix')
        self.assertEqual(out, {"a": {"b": 1}, "c": 2})

    def test_prose_reply_yields_none_not_garbage(self) -> None:
        self.assertIsNone(parse_json_reply("It looks like a wardrobe to me."))

    def test_none_and_empty(self) -> None:
        self.assertIsNone(parse_json_reply(None))
        self.assertIsNone(parse_json_reply(""))

    def test_unbalanced_braces(self) -> None:
        self.assertIsNone(parse_json_reply('{"name": "half'))


class SameObjectVerdictTest(unittest.TestCase):
    def test_true_and_false(self) -> None:
        self.assertIs(same_object_verdict({"same": True}), True)
        self.assertIs(same_object_verdict({"same": False}), False)

    def test_string_answers(self) -> None:
        self.assertIs(same_object_verdict({"same": "true"}), True)
        self.assertIs(same_object_verdict({"same": "no"}), False)

    def test_no_answer_is_none_not_false(self) -> None:
        # The distinction that matters: an unreachable model must not read as
        # "different objects", which would double every object in the room.
        self.assertIsNone(same_object_verdict(None))
        self.assertIsNone(same_object_verdict({}))
        self.assertIsNone(same_object_verdict({"why": "unclear"}))


class CropTest(unittest.TestCase):
    def test_outline_is_drawn_and_margin_applied(self) -> None:
        bgr = np.zeros((200, 300, 3), dtype=np.uint8)
        mask = np.zeros((200, 300), dtype=np.uint8)
        mask[50:150, 100:200] = 1
        crop = crop_with_outline(bgr, mask, [100, 50, 200, 150], margin_frac=0.1)
        self.assertIsNotNone(crop)
        # 100x100 box + 10% margin each side, clipped to the frame.
        self.assertEqual(crop.shape[0], 120)
        self.assertEqual(crop.shape[1], 120)
        # Something green was drawn — the contour the VLM needs to know which
        # object is meant.
        self.assertGreater(int((crop[:, :, 1] > 200).sum()), 0)

    def test_degenerate_box_returns_none(self) -> None:
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(crop_with_outline(bgr, None, [50, 50, 50, 50], margin_frac=0.0))

    def test_margin_is_clipped_at_the_frame_edge(self) -> None:
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = crop_with_outline(bgr, None, [0, 0, 50, 50], margin_frac=0.5)
        self.assertEqual(crop.shape[:2], (75, 75))


if __name__ == "__main__":
    unittest.main()
