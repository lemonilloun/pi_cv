"""Server-side materialization of VLA behaviour-cloning episodes.

    python3 -m unittest server.tests.test_vla_episodes -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from mac_server.handlers import handle_message  # noqa: E402
from shared.messages import Message  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0" + b"x" * 512 + b"\xff\xd9"


def start(episode_id="ep1", task="go to the chair"):
    return Message(device_id="pi", type="vla_session_start", payload={
        "episode_id": episode_id, "task": task,
        "settings": {"hz": 10, "width": 640, "height": 360},
        "clock": {"start": {"offset_ns": 1234, "quality_ms": 0.1}},
    })


def frame(idx, episode_id="ep1", jpeg=JPEG, yaw=0.0):
    return Message(device_id="pi", type="vla_frame", payload={
        "episode_id": episode_id, "frame_idx": idx, "t_pi_mono_ns": 1000 * idx,
        "jpeg_bytes": len(jpeg),
        "imu": {"yaw_deg": yaw, "pitch_deg": -0.5, "roll_deg": 0.5, "age_s": 0.01},
    })


class EpisodeMaterializationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = Path(self._tmp.name) / "received"
        self.storage.mkdir(parents=True)
        self.episode = Path(self._tmp.name) / "vla_episodes" / "ep1"

    def tearDown(self):
        self._tmp.cleanup()

    def test_writes_frames_and_one_jsonl_row_each(self):
        handle_message(start(), b"", storage_dir=self.storage)
        for i in range(1, 6):
            handle_message(frame(i, yaw=i * 3.0), JPEG, storage_dir=self.storage)
        self.assertEqual(len(list((self.episode / "frames").glob("*.jpg"))), 5)
        rows = [json.loads(l) for l in (self.episode / "frames.jsonl").read_text().splitlines()]
        self.assertEqual([r["frame_idx"] for r in rows], [1, 2, 3, 4, 5])
        self.assertEqual(rows[2]["imu"]["yaw_deg"], 9.0)
        self.assertEqual(rows[2]["t_pi_mono_ns"], 3000)

    def test_frame_names_sort_lexically_in_capture_order(self):
        """Zero-padded, so a glob-and-sort in build_dataset gives the right
        order without parsing — 10 must not sort before 9."""
        handle_message(start(), b"", storage_dir=self.storage)
        for i in (9, 10, 11):
            handle_message(frame(i), JPEG, storage_dir=self.storage)
        names = sorted(p.name for p in (self.episode / "frames").glob("*.jpg"))
        self.assertEqual(names, ["000009.jpg", "000010.jpg", "000011.jpg"])

    def test_byte_count_mismatch_is_refused(self):
        """A truncated frame must fail loudly: a dataset with silently
        corrupt images trains a policy on noise."""
        handle_message(start(), b"", storage_dir=self.storage)
        with self.assertRaises(ValueError):
            handle_message(frame(1), JPEG[:-10], storage_dir=self.storage)

    def test_clock_offset_survives_into_the_metadata(self):
        """Without it the frames cannot be joined to the action log at all."""
        handle_message(start(), b"", storage_dir=self.storage)
        meta = json.loads((self.episode / "episode_meta.json").read_text())
        self.assertEqual(meta["clock"]["start"]["offset_ns"], 1234)
        self.assertEqual(meta["task"], "go to the chair")

    def test_session_end_merges_rather_than_replaces(self):
        handle_message(start(), b"", storage_dir=self.storage)
        handle_message(Message(device_id="pi", type="vla_session_end", payload={
            "episode_id": "ep1",
            "episode_meta": {"frames_captured": 42, "interval_ms_median": 100.2},
        }), b"", storage_dir=self.storage)
        meta = json.loads((self.episode / "episode_meta.json").read_text())
        self.assertEqual(meta["frames_captured"], 42)
        self.assertEqual(meta["task"], "go to the chair")     # from the start message
        self.assertIn("ended_wall", meta)

    def test_an_unclean_episode_id_is_rejected_not_sanitized(self):
        """Sanitizing would map several distinct ids onto one directory, and
        two demonstrations merged into one episode is a silent data bug."""
        for bad in ("../../etc", "ep 1", "ep/1", "ep-1", ""):
            with self.assertRaises(ValueError, msg=bad):
                handle_message(start(episode_id=bad), b"", storage_dir=self.storage)

    def test_the_recorders_own_id_format_is_accepted(self):
        handle_message(start(episode_id="ep_20260803_154210"), b"", storage_dir=self.storage)
        self.assertTrue((self.storage.parent / "vla_episodes" / "ep_20260803_154210").exists())

    def test_a_reconnect_resending_start_does_not_wipe_frames(self):
        """The uplink resends session_start on every reconnect, exactly as
        the scene recorder does, so this has to be idempotent."""
        handle_message(start(), b"", storage_dir=self.storage)
        handle_message(frame(1), JPEG, storage_dir=self.storage)
        handle_message(start(), b"", storage_dir=self.storage)
        self.assertEqual(len(list((self.episode / "frames").glob("*.jpg"))), 1)
