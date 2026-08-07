"""Unit tests for mac_server.vla.build_dataset — joining frames to actions.

The join is where the whole dataset's correctness is decided, and it is not
verifiable by eye afterwards: a systematically shifted label produces a
plausible-looking dataset that trains a subtly wrong policy.

    python3 -m unittest server.tests.test_build_dataset -v
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from mac_server.vla.build_dataset import (  # noqa: E402
    ActionEvent,
    actions_to_pi_timeline,
    check_episode,
    mirror_sample,
    observation_state,
    resample_actions,
    yaw_rate_series,
)

S = 1_000_000_000   # one second in ns


class TimelineTests(unittest.TestCase):
    def test_subtracts_the_offset_to_reach_the_pi_clock(self):
        """`pi + offset = mac`, so the inverse is applied here. A sign error
        would shift every label by the offset — hours, on real hardware."""
        events = actions_to_pi_timeline(
            [{"kind": "drive", "t_mono_ns": 100 * S, "left": 10, "right": 20}], offset_ns=40 * S
        )
        self.assertEqual(events[0].t_pi_ns, 60 * S)

    def test_heartbeats_are_not_labels(self):
        records = [
            {"kind": "drive", "t_mono_ns": S, "left": 1, "right": 1},
            {"kind": "heartbeat", "t_mono_ns": 2 * S, "text": "HB up=1 L=1 R=1"},
        ]
        self.assertEqual(len(actions_to_pi_timeline(records, 0)), 1)

    def test_sorts_out_of_order_records(self):
        records = [
            {"kind": "drive", "t_mono_ns": 3 * S, "left": 3, "right": 3},
            {"kind": "drive", "t_mono_ns": 1 * S, "left": 1, "right": 1},
        ]
        self.assertEqual([e.left for e in actions_to_pi_timeline(records, 0)], [1, 3])


class ResampleTests(unittest.TestCase):
    def test_zero_order_hold_between_commands(self):
        """Not interpolation: a drive command is a step that persists, and
        _repeat_loop literally retransmits it every 150 ms."""
        events = [ActionEvent(0, 100, 100), ActionEvent(1 * S, -50, -50)]
        frames = [0, S // 2, S, 3 * S // 2]
        self.assertEqual(
            resample_actions(frames, events),
            [(100, 100), (100, 100), (-50, -50), (-50, -50)],
        )

    def test_ttl_expiry_becomes_a_real_stop(self):
        """After COMMAND_TTL_S the server itself sends STOP, so the recorded
        reality is (0, 0) — those frames are labels, not gaps."""
        events = [ActionEvent(0, 200, 200)]
        frames = [0, int(0.5 * S), int(0.7 * S), 5 * S]
        self.assertEqual(
            resample_actions(frames, events),
            [(200, 200), (200, 200), (0, 0), (0, 0)],
        )

    def test_frames_before_the_first_command_have_no_label(self):
        """Fabricating a stop there would put a wrong label on real pixels;
        the operator may well have been holding a key already."""
        events = [ActionEvent(2 * S, 100, 100)]
        self.assertEqual(
            resample_actions([0, S, 2 * S], events), [None, None, (100, 100)]
        )

    def test_no_actions_at_all_labels_nothing(self):
        self.assertEqual(resample_actions([0, S], []), [None, None])

    def test_several_commands_inside_one_frame_interval_keep_the_last(self):
        """The panel posts at 10 Hz and frames arrive at 10 Hz; jitter can put
        two commands in one interval. The one in force at the shutter is the
        last one issued."""
        events = [ActionEvent(0, 1, 1), ActionEvent(10, 2, 2), ActionEvent(20, 3, 3)]
        self.assertEqual(resample_actions([30], events), [(3, 3)])


class ObservationStateTests(unittest.TestCase):
    def test_shape_and_relative_yaw(self):
        state = observation_state({"yaw_deg": 90.0, "pitch_deg": 0.0, "roll_deg": 0.0},
                                  yaw0_deg=0.0, prev_action=(0, 0), yaw_rate_dps=0.0)
        self.assertEqual(len(state), 7)
        self.assertAlmostEqual(state[0], 1.0, places=6)   # sin(90 deg)
        self.assertAlmostEqual(state[1], 0.0, places=6)   # cos(90 deg)

    def test_absolute_yaw_never_leaks_in(self):
        """The RVC datum is whatever the sensor powered up facing, so the same
        physical heading has a different absolute yaw in every episode. Two
        episodes at the same relative heading must produce the same state."""
        a = observation_state({"yaw_deg": 30.0, "pitch_deg": 0, "roll_deg": 0},
                              yaw0_deg=10.0, prev_action=(0, 0), yaw_rate_dps=0.0)
        b = observation_state({"yaw_deg": 220.0, "pitch_deg": 0, "roll_deg": 0},
                              yaw0_deg=200.0, prev_action=(0, 0), yaw_rate_dps=0.0)
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=9)

    def test_no_wrap_discontinuity(self):
        """sin/cos rather than the angle, so +179 and -179 deg are adjacent."""
        a = observation_state({"yaw_deg": 179.0, "pitch_deg": 0, "roll_deg": 0},
                              0.0, (0, 0), 0.0)
        b = observation_state({"yaw_deg": -179.0, "pitch_deg": 0, "roll_deg": 0},
                              0.0, (0, 0), 0.0)
        self.assertLess(math.dist(a[:2], b[:2]), 0.1)

    def test_previous_action_is_carried(self):
        """The ESP ramps at 8 PWM per 15 ms, so what the motors are doing
        depends on the previous command — hidden state the policy needs."""
        state = observation_state(None, None, prev_action=(255, -255), yaw_rate_dps=0.0)
        self.assertAlmostEqual(state[5], 1.0)
        self.assertAlmostEqual(state[6], -1.0)

    def test_yaw_rate_is_clamped(self):
        state = observation_state(None, None, (0, 0), yaw_rate_dps=9999.0)
        self.assertEqual(state[4], 1.0)

    def test_survives_a_missing_imu(self):
        self.assertEqual(len(observation_state(None, None, (0, 0), 0.0)), 7)


class YawRateTests(unittest.TestCase):
    def test_differences_the_fused_yaw(self):
        frames = [
            {"t_pi_mono_ns": 0, "imu": {"yaw_deg": 0.0}},
            {"t_pi_mono_ns": S // 10, "imu": {"yaw_deg": 3.0}},
        ]
        self.assertAlmostEqual(yaw_rate_series(frames)[1], 30.0, places=3)

    def test_holds_the_last_rate_across_a_missing_sample(self):
        frames = [
            {"t_pi_mono_ns": 0, "imu": {"yaw_deg": 0.0}},
            {"t_pi_mono_ns": S // 10, "imu": {"yaw_deg": 3.0}},
            {"t_pi_mono_ns": 2 * S // 10, "imu": None},
        ]
        rates = yaw_rate_series(frames)
        self.assertAlmostEqual(rates[2], rates[1])


class MirrorTests(unittest.TestCase):
    """An exact symmetry of a differential drive, so a free and correct 2x."""

    def test_swaps_wheels_and_negates_heading(self):
        state = [0.5, 0.86, 0.1, 0.2, 0.4, 0.3, -0.7]
        mirrored, action = mirror_sample(state, (120, -40))
        self.assertEqual(action, (-40, 120))
        self.assertEqual(mirrored[0], -0.5)     # sin(yaw) negated
        self.assertEqual(mirrored[1], 0.86)     # cos(yaw) unchanged
        self.assertEqual(mirrored[2], 0.1)      # pitch unchanged
        self.assertEqual(mirrored[3], -0.2)     # roll flips with the world
        self.assertEqual(mirrored[4], -0.4)     # turn rate reverses
        self.assertEqual(mirrored[5:], [-0.7, 0.3])

    def test_mirroring_twice_is_the_identity(self):
        state = [0.5, 0.86, 0.1, 0.2, 0.4, 0.3, -0.7]
        once = mirror_sample(state, (120, -40))
        twice = mirror_sample(*once)
        self.assertEqual(twice[0], state)
        self.assertEqual(twice[1], (120, -40))

    def test_driving_straight_is_unchanged_by_mirroring(self):
        _state, action = mirror_sample([0.0, 1.0, 0, 0, 0, 0.5, 0.5], (140, 140))
        self.assertEqual(action, (140, 140))


class CheckEpisodeTests(unittest.TestCase):
    @staticmethod
    def _report(**kw):
        base = {"samples": [{"raw_action": (100, 100)}], "frames_total": 100,
                "frames_unlabeled_dropped": 0, "clock_quality_ms": 1.0,
                "action_events": 50, "stopped_frac": 0.2}
        base.update(kw)
        return base

    def test_clean_episode_passes(self):
        self.assertEqual(check_episode(self._report()), [])

    def test_flags_a_loose_clock(self):
        problems = check_episode(self._report(clock_quality_ms=45.0))
        self.assertTrue(any("20 ms budget" in p for p in problems))

    def test_flags_a_late_armed_action_log(self):
        problems = check_episode(self._report(frames_unlabeled_dropped=40))
        self.assertTrue(any("no action yet" in p for p in problems))

    def test_flags_an_episode_where_nothing_happened(self):
        problems = check_episode(self._report(stopped_frac=0.97))
        self.assertTrue(any("almost nothing happened" in p for p in problems))

    def test_flags_a_robot_that_was_never_driven(self):
        problems = check_episode(self._report(action_events=2))
        self.assertTrue(any("actually driven" in p for p in problems))

    def test_empty_episode_short_circuits(self):
        self.assertEqual(check_episode(self._report(samples=[])), ["no labelled frames at all"])


class ExportTests(unittest.TestCase):
    """The recording is the expensive, unrepeatable part. The export must not
    be the thing that makes it unreadable."""

    @staticmethod
    def _report(episode_id, n=4, task="go to the bed"):
        return {
            "episode_id": episode_id,
            "task": task,
            "samples": [{
                "frame": Path(f"/tmp/{episode_id}/{i:06d}.jpg"),
                "state": [0.5, 0.86, 0.1, 0.2, 0.4, 0.3, -0.7],
                "action": [140 / 255.0, 40 / 255.0],
                "raw_action": (140, 40),
                "t_pi_mono_ns": i * 100_000_000,
            } for i in range(n)],
            "frames_total": n, "frames_unlabeled_dropped": 0,
            "stopped_frac": 0.2, "action_events": 20, "clock_quality_ms": 1.2,
        }

    def test_splits_are_by_whole_episode(self):
        """Adjacent frames at 10 Hz are near-duplicates, so a frame-level
        split puts almost every validation frame within 100 ms of a training
        one — a beautiful, meaningless curve."""
        from mac_server.vla.build_dataset import episode_splits

        reports = [self._report(f"ep_{i:03d}") for i in range(10)]
        splits = episode_splits(reports, holdout_frac=0.2)
        self.assertEqual(len(splits["val"]), 2)
        self.assertEqual(set(splits["train"]) & set(splits["val"]), set())
        self.assertEqual(len(splits["train"]) + len(splits["val"]), 10)

    def test_splits_are_deterministic(self):
        from mac_server.vla.build_dataset import episode_splits

        reports = [self._report(f"ep_{i:03d}") for i in range(10)]
        self.assertEqual(episode_splits(reports), episode_splits(reports))

    def test_manifest_doubles_the_rows_when_mirroring(self):
        from mac_server.vla.build_dataset import write_manifest
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            plain = write_manifest([self._report("ep_a", n=5)], out, mirror=False)
            mirrored = write_manifest([self._report("ep_a", n=5)], out, mirror=True)
            self.assertEqual(plain["rows"], 5)
            self.assertEqual(mirrored["rows"], 10)

    def test_mirrored_rows_carry_the_swapped_action(self):
        from mac_server.vla.build_dataset import write_manifest
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            write_manifest([self._report("ep_a", n=1)], out, mirror=True)
            rows = [json.loads(l) for l in (out / "samples.jsonl").read_text().splitlines()]
            original = next(r for r in rows if not r["mirrored"])
            mirrored = next(r for r in rows if r["mirrored"])
            self.assertEqual(mirrored["action"], list(reversed(original["action"])))
            self.assertEqual(mirrored["state"][0], -original["state"][0])

    def test_manifest_documents_its_own_layout(self):
        """A JSONL of bare float lists is unreadable in six weeks unless it
        says what the columns mean."""
        from mac_server.vla.build_dataset import write_manifest
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            manifest = write_manifest([self._report("ep_a")], Path(td))
            self.assertEqual(len(manifest["state_layout"]), 7)
            self.assertEqual(len(manifest["action_layout"]), 2)
            self.assertIn("manifest_version", manifest)

    def test_lerobot_absence_is_reported_not_raised(self):
        """Losing an optional export is not a reason to lose a session."""
        from mac_server.vla.build_dataset import write_lerobot
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            result = write_lerobot([self._report("ep_a")], Path(td))
            self.assertIn("written", result)
            if not result["written"]:
                self.assertIn("reason", result)

    def test_export_excludes_bad_episodes_and_says_which(self):
        from mac_server.vla.build_dataset import export
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            episodes = Path(td) / "episodes"
            (episodes / "ep_broken").mkdir(parents=True)
            result = export(episodes, Path(td) / "out")
            self.assertEqual(result["episodes_used"], 0)
            self.assertEqual(result["rejected"][0]["episode_id"], "ep_broken")
