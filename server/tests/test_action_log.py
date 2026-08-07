"""Unit tests for mac_server.vla.action_log.

Until this existed, actions were not recorded at all — RobocarService kept
only `_last_drive`, the most recent pair, with no timestamp and no history.
A behaviour-cloning dataset cannot be built from that.

    python3 -m unittest server.tests.test_action_log -v
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

from mac_server.vla.action_log import ActionLog, parse_heartbeat  # noqa: E402


class ParseHeartbeatTests(unittest.TestCase):
    """The ESP reports its APPLIED post-ramp PWM at 0.5 Hz. It should lag the
    command by ~one ramp time and nothing more, which makes it an independent
    check on the whole Pi/laptop timestamp join."""

    def test_reads_the_applied_wheels(self):
        left, right = parse_heartbeat("HB up=42 L=138 R=-138 rssi=-52 heap=21000")
        self.assertEqual((left, right), (138, -138))

    def test_tolerates_a_missing_field(self):
        """The sketch's heartbeat format is not a contract — degrade, do not
        break the recording."""
        self.assertEqual(parse_heartbeat("HB up=42 rssi=-52"), (None, None))
        self.assertEqual(parse_heartbeat("HB L=10"), (10, None))

    def test_tolerates_garbage(self):
        self.assertEqual(parse_heartbeat("HB L=abc R=99"), (None, 99))
        self.assertEqual(parse_heartbeat(""), (None, None))


class RingBufferTests(unittest.TestCase):
    def test_records_intent_alongside_wheels(self):
        log = ActionLog()
        log.record_drive(140, -140, source="panel", max_pwm=150,
                         throttle=0.0, steer=-1.0, speed=140)
        (record,) = log.records()
        self.assertEqual((record.left, record.right), (140, -140))
        self.assertEqual((record.throttle, record.steer), (0.0, -1.0))
        self.assertEqual(record.source, "panel")
        self.assertEqual(record.max_pwm, 150)

    def test_timestamps_are_monotonic_and_increasing(self):
        log = ActionLog()
        for _ in range(5):
            log.record_drive(10, 10, source="panel", max_pwm=150)
        stamps = [r.t_mono_ns for r in log.records()]
        self.assertEqual(stamps, sorted(stamps))

    def test_overflow_drops_oldest_and_counts_it(self):
        """Silently losing the head of an episode would produce a dataset
        with a plausible-looking but truncated first second."""
        log = ActionLog(capacity=10)
        for i in range(25):
            log.record_drive(i, i, source="panel", max_pwm=150)
        self.assertEqual(len(log.records()), 10)
        self.assertEqual(log.status()["dropped"], 15)
        self.assertEqual(log.records()[0].left, 15)   # oldest survivor

    def test_status_separates_drives_from_heartbeats(self):
        log = ActionLog()
        log.record_drive(1, 1, source="panel", max_pwm=150)
        log.record_heartbeat("HB up=1 L=0 R=0")
        status = log.status()
        self.assertEqual((status["drives"], status["heartbeats"]), (1, 1))
        self.assertFalse(status["recording"])
        json.dumps(status)


class EpisodeFileTests(unittest.TestCase):
    def test_writes_jsonl_and_reports_the_path(self):
        log = ActionLog()
        with tempfile.TemporaryDirectory() as td:
            path = log.start_episode("ep1", Path(td) / "ep1")
            log.record_drive(140, 140, source="panel", max_pwm=150,
                             throttle=1.0, steer=0.0, speed=140)
            log.record_heartbeat("HB up=3 L=140 R=140")
            self.assertTrue(log.status()["recording"])
            log.stop_episode()

            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r["kind"] for r in rows], ["drive", "heartbeat"])
            self.assertEqual(rows[0]["throttle"], 1.0)
            self.assertEqual(rows[1]["applied_left"], 140)

    def test_starting_a_second_episode_closes_the_first(self):
        log = ActionLog()
        with tempfile.TemporaryDirectory() as td:
            first = log.start_episode("ep1", Path(td) / "ep1")
            log.record_drive(1, 1, source="panel", max_pwm=150)
            second = log.start_episode("ep2", Path(td) / "ep2")
            log.record_drive(2, 2, source="panel", max_pwm=150)
            log.stop_episode()
            self.assertEqual(len(first.read_text().splitlines()), 1)
            self.assertEqual(len(second.read_text().splitlines()), 1)
            # A new episode must not inherit the previous one's buffer.
            self.assertEqual(len(log.records()), 1)

    def test_records_survive_without_an_episode(self):
        """The buffer is always on so the panel can show recent activity;
        only disk mirroring is episode-scoped."""
        log = ActionLog()
        log.record_drive(5, 5, source="spin", max_pwm=150)
        self.assertEqual(len(log.records()), 1)
        self.assertIsNone(log.stop_episode())

    def test_flushes_per_record_so_a_crash_keeps_what_happened(self):
        """An episode that ends in a yanked battery is exactly when the
        interesting failure data lives."""
        log = ActionLog()
        with tempfile.TemporaryDirectory() as td:
            path = log.start_episode("ep1", Path(td) / "ep1")
            log.record_drive(1, 1, source="panel", max_pwm=150)
            self.assertEqual(len(path.read_text().splitlines()), 1)  # before stop
            log.stop_episode()
