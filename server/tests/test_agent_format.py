"""Unit tests for the apfel agent layer: event formatting, budgeting, and the
HTTP client against an in-process fake OpenAI-compatible server."""

from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mac_server.monitoring.agent import (  # noqa: E402
    ApfelClient,
    budget_lines,
    chunk_lines,
    format_event_line,
)


def event(etype: str, subject: str, t_start=1000.0, t_end=None, **details) -> dict:
    return {
        "type": etype,
        "subject_label": subject,
        "object_label": details.pop("object_label", None),
        "t_start": t_start,
        "t_end": t_end,
        "details": details,
    }


class FormatTest(unittest.TestCase):
    def test_entered(self) -> None:
        line = format_event_line(event("entered", "Person#1"))
        self.assertIn("Person#1 entered", line)

    def test_exited_with_duration(self) -> None:
        line = format_event_line(event("exited", "Cat#1", visible_s=125.0))
        self.assertIn("Cat#1 exited", line)
        self.assertIn("2m", line)

    def test_on_furniture_closed(self) -> None:
        line = format_event_line(
            event(
                "on_furniture", "Person#1", t_start=1000.0, t_end=1900.0,
                object_label="couch_1", posture="sitting", duration_s=900.0,
            )
        )
        self.assertIn("Person#1 on couch_1, sitting (15m)", line)

    def test_on_furniture_ongoing(self) -> None:
        line = format_event_line(
            event("on_furniture", "Person#1", object_label="couch_1", posture="lying")
        )
        self.assertIn("(ongoing)", line)
        self.assertIn("lying", line)

    def test_unclear_posture_omitted(self) -> None:
        line = format_event_line(
            event("on_furniture", "Cat#1", object_label="bed_1", posture="unclear")
        )
        self.assertNotIn("unclear", line)


class BudgetTest(unittest.TestCase):
    def test_keeps_newest(self) -> None:
        lines = [f"line {i:03d}" for i in range(100)]
        kept = budget_lines(lines, max_chars=100)
        self.assertLess(len(kept), 100)
        self.assertEqual(kept[-1], "line 099")  # newest survives
        self.assertEqual(kept, sorted(kept))  # order preserved

    def test_all_fit(self) -> None:
        lines = ["a", "b", "c"]
        self.assertEqual(budget_lines(lines, 100), lines)

    def test_chunking_respects_budget(self) -> None:
        lines = ["x" * 50 for _ in range(20)]
        chunks = chunk_lines(lines, max_chars=120)
        self.assertTrue(all(sum(len(l) + 1 for l in c) <= 120 for c in chunks))
        self.assertEqual(sum(len(c) for c in chunks), 20)


class _FakeApfel(BaseHTTPRequestHandler):
    calls: list[dict] = []
    fail = False

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeApfel.calls.append(body)
        if _FakeApfel.fail:
            self.send_error(503)
            return
        response = json.dumps(
            {"choices": [{"message": {"content": "Person#1 sat on the couch for 15 minutes."}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *args):
        pass


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.httpd = HTTPServer(("127.0.0.1", 0), _FakeApfel)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()

    def setUp(self) -> None:
        _FakeApfel.calls = []
        _FakeApfel.fail = False
        self.client = ApfelClient(base_url=f"http://127.0.0.1:{self.port}", cooldown_s=0.0)

    def test_digest_roundtrip(self) -> None:
        events = [
            event("entered", "Person#1", t_start=1000.0),
            event("on_furniture", "Person#1", t_start=1030.0, t_end=1930.0,
                  object_label="couch_1", posture="sitting", duration_s=900.0),
        ]
        text = self.client.digest("Living room", events)
        self.assertIn("couch", text)
        self.assertTrue(self.client.healthy)
        sent = _FakeApfel.calls[0]
        self.assertIn("Person#1 on couch_1", sent["messages"][1]["content"])

    def test_rolling_summary_first_update(self) -> None:
        events = [event("entered", "Person#1", t_start=1000.0)]
        updated = self.client.update_rolling_summary("Living room", None, events)
        self.assertIsNotNone(updated)
        sent = _FakeApfel.calls[0]
        self.assertIn("none yet", sent["messages"][1]["content"])
        self.assertIn("Person#1 entered", sent["messages"][1]["content"])

    def test_rolling_summary_incorporates_previous(self) -> None:
        events = [event("on_furniture", "Person#1", object_label="couch_1", posture="sitting")]
        self.client.update_rolling_summary("Living room", "Person#1 entered the room.", events)
        sent = _FakeApfel.calls[0]
        self.assertIn("Person#1 entered the room.", sent["messages"][1]["content"])

    def test_rolling_summary_falls_back_on_failure(self) -> None:
        _FakeApfel.fail = True
        result = self.client.update_rolling_summary(
            "Living room", "Existing summary.", [event("entered", "Cat#1")]
        )
        self.assertEqual(result, "Existing summary.")

    def test_rolling_summary_no_new_events_keeps_previous(self) -> None:
        result = self.client.update_rolling_summary("Living room", "Existing summary.", [])
        self.assertEqual(result, "Existing summary.")
        self.assertEqual(len(_FakeApfel.calls), 0)

    def test_failure_marks_unhealthy(self) -> None:
        _FakeApfel.fail = True
        result = self.client.chat("s", "u")
        self.assertIsNone(result)
        self.assertFalse(self.client.healthy)

    def test_summary_hierarchical_under_budget(self) -> None:
        self.client.max_input_chars = 300
        digests = [
            {"t_start": 0.0, "t_end": 300.0, "text": "d" * 100} for _ in range(10)
        ]
        summary = self.client.summarize("Living room", "last 1h", digests)
        self.assertIsNotNone(summary)
        # More than one call means the hierarchical path engaged.
        self.assertGreater(len(_FakeApfel.calls), 1)
        for call in _FakeApfel.calls:
            self.assertLess(len(call["messages"][1]["content"]), 500)


if __name__ == "__main__":
    unittest.main()
