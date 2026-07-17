"""Unit tests for the apfel agent layer: edge formatting, budgeting, key
moments, Q&A prompt assembly — against an in-process fake server."""

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
    format_edge_line,
    format_now_state,
    parse_key_moment_indices,
)


def edge(relation: str, subject: str, t_start=1000.0, t_end=None, **details) -> dict:
    return {
        "relation": relation,
        "subject_label": subject,
        "object_label": details.pop("object_label", None),
        "t_start": t_start,
        "t_end": t_end,
        "details": details,
    }


class FormatTest(unittest.TestCase):
    def test_entered_plain_label(self) -> None:
        line = format_edge_line(edge("entered", "person"))
        self.assertIn("person entered the frame", line)
        self.assertNotIn("#", line)

    def test_exited_with_duration(self) -> None:
        line = format_edge_line(edge("exited", "cat", visible_s=125.0))
        self.assertIn("cat left the frame", line)
        self.assertIn("2m", line)

    def test_on_closed_with_posture_and_duration(self) -> None:
        line = format_edge_line(
            edge("on", "person", t_start=1000.0, t_end=1900.0,
                 object_label="couch_1", posture="sitting", duration_s=900.0)
        )
        self.assertIn("person on couch_1", line)
        self.assertIn("sitting", line)
        self.assertIn("15m", line)

    def test_near_carries_geometry(self) -> None:
        line = format_edge_line(
            edge("near", "laptop", object_label="couch_1",
                 distance_m=0.4, arrangement="left-of")
        )
        self.assertIn("laptop near couch_1", line)
        self.assertIn("0.4m apart", line)
        self.assertIn("left-of", line)
        self.assertIn("(ongoing)", line)

    def test_moved_zones_and_anchor(self) -> None:
        line = format_edge_line(
            edge("moved", "laptop", t_end=1000.0, from_zone="left",
                 to_zone="right", nearest_anchor="couch_1")
        )
        self.assertIn("laptop moved from left to right", line)
        self.assertIn("near couch_1", line)

    def test_unclear_posture_omitted(self) -> None:
        line = format_edge_line(edge("on", "cat", object_label="bed_1", posture="unclear"))
        self.assertNotIn("unclear", line)

    def test_now_state_lines(self) -> None:
        lines = format_now_state(
            [
                {"label": "person", "relations": ["on couch_1"], "depth_m": 2.9, "state": "present"},
                {"label": "laptop", "relations": [], "depth_m": None, "state": "occluded"},
            ]
        )
        self.assertEqual(lines[0], "person: on couch_1 (2.9m away)")
        self.assertIn("laptop: in view", lines[1])
        self.assertIn("[occluded]", lines[1])


class KeyMomentParseTest(unittest.TestCase):
    def test_parses_comma_list(self) -> None:
        self.assertEqual(parse_key_moment_indices("2,5", count=6, max_moments=3), [1, 4])

    def test_none_answer(self) -> None:
        self.assertEqual(parse_key_moment_indices("none", 5, 3), [])
        self.assertEqual(parse_key_moment_indices(None, 5, 3), [])

    def test_out_of_range_and_dupes_dropped(self) -> None:
        self.assertEqual(parse_key_moment_indices("1, 99, 1, 3", count=4, max_moments=3), [0, 2])

    def test_caps_at_max(self) -> None:
        self.assertEqual(parse_key_moment_indices("1,2,3,4", count=6, max_moments=2), [0, 1])


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
    answer = "A person sat on the couch for 15 minutes."

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeApfel.calls.append(body)
        if _FakeApfel.fail:
            self.send_error(503)
            return
        response = json.dumps(
            {"choices": [{"message": {"content": _FakeApfel.answer}}]}
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
        _FakeApfel.answer = "A person sat on the couch for 15 minutes."
        self.client = ApfelClient(base_url=f"http://127.0.0.1:{self.port}", cooldown_s=0.0)

    def test_digest_roundtrip(self) -> None:
        edges = [
            edge("entered", "person", t_start=1000.0),
            edge("on", "person", t_start=1030.0, t_end=1930.0,
                 object_label="couch_1", posture="sitting", duration_s=900.0),
        ]
        text = self.client.digest("Living room", edges)
        self.assertIn("couch", text)
        self.assertTrue(self.client.healthy)
        sent = _FakeApfel.calls[0]
        self.assertIn("person on couch_1", sent["messages"][1]["content"])

    def test_pick_key_moments_roundtrip(self) -> None:
        _FakeApfel.answer = "1,3"
        edges = [
            edge("entered", "person", t_start=1000.0),
            edge("near", "laptop", object_label="couch_1", t_start=1010.0),
            edge("exited", "person", t_start=1200.0, visible_s=200.0),
        ]
        picked = self.client.pick_key_moments(edges, max_moments=3)
        self.assertEqual([e["relation"] for e in picked], ["entered", "exited"])
        # The prompt numbers lines from 1.
        self.assertIn("1. ", _FakeApfel.calls[0]["messages"][1]["content"])

    def test_pick_key_moments_none(self) -> None:
        _FakeApfel.answer = "none"
        picked = self.client.pick_key_moments([edge("entered", "person")])
        self.assertEqual(picked, [])

    def test_pick_key_moments_survives_failure(self) -> None:
        _FakeApfel.fail = True
        picked = self.client.pick_key_moments([edge("entered", "person")])
        self.assertEqual(picked, [])

    def test_rolling_summary_first_update(self) -> None:
        edges = [edge("entered", "person", t_start=1000.0)]
        updated = self.client.update_rolling_summary("Living room", None, edges)
        self.assertIsNotNone(updated)
        sent = _FakeApfel.calls[0]
        self.assertIn("none yet", sent["messages"][1]["content"])
        self.assertIn("person entered", sent["messages"][1]["content"])

    def test_rolling_summary_falls_back_on_failure(self) -> None:
        _FakeApfel.fail = True
        result = self.client.update_rolling_summary(
            "Living room", "Existing summary.", [edge("entered", "cat")]
        )
        self.assertEqual(result, "Existing summary.")

    def test_rolling_summary_no_new_edges_keeps_previous(self) -> None:
        result = self.client.update_rolling_summary("Living room", "Existing summary.", [])
        self.assertEqual(result, "Existing summary.")
        self.assertEqual(len(_FakeApfel.calls), 0)

    def test_failure_marks_unhealthy(self) -> None:
        _FakeApfel.fail = True
        result = self.client.chat("s", "u")
        self.assertIsNone(result)
        self.assertFalse(self.client.healthy)

    def test_ask_assembles_sections(self) -> None:
        answer = self.client.ask(
            scene_name="Living room",
            question="Where is the laptop?",
            now_lines=["laptop: near couch_1 (2.5m away)"],
            edges=[edge("moved", "laptop", t_end=1000.0, from_zone="left", to_zone="center")],
            digests=[{"t_start": 0.0, "t_end": 300.0, "text": "A laptop was moved."}],
            rolling_summary="A quiet morning.",
        )
        self.assertIsNotNone(answer)
        prompt = _FakeApfel.calls[0]["messages"][1]["content"]
        self.assertIn("Question: Where is the laptop?", prompt)
        self.assertIn("Current state:", prompt)
        self.assertIn("laptop: near couch_1", prompt)
        self.assertIn("A laptop was moved.", prompt)
        self.assertIn("A quiet morning.", prompt)

    def test_ask_respects_budget(self) -> None:
        self.client.max_input_chars = 600
        edges = [
            edge("moved", "laptop", t_start=1000.0 + i, t_end=1000.0 + i,
                 from_zone="left", to_zone="right")
            for i in range(100)
        ]
        self.client.ask("Living room", "What happened?", [], edges, [])
        prompt = _FakeApfel.calls[0]["messages"][1]["content"]
        self.assertLess(len(prompt), 1200)


if __name__ == "__main__":
    unittest.main()
