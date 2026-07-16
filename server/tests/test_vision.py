"""Unit tests for vision captioning: crop geometry + Ollama client against
an in-process fake server (same pattern as test_agent_format.py)."""

from __future__ import annotations

import base64
import json
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mac_server.monitoring.vision import (  # noqa: E402
    OllamaVisionClient,
    union_bbox_with_margin,
)


class UnionBboxTest(unittest.TestCase):
    def test_union_of_disjoint_boxes(self) -> None:
        subject = (100.0, 100.0, 200.0, 300.0)
        anchor = (400.0, 200.0, 600.0, 450.0)
        x0, y0, x1, y1 = union_bbox_with_margin(subject, anchor, 0.0, 1280, 720)
        self.assertEqual((x0, y0, x1, y1), (100, 100, 600, 450))

    def test_margin_expands_and_clamps(self) -> None:
        subject = (0.0, 0.0, 100.0, 100.0)
        anchor = (0.0, 0.0, 100.0, 100.0)
        x0, y0, x1, y1 = union_bbox_with_margin(subject, anchor, 0.5, 1280, 720)
        # Union is 100x100; 0.5 margin -> 50px each side, clamped at frame edge (0).
        self.assertEqual(x0, 0)
        self.assertEqual(y0, 0)
        self.assertEqual(x1, 150)
        self.assertEqual(y1, 150)

    def test_clamped_to_frame_bounds(self) -> None:
        subject = (1200.0, 650.0, 1280.0, 720.0)
        anchor = (1200.0, 650.0, 1280.0, 720.0)
        x0, y0, x1, y1 = union_bbox_with_margin(subject, anchor, 0.5, 1280, 720)
        self.assertLessEqual(x1, 1280)
        self.assertLessEqual(y1, 720)


class _FakeOllama(BaseHTTPRequestHandler):
    calls: list[dict] = []
    fail = False
    delay_s = 0.0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOllama.calls.append(body)
        if _FakeOllama.fail:
            self.send_error(500)
            return
        if _FakeOllama.delay_s:
            time.sleep(_FakeOllama.delay_s)
        response = json.dumps(
            {"message": {"role": "assistant", "content": "A person is sitting on the couch."}}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *args):
        pass


class OllamaVisionClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.httpd = HTTPServer(("127.0.0.1", 0), _FakeOllama)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()

    def setUp(self) -> None:
        _FakeOllama.calls = []
        _FakeOllama.fail = False
        _FakeOllama.delay_s = 0.0
        self.client = OllamaVisionClient(
            base_url=f"http://127.0.0.1:{self.port}", cooldown_s=0.0, keep_alive="0s"
        )

    def test_caption_roundtrip(self) -> None:
        caption = self.client.caption(b"fake-jpeg-bytes", prompt="Describe this.")
        self.assertEqual(caption, "A person is sitting on the couch.")
        self.assertTrue(self.client.healthy)

    def test_sends_base64_image_and_keep_alive(self) -> None:
        self.client.caption(b"fake-jpeg-bytes")
        sent = _FakeOllama.calls[0]
        self.assertEqual(sent["keep_alive"], "0s")
        self.assertEqual(sent["stream"], False)
        images = sent["messages"][0]["images"]
        self.assertEqual(len(images), 1)
        self.assertEqual(base64.b64decode(images[0]), b"fake-jpeg-bytes")

    def test_failure_marks_unhealthy_and_returns_none(self) -> None:
        _FakeOllama.fail = True
        result = self.client.caption(b"fake-jpeg-bytes")
        self.assertIsNone(result)
        self.assertFalse(self.client.healthy)

    def test_cooldown_skips_calls_while_unhealthy(self) -> None:
        client = OllamaVisionClient(base_url=f"http://127.0.0.1:{self.port}", cooldown_s=60.0)
        _FakeOllama.fail = True
        client.caption(b"x")
        self.assertEqual(len(_FakeOllama.calls), 1)
        _FakeOllama.fail = False
        result = client.caption(b"x")  # still in cooldown -> skipped, no new call
        self.assertIsNone(result)
        self.assertEqual(len(_FakeOllama.calls), 1)

    def test_concurrent_call_is_dropped_not_queued(self) -> None:
        """Regression guard: two overlapping captions must never both reach
        the model (would race two 6 GB loads on an 8 GB Mac). The second
        caller gets None immediately instead of waiting its turn."""
        _FakeOllama.delay_s = 0.5
        results: list[str | None] = []

        def slow_call():
            results.append(self.client.caption(b"first"))

        first = threading.Thread(target=slow_call)
        first.start()
        time.sleep(0.15)  # ensure the first call has acquired the lock

        started = time.perf_counter()
        second_result = self.client.caption(b"second")
        second_elapsed = time.perf_counter() - started

        first.join(timeout=5)

        self.assertIsNone(second_result)
        self.assertLess(second_elapsed, 0.3)  # dropped immediately, not queued
        self.assertEqual(results, ["A person is sitting on the couch."])
        self.assertEqual(len(_FakeOllama.calls), 1)  # only the first ever hit the server


if __name__ == "__main__":
    unittest.main()
