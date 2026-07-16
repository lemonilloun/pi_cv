"""Optional vision captioning for monitoring events, via a local Ollama
multimodal model (e.g. Gemma 4 E4B).

This is deliberately separate from `agent.py` (apfel): apfel is text-only, so
image-grounded descriptions need a different local model. Captioning only
happens on event OPEN transitions for configured event types (sparse — a
handful of calls per hour, not per frame), so a 6 GB resident model spinning
up on demand (via Ollama's `keep_alive`) is a reasonable trade for an 8 GB
Mac: `keep_alive: "0s"` unloads it immediately after each call, trading
latency (the caption arrives a few seconds after the event, asynchronously)
for not holding multiple GB resident between calls.

Degrades gracefully exactly like ApfelClient: monitoring never depends on
this being available.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any


logger = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "This is a cropped frame from a home security camera. In ONE short "
    "factual sentence (max 20 words), describe what is happening — who or "
    "what is present and what they are doing, in plain language suitable "
    "for a monitoring log. Answer with only that sentence: no preamble, no "
    "questions, no mention of image quality or cropping."
)


def union_bbox_with_margin(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
    margin_frac: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Union of two bboxes (subject + anchor) expanded by margin_frac of the
    union's own size, clamped to frame bounds. Used to crop a single image
    showing both objects for the vision model."""
    x0 = min(bbox_a[0], bbox_b[0])
    y0 = min(bbox_a[1], bbox_b[1])
    x1 = max(bbox_a[2], bbox_b[2])
    y1 = max(bbox_a[3], bbox_b[3])

    margin_x = (x1 - x0) * margin_frac
    margin_y = (y1 - y0) * margin_frac
    x0 = max(0, int(x0 - margin_x))
    y0 = max(0, int(y0 - margin_y))
    x1 = min(frame_width, int(x1 + margin_x))
    y1 = min(frame_height, int(y1 + margin_y))
    return x0, y0, x1, y1


class OllamaVisionClient:
    """One caption call runs at a time. `keep_alive="0s"` means every call
    is a cold model load (~40-50s for a 6 GB model); if a second event opens
    while one is still loading/running, firing a second concurrent load
    would compete for the same RAM on an 8 GB Mac and could hang it. The
    non-blocking lock below drops (never queues) any overlapping request —
    captions are a best-effort enrichment, not something worth blocking or
    backing up for."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "gemma4:e4b-it-qat",
        timeout_s: float = 120.0,
        keep_alive: str = "0s",
        cooldown_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.keep_alive = keep_alive
        self.cooldown_s = cooldown_s
        self.healthy = True
        self._last_failure = 0.0
        self._call_lock = threading.Lock()

    def caption(self, jpeg_bytes: bytes, prompt: str = CAPTION_PROMPT) -> str | None:
        if not self.healthy and time.monotonic() - self._last_failure < self.cooldown_s:
            return None

        if not self._call_lock.acquire(blocking=False):
            logger.info("Dropping vision caption request: a call is already in flight")
            return None
        try:
            return self._call(jpeg_bytes, prompt)
        finally:
            self._call_lock.release()

    def _call(self, jpeg_bytes: bytes, prompt: str) -> str | None:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [base64.b64encode(jpeg_bytes).decode("ascii")],
                    }
                ],
                "stream": False,
                "keep_alive": self.keep_alive,
                # Force a short answer: a free-running caption call can
                # otherwise generate hundreds of tokens (~10 tok/s on an M2 =
                # tens of extra seconds) despite being asked for one sentence.
                "options": {"num_predict": 60},
            }
        ).encode("utf-8")

        try:
            request = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = payload["message"]["content"].strip()
            self.healthy = True
            return text or None
        except Exception as exc:
            logger.warning("Ollama vision caption failed: %s", exc)
            self.healthy = False
            self._last_failure = time.monotonic()
            return None
