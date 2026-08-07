"""Optional vision captioning for monitoring events.

Two backends, same `.caption(jpeg_bytes, prompt) -> str | None` interface
(controller.py's `_caption_worker` doesn't care which is configured):

- `OllamaVisionClient` — a local Ollama multimodal model (e.g. Gemma 4
  E4B). Historically the only option, since apfel (agent.py) is text-only
  and on an 8 GB Mac a resident multimodal model wasn't affordable
  alongside everything else — hence `keep_alive: "0s"` trading latency for
  not holding multiple GB resident between calls. Still useful standalone
  (no GPU-service dependency, works offline).
- `QwenVisionClient` — the same Qwen/Qwen3.5-9B vLLM instance agent.py now
  talks to by default (confirmed multimodal: it correctly described a real
  test photo when sent as an OpenAI `image_url` content part), reached the
  same way (direct network, not through the gpu_service SSH tunnel). Since
  it's one already-running model serving both digests and captions, no
  per-call cold-load tax applies here, and `monitoring.vision.backend` in
  config/default.json defaults to this now that it's available.

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


class QwenVisionClient:
    """Same interface as OllamaVisionClient, OpenAI-compatible wire format
    (`/v1/chat/completions` with an `image_url` content part) instead of
    Ollama's native `/api/chat`. No call-serializing lock: this hits an
    already-running, request-queuing vLLM server rather than a cold-loading
    local model on shared Mac RAM, so there's no reason to drop overlapping
    requests the way OllamaVisionClient does."""

    def __init__(
        self,
        base_url: str = "http://172.25.6.176:8000",
        model: str = "Qwen/Qwen3.5-9B",
        timeout_s: float = 30.0,
        cooldown_s: float = 60.0,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.cooldown_s = cooldown_s
        self.chat_template_kwargs = chat_template_kwargs
        self.healthy = True
        self._last_failure = 0.0

    def caption(self, jpeg_bytes: bytes, prompt: str = CAPTION_PROMPT) -> str | None:
        return self.ask([jpeg_bytes], prompt, max_tokens=80)

    def ask(self, images: list[bytes], prompt: str,
            max_tokens: int = 200) -> str | None:
        """One prompt over one or more images.

        Multi-image is what makes this useful beyond captioning: showing the
        model two crops and asking "same physical object?" is a far better
        arbiter for a borderline merge than a cosine threshold, and describing
        an object from three views beats describing it from its blurriest one.

        Note this vLLM belongs to another user of the GPU box. It queues
        requests rather than loading per call, so overlapping requests are
        fine, but the cooldown below still matters: when it is down or busy we
        back off for a minute instead of hammering someone else's service.
        """
        if not self.healthy and time.monotonic() - self._last_failure < self.cooldown_s:
            return None
        if not images:
            return None
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for jpeg_bytes in images:
            image_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            })
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            # A free-running answer can otherwise spend far more than the
            # short reply these prompts ask for.
            "max_tokens": max_tokens,
        }
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        body = json.dumps(payload).encode("utf-8")

        try:
            request = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            message = response_payload["choices"][0]["message"]
            text = message.get("content") or message.get("reasoning")
            self.healthy = True
            return text.strip() if text else None
        except Exception as exc:
            logger.warning("Qwen vision caption failed: %s", exc)
            self.healthy = False
            self._last_failure = time.monotonic()
            return None
