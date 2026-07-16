"""Local AI agent layer over apfel (Apple Intelligence, on-device).

apfel runs an OpenAI-compatible HTTP server (default 127.0.0.1:11434). The
on-device model has a 4096-token context and is text-only, so this module is
strict about prompt budgets: events are compressed into one-line strings,
digests are the compression layer for summaries, and oversized summary
requests fall back to hierarchical chunk-merge.

Everything degrades gracefully: when apfel is down, `healthy` flips false
(with a cooldown before retrying) and monitoring continues without digests.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any


logger = logging.getLogger(__name__)

DIGEST_SYSTEM_PROMPT = (
    "You are a home-monitoring narrator. You receive a log of events from a "
    "fixed camera. Names like Person#1 and Cat#1 are persistent identities; "
    "labels like couch_1 are furniture. Write 1-3 plain factual sentences "
    "about what happened. Refer to identities by their names. No speculation, "
    "no preamble."
)

ROLLING_SUMMARY_SYSTEM_PROMPT = (
    "You maintain a running summary of what has happened in a room, based on "
    "events from a fixed camera. You will be given the CURRENT summary (may "
    "be empty, if this is the first update) and NEW events observed since "
    "that summary was last updated. Produce an UPDATED summary that "
    "incorporates the new events. Do not just append — actually merge and "
    "compress: keep older details only if still relevant, drop stale ones "
    "(e.g. someone who left long ago and hasn't returned). Identities "
    "(Person#1, Cat#1) persist across updates — refer to them by name. Keep "
    "the whole summary to at most a short paragraph. Factual, chronological, "
    "no preamble, no speculation."
)

SUMMARY_SYSTEM_PROMPT = (
    "You summarize home-monitoring digests. Identities (Person#1, Cat#1) "
    "persist across digests. Answer: who was present, when, and what they "
    "did. 3-6 sentences, chronological, factual. No preamble."
)


def format_event_line(event: dict[str, Any]) -> str:
    """One compact line per event (~12-18 tokens)."""

    def hhmm(ts: float | None) -> str:
        if ts is None:
            return "?"
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

    subject = event.get("subject_label", "?")
    etype = event.get("type", "?")
    details = event.get("details") or {}
    t_start, t_end = event.get("t_start"), event.get("t_end")

    if etype == "entered":
        return f"{hhmm(t_start)} {subject} entered"
    if etype == "exited":
        visible = details.get("visible_s")
        extra = f" (visible {_fmt_duration(visible)})" if visible else ""
        return f"{hhmm(t_start)} {subject} exited{extra}"
    if etype == "on_furniture":
        posture = details.get("posture", "")
        posture_part = f", {posture}" if posture and posture != "unclear" else ""
        duration = details.get("duration_s")
        if t_end is not None and duration:
            return (
                f"{hhmm(t_start)}→{hhmm(t_end)} {subject} on "
                f"{event.get('object_label', '?')}{posture_part} ({_fmt_duration(duration)})"
            )
        return f"{hhmm(t_start)} {subject} on {event.get('object_label', '?')}{posture_part} (ongoing)"
    if etype in {"stationary", "moving"}:
        if t_end is not None:
            return f"{hhmm(t_start)}→{hhmm(t_end)} {subject} {etype}"
        return f"{hhmm(t_start)} {subject} {etype} (ongoing)"
    return f"{hhmm(t_start)} {subject} {etype}"


def _fmt_duration(seconds: Any) -> str:
    seconds = float(seconds)
    if seconds < 90:
        return f"{int(seconds)}s"
    return f"{int(round(seconds / 60))}m"


def budget_lines(lines: list[str], max_chars: int) -> list[str]:
    """Keep the newest lines that fit the character budget (oldest dropped)."""
    kept: list[str] = []
    total = 0
    for line in reversed(lines):
        cost = len(line) + 1
        if total + cost > max_chars:
            break
        kept.append(line)
        total += cost
    return list(reversed(kept))


def chunk_lines(lines: list[str], max_chars: int) -> list[list[str]]:
    """Split lines into consecutive chunks each fitting the budget."""
    chunks: list[list[str]] = []
    current: list[str] = []
    total = 0
    for line in lines:
        cost = len(line) + 1
        if current and total + cost > max_chars:
            chunks.append(current)
            current, total = [], 0
        current.append(line)
        total += cost
    if current:
        chunks.append(current)
    return chunks


class ApfelClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "apple-foundationmodel",
        timeout_s: float = 20.0,
        max_input_tokens: int = 2500,
        max_output_tokens: int = 400,
        cooldown_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_input_chars = max_input_tokens * 4
        self.max_output_tokens = max_output_tokens
        self.cooldown_s = cooldown_s
        self.healthy = True
        self._last_failure = 0.0

    # ----------------------------------------------------------- plumbing

    def chat(self, system: str, user: str, max_tokens: int | None = None) -> str | None:
        if not self.healthy and time.monotonic() - self._last_failure < self.cooldown_s:
            return None
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens or self.max_output_tokens,
                "temperature": 0.3,
            }
        ).encode("utf-8")

        for attempt in range(2):
            try:
                request = urllib.request.Request(
                    f"{self.base_url}/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                text = payload["choices"][0]["message"]["content"].strip()
                self.healthy = True
                return text
            except Exception as exc:
                logger.warning("apfel call failed (attempt %d): %s", attempt + 1, exc)
                if attempt == 0:
                    time.sleep(2)
        self.healthy = False
        self._last_failure = time.monotonic()
        return None

    # ------------------------------------------------------------- digest

    def digest(self, scene_name: str, events: list[dict[str, Any]]) -> str | None:
        lines = budget_lines([format_event_line(e) for e in events], self.max_input_chars)
        if not lines:
            return None
        t0 = datetime.fromtimestamp(events[0]["t_start"]).strftime("%H:%M")
        t1 = datetime.fromtimestamp(events[-1]["t_start"]).strftime("%H:%M")
        user = f"Scene: {scene_name}. Window {t0}-{t1}.\n" + "\n".join(lines)
        return self.chat(DIGEST_SYSTEM_PROMPT, user, max_tokens=160)

    # -------------------------------------------------- rolling summary

    def update_rolling_summary(
        self,
        scene_name: str,
        previous_summary: str | None,
        new_events: list[dict[str, Any]],
    ) -> str | None:
        """Fold new_events into previous_summary via one incremental call —
        the summary itself stays roughly constant-sized over an arbitrarily
        long session, since each update explicitly asks the model to compress
        rather than append. This is the efficient alternative to re-merging
        a growing pile of independent digests on every summary request."""
        lines = budget_lines([format_event_line(e) for e in new_events], self.max_input_chars // 2)
        if not lines:
            return previous_summary

        current = previous_summary.strip() if previous_summary else "(none yet — first update)"
        user = f"Scene: {scene_name}.\nCurrent summary:\n{current}\n\nNew events:\n" + "\n".join(lines)
        updated = self.chat(ROLLING_SUMMARY_SYSTEM_PROMPT, user, max_tokens=220)
        return updated if updated else previous_summary

    # ------------------------------------------------------------ summary

    def summarize(
        self, scene_name: str, period_label: str, digests: list[dict[str, Any]]
    ) -> str | None:
        lines = []
        for digest in digests:
            t0 = datetime.fromtimestamp(digest["t_start"]).strftime("%H:%M")
            t1 = datetime.fromtimestamp(digest["t_end"]).strftime("%H:%M")
            lines.append(f"[{t0}-{t1}] {digest['text']}")
        if not lines:
            return None

        def merge_call(chunk: list[str]) -> str | None:
            user = f"Scene: {scene_name}. Period: {period_label}.\nDigests:\n" + "\n".join(chunk)
            return self.chat(SUMMARY_SYSTEM_PROMPT, user)

        total_chars = sum(len(line) + 1 for line in lines)
        if total_chars <= self.max_input_chars:
            return merge_call(lines)

        # Hierarchical: merge chunks, then merge the chunk summaries.
        chunk_summaries: list[str] = []
        for chunk in chunk_lines(lines, self.max_input_chars):
            summary = merge_call(chunk)
            if summary:
                chunk_summaries.append(summary)
        if not chunk_summaries:
            return None
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]
        return merge_call(budget_lines(chunk_summaries, self.max_input_chars))
