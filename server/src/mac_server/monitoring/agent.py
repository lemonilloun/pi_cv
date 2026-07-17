"""Local AI agent layer over apfel (Apple Intelligence, on-device).

apfel runs an OpenAI-compatible HTTP server (default 127.0.0.1:11500). The
on-device model has a 4096-token context and is text-only, so this module is
strict about prompt budgets: graph edges are compressed into one-line
strings, digests are the compression layer for longer windows, and the
rolling summary keeps a constant-size running narrative.

v2 semantics: no persistent numbering — subjects are plain class labels
("person", "laptop", "person_2" only during concurrency). Digests speak in
aggregates ("A person spent 25 minutes on the couch; a laptop sat nearby"),
the model also picks 0-3 key moments per digest cycle (they get participant
snapshots), and free-form questions are answered over the graph slice
(`ask`) instead of the old one-button summary.

Everything degrades gracefully: when apfel is down, `healthy` flips false
(with a cooldown before retrying) and monitoring continues without digests.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any


logger = logging.getLogger(__name__)

DIGEST_SYSTEM_PROMPT = (
    "You are a home-monitoring narrator. You receive a log of observations "
    "from a fixed camera: subjects are object classes (person, laptop, cat; "
    "person_2 only means a second person was in view at the same time), and "
    "labels like couch_1 are furniture. Write 1-3 compact factual sentences "
    "in aggregate style: total time spent where, what was nearby, what "
    "moved. Example tone: 'A person spent 25 minutes sitting on the couch; "
    "a laptop sat nearby and was moved twice.' No numbering of objects, no "
    "speculation, no preamble."
)

KEY_MOMENTS_SYSTEM_PROMPT = (
    "You review a numbered log of observations from a fixed home camera and "
    "pick the few genuinely notable moments — arrivals, departures, real "
    "movements or interactions. Ignore routine continuations. Answer with "
    "ONLY the numbers of at most {max_moments} notable lines, comma-"
    "separated (e.g. '2,5'). If nothing is notable, answer 'none'."
)

ROLLING_SUMMARY_SYSTEM_PROMPT = (
    "You maintain a running summary of what has happened in a room, based on "
    "observations from a fixed camera. You will be given the CURRENT summary "
    "(may be empty, if this is the first update) and NEW observations since "
    "that summary was last updated. Produce an UPDATED summary that "
    "incorporates the new observations. Do not just append — actually merge "
    "and compress: keep older details only if still relevant, drop stale "
    "ones. Subjects are plain object classes (person, laptop); do not invent "
    "numbering. Keep the whole summary to at most a short paragraph. "
    "Factual, chronological, no preamble, no speculation."
)

ASK_SYSTEM_PROMPT = (
    "You answer questions about what happened in a room watched by a fixed "
    "camera. You are given: the CURRENT state of the scene (where every "
    "object is right now), a log of notable observations, and periodic "
    "digests. Subjects are plain object classes (person, laptop, cat); "
    "labels like couch_1 are furniture. Answer the question directly and "
    "factually from this data, mention times when relevant, and say plainly "
    "when the data does not contain the answer. No speculation, no preamble."
)


def format_edge_line(edge: dict[str, Any]) -> str:
    """One compact line per graph edge (~12-20 tokens), geometry included
    so the model can describe interactions concretely."""

    def hhmm(ts: float | None) -> str:
        if ts is None:
            return "?"
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

    subject = edge.get("subject_label", "?")
    relation = edge.get("relation", "?")
    obj = edge.get("object_label")
    details = edge.get("details") or {}
    t_start, t_end = edge.get("t_start"), edge.get("t_end")

    if relation == "entered":
        return f"{hhmm(t_start)} {subject} entered the frame"
    if relation == "exited":
        visible = details.get("visible_s")
        extra = f" (was visible {_fmt_duration(visible)})" if visible else ""
        return f"{hhmm(t_start)} {subject} left the frame{extra}"
    if relation == "moved":
        parts = [f"{hhmm(t_start)} {subject} moved"]
        if details.get("from_zone") and details.get("to_zone"):
            parts.append(f"from {details['from_zone']} to {details['to_zone']}")
        if details.get("nearest_anchor"):
            parts.append(f"(now near {details['nearest_anchor']})")
        return " ".join(parts)

    geo = []
    if details.get("posture") and details["posture"] != "unclear":
        geo.append(details["posture"])
    if details.get("distance_m") is not None:
        geo.append(f"{details['distance_m']}m apart")
    if details.get("arrangement") and details.get("arrangement") != "overlapping":
        geo.append(str(details["arrangement"]))
    if t_end is not None and details.get("duration_s"):
        geo.append(_fmt_duration(details["duration_s"]))
    geo_part = f" ({', '.join(geo)})" if geo else ""

    span = f"{hhmm(t_start)}→{hhmm(t_end)}" if t_end is not None else f"{hhmm(t_start)}"
    ongoing = "" if t_end is not None else " (ongoing)"
    return f"{span} {subject} {relation} {obj or '?'}{geo_part}{ongoing}"


def _fmt_duration(seconds: Any) -> str:
    seconds = float(seconds)
    if seconds < 90:
        return f"{int(seconds)}s"
    return f"{int(round(seconds / 60))}m"


def format_now_state(state: list[dict[str, Any]]) -> list[str]:
    """Current-state lines for Q&A prompts: `person: on couch_1 (2.9m)`."""
    lines = []
    for item in state:
        relations = ", ".join(item.get("relations") or []) or "in view"
        depth = f" ({item['depth_m']}m away)" if item.get("depth_m") is not None else ""
        marker = " [occluded]" if item.get("state") == "occluded" else ""
        lines.append(f"{item.get('label', '?')}: {relations}{depth}{marker}")
    return lines


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


def parse_key_moment_indices(answer: str | None, count: int, max_moments: int) -> list[int]:
    """Parse the model's '2,5' style answer into valid 0-based indices."""
    if not answer:
        return []
    if "none" in answer.lower():
        return []
    indices: list[int] = []
    for token in re.findall(r"\d+", answer):
        idx = int(token) - 1  # the prompt numbers lines from 1
        if 0 <= idx < count and idx not in indices:
            indices.append(idx)
        if len(indices) >= max_moments:
            break
    return indices


class ApfelClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11500",
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

    def digest(self, scene_name: str, edges: list[dict[str, Any]]) -> str | None:
        lines = budget_lines([format_edge_line(e) for e in edges], self.max_input_chars)
        if not lines:
            return None
        t0 = datetime.fromtimestamp(edges[0]["t_start"]).strftime("%H:%M")
        t1 = datetime.fromtimestamp(edges[-1]["t_start"]).strftime("%H:%M")
        user = f"Scene: {scene_name}. Window {t0}-{t1}.\n" + "\n".join(lines)
        return self.chat(DIGEST_SYSTEM_PROMPT, user, max_tokens=160)

    # -------------------------------------------------------- key moments

    def pick_key_moments(
        self, edges: list[dict[str, Any]], max_moments: int = 3
    ) -> list[dict[str, Any]]:
        """Ask the model which edges of the cycle are notable; returns the
        chosen edge dicts (possibly empty). Fail-safe: on any model trouble
        the answer is just 'no key moments'."""
        if not edges:
            return []
        lines = [f"{i + 1}. {format_edge_line(e)}" for i, e in enumerate(edges)]
        lines = budget_lines(lines, self.max_input_chars)
        answer = self.chat(
            KEY_MOMENTS_SYSTEM_PROMPT.format(max_moments=max_moments),
            "\n".join(lines),
            max_tokens=30,
        )
        indices = parse_key_moment_indices(answer, len(edges), max_moments)
        return [edges[i] for i in indices]

    # -------------------------------------------------- rolling summary

    def update_rolling_summary(
        self,
        scene_name: str,
        previous_summary: str | None,
        new_edges: list[dict[str, Any]],
    ) -> str | None:
        """Fold new edges into the running summary via one incremental call —
        the summary stays roughly constant-sized over an arbitrarily long
        session. Kept as internal context for `ask` (no UI button)."""
        lines = budget_lines([format_edge_line(e) for e in new_edges], self.max_input_chars // 2)
        if not lines:
            return previous_summary

        current = previous_summary.strip() if previous_summary else "(none yet — first update)"
        user = f"Scene: {scene_name}.\nCurrent summary:\n{current}\n\nNew observations:\n" + "\n".join(lines)
        updated = self.chat(ROLLING_SUMMARY_SYSTEM_PROMPT, user, max_tokens=220)
        return updated if updated else previous_summary

    # ---------------------------------------------------------------- ask

    def ask(
        self,
        scene_name: str,
        question: str,
        now_lines: list[str],
        edges: list[dict[str, Any]],
        digests: list[dict[str, Any]],
        rolling_summary: str | None = None,
    ) -> str | None:
        """Answer a free-form question over the graph slice. Budget order:
        the question and current state always fit; recent edges get half the
        remaining budget, digests (older, already compressed) the rest."""
        sections = [f"Scene: {scene_name}.", f"Question: {question}", ""]
        sections.append("Current state:")
        sections.extend(now_lines or ["(nothing in view)"])

        remaining = self.max_input_chars - sum(len(s) + 1 for s in sections)
        edge_lines = budget_lines(
            [format_edge_line(e) for e in edges], max(0, remaining // 2)
        )
        digest_lines = budget_lines(
            [
                f"[{datetime.fromtimestamp(d['t_start']).strftime('%H:%M')}-"
                f"{datetime.fromtimestamp(d['t_end']).strftime('%H:%M')}] {d['text']}"
                for d in digests
            ],
            max(0, remaining - sum(len(line) + 1 for line in edge_lines)),
        )
        if rolling_summary:
            sections += ["", "Session summary so far:", rolling_summary]
        if digest_lines:
            sections += ["", "Digests:"] + digest_lines
        if edge_lines:
            sections += ["", "Observations:"] + edge_lines
        return self.chat(ASK_SYSTEM_PROMPT, "\n".join(sections))
