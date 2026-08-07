"""Timestamped record of every wheel command, for behaviour-cloning datasets.

The robot's observations are produced on the Pi and its actions on this
laptop, and until now the actions were not recorded at all — `RobocarService`
kept only `_last_drive`, the most recent pair, with no timestamp and no
history. A policy cannot be trained from that.

`RobocarService.drive()` is the single funnel every wheel command passes
through — the panel, the localization spin, and anything added later — so one
callback there captures the complete action stream with no risk of a second
path being forgotten.

**What gets written, and why each field is here:**

* `left`/`right` — post-`mix_drive` wheel PWMs. Ground truth for what the
  hardware was told to do, and the quantity a policy predicts.
* `throttle`/`steer`/`speed` — the pre-mix operator intent, when the command
  came in that form. Kept because `mix_drive` deliberately swaps the wheels
  to compensate this chassis' mirrored motor leads; if that wiring is ever
  corrected, a wheels-only recording silently becomes wrong while the intent
  stays true.
* `source` — panel / spin / harness. Lets a later filter separate teleop from
  automation without guessing.
* `max_pwm` — the ESP's `scaleCmd` maps a request onto `[MIN_PWM, MAX_PWM]`,
  so the physical meaning of the number `120` depends on it. Episodes
  recorded under different limits are not comparable and must not be merged.
* `heartbeat` rows — the ESP reports its *applied* post-ramp PWM at 0.5 Hz.
  Since `updateRamp` moves by 8 every 15 ms, the applied value should lag the
  command by roughly one ramp time and nothing else. That makes the
  heartbeats a free, independent check on the whole timestamp join: if they
  lag by a second, the join is wrong.

Timestamps are `time.monotonic_ns()` on this machine, converted into the Pi's
timeline at dataset-build time with the offset measured by
`shared.clock_sync`. Wall clock is recorded alongside for human readability
only — never use it for the join, it can step.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Enough to cover a long episode at the panel's 10 Hz refresh (plus the
# spin's own commands) without unbounded growth if nobody is recording.
DEFAULT_CAPACITY = 20_000


@dataclass(frozen=True)
class DriveRecord:
    t_mono_ns: int
    t_wall_ns: int
    left: int
    right: int
    source: str
    max_pwm: int
    throttle: float | None = None
    steer: float | None = None
    speed: int | None = None
    kind: str = "drive"


@dataclass(frozen=True)
class HeartbeatRecord:
    t_mono_ns: int
    t_wall_ns: int
    text: str
    applied_left: int | None = None
    applied_right: int | None = None
    kind: str = "heartbeat"


def parse_heartbeat(text: str) -> tuple[int | None, int | None]:
    """Pull the applied PWMs out of `HB up=123 L=140 R=-140 rssi=-52 heap=…`.

    Tolerant by design: the sketch's heartbeat format is not a contract, and
    a missing field must degrade to None rather than break recording.
    """
    left = right = None
    for token in text.split():
        try:
            if token.startswith("L="):
                left = int(token[2:])
            elif token.startswith("R="):
                right = int(token[2:])
        except ValueError:
            continue
    return left, right


@dataclass
class ActionLog:
    """Ring buffer of drive commands, optionally mirrored to an episode file."""

    capacity: int = DEFAULT_CAPACITY
    _records: list[Any] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _episode_id: str | None = None
    _episode_path: Path | None = None
    _handle: Any = None
    _dropped: int = 0

    # ------------------------------------------------------------ recording

    def record_drive(
        self,
        left: int,
        right: int,
        source: str,
        max_pwm: int,
        throttle: float | None = None,
        steer: float | None = None,
        speed: int | None = None,
    ) -> None:
        self._append(DriveRecord(
            t_mono_ns=time.monotonic_ns(), t_wall_ns=time.time_ns(),
            left=int(left), right=int(right), source=source, max_pwm=int(max_pwm),
            throttle=throttle, steer=steer, speed=speed,
        ))

    def record_heartbeat(self, text: str) -> None:
        applied_left, applied_right = parse_heartbeat(text)
        self._append(HeartbeatRecord(
            t_mono_ns=time.monotonic_ns(), t_wall_ns=time.time_ns(),
            text=text, applied_left=applied_left, applied_right=applied_right,
        ))

    def _append(self, record: Any) -> None:
        with self._lock:
            self._records.append(record)
            if len(self._records) > self.capacity:
                # Drop oldest. Counted, because silently losing the head of an
                # episode would produce a dataset with a plausible-looking but
                # truncated first second.
                overflow = len(self._records) - self.capacity
                del self._records[:overflow]
                self._dropped += overflow
            handle = self._handle
        if handle is not None:
            try:
                handle.write(json.dumps(asdict(record)) + "\n")
            except OSError as exc:      # disk full, unplugged, …
                logger.warning("Action log write failed: %s", exc)

    # ------------------------------------------------------------- episodes

    def start_episode(self, episode_id: str, episode_dir: Path) -> Path:
        """Begin mirroring to `<episode_dir>/actions.jsonl`.

        Line-buffered and flushed per record: an episode that ends in a crash
        or a yanked battery should still yield everything up to that moment,
        which is exactly when the interesting failure data lives.
        """
        self.stop_episode()
        episode_dir.mkdir(parents=True, exist_ok=True)
        path = episode_dir / "actions.jsonl"
        with self._lock:
            self._episode_id = episode_id
            self._episode_path = path
            self._handle = path.open("w", encoding="utf-8", buffering=1)
            self._records.clear()
            self._dropped = 0
        logger.info("Action log recording to %s", path)
        return path

    def stop_episode(self) -> Path | None:
        with self._lock:
            handle, path = self._handle, self._episode_path
            self._handle = None
            self._episode_id = None
            self._episode_path = None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        return path

    # ----------------------------------------------------------------- read

    def records(self) -> list[Any]:
        with self._lock:
            return list(self._records)

    def status(self) -> dict[str, Any]:
        with self._lock:
            drives = sum(1 for r in self._records if r.kind == "drive")
            return {
                "recording": self._handle is not None,
                "episode_id": self._episode_id,
                "path": str(self._episode_path) if self._episode_path else None,
                "records": len(self._records),
                "drives": drives,
                "heartbeats": len(self._records) - drives,
                "dropped": self._dropped,
            }
