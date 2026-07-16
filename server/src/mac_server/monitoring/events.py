"""Semantic event engine with symmetric hysteresis.

Evaluated once per tracker tick over confirmed tracks and frozen scene
anchors. Durational conditions must hold `open_hold_s` to open an event and
be false `close_hold_s` to close it (bridges YOLO flicker).

Event types (v1):
- entered  (point): a mobile track became confirmed
- exited   (point): a confirmed track ended; details.visible_s
- stationary / moving (durational): centroid spread over a window
- on_furniture (durational): mobile track overlaps an anchor at similar
  depth; details.posture from bbox aspect ratio (majority vote at close)

Pure logic — unit-testable with scripted track snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mac_server.monitoring.tracker import Track, is_stationary


@dataclass
class EventRuleConfig:
    open_hold_s: float = 2.0
    close_hold_s: float = 3.0
    overlap_on_ratio: float = 0.3
    depth_agree_m: float = 0.7
    stationary_window_s: float = 5.0
    stationary_px_frac: float = 0.02
    frame_diag: float = 1468.6
    posture_sitting_ratio: float = 1.15  # h/w above this = sitting/upright
    posture_lying_ratio: float = 0.85  # h/w below this = lying


@dataclass
class ActiveEvent:
    key: tuple  # (type, track_id, object_id)
    event_type: str
    track: Track
    object_label: str | None
    t_start: float
    details: dict[str, Any] = field(default_factory=dict)
    posture_votes: dict[str, int] = field(default_factory=dict)
    db_id: int | None = None


@dataclass
class EventTransition:
    action: str  # "open" | "close" | "point"
    event_type: str
    track: Track
    object_label: str | None
    t_start: float
    t_end: float | None
    details: dict[str, Any]
    active: ActiveEvent | None = None  # for "open", so the caller can stash db ids


def bbox_posture(bbox: tuple[float, float, float, float], cfg: EventRuleConfig) -> str:
    width = max(1e-6, bbox[2] - bbox[0])
    height = max(1e-6, bbox[3] - bbox[1])
    ratio = height / width
    if ratio >= cfg.posture_sitting_ratio:
        return "sitting"
    if ratio <= cfg.posture_lying_ratio:
        return "lying"
    return "unclear"


def _overlap_ratio(subject: tuple[float, float, float, float], anchor: list[float]) -> float:
    ix0, iy0 = max(subject[0], anchor[0]), max(subject[1], anchor[1])
    ix1, iy1 = min(subject[2], anchor[2]), min(subject[3], anchor[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    subject_area = max(1e-6, (subject[2] - subject[0]) * (subject[3] - subject[1]))
    return inter / subject_area


class EventEngine:
    def __init__(self, config: EventRuleConfig) -> None:
        self.config = config
        self._active: dict[tuple, ActiveEvent] = {}
        self._pending_open: dict[tuple, float] = {}  # condition true since
        self._pending_close: dict[tuple, float] = {}  # condition false since

    @property
    def open_events(self) -> list[ActiveEvent]:
        return list(self._active.values())

    def track_confirmed(self, track: Track, now: float) -> EventTransition:
        return EventTransition(
            action="point",
            event_type="entered",
            track=track,
            object_label=None,
            t_start=now,
            t_end=now,
            details={},
        )

    def track_ended(self, track: Track, now: float) -> list[EventTransition]:
        """Exited point event + force-close of the track's open events."""
        transitions = [
            EventTransition(
                action="point",
                event_type="exited",
                track=track,
                object_label=None,
                t_start=now,
                t_end=now,
                details={"visible_s": round(track.last_seen - track.first_seen, 1)},
            )
        ]
        for key in [k for k in self._active if k[1] == track.track_id]:
            transitions.append(self._close(key, track.last_seen))
        for pending in (self._pending_open, self._pending_close):
            for key in [k for k in pending if k[1] == track.track_id]:
                del pending[key]
        return transitions

    def evaluate(
        self,
        tracks: list[Track],
        anchors: list[dict[str, Any]],
        now: float,
    ) -> list[EventTransition]:
        transitions: list[EventTransition] = []
        conditions: dict[tuple, dict[str, Any]] = {}

        for track in tracks:
            if track.state != "confirmed":
                continue

            stationary = is_stationary(
                track.centroid_history,
                now,
                self.config.stationary_window_s,
                self.config.stationary_px_frac,
                self.config.frame_diag,
            )
            if stationary is True:
                conditions[("stationary", track.track_id, None)] = {"_track": track}
            elif stationary is False:
                conditions[("moving", track.track_id, None)] = {"_track": track}

            for anchor in anchors:
                anchor_bbox = anchor.get("bbox_xyxy")
                if not anchor_bbox:
                    continue
                overlap = _overlap_ratio(track.bbox, anchor_bbox)
                if overlap < self.config.overlap_on_ratio:
                    continue
                anchor_depth = anchor.get("depth_m")
                if (
                    anchor_depth is not None
                    and track.depth_m is not None
                    and abs(track.depth_m - float(anchor_depth)) > self.config.depth_agree_m
                ):
                    continue
                conditions[("on_furniture", track.track_id, anchor["anchor_id"])] = {
                    "_track": track,
                    "overlap": round(overlap, 2),
                    "posture": bbox_posture(track.bbox, self.config),
                }

        # Open logic: condition must hold open_hold_s.
        for key, info in conditions.items():
            if key in self._active:
                self._pending_close.pop(key, None)
                active = self._active[key]
                posture = info.get("posture")
                if posture:
                    active.posture_votes[posture] = active.posture_votes.get(posture, 0) + 1
                continue
            since = self._pending_open.setdefault(key, now)
            if now - since >= self.config.open_hold_s:
                del self._pending_open[key]
                track = info.pop("_track")
                active = ActiveEvent(
                    key=key,
                    event_type=key[0],
                    track=track,
                    object_label=key[2],
                    t_start=since,
                    details=dict(info),
                )
                posture = info.get("posture")
                if posture:
                    active.posture_votes[posture] = 1
                self._active[key] = active
                transitions.append(
                    EventTransition(
                        action="open",
                        event_type=active.event_type,
                        track=track,
                        object_label=active.object_label,
                        t_start=active.t_start,
                        t_end=None,
                        details=dict(info),
                        active=active,
                    )
                )

        # Drop pending opens whose condition broke before maturing.
        for key in [k for k in self._pending_open if k not in conditions]:
            del self._pending_open[key]

        # Close logic: condition must stay false close_hold_s.
        for key in list(self._active.keys()):
            if key in conditions:
                continue
            since = self._pending_close.setdefault(key, now)
            if now - since >= self.config.close_hold_s:
                transitions.append(self._close(key, since))

        return transitions

    def force_close_all(self, now: float) -> list[EventTransition]:
        return [self._close(key, now) for key in list(self._active.keys())]

    def _close(self, key: tuple, t_end: float) -> EventTransition:
        active = self._active.pop(key)
        self._pending_close.pop(key, None)
        details = dict(active.details)
        if active.posture_votes:
            details["posture"] = max(active.posture_votes.items(), key=lambda kv: kv[1])[0]
        details["duration_s"] = round(t_end - active.t_start, 1)
        return EventTransition(
            action="close",
            event_type=active.event_type,
            track=active.track,
            object_label=active.object_label,
            t_start=active.t_start,
            t_end=t_end,
            details=details,
            active=active,
        )
