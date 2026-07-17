"""Presence layer: class-keyed entities that survive detection flicker.

Sits between the short-term tracker and the scene graph. The tracker's job
is frame-to-frame association; tracks die quickly (occlusion, missed
detections). A PresentEntity is the durable thing the user cares about:
"the person", "the laptop" — it stays present across track deaths and only
*leaves* when its last known bbox was near a frame border AND it has been
unseen for `exit_absent_s` (an object that vanished mid-frame is occluded,
not gone; `presence_timeout_s` is the honesty fallback).

Labels carry no permanent numbering: a lone entity of a class is just
"person"; only while several same-class entities are simultaneously alive
do they get "_1"/"_2" suffixes (rank by first_seen).

Pure logic (math only) — unit-testable with scripted tracks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from mac_server.monitoring.tracker import Track, centroid


@dataclass
class PresenceConfig:
    exit_edge_frac: float = 0.10
    exit_absent_s: float = 25.0
    presence_timeout_s: float = 300.0
    frame_width: float = 1280.0
    frame_height: float = 720.0
    # Re-bind scoring: position proximity decay scale as a fraction of the
    # frame diagonal, and the weight split vs signature similarity.
    rebind_position_scale_frac: float = 0.3
    rebind_signature_weight: float = 0.6

    @property
    def frame_diag(self) -> float:
        return math.hypot(self.frame_width, self.frame_height)


@dataclass
class PresentEntity:
    entity_key: int
    class_name: str
    state: str  # present | occluded | left
    first_seen: float
    last_seen: float
    last_bbox: tuple[float, float, float, float]
    track_id: int | None = None
    depth_m: float | None = None
    lateral_m: float | None = None
    forward_m: float | None = None
    signature: dict[str, Any] | None = None
    node_db_id: int | None = None  # graph node backing this entity
    total_visible_s: float = 0.0
    _visible_since: float | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "entity_key": self.entity_key,
            "class": self.class_name,
            "state": self.state,
            "bbox_xyxy": list(self.last_bbox),
            "depth_m": round(self.depth_m, 2) if self.depth_m is not None else None,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass
class PresenceEvent:
    event_type: str  # entered | exited
    entity: PresentEntity
    t: float
    details: dict[str, Any] = field(default_factory=dict)


def bbox_near_edge(
    bbox: tuple[float, float, float, float],
    frame_width: float,
    frame_height: float,
    frac: float,
) -> bool:
    """True when any side of the bbox is within `frac` of the corresponding
    frame border — the standard "leaving the frame" heuristic."""
    x0, y0, x1, y1 = bbox
    margin_x = frac * frame_width
    margin_y = frac * frame_height
    return (
        x0 <= margin_x
        or y0 <= margin_y
        or x1 >= frame_width - margin_x
        or y1 >= frame_height - margin_y
    )


class PresenceManager:
    """Maintains PresentEntity instances for one scene session.

    `similarity` is an optional callable (sig_a, sig_b) -> 0..1 used to pick
    the best occluded candidate on re-bind when several exist (CLIP cosine
    in production, anything in tests).
    """

    def __init__(
        self,
        config: PresenceConfig,
        similarity: Callable[[dict[str, Any], dict[str, Any]], float] | None = None,
    ) -> None:
        self.config = config
        self.similarity = similarity
        self.entities: dict[int, PresentEntity] = {}
        self._by_track: dict[int, int] = {}  # track_id -> entity_key
        self._next_key = 1

    # ------------------------------------------------------------- labels

    def label_of(self, entity: PresentEntity) -> str:
        """`person` when it is the only alive same-class entity, else
        `person_1`/`person_2` ranked by first_seen. Ranks are recomputed
        live: labels are display/log names, not identities."""
        alive = [
            e
            for e in self.entities.values()
            if e.class_name == entity.class_name and e.state != "left"
        ]
        if len(alive) <= 1:
            return entity.class_name
        alive.sort(key=lambda e: (e.first_seen, e.entity_key))
        rank = alive.index(entity) + 1
        return f"{entity.class_name}_{rank}"

    # ------------------------------------------------------ track lifecycle

    def on_track_confirmed(self, track: Track, now: float) -> PresenceEvent | None:
        """A confirmed track either re-binds to an occluded entity of the
        same class (no event — flicker/occlusion recovery) or starts a new
        presence (`entered`)."""
        candidates = [
            e
            for e in self.entities.values()
            if e.class_name == track.class_name and e.state == "occluded"
        ]
        if candidates:
            best = max(candidates, key=lambda e: self._rebind_score(e, track))
            self._bind(best, track, now)
            return None

        entity = PresentEntity(
            entity_key=self._next_key,
            class_name=track.class_name,
            state="present",
            first_seen=now,
            last_seen=now,
            last_bbox=track.bbox,
        )
        self._next_key += 1
        self.entities[entity.entity_key] = entity
        self._bind(entity, track, now)
        return PresenceEvent(event_type="entered", entity=entity, t=now)

    def on_track_ended(self, track: Track, now: float) -> None:
        """Track death is occlusion, never departure."""
        key = self._by_track.pop(track.track_id, None)
        if key is None:
            return
        entity = self.entities.get(key)
        if entity is None or entity.state != "present":
            return
        entity.state = "occluded"
        entity.track_id = None
        if entity._visible_since is not None:
            entity.total_visible_s += max(0.0, now - entity._visible_since)
            entity._visible_since = None

    # --------------------------------------------------------------- tick

    def update(self, tracks: list[Track], now: float) -> list[PresenceEvent]:
        """Sync bound entities from live tracks; expire occluded ones."""
        by_id = {t.track_id: t for t in tracks if t.state == "confirmed"}
        for track_id, key in list(self._by_track.items()):
            entity = self.entities.get(key)
            track = by_id.get(track_id)
            if entity is None:
                del self._by_track[track_id]
                continue
            if track is None:
                continue  # lost this tick; tracker will end it eventually
            entity.last_bbox = track.bbox
            entity.last_seen = now
            entity.depth_m = track.depth_m
            entity.lateral_m = track.lateral_m
            entity.forward_m = track.forward_m
            if track.signature is not None:
                entity.signature = track.signature

        events: list[PresenceEvent] = []
        cfg = self.config
        for entity in self.entities.values():
            if entity.state != "occluded":
                continue
            absent = now - entity.last_seen
            near_edge = bbox_near_edge(
                entity.last_bbox, cfg.frame_width, cfg.frame_height, cfg.exit_edge_frac
            )
            if near_edge and absent >= cfg.exit_absent_s:
                events.append(self._leave(entity, now, reason="left_frame"))
            elif absent >= cfg.presence_timeout_s:
                events.append(self._leave(entity, now, reason="timeout"))
        return events

    def force_leave_all(self, now: float) -> list[PresenceEvent]:
        """Monitoring stop: everything still around leaves."""
        events = []
        for entity in self.entities.values():
            if entity.state != "left":
                events.append(self._leave(entity, now, reason="session_stop"))
        self._by_track.clear()
        return events

    def prune_left(self) -> None:
        """Drop `left` entities once the caller has persisted their exit
        (their durable identity lives on as a graph node)."""
        for key in [k for k, e in self.entities.items() if e.state == "left"]:
            del self.entities[key]

    # ------------------------------------------------------------ queries

    def present_entities(self) -> list[PresentEntity]:
        return [e for e in self.entities.values() if e.state == "present"]

    def alive_entities(self) -> list[PresentEntity]:
        return [e for e in self.entities.values() if e.state != "left"]

    def entity_for_track(self, track_id: int) -> PresentEntity | None:
        key = self._by_track.get(track_id)
        return self.entities.get(key) if key is not None else None

    # ------------------------------------------------------------ helpers

    def _bind(self, entity: PresentEntity, track: Track, now: float) -> None:
        entity.state = "present"
        entity.track_id = track.track_id
        entity.last_bbox = track.bbox
        entity.last_seen = now
        entity.depth_m = track.depth_m
        entity.lateral_m = track.lateral_m
        entity.forward_m = track.forward_m
        if entity._visible_since is None:
            entity._visible_since = now
        self._by_track[track.track_id] = entity.entity_key

    def _rebind_score(self, entity: PresentEntity, track: Track) -> float:
        cfg = self.config
        tcx, tcy = centroid(track.bbox)
        ecx, ecy = centroid(entity.last_bbox)
        dist = math.hypot(tcx - ecx, tcy - ecy)
        position = math.exp(-dist / max(1.0, cfg.rebind_position_scale_frac * cfg.frame_diag))

        sig_sim = 0.5  # neutral when either side has no signature
        if (
            self.similarity is not None
            and entity.signature is not None
            and track.signature is not None
        ):
            sig_sim = self.similarity(entity.signature, track.signature)

        w = cfg.rebind_signature_weight
        return w * sig_sim + (1 - w) * position

    def _leave(self, entity: PresentEntity, now: float, reason: str) -> PresenceEvent:
        if entity.track_id is not None:
            self._by_track.pop(entity.track_id, None)
        entity.state = "left"
        entity.track_id = None
        if entity._visible_since is not None:
            entity.total_visible_s += max(0.0, entity.last_seen - entity._visible_since)
            entity._visible_since = None
        return PresenceEvent(
            event_type="exited",
            entity=entity,
            t=now,
            details={
                "reason": reason,
                "visible_s": round(entity.total_visible_s, 1),
            },
        )
