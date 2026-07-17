"""Relation engine: durational graph edges with symmetric hysteresis.

Consumes *present* entities (presence.py) and frozen scene anchors once per
tick and maintains episodic relations — the edges of the scene graph:

- on(entity, anchor)     durational: bbox overlap at agreeing (frozen) depth;
                         posture from bbox aspect ratio, majority-voted.
- near(A, B)             durational: small bbox gap + agreeing depth, between
                         an entity and an anchor or another entity. The feed
                         phrases its open as "A approached B" and its close
                         as "A moved away from B".
- moved(entity)          point: sustained centroid displacement beyond a
                         threshold — micro-jitter never fires it.

Hysteresis: a condition must hold `open_hold_s` to open and stay false
`close_hold_s` to close. Occluded subjects PAUSE their close countdowns
instead of closing (an occluded object presumably hasn't moved — closing
and reopening on flicker is exactly the spam this design removes); edges
only force-close when the entity actually leaves.

Every transition's details carry geometry facts (depth, gap, mutual bbox
arrangement) so the LLM can describe interactions concretely.

Pure logic — unit-testable with scripted entities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from mac_server.monitoring.presence import PresentEntity


@dataclass
class RelationConfig:
    open_hold_s: float = 2.0
    close_hold_s: float = 3.0
    overlap_on_ratio: float = 0.3
    depth_agree_m: float = 0.7
    near_gap_frac: float = 0.05
    near_depth_agree_m: float = 1.0
    move_min_frac: float = 0.15
    frame_width: float = 1280.0
    frame_height: float = 720.0
    posture_sitting_ratio: float = 1.15  # h/w above this = sitting/upright
    posture_lying_ratio: float = 0.85  # h/w below this = lying

    @property
    def frame_diag(self) -> float:
        return math.hypot(self.frame_width, self.frame_height)


@dataclass
class ActiveRelation:
    key: tuple  # (relation, subject_key, object_id)
    relation: str
    subject_key: int
    subject_label: str
    object_key: int | None  # entity_key when the object is another entity
    object_label: str | None
    t_start: float
    details: dict[str, Any] = field(default_factory=dict)
    posture_votes: dict[str, int] = field(default_factory=dict)
    db_id: int | None = None  # graph_edges row


@dataclass
class RelationTransition:
    action: str  # "open" | "close" | "point"
    relation: str
    subject_key: int
    subject_label: str
    object_key: int | None
    object_label: str | None
    t_start: float
    t_end: float | None
    details: dict[str, Any]
    active: ActiveRelation | None = None  # for "open", to stash db ids


# ------------------------------------------------------------- geometry


def bbox_gap_frac(
    a: tuple[float, float, float, float],
    b: list[float] | tuple[float, float, float, float],
    frame_diag: float,
) -> float:
    """Distance between the closest edges of two boxes as a fraction of the
    frame diagonal; 0.0 when they touch or overlap."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy) / max(1.0, frame_diag)


def bbox_relation(
    subject: tuple[float, float, float, float],
    obj: list[float] | tuple[float, float, float, float],
) -> str:
    """Coarse mutual arrangement of subject vs object, for LLM prompts."""
    ix = min(subject[2], obj[2]) - max(subject[0], obj[0])
    iy = min(subject[3], obj[3]) - max(subject[1], obj[1])
    if ix > 0 and iy > 0:
        return "overlapping"
    scx, scy = (subject[0] + subject[2]) / 2, (subject[1] + subject[3]) / 2
    ocx, ocy = (obj[0] + obj[2]) / 2, (obj[1] + obj[3]) / 2
    if abs(scx - ocx) >= abs(scy - ocy):
        return "left-of" if scx < ocx else "right-of"
    return "above" if scy < ocy else "below"


def bbox_posture(bbox: tuple[float, float, float, float], cfg: RelationConfig) -> str:
    width = max(1e-6, bbox[2] - bbox[0])
    height = max(1e-6, bbox[3] - bbox[1])
    ratio = height / width
    if ratio >= cfg.posture_sitting_ratio:
        return "sitting"
    if ratio <= cfg.posture_lying_ratio:
        return "lying"
    return "unclear"


def _overlap_ratio(
    subject: tuple[float, float, float, float], anchor: list[float]
) -> float:
    ix0, iy0 = max(subject[0], anchor[0]), max(subject[1], anchor[1])
    ix1, iy1 = min(subject[2], anchor[2]), min(subject[3], anchor[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    subject_area = max(1e-6, (subject[2] - subject[0]) * (subject[3] - subject[1]))
    return inter / subject_area


def entity_distance_m(a: PresentEntity, b: PresentEntity) -> float | None:
    """Metric distance in the camera plane when both entities carry
    lateral/forward coordinates (from the Mac depth worker)."""
    if None in (a.lateral_m, a.forward_m, b.lateral_m, b.forward_m):
        return None
    return math.hypot(a.lateral_m - b.lateral_m, a.forward_m - b.forward_m)


def frame_zone(cx: float, frame_width: float) -> str:
    third = frame_width / 3.0
    if cx < third:
        return "left"
    if cx > 2 * third:
        return "right"
    return "center"


# --------------------------------------------------------------- engine


class RelationEngine:
    def __init__(self, config: RelationConfig) -> None:
        self.config = config
        self._active: dict[tuple, ActiveRelation] = {}
        self._pending_open: dict[tuple, float] = {}  # condition true since
        self._pending_close: dict[tuple, float] = {}  # condition false since
        self._rest_position: dict[int, tuple[float, float]] = {}  # entity_key
        self._pending_move: dict[int, float] = {}  # displaced since

    @property
    def open_relations(self) -> list[ActiveRelation]:
        return list(self._active.values())

    def evaluate(
        self,
        entities: list[PresentEntity],
        labels: dict[int, str],
        anchors: list[dict[str, Any]],
        now: float,
        occluded_keys: set[int] | None = None,
    ) -> list[RelationTransition]:
        cfg = self.config
        occluded_keys = occluded_keys or set()
        transitions: list[RelationTransition] = []
        conditions: dict[tuple, dict[str, Any]] = {}

        on_pairs: set[tuple[int, str]] = set()
        for entity in entities:
            self._collect_on(entity, anchors, conditions, on_pairs)
        for entity in entities:
            self._collect_near_anchor(entity, anchors, conditions, on_pairs)
        self._collect_near_entities(entities, conditions)

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
            if now - since >= cfg.open_hold_s:
                del self._pending_open[key]
                subject_key = key[1]
                details = {k: v for k, v in info.items() if not k.startswith("_")}
                active = ActiveRelation(
                    key=key,
                    relation=key[0],
                    subject_key=subject_key,
                    subject_label=labels.get(subject_key, "?"),
                    object_key=info.get("_object_key"),
                    object_label=info.get("_object_label"),
                    t_start=since,
                    details=details,
                )
                posture = info.get("posture")
                if posture:
                    active.posture_votes[posture] = 1
                self._active[key] = active
                transitions.append(
                    RelationTransition(
                        action="open",
                        relation=active.relation,
                        subject_key=active.subject_key,
                        subject_label=active.subject_label,
                        object_key=active.object_key,
                        object_label=active.object_label,
                        t_start=active.t_start,
                        t_end=None,
                        details=dict(details),
                        active=active,
                    )
                )

        # Drop pending opens whose condition broke before maturing.
        for key in [k for k in self._pending_open if k not in conditions]:
            del self._pending_open[key]

        # Close logic: condition must stay false close_hold_s. An occluded
        # subject/object pauses the countdown instead of closing.
        for key in list(self._active.keys()):
            if key in conditions:
                continue
            active = self._active[key]
            if active.subject_key in occluded_keys or (
                active.object_key is not None and active.object_key in occluded_keys
            ):
                self._pending_close.pop(key, None)
                continue
            since = self._pending_close.setdefault(key, now)
            if now - since >= cfg.close_hold_s:
                transitions.append(self._close(key, since))

        transitions.extend(self._evaluate_moves(entities, labels, anchors, now))
        return transitions

    # ---------------------------------------------------------- conditions

    def _collect_on(
        self,
        entity: PresentEntity,
        anchors: list[dict[str, Any]],
        conditions: dict[tuple, dict[str, Any]],
        on_pairs: set[tuple[int, str]],
    ) -> None:
        cfg = self.config
        for anchor in anchors:
            anchor_bbox = anchor.get("bbox_xyxy")
            if not anchor_bbox:
                continue
            overlap = _overlap_ratio(entity.last_bbox, anchor_bbox)
            if overlap < cfg.overlap_on_ratio:
                continue
            anchor_depth = anchor.get("depth_m")
            if (
                anchor_depth is not None
                and entity.depth_m is not None
                and abs(entity.depth_m - float(anchor_depth)) > cfg.depth_agree_m
            ):
                continue
            conditions[("on", entity.entity_key, anchor["anchor_id"])] = {
                "_object_label": anchor["anchor_id"],
                "overlap": round(overlap, 2),
                "posture": bbox_posture(entity.last_bbox, cfg),
                "arrangement": bbox_relation(entity.last_bbox, anchor_bbox),
                "depth_m": round(entity.depth_m, 2) if entity.depth_m is not None else None,
            }
            on_pairs.add((entity.entity_key, str(anchor["anchor_id"])))

    def _collect_near_anchor(
        self,
        entity: PresentEntity,
        anchors: list[dict[str, Any]],
        conditions: dict[tuple, dict[str, Any]],
        on_pairs: set[tuple[int, str]],
    ) -> None:
        cfg = self.config
        for anchor in anchors:
            anchor_bbox = anchor.get("bbox_xyxy")
            if not anchor_bbox:
                continue
            if (entity.entity_key, str(anchor["anchor_id"])) in on_pairs:
                continue  # `on` subsumes `near` for the same pair
            gap = bbox_gap_frac(entity.last_bbox, anchor_bbox, cfg.frame_diag)
            if gap > cfg.near_gap_frac:
                continue
            anchor_depth = anchor.get("depth_m")
            if (
                anchor_depth is not None
                and entity.depth_m is not None
                and abs(entity.depth_m - float(anchor_depth)) > cfg.near_depth_agree_m
            ):
                continue
            conditions[("near", entity.entity_key, anchor["anchor_id"])] = {
                "_object_label": anchor["anchor_id"],
                "gap_frac": round(gap, 3),
                "arrangement": bbox_relation(entity.last_bbox, anchor_bbox),
                "depth_m": round(entity.depth_m, 2) if entity.depth_m is not None else None,
            }

    def _collect_near_entities(
        self, entities: list[PresentEntity], conditions: dict[tuple, dict[str, Any]]
    ) -> None:
        cfg = self.config
        for i, a in enumerate(entities):
            for b in entities[i + 1 :]:
                # Symmetric pair: the lower key is the canonical subject.
                subject, obj = (a, b) if a.entity_key < b.entity_key else (b, a)
                gap = bbox_gap_frac(subject.last_bbox, obj.last_bbox, cfg.frame_diag)
                if gap > cfg.near_gap_frac:
                    continue
                if (
                    subject.depth_m is not None
                    and obj.depth_m is not None
                    and abs(subject.depth_m - obj.depth_m) > cfg.near_depth_agree_m
                ):
                    continue
                distance = entity_distance_m(subject, obj)
                conditions[("near", subject.entity_key, ("entity", obj.entity_key))] = {
                    "_object_key": obj.entity_key,
                    "gap_frac": round(gap, 3),
                    "arrangement": bbox_relation(subject.last_bbox, obj.last_bbox),
                    "distance_m": round(distance, 2) if distance is not None else None,
                }

    # -------------------------------------------------------------- moves

    def _evaluate_moves(
        self,
        entities: list[PresentEntity],
        labels: dict[int, str],
        anchors: list[dict[str, Any]],
        now: float,
    ) -> list[RelationTransition]:
        cfg = self.config
        threshold = cfg.move_min_frac * cfg.frame_diag
        transitions: list[RelationTransition] = []
        seen: set[int] = set()

        for entity in entities:
            seen.add(entity.entity_key)
            cx = (entity.last_bbox[0] + entity.last_bbox[2]) / 2
            cy = (entity.last_bbox[1] + entity.last_bbox[3]) / 2
            rest = self._rest_position.setdefault(entity.entity_key, (cx, cy))
            displaced = math.hypot(cx - rest[0], cy - rest[1]) > threshold
            if not displaced:
                self._pending_move.pop(entity.entity_key, None)
                continue
            since = self._pending_move.setdefault(entity.entity_key, now)
            if now - since < cfg.open_hold_s:
                continue
            del self._pending_move[entity.entity_key]
            details = {
                "from_zone": frame_zone(rest[0], cfg.frame_width),
                "to_zone": frame_zone(cx, cfg.frame_width),
                "shift_frac": round(
                    math.hypot(cx - rest[0], cy - rest[1]) / cfg.frame_diag, 3
                ),
                "depth_m": round(entity.depth_m, 2) if entity.depth_m is not None else None,
            }
            nearest = self._nearest_anchor(entity, anchors)
            if nearest is not None:
                details["nearest_anchor"] = nearest
            self._rest_position[entity.entity_key] = (cx, cy)
            transitions.append(
                RelationTransition(
                    action="point",
                    relation="moved",
                    subject_key=entity.entity_key,
                    subject_label=labels.get(entity.entity_key, "?"),
                    object_key=None,
                    object_label=None,
                    t_start=now,
                    t_end=now,
                    details=details,
                )
            )

        for key in [k for k in self._pending_move if k not in seen]:
            del self._pending_move[key]
        return transitions

    def _nearest_anchor(
        self, entity: PresentEntity, anchors: list[dict[str, Any]]
    ) -> str | None:
        best_id, best_gap = None, float("inf")
        for anchor in anchors:
            bbox = anchor.get("bbox_xyxy")
            if not bbox:
                continue
            gap = bbox_gap_frac(entity.last_bbox, bbox, self.config.frame_diag)
            if gap < best_gap:
                best_id, best_gap = str(anchor["anchor_id"]), gap
        return best_id

    # ------------------------------------------------------------ closing

    def entity_left(self, entity_key: int, now: float) -> list[RelationTransition]:
        """Force-close everything involving an entity that left the scene."""
        transitions = []
        for key in [
            k
            for k, a in self._active.items()
            if a.subject_key == entity_key or a.object_key == entity_key
        ]:
            transitions.append(self._close(key, now))
        for pending in (self._pending_open, self._pending_close):
            for key in [
                k for k in pending if k[1] == entity_key or k[2] == ("entity", entity_key)
            ]:
                del pending[key]
        self._rest_position.pop(entity_key, None)
        self._pending_move.pop(entity_key, None)
        return transitions

    def force_close_all(self, now: float) -> list[RelationTransition]:
        return [self._close(key, now) for key in list(self._active.keys())]

    def _close(self, key: tuple, t_end: float) -> RelationTransition:
        active = self._active.pop(key)
        self._pending_close.pop(key, None)
        details = dict(active.details)
        if active.posture_votes:
            details["posture"] = max(active.posture_votes.items(), key=lambda kv: kv[1])[0]
        details["duration_s"] = round(t_end - active.t_start, 1)
        return RelationTransition(
            action="close",
            relation=active.relation,
            subject_key=active.subject_key,
            subject_label=active.subject_label,
            object_key=active.object_key,
            object_label=active.object_label,
            t_start=active.t_start,
            t_end=t_end,
            details=details,
            active=active,
        )
