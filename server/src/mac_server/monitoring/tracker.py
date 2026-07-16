"""Greedy multi-object tracker for a fixed camera.

Consumes enriched object dicts (class, confidence, bbox_xyxy, depth_median_m,
lateral_m, forward_m) at ~8-15 Hz and maintains short-term identities.

Design notes:
- No Kalman filter: with a fixed camera and indoor subjects at this tick
  rate, frame-to-frame IoU of the same object is high; the `lost` state
  (holding the last bbox as an association candidate) covers occlusions.
- Association is greedy per class: IoU pass first, then a centroid-distance
  pass for leftovers (fast movers / bbox jitter).
- Tentative tracks that miss a single tick are dropped — kills YOLO flicker.

Pure Python (math/deque only) — unit-testable with scripted streams.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackerConfig:
    min_confidence: float = 0.35
    iou_gate: float = 0.1
    centroid_gate_frac: float = 0.15
    confirm_hits: int = 3
    lost_after_s: float = 2.0
    end_after_s: float = 10.0
    history_len: int = 64
    ema_alpha: float = 0.3
    frame_width: float = 1280.0
    frame_height: float = 720.0

    @property
    def frame_diag(self) -> float:
        return math.hypot(self.frame_width, self.frame_height)


@dataclass
class Track:
    track_id: int
    class_name: str
    state: str  # tentative | confirmed | lost | ended
    first_seen: float
    last_seen: float
    hits: int
    bbox: tuple[float, float, float, float]
    bbox_history: deque = field(default_factory=deque)
    centroid_history: deque = field(default_factory=deque)
    depth_m: float | None = None
    lateral_m: float | None = None
    forward_m: float | None = None
    confidence: float = 0.0
    entity_id: int | None = None
    entity_label: str | None = None
    signature: dict[str, Any] | None = None
    signature_updated_at: float = 0.0
    db_id: int | None = None  # row id in the tracks table

    def snapshot(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class": self.class_name,
            "state": self.state,
            "bbox_xyxy": list(self.bbox),
            "depth_m": round(self.depth_m, 2) if self.depth_m is not None else None,
            "lateral_m": round(self.lateral_m, 2) if self.lateral_m is not None else None,
            "forward_m": round(self.forward_m, 2) if self.forward_m is not None else None,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "entity_label": self.entity_label,
        }


@dataclass
class TrackUpdates:
    confirmed_new: list[Track] = field(default_factory=list)  # just became confirmed
    ended: list[Track] = field(default_factory=list)  # just ended


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def is_stationary(
    centroid_history: Any,
    now: float,
    window_s: float,
    px_frac: float,
    frame_diag: float,
) -> bool | None:
    """True when max centroid displacement within the window is below
    px_frac * frame_diag. None when the window has too few samples."""
    recent = [(t, cx, cy) for (t, cx, cy) in centroid_history if now - t <= window_s]
    if len(recent) < 3 or (recent[-1][0] - recent[0][0]) < window_s * 0.6:
        return None
    xs = [cx for _, cx, _ in recent]
    ys = [cy for _, _, cy in recent]
    spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return spread <= px_frac * frame_diag


class GreedyTracker:
    """Tracks every YOLO class EXCEPT `excluded_classes` (furniture/anchor
    classes — those are handled as frozen reference points instead, see
    scenes.py). No hand-picked allowlist: whatever the detector tags gets
    tracked, native to how YOLO already works."""

    def __init__(self, config: TrackerConfig, excluded_classes: set[str]) -> None:
        self.config = config
        self.excluded_classes = excluded_classes
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list[dict[str, Any]], now: float) -> TrackUpdates:
        cfg = self.config
        updates = TrackUpdates()

        candidates = [
            det
            for det in detections
            if det.get("class") not in self.excluded_classes
            and float(det.get("confidence", 0.0)) >= cfg.min_confidence
            and isinstance(det.get("bbox_xyxy"), list)
            and len(det["bbox_xyxy"]) == 4
        ]

        alive = [t for t in self.tracks.values() if t.state != "ended"]
        unmatched_dets = list(range(len(candidates)))
        unmatched_tracks = {t.track_id for t in alive}

        # Pass 1: greedy IoU per class.
        pairs: list[tuple[float, int, int]] = []
        for track in alive:
            for di in unmatched_dets:
                det = candidates[di]
                if det["class"] != track.class_name:
                    continue
                overlap = iou(track.bbox, tuple(det["bbox_xyxy"]))
                if overlap >= cfg.iou_gate:
                    pairs.append((overlap, track.track_id, di))
        pairs.sort(reverse=True)
        matched_dets: set[int] = set()
        for overlap, track_id, di in pairs:
            if track_id not in unmatched_tracks or di in matched_dets:
                continue
            self._apply_match(self.tracks[track_id], candidates[di], now, updates)
            unmatched_tracks.discard(track_id)
            matched_dets.add(di)

        # Pass 2: centroid distance for leftovers.
        gate_px = cfg.centroid_gate_frac * cfg.frame_diag
        pairs2: list[tuple[float, int, int]] = []
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            tcx, tcy = centroid(track.bbox)
            for di in unmatched_dets:
                if di in matched_dets:
                    continue
                det = candidates[di]
                if det["class"] != track.class_name:
                    continue
                dcx, dcy = centroid(tuple(det["bbox_xyxy"]))
                dist = math.hypot(dcx - tcx, dcy - tcy)
                if dist <= gate_px:
                    pairs2.append((dist, track_id, di))
        pairs2.sort()
        for dist, track_id, di in pairs2:
            if track_id not in unmatched_tracks or di in matched_dets:
                continue
            self._apply_match(self.tracks[track_id], candidates[di], now, updates)
            unmatched_tracks.discard(track_id)
            matched_dets.add(di)

        # New tentative tracks for unmatched detections.
        for di in range(len(candidates)):
            if di in matched_dets:
                continue
            det = candidates[di]
            track = Track(
                track_id=self._next_id,
                class_name=str(det["class"]),
                state="tentative",
                first_seen=now,
                last_seen=now,
                hits=1,
                bbox=tuple(det["bbox_xyxy"]),
                bbox_history=deque(maxlen=self.config.history_len),
                centroid_history=deque(maxlen=self.config.history_len),
                confidence=float(det.get("confidence", 0.0)),
            )
            self._record_position(track, det, now)
            self.tracks[self._next_id] = track
            self._next_id += 1

        # Age out unmatched tracks.
        for track_id in list(unmatched_tracks):
            track = self.tracks[track_id]
            if track.state == "tentative":
                # A tentative track missing even one tick is flicker — drop.
                del self.tracks[track_id]
            elif track.state == "confirmed":
                if now - track.last_seen > cfg.lost_after_s:
                    track.state = "lost"
            elif track.state == "lost":
                if now - track.last_seen > cfg.end_after_s:
                    track.state = "ended"
                    updates.ended.append(track)
                    del self.tracks[track_id]

        return updates

    def _apply_match(
        self, track: Track, det: dict[str, Any], now: float, updates: TrackUpdates
    ) -> None:
        track.bbox = tuple(det["bbox_xyxy"])
        track.confidence = float(det.get("confidence", 0.0))
        track.last_seen = now
        track.hits += 1
        self._record_position(track, det, now)
        if track.state == "tentative" and track.hits >= self.config.confirm_hits:
            track.state = "confirmed"
            updates.confirmed_new.append(track)
        elif track.state == "lost":
            track.state = "confirmed"

    def _record_position(self, track: Track, det: dict[str, Any], now: float) -> None:
        alpha = self.config.ema_alpha
        track.bbox_history.append((now, track.bbox))
        track.centroid_history.append((now, *centroid(track.bbox)))
        for attr, key in (
            ("depth_m", "depth_median_m"),
            ("lateral_m", "lateral_m"),
            ("forward_m", "forward_m"),
        ):
            value = det.get(key)
            if value is None:
                continue
            value = float(value)
            current = getattr(track, attr)
            setattr(track, attr, value if current is None else alpha * value + (1 - alpha) * current)

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.state in {"confirmed", "lost"}]

    def force_end_all(self, now: float) -> list[Track]:
        """End every live track (monitoring stop). Returns confirmed ones."""
        ended = []
        for track in list(self.tracks.values()):
            if track.state in {"confirmed", "lost"}:
                track.state = "ended"
                track.last_seen = min(track.last_seen, now)
                ended.append(track)
        self.tracks.clear()
        return ended
