"""Runtime place-recognition: match a live CLIP embedding from the Pi
against every session's place_index (built by navindex_step).

Pure numpy at query time — one cosine sweep over a few hundred vectors per
room, microseconds. Indexes are cached per session and reloaded when the
npz mtime changes (e.g. after a pipeline re-run). The last fix is kept for
the panel (GET /api/nav/last draws it on the floor plan).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: dict[str, dict[str, Any]] = {}  # session_id -> {mtime, data...}
last_result: dict[str, Any] | None = None


def _load_session_index(session_dir: Path) -> dict[str, Any] | None:
    import numpy as np

    npz_path = session_dir / "derived" / "place_index.npz"
    if not npz_path.exists():
        return None
    mtime = npz_path.stat().st_mtime
    cached = _cache.get(session_dir.name)
    if cached is not None and cached["mtime"] == mtime:
        return cached

    data = np.load(npz_path)
    entry: dict[str, Any] = {
        "mtime": mtime,
        "kf": data["kf"],
        "embs": data["embs"].astype(np.float32),  # (N, D), L2-normalized
        "positions": data["positions"],
        "forwards": data["forwards"],
        "plan_frame": None,
        "name": None,
        "stale": False,
    }
    # Refuse to localize against an index built for different geometry. The
    # alternative is what actually happened: positions from a 17x-larger world
    # reported against a 4 m floor plan, with the marker quietly off-canvas.
    built = data["built_scale"][0] if "built_scale" in data.files else None
    poses_path = session_dir / "derived" / "poses.json"
    if built is not None and built > 0 and poses_path.exists():
        try:
            current = json.loads(poses_path.read_text(encoding="utf-8")).get("scale")
        except (OSError, ValueError):
            current = None
        if current and abs(float(current) - float(built)) > 0.01 * float(current):
            logger.warning(
                "place_index for %s was built at scale %.4f but poses.json is now "
                "%.4f — re-run the navindex step; localization is disabled for this "
                "session until then.", session_dir.name, float(built), float(current))
            entry["stale"] = True
    frame_path = session_dir / "derived" / "plan_frame.json"
    if frame_path.exists():
        try:
            entry["plan_frame"] = json.loads(frame_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    meta_path = session_dir / "session_meta.json"
    if meta_path.exists():
        try:
            entry["name"] = json.loads(meta_path.read_text(encoding="utf-8")).get("name")
        except (OSError, ValueError):
            pass
    _cache[session_dir.name] = entry
    logger.info("Loaded place index %s (%d frames)", session_dir.name, len(entry["kf"]))
    return entry


def _heading_deg(plan: dict[str, Any], forward) -> float | None:
    """Camera +Z projected onto the floor plan's horizontal axes -> degrees,
    0 = plan +x, counterclockwise."""
    import numpy as np

    up = np.asarray(plan["up"])
    a, b = np.asarray(plan["axis_a"]), np.asarray(plan["axis_b"])
    fwd_h = forward - (forward @ up) * up
    norm = float(np.linalg.norm(fwd_h))
    if norm <= 1e-6:
        return None
    fwd_h = fwd_h / norm
    return math.degrees(math.atan2(float(fwd_h @ b), float(fwd_h @ a)))


def _plan_frac(plan: dict[str, Any], position) -> list[float] | None:
    if not plan.get("grid_w"):
        return None
    import numpy as np

    a, b = np.asarray(plan["axis_a"]), np.asarray(plan["axis_b"])
    ox, oz = plan["origin"]
    res = float(plan["resolution_m"])
    return [
        round(((float(position @ a)) - ox) / res / plan["grid_w"], 4),
        round(((float(position @ b)) - oz) / res / plan["grid_h"], 4),
    ]


def _plan_position_m(plan: dict[str, Any], position) -> list[float]:
    """Position along (axis_a, axis_b) in METRES, not `plan_frac`'s grid-
    normalized fraction. The two plan_frac components are independently
    normalized by grid_w and grid_h, which are only equal-scale when the
    room is square — a live consumer that needs Euclidean math (the EKF's
    odometry model, which assumes an isotropic metre-space) must use this,
    not plan_frac. plan_frac stays as the panel's display convention."""
    import numpy as np

    a, b = np.asarray(plan["axis_a"]), np.asarray(plan["axis_b"])
    return [round(float(position @ a), 4), round(float(position @ b), 4)]


def fused_to_plan(
    sessions_dir: Path,
    session_id: str,
    position_m: list[float],
    heading_deg: float | None = None,
    std_m: float | None = None,
) -> dict[str, Any] | None:
    """An EKF fused state (metres along axis_a/axis_b) -> the same
    `plan_frac` display convention the panel already draws the raw visual
    fix with, so both can be shown on one floor plan.

    Inverse of `_plan_position_m` composed with `_plan_frac`: the filter
    works in metres (isotropic, which its motion model requires) while the
    panel positions markers by grid fraction, and the two are only equal up
    to grid_w/grid_h — which differ whenever the room isn't square.
    """
    with _lock:
        entry = _load_session_index(sessions_dir / session_id)
    if entry is None or entry.get("stale"):
        return None
    plan = entry.get("plan_frame")
    if not plan or not plan.get("grid_w"):
        return None
    ox, oz = plan["origin"]
    res = float(plan["resolution_m"])
    out: dict[str, Any] = {
        "plan_frac": [
            round((float(position_m[0]) - ox) / res / plan["grid_w"], 4),
            round((float(position_m[1]) - oz) / res / plan["grid_h"], 4),
        ],
        "position_m": [round(float(v), 4) for v in position_m[:2]],
    }
    if heading_deg is not None:
        out["heading_deg"] = round(float(heading_deg), 1)
    if std_m is not None:
        # Radius in grid fractions so the panel can size an uncertainty
        # ellipse without knowing the metric scale. Both axes reported
        # because a non-square grid stretches them differently.
        out["std_m"] = round(float(std_m), 3)
        out["std_frac"] = [
            round(float(std_m) / res / plan["grid_w"], 4),
            round(float(std_m) / res / plan["grid_h"], 4),
        ]
    return out


def _plan_location(entry: dict[str, Any], position, forward) -> dict[str, Any]:
    """Project a single stored pose into the session's floor-plan frame —
    used for the (cheap, single-nearest-neighbor) alternative-room matches."""
    plan = entry.get("plan_frame")
    if not plan:
        return {}
    out: dict[str, Any] = {}
    heading = _heading_deg(plan, forward)
    if heading is not None:
        out["heading_deg"] = round(heading, 1)
    out["position_m"] = _plan_position_m(plan, position)
    frac = _plan_frac(plan, position)
    if frac is not None:
        out["plan_frac"] = frac
    return out


def _circular_weighted_mean_deg(headings_deg, weights) -> tuple[float, float]:
    """Similarity-weighted circular mean + spread (0 = neighbors fully
    agree on facing direction, larger = they disagree — an honest
    confidence signal, since naive linear averaging of angles is wrong
    across the 0/360 wrap and would silently produce nonsense)."""
    import numpy as np

    rad = np.radians(headings_deg)
    w = weights / (weights.sum() or 1.0)
    x = float((w * np.cos(rad)).sum())
    y = float((w * np.sin(rad)).sum())
    mean_deg = math.degrees(math.atan2(y, x))
    resultant = math.hypot(x, y)  # 1.0 = perfect agreement, -> 0 = scattered
    spread_deg = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(resultant, 1e-6)))))
    return round(mean_deg, 1), round(min(spread_deg, 180.0), 1)


def _refine_winner(entry: dict[str, Any], emb, k: int = 7) -> dict[str, Any]:
    """The single nearest-neighbor pose is noisy: CLIP embeddings are
    deliberately somewhat viewpoint-invariant (that's what makes them good
    for "which room", but it means the ONE best match's stored heading can
    easily be facing a different way than the live camera). Average the
    pose of the top-k most similar keyframes IN THIS SESSION instead,
    weighted by similarity, and report how much they agree
    (heading_spread_deg) so a genuinely ambiguous fix is visibly uncertain
    rather than silently wrong."""
    import numpy as np

    plan = entry.get("plan_frame")
    if not plan:
        return {}

    sims = entry["embs"] @ emb
    order = np.argsort(-sims)[:k]
    top_sim = float(sims[order[0]])
    # Drop neighbors that are meaningfully less similar than the best one —
    # they're a different spot, not a noisy re-observation of this one.
    order = [i for i in order if sims[i] >= top_sim - 0.05]
    weights = np.clip(sims[order].astype(np.float64), 1e-6, None)

    out: dict[str, Any] = {}
    headings, hweights = [], []
    pos_acc = np.zeros(3, dtype=np.float64)
    for idx, wgt in zip(order, weights):
        heading = _heading_deg(plan, entry["forwards"][idx])
        if heading is not None:
            headings.append(heading)
            hweights.append(wgt)
        pos_acc += entry["positions"][idx].astype(np.float64) * wgt

    if headings:
        mean_h, spread_h = _circular_weighted_mean_deg(np.array(headings), np.array(hweights))
        out["heading_deg"] = mean_h
        out["heading_spread_deg"] = spread_h
    avg_pos = pos_acc / weights.sum()
    out["position_m"] = _plan_position_m(plan, avg_pos)
    frac = _plan_frac(plan, avg_pos)
    if frac is not None:
        out["plan_frac"] = frac
    out["neighbors_used"] = len(order)
    return out


def query(
    sessions_dir: Path, embedding: list[float], top_k: int = 3, intra_k: int = 7
) -> dict[str, Any]:
    """Best-matching room + pose for a live embedding. Thread-safe.

    `top_k` ranks candidate ROOMS (one nearest keyframe each, for the
    "alternatives" list — cheap disambiguation). Within the winning room,
    `intra_k` nearest keyframes are aggregated for a steadier heading/
    position estimate (see `_refine_winner`)."""
    import numpy as np

    global last_result
    emb = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    if norm <= 0:
        raise ValueError("Zero embedding")
    emb = emb / norm

    candidates = []
    with _lock:
        for session_dir in sorted(p for p in sessions_dir.iterdir() if p.is_dir()):
            entry = _load_session_index(session_dir)
            if entry is None or entry.get("stale"):
                # Built against different geometry — its coordinates would be
                # confidently wrong, which is worse than reporting no fix.
                continue
            if entry["embs"].shape[1] != emb.shape[0]:
                continue
            sims = entry["embs"] @ emb
            best = int(np.argmax(sims))
            candidates.append((float(sims[best]), session_dir.name, best, entry))

    if not candidates:
        return {"located": False, "reason": "no place indexes built (run the navindex step)"}

    candidates.sort(reverse=True, key=lambda c: c[0])
    matches = []
    for i, (sim, session_id, best, entry) in enumerate(candidates[:top_k]):
        match = {
            "session_id": session_id,
            "name": entry["name"],
            "keyframe": int(entry["kf"][best]),
            "similarity": round(sim, 4),
        }
        if i == 0:
            match.update(_refine_winner(entry, emb, k=intra_k))
        else:
            match.update(_plan_location(entry, entry["positions"][best], entry["forwards"][best]))
        matches.append(match)

    result = {
        "located": True,
        "best": matches[0],
        "alternatives": matches[1:],
        "at": time.time(),
    }
    with _lock:
        last_result = result
    return result


def get_last() -> dict[str, Any] | None:
    with _lock:
        return last_result
