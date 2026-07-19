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
    }
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


def _plan_location(entry: dict[str, Any], position, forward) -> dict[str, Any]:
    """Project a world pose into the session's floor-plan frame: fractional
    x/y on the plan image + heading in plan degrees (0 = plan +x, CCW)."""
    import numpy as np

    plan = entry.get("plan_frame")
    out: dict[str, Any] = {}
    if not plan:
        return out
    a, up, b = np.asarray(plan["axis_a"]), np.asarray(plan["up"]), np.asarray(plan["axis_b"])
    ox, oz = plan["origin"]
    res = float(plan["resolution_m"])
    fwd_h = forward - (forward @ up) * up
    norm = np.linalg.norm(fwd_h)
    if norm > 1e-6:
        fwd_h = fwd_h / norm
        out["heading_deg"] = round(math.degrees(math.atan2(float(fwd_h @ b), float(fwd_h @ a))), 1)
    if plan.get("grid_w"):
        out["plan_frac"] = [
            round(((float(position @ a)) - ox) / res / plan["grid_w"], 4),
            round(((float(position @ b)) - oz) / res / plan["grid_h"], 4),
        ]
    return out


def query(sessions_dir: Path, embedding: list[float], top_k: int = 3) -> dict[str, Any]:
    """Best-matching room + pose for a live embedding. Thread-safe."""
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
            if entry is None or entry["embs"].shape[1] != emb.shape[0]:
                continue
            sims = entry["embs"] @ emb
            best = int(np.argmax(sims))
            candidates.append((float(sims[best]), session_dir.name, best, entry))

    if not candidates:
        return {"located": False, "reason": "no place indexes built (run the navindex step)"}

    candidates.sort(reverse=True, key=lambda c: c[0])
    matches = []
    for sim, session_id, best, entry in candidates[:top_k]:
        match = {
            "session_id": session_id,
            "name": entry["name"],
            "keyframe": int(entry["kf"][best]),
            "similarity": round(sim, 4),
        }
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
