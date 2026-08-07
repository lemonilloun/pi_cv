"""Step 5: open-vocabulary labels + scene graph.

Labels: each object's CLIP embedding (averaged from the Pi's on-NPU
RN50x4 embeddings, or computed here as a fallback) against text embeddings
of an indoor vocabulary. IMPORTANT: the text tower must be the SAME CLIP
variant as the Pi hef — RN50x4 with OpenAI weights (open_clip
model='RN50x4', pretrained='openai'). The COCO class vote from the Pi is a
prior: agreement with the CLIP top-3 boosts confidence.

Edges (geometric heuristics over OBBs/centroids):
    near(A,B)  centroid distance < near_max_m
    on(A,B)    A's bottom within on_gap_m of B's top + horizontal overlap
    in_room    every object -> the room node

Outputs scene_graph.json + a labeled floor plan PNG.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from mac_server.scene3d.session_io import SceneSession


logger = logging.getLogger(__name__)

INDOOR_VOCAB = [
    "chair", "armchair", "office chair", "sofa", "couch", "table", "desk",
    "coffee table", "dining table", "bed", "pillow", "blanket", "monitor",
    "computer monitor", "laptop", "keyboard", "computer mouse", "tv",
    "television", "lamp", "floor lamp", "ceiling light", "plant",
    "potted plant", "flower pot", "book", "bookshelf", "shelf", "cabinet",
    "wardrobe", "drawer", "door", "window", "curtain", "mirror",
    "picture frame", "painting", "clock", "vase", "bottle", "cup",
    "mug", "glass", "plate", "bowl", "backpack", "bag", "shoe", "jacket",
    "guitar", "piano", "speaker", "headphones", "phone", "smartphone",
    "tablet", "camera", "printer", "router", "fan", "heater",
    "air conditioner", "refrigerator", "microwave", "oven", "sink",
    "trash can", "box", "carpet", "rug", "person", "cat", "dog",
]


def label_objects(
    objects: list[dict[str, Any]],
    text_embs: dict[str, list[float]],
    coco_boost: float = 0.1,
) -> list[dict[str, Any]]:
    """Pure logic: cosine object clip_emb vs vocabulary; top-3 with COCO
    prior boost when the Pi's class agrees."""
    import numpy as np

    words = list(text_embs)
    matrix = np.asarray([text_embs[w] for w in words])  # (V, D) normalized
    labeled = []
    for obj in objects:
        entry = dict(obj)
        emb = obj.get("clip_emb")
        if emb is None:
            entry["labels"] = [{"label": obj.get("class_top", "?"), "score": None}]
            entry["label_top"] = obj.get("class_top", "?")
            labeled.append(entry)
            continue
        sims = matrix @ np.asarray(emb)
        coco = str(obj.get("class_top", ""))
        boosted = sims.copy()
        for i, word in enumerate(words):
            if coco and (coco in word or word in coco):
                boosted[i] += coco_boost
        top = np.argsort(-boosted)[:3]
        entry["labels"] = [
            {"label": words[i], "score": round(float(sims[i]), 4)} for i in top
        ]
        entry["label_top"] = words[top[0]]
        labeled.append(entry)
    return labeled


def obb_top_bottom(obb: dict[str, Any], up) -> tuple[float, float]:
    """Object extent along the up axis from its OBB corners."""
    import numpy as np

    center = np.asarray(obb["center"])
    extent = np.asarray(obb["extent"]) / 2.0
    rotation = np.asarray(obb["rotation"])
    signs = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    corners = center + (signs * extent) @ rotation.T
    heights = corners @ np.asarray(up)
    return float(heights.max()), float(heights.min())


def obb_support(obb: dict[str, Any], direction) -> float:
    """Half-width of the box along a unit direction (its support function)."""
    import numpy as np

    rotation = np.asarray(obb["rotation"])  # columns are the box axes
    half = np.asarray(obb["extent"], dtype=np.float64) / 2.0
    return float(np.sum(np.abs(rotation.T @ np.asarray(direction)) * half))


def obb_gap_m(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Distance between two boxes' *surfaces* along the line joining them.

    Centroid distance is the wrong measure for furniture: a lamp standing
    against a 2 m sofa is 1 m from its centre and touching its side. With a
    0.5 m centroid threshold that room produced 1 and 4 `near` edges for 8
    and 10 objects — almost no graph at all. Surface gap is what "near"
    means to anyone reading the graph.
    """
    import numpy as np

    ca = np.asarray(a["centroid"], dtype=np.float64)
    cb = np.asarray(b["centroid"], dtype=np.float64)
    delta = cb - ca
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        return 0.0
    if not a.get("obb") or not b.get("obb"):
        return length
    direction = delta / length
    return max(0.0, length - obb_support(a["obb"], direction)
               - obb_support(b["obb"], direction))


def horizontal_half_extent(obb: dict[str, Any]) -> float:
    """Half-footprint of a gravity-aligned box (its first two axes are
    horizontal by construction; the third is vertical)."""
    extent = obb["extent"]
    return max(float(extent[0]), float(extent[1])) / 2.0


def build_edges(
    objects: list[dict[str, Any]],
    up,
    near_max_m: float = 0.5,
    on_gap_m: float = 0.08,
    skip_implausible: bool = True,
) -> list[dict[str, Any]]:
    import numpy as np

    edges = []
    up = np.asarray(up)
    usable = [
        o for o in objects
        if not (skip_implausible and o.get("size_verdict") in ("too_large", "too_small"))
    ]
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            gap = obb_gap_m(a, b)
            if gap < near_max_m:
                edges.append(
                    {
                        "src": a["object_id"], "dst": b["object_id"],
                        "relation": "near", "distance_m": round(gap, 2),
                    }
                )
    for a in usable:
        for b in usable:
            if a is b or not a.get("obb") or not b.get("obb"):
                continue
            _, a_bottom = obb_top_bottom(a["obb"], up)
            b_top, _ = obb_top_bottom(b["obb"], up)
            offset = np.asarray(a["centroid"]) - np.asarray(b["centroid"])
            horizontal = float(np.linalg.norm(offset - (offset @ up) * up))
            if (abs(a_bottom - b_top) <= on_gap_m
                    and horizontal <= horizontal_half_extent(b["obb"])):
                edges.append(
                    {"src": a["object_id"], "dst": b["object_id"], "relation": "on"}
                )
    return edges


def run_graph_step(
    session: SceneSession,
    config: dict[str, Any],
    repo_root: Path,
    progress: Callable[[str], None] = lambda msg: None,
) -> dict[str, Any]:
    import numpy as np

    graph_cfg = config.get("graph", {})
    emb_cfg = config.get("embeddings", {})
    objects_data = json.loads(session.objects_path().read_text(encoding="utf-8"))
    objects = objects_data["objects"]

    # ------------------------------------------------------ CLIP labels
    # MUST stay RN50x4/openai regardless of backend — this is compared
    # against clip_emb vectors the Pi already computed on-device via its
    # Hailo hef (see module docstring above).
    progress("computing CLIP text embeddings")
    vocab = list(dict.fromkeys(INDOOR_VOCAB))
    text_embs: dict[str, list[float]] = {}
    try:
        if str(emb_cfg.get("backend", "remote")).lower() == "remote":
            from mac_server.scene3d.gpu_client import remote_clip_text_embed

            vectors = remote_clip_text_embed(
                str(emb_cfg.get("gpu_service_url", "http://127.0.0.1:8700")),
                [f"a photo of a {w}" for w in vocab],
            )
        else:
            import open_clip
            import torch

            model_name = str(graph_cfg.get("clip_text_model", "RN50x4"))
            pretrained = str(graph_cfg.get("clip_text_pretrained", "openai"))
            model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            tokenizer = open_clip.get_tokenizer(model_name)
            model.eval()
            with torch.no_grad():
                tokens = tokenizer([f"a photo of a {w}" for w in vocab])
                features = model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            vectors = [[float(v) for v in vec] for vec in features.float().numpy()]
        text_embs = dict(zip(vocab, vectors))
    except Exception as exc:
        logger.warning("CLIP text encoder unavailable (%s) — COCO labels only", exc)

    labeled = label_objects(objects, text_embs) if text_embs else [
        {**o, "labels": [{"label": o.get("class_top", "?"), "score": None}],
         "label_top": o.get("class_top", "?")}
        for o in objects
    ]

    # -------------------------------------------------------------- edges
    plan_frame = {}
    frame_path = session.derived / "plan_frame.json"
    if frame_path.exists():
        plan_frame = json.loads(frame_path.read_text(encoding="utf-8"))
    up = plan_frame.get("up", [0.0, -1.0, 0.0])

    edges = build_edges(
        labeled, up,
        near_max_m=float(graph_cfg.get("near_max_m", 0.5)),
        on_gap_m=float(graph_cfg.get("on_gap_m", 0.08)),
    )

    graph = {
        "session_id": session.session_id,
        "nodes": [
            {
                "object_id": o["object_id"],
                "label": o["label_top"],
                "labels": o["labels"],
                "class_coco": o.get("class_top"),
                "centroid": o["centroid"],
                "obb": o.get("obb"),
                "n_observations": o["n_observations"],
            }
            for o in labeled
        ],
        "edges": edges + [
            {"src": o["object_id"], "dst": "room", "relation": "in_room"}
            for o in labeled
        ],
    }
    session.graph_path().write_text(json.dumps(graph, indent=1), encoding="utf-8")

    self_plan = _draw_labeled_plan(session, labeled, plan_frame)
    report = {
        "objects": len(labeled),
        "labels": [o["label_top"] for o in labeled],
        "edges": len(edges),
        "plan": self_plan,
    }
    logger.info("Graph step done: %s", report)
    return report


def _draw_labeled_plan(
    session: SceneSession, objects: list[dict[str, Any]], plan_frame: dict[str, Any]
) -> bool:
    """Objects drawn onto the floor plan (returns False when no plan)."""
    import cv2
    import numpy as np

    if not session.plan_path().exists() or not plan_frame:
        return False
    plan = cv2.imread(str(session.plan_path()))
    if plan is None:
        return False
    a = np.asarray(plan_frame["axis_a"])
    b = np.asarray(plan_frame["axis_b"])
    ox, oz = plan_frame["origin"]
    res = float(plan_frame["resolution_m"])
    h, w = plan.shape[:2]
    scale = 3
    plan = cv2.resize(plan, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    for obj in objects:
        c = np.asarray(obj["centroid"])
        px = int((float(c @ a) - ox) / res) * scale
        pz = int((float(c @ b) - oz) / res) * scale
        if not (0 <= px < w * scale and 0 <= pz < h * scale):
            continue
        cv2.circle(plan, (px, pz), 6, (0, 200, 255), -1)
        cv2.putText(
            plan, obj["label_top"], (px + 8, pz + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA,
        )
    cv2.imwrite(str(session.plan_labeled_path()), plan)
    return True
