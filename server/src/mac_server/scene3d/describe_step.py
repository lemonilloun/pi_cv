"""Step: turn each reconstructed object into something a policy can read —
a picture, a place, and a sentence.

`objects.json` says `{"class_top": "cabinet", "centroid": [2.1, -0.4, 0.55]}`.
That is enough to draw a dot on a plan and nothing like enough to act on: a
robot told "cabinet at 2.1, -0.4" cannot tell the tall white wardrobe by the
door from the low chest under the window, and neither can a VLA policy reading
the same line. This step attaches the two things that close that gap — the
crops the object was actually seen in, and a description written from them.

Three decisions worth stating, because each has a cheaper wrong version:

- **Crops are chosen, not taken.** The first observation of an object is
  usually its worst: the object is entering the frame, half of it is cut off.
  Views are ranked by mask area and penalised for touching the frame border,
  so what reaches the model is the view where the object is most fully
  visible.
- **The instance is outlined in the crop.** A crop of a cabinet in a room is
  also a crop of the wall, the floor and whatever sits on it, and a VLM asked
  to describe it will happily describe the room. A contour drawn round the
  mask is the difference between "a bedroom" and "a white two-door wardrobe".
- **Failure is a null, not an exception.** The vLLM belongs to another user of
  the GPU host. If it is down the pipeline must still produce objects with
  positions and pictures — a scene with no sentences is degraded, a scene that
  failed to build is useless.

Descriptions are cached per object under `derived/descriptions/`, so re-running
the step after a crash costs nothing and a deliberate re-description is an
explicit `--force`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mac_server.scene3d.session_io import SceneSession

logger = logging.getLogger(__name__)

# Asking for JSON and asking for prose are different tasks; this asks for both
# in a shape that survives a model that ignores half the instruction.
DESCRIBE_PROMPT = (
    "You are looking at {n} photo(s) of ONE object in a room, outlined in "
    "green. A detector labelled it \"{label}\"{hedge}.\n"
    "Reply with STRICT JSON and nothing else:\n"
    '{{"name": "<2-4 words, what it actually is>", '
    '"description": "<one sentence: colour, material, size, distinguishing '
    'features>", '
    '"label_ok": <true if the detector label fits, false if it is wrong>}}'
)

SAME_OBJECT_PROMPT = (
    "Two photos, each showing one object outlined in green. Are they the SAME "
    "physical object seen from different angles, or two different objects?\n"
    "The camera moved between shots, so lighting and viewpoint differ; judge "
    "by what the object IS, not by the background.\n"
    'Reply with STRICT JSON only: {"same": true|false, "why": "<5 words>"}'
)

MAX_VIEWS_PER_OBJECT = 3
CROP_MARGIN_FRAC = 0.15
CONTOUR_COLOR_BGR = (0, 255, 0)


def rank_views(views: list[dict[str, Any]], frame_shape: tuple[int, int],
               edge_tol_px: float = 4.0) -> list[dict[str, Any]]:
    """Best-first ordering of an object's observations.

    Score is mask area, halved when the box touches a frame border. A clipped
    view of a big object can easily have more visible pixels than a complete
    view of it from further away, and describing the half you can see as if it
    were the whole thing is exactly the error to avoid.
    """
    height, width = float(frame_shape[0]), float(frame_shape[1])

    def score(view: dict[str, Any]) -> float:
        area = float(view.get("mask_area", 0))
        bbox = view.get("bbox_xyxy") or [0, 0, 0, 0]
        x0, y0, x1, y1 = (float(v) for v in bbox)
        touching = (
            x0 <= edge_tol_px or y0 <= edge_tol_px
            or x1 >= width - edge_tol_px or y1 >= height - edge_tol_px
        )
        return area * (0.5 if touching else 1.0)

    return sorted(views, key=score, reverse=True)


def crop_with_outline(bgr, mask, bbox_xyxy, margin_frac: float = CROP_MARGIN_FRAC):
    """Crop around the object with its instance outlined.

    The outline is drawn BEFORE cropping so it follows the true mask boundary,
    then the crop is taken with a margin — context helps the model judge scale
    ("a wardrobe" vs "a doll's wardrobe") while the contour keeps it clear
    which thing is being asked about.
    """
    import cv2
    import numpy as np

    annotated = bgr.copy()
    if mask is not None and mask.any():
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        if mask_u8.shape[:2] != bgr.shape[:2]:
            mask_u8 = cv2.resize(mask_u8, (bgr.shape[1], bgr.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, CONTOUR_COLOR_BGR, 2)

    height, width = bgr.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in bbox_xyxy)
    mx, my = (x1 - x0) * margin_frac, (y1 - y0) * margin_frac
    x0 = int(max(0, x0 - mx)); y0 = int(max(0, y0 - my))
    x1 = int(min(width, x1 + mx)); y1 = int(min(height, y1 + my))
    if x1 <= x0 or y1 <= y0:
        return None
    return annotated[y0:y1, x0:x1]


def parse_json_reply(text: str | None) -> dict[str, Any] | None:
    """Defensive parse of a 'STRICT JSON' reply.

    Models prepend explanations, wrap output in ```json fences, and sometimes
    answer in prose. The first balanced object in the text is taken; anything
    unparseable returns None so the caller records "no description" rather
    than a corrupted one.
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def same_object_verdict(reply: dict[str, Any] | None) -> bool | None:
    """True/False from a merge arbitration, or None when it did not answer.

    None is deliberately distinct from False: "the model is unreachable" must
    not read as "these are different objects", or an outage would silently
    double every object in the room.
    """
    if not reply or "same" not in reply:
        return None
    value = reply["same"]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "same"}
    return None


def run_describe_step(
    session: SceneSession,
    client: Any | None = None,
    force: bool = False,
    max_views: int = MAX_VIEWS_PER_OBJECT,
) -> dict[str, Any]:
    """Attach crops + descriptions to `objects.json`, writing `objects_described.json`.

    `client` is anything with `.ask(images, prompt)` — normally
    `monitoring.vision.QwenVisionClient`. Passing None (or an unreachable
    server) produces objects with crops and coordinates and null descriptions,
    which is a usable scene.
    """
    import cv2

    objects_path = session.derived / "objects.json"
    if not objects_path.exists():
        raise FileNotFoundError(f"run the objects step first: {objects_path} missing")
    objects = json.loads(objects_path.read_text(encoding="utf-8"))
    if isinstance(objects, dict):
        objects = objects.get("objects", [])

    crops_dir = session.derived / "object_crops"
    cache_dir = session.derived / "descriptions"
    crops_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    keyframes = {kf.index: kf for kf in session.keyframes()}
    described = 0
    failed = 0
    out: list[dict[str, Any]] = []

    for obj_id, obj in enumerate(objects):
        views: list[dict[str, Any]] = []
        for kf_index in obj.get("keyframes", [])[:12]:
            kf = keyframes.get(kf_index)
            if kf is None:
                continue
            meta = kf.meta()
            for det in meta.get("detections", []):
                det_class = str(det.get("class_coco") or det.get("class") or "")
                if det_class not in obj.get("class_votes", {obj.get("class_top"): 1}):
                    continue
                views.append({
                    "keyframe": kf_index,
                    "instance_id": det.get("instance_id"),
                    "bbox_xyxy": det.get("bbox_xyxy", [0, 0, 0, 0]),
                    "mask_area": _bbox_area(det.get("bbox_xyxy")),
                })
                break

        frame_shape = tuple(
            (keyframes[views[0]["keyframe"]].meta().get("frame_shape") or [864, 1536])
        ) if views else (864, 1536)
        chosen = rank_views(views, frame_shape)[:max_views]

        crop_paths: list[str] = []
        crop_bytes: list[bytes] = []
        for rank, view in enumerate(chosen):
            kf = keyframes[view["keyframe"]]
            try:
                bgr = kf.rgb_bgr()
                mask = kf.instance_mask(view["instance_id"], bgr.shape[:2])
                crop = crop_with_outline(bgr, mask, view["bbox_xyxy"])
            except Exception as exc:  # noqa: BLE001 - one bad crop is not fatal
                logger.warning("object %d view %d: %s", obj_id, rank, exc)
                continue
            if crop is None or crop.size == 0:
                continue
            path = crops_dir / f"obj_{obj_id:03d}_v{rank}.jpg"
            ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                continue
            path.write_bytes(encoded.tobytes())
            crop_paths.append(str(path.relative_to(session.root)))
            crop_bytes.append(encoded.tobytes())

        cache_path = cache_dir / f"obj_{obj_id:03d}.json"
        description: dict[str, Any] | None = None
        if cache_path.exists() and not force:
            description = json.loads(cache_path.read_text(encoding="utf-8"))
        elif client is not None and crop_bytes:
            agreement = float(obj.get("label_agreement", 1.0))
            hedge = "" if agreement >= 0.7 else (
                f", but only {agreement:.0%} of the evidence agreed, so treat "
                f"that label as a guess")
            prompt = DESCRIBE_PROMPT.format(
                n=len(crop_bytes), label=obj.get("class_top", "object"), hedge=hedge)
            description = parse_json_reply(client.ask(crop_bytes, prompt))
            if description is not None:
                cache_path.write_text(json.dumps(description, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
            else:
                failed += 1
        if description is not None:
            described += 1

        centroid = obj.get("centroid") or [0.0, 0.0, 0.0]
        enriched = dict(obj)
        enriched.update({
            "object_id": obj_id,
            # The plan frame is x-y; z is height above the floor plane. Split
            # out so consumers stop re-deriving it and disagreeing about which
            # axis is up.
            "position_xy": [round(float(centroid[0]), 3), round(float(centroid[1]), 3)],
            "height_m": round(float(centroid[2]), 3),
            "crops": crop_paths,
            "description": description,
        })
        # Drop the bulky embeddings from the described copy: this file is what
        # the panel and the VLA context layer read, and a 384-d vector per
        # object makes it unreadable for no benefit.
        enriched.pop("dino_emb", None)
        enriched.pop("points", None)
        out.append(enriched)

    output_path = session.derived / "objects_described.json"
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    report = {
        "objects": len(out),
        "described": described,
        "description_failed": failed,
        "crops": sum(len(o["crops"]) for o in out),
        "output": str(output_path),
    }
    logger.info("Describe step done: %s", report)
    return report


def _bbox_area(bbox: Any) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def run_step(session: SceneSession, config: dict[str, Any], repo_root: Path,
             progress: Any = None) -> dict[str, Any]:
    """Pipeline adapter: `(session, config, repo_root, progress)`.

    Builds the vision client from the SAME config block the monitoring stack
    uses, so there is one place that says where the vLLM lives. A missing or
    disabled block means descriptions are skipped and the step still writes
    crops and coordinates — which is the degraded-but-usable outcome, not a
    failure.
    """
    # The pipeline hands each step ONLY the `scene3d` config section, so
    # `config["monitoring"]["vision"]` is absent and this used to silently
    # produce crops with no descriptions — measured on
    # session_20260807_165420: 18 crops, 0 descriptions, 0 failures, which is
    # the signature of a client that was never built rather than one that
    # errored. Accept it from the step config when a full config is passed
    # (tests, CLI) and otherwise read the project config file.
    vision_cfg = (config.get("monitoring") or {}).get("vision") or {}
    if not vision_cfg:
        try:
            from shared.config import load_config

            full = load_config(Path(repo_root) / "config/default.json")
            vision_cfg = (full.get("monitoring") or {}).get("vision") or {}
        except Exception as exc:  # noqa: BLE001 - descriptions are optional
            logger.warning("Could not load vision config: %s", exc)
    client = None
    if vision_cfg.get("enabled", True) and vision_cfg.get("backend") == "qwen":
        from mac_server.monitoring.vision import QwenVisionClient

        client = QwenVisionClient(
            base_url=vision_cfg.get("base_url", "http://127.0.0.1:8000"),
            model=vision_cfg.get("model", "Qwen/Qwen3.5-9B"),
            timeout_s=float(vision_cfg.get("timeout_s", 30)),
            chat_template_kwargs=vision_cfg.get("chat_template_kwargs"),
        )
    else:
        logger.warning("No vision backend configured (%s) — objects will have "
                       "crops and coordinates but no descriptions",
                       vision_cfg or "empty config")
    if progress:
        progress("describing objects")
    return run_describe_step(session, client=client, force=bool(config.get("force")))


def make_vlm_arbiter(client: Any) -> Any:
    """A `(crop_a, crop_b) -> bool | None` merge arbiter backed by the VLM.

    Used by `ObjectBank` for pairs its embeddings cannot separate. Returns
    None on any failure, which the bank treats as "no opinion" rather than
    "different" — see ObjectBank.__init__ for why that distinction is not
    cosmetic.
    """
    if client is None:
        return None

    def arbiter(crop_a: bytes, crop_b: bytes) -> bool | None:
        return same_object_verdict(
            parse_json_reply(client.ask([crop_a, crop_b], SAME_OBJECT_PROMPT, max_tokens=60))
        )

    return arbiter
