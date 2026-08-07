"""What the reconstruction should DO with a detected class, and how to decide
that two observations are the same physical thing.

Pure logic, no I/O beyond reading the class config — so the rules that decide
whether a room ends up with one cabinet or eleven are unit-testable without a
robot, a camera or a GPU.

Two problems live here, both observed on real output rather than imagined:

1. **`wall`/`floor`/`ceiling` are not objects.** On the 53-keyframe session the
   fine-tuned detector returned 50 `wall` detections out of 112. Treated as
   instances they become dozens of "objects" that clutter the floor plan and
   say nothing, while the information they carry — where the room ends, where
   the robot can drive — already lives in the occupancy grid and the mesh.
   `floor` is kept as a single aggregate record because the robot's drivable
   area is worth one entry with a nominal centre, not one per fragment.

2. **The label flips between frames.** Measured on one snapshot: the same box
   came back as `cabinet 0.339` AND `door 0.339`. An association rule that
   requires the class to match exactly turns that into two objects sitting in
   the same place. So class is *evidence*, not a gate: geometry and appearance
   decide identity, and the name is voted on afterwards from every observation.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

# Roles a class can play in the reconstruction.
ROLE_AREA = "area"        # a region, not a thing: floor/wall/ceiling
ROLE_OBJECT = "object"    # gets an instance, a position and a crop

# Aggregated into ONE record rather than dropped: the robot drives on it, so
# "where is the floor" is a question worth one answer.
AGGREGATE_CLASSES = frozenset({"floor"})

# Classes the detector genuinely confuses because they are all large flat
# vertical surfaces, confirmed on real frames (a curtain came back as `wall`,
# a cabinet front as `door`). Within a group a label disagreement is NOT
# evidence against a merge; across groups it still is.
CONFUSABLE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"wall", "curtain", "door", "window", "painting", "mirror"}),
    frozenset({"cabinet", "shelf", "counter", "refrigerator", "door"}),
    frozenset({"chair", "sofa", "bed", "pillow"}),
    frozenset({"table", "counter", "desk"}),
)


def load_roles(path: Path) -> dict[str, str]:
    """class name -> role, from the same file that defines the class ids.

    Absent `role` defaults to `object`: a class someone added without thinking
    about it should show up on the plan and be noticed, not silently vanish.
    """
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(entry["name"]): str(entry.get("role", ROLE_OBJECT))
        for entry in config["classes"]
    }


def classes_may_be_same(a: str, b: str) -> bool:
    """Is a label disagreement between two observations forgivable?

    Identical names always are. Different names are only when the detector is
    known to confuse them — otherwise a chair and a refrigerator half a metre
    apart could merge on appearance alone, and the resulting object would be
    named by whichever label happened to win the vote.
    """
    if a == b:
        return True
    return any(a in group and b in group for group in CONFUSABLE_GROUPS)


def vote_label(class_counts: dict[str, int], confidence_sums: dict[str, float] | None = None
               ) -> tuple[str, float]:
    """The object's name, from every observation rather than the first one.

    Ranked by summed confidence when available and by count otherwise, because
    six weak `wall` guesses should not outrank two strong `cabinet` ones.
    Returns (name, agreement) where agreement is the winner's share — a low
    value is the honest signal that the detector never settled, and is worth
    surfacing to the VLM rather than hiding behind a confident-looking label.
    """
    if not class_counts:
        return ("unknown", 0.0)
    if confidence_sums:
        winner = max(class_counts, key=lambda c: confidence_sums.get(c, 0.0))
        total = sum(confidence_sums.get(c, 0.0) for c in class_counts) or 1.0
        return (winner, confidence_sums.get(winner, 0.0) / total)
    counts = Counter(class_counts)
    winner, n = counts.most_common(1)[0]
    return (winner, n / max(1, sum(counts.values())))


def partition_detections(
    detections: list[dict[str, Any]], roles: dict[str, str]
) -> tuple[list[int], list[int], list[int]]:
    """Split detection indices into (objects, aggregate, ignored).

    Index lists rather than filtered copies, so callers can keep masks, points
    and embeddings positionally aligned with whatever they already have.
    """
    objects: list[int] = []
    aggregate: list[int] = []
    ignored: list[int] = []
    for i, det in enumerate(detections):
        name = str(det.get("class") or det.get("class_coco") or "")
        role = roles.get(name, ROLE_OBJECT)
        if name in AGGREGATE_CLASSES:
            aggregate.append(i)
        elif role == ROLE_AREA:
            ignored.append(i)
        else:
            objects.append(i)
    return objects, aggregate, ignored


# Rough real-world largest dimension per class, metres. NOT ground truth — a
# coarse plausibility band, used only to answer "is the whole reconstruction
# the wrong size" rather than to measure any individual object.
TYPICAL_MAX_DIM_M = {
    "bed": 2.1, "sofa": 2.2, "table": 1.8, "cabinet": 2.2, "shelf": 2.0,
    "chair": 1.2, "door": 2.1, "window": 1.6, "refrigerator": 2.0,
    "person": 1.9, "curtain": 2.6, "painting": 1.2, "mirror": 1.5,
    "tv": 1.6, "lamp": 1.8, "toilet": 0.9, "sink": 1.2, "stove": 1.0,
    "box": 0.8, "pillow": 0.8, "plant": 1.6, "rug": 3.0, "book": 0.4,
    "clock": 0.6, "counter": 3.0, "fireplace": 2.0, "stairs": 4.0,
}

# Below this many usable objects the estimate is noise, not evidence.
MIN_OBJECTS_FOR_SCALE_CHECK = 3
# The tolerance is ASYMMETRIC, and the asymmetry is the point. A reconstructed
# object is built from the surfaces the camera actually saw, which is never the
# whole thing — you see the front of a cabinet, one side of a table, part of a
# door. So measured extents are biased DOWNWARD and a ratio below 1 is normal.
# Measured on session_20260807_171938 at a scale independently confirmed by the
# room's real dimensions (4.09 x 3.58 m): median ratio 0.61 across 12 objects,
# every one of which passed its own size verdict. An earlier symmetric 1.5x
# band flagged that correct reconstruction as broken.
#
# Nothing biases extents UPWARD except depth bleeding from the background and a
# genuinely wrong scale, so the upper bound stays tight.
SCALE_RATIO_MAX = 1.5
SCALE_RATIO_MIN = 0.35


def scale_plausibility(objects: list[dict[str, Any]]) -> dict[str, Any]:
    """Does the finished reconstruction have room-sized things in it?

    Every metric number downstream — the mesh, the floor plan, every position
    navigation reports — is the global scale times something, and the scale can
    be wrong by a large factor while every internal consistency check passes.
    Measured on session_20260807_165420: `scale_imu` had 114 intervals and a
    13% IQR, which reads as a confident estimate, and the reconstruction still
    came out 2x too big — a 4.08 m bed, a 3.12 m window, a 2.5 m person. The
    IMU's spread describes its own repeatability, not its accuracy, and the
    only independent check (camera height over the floor plane) had already
    failed at 14% inliers because the camera barely sees the floor.

    So this compares object extents against a coarse table of what such things
    actually measure and reports the median ratio. It is deliberately a
    WARNING, never a correction: calibrating the scale from a sanity table
    would turn a rough prior into a measurement, and the real fix is a tape
    measure (see `baseline.json`).
    """
    ratios: list[float] = []
    for obj in objects:
        name = str(obj.get("class_top") or "")
        expected = TYPICAL_MAX_DIM_M.get(name)
        extent = obj.get("extent") or (obj.get("obb") or {}).get("extent")
        if not expected or not extent:
            continue
        largest = max(float(v) for v in extent)
        if largest > 0:
            ratios.append(largest / expected)

    if len(ratios) < MIN_OBJECTS_FOR_SCALE_CHECK:
        return {"checked": False,
                "reason": f"only {len(ratios)} objects with a known typical size"}

    ratios.sort()
    median = ratios[len(ratios) // 2]
    implausible = median > SCALE_RATIO_MAX or median < SCALE_RATIO_MIN
    result: dict[str, Any] = {
        "checked": True,
        "n_objects": len(ratios),
        "median_ratio": round(median, 2),
        "range": [round(ratios[0], 2), round(ratios[-1], 2)],
        "plausible": not implausible,
    }
    if implausible:
        direction = "larger" if median > SCALE_RATIO_MAX else "smaller"
        result["warning"] = (
            f"Objects are {median:.2f}x their typical real size ({direction} "
            f"than a partial view can explain). The global "
            f"scale is probably wrong by about that factor — everything metric "
            f"downstream is affected, and depth truncation will have cropped "
            f"the room if the error is upward. Fix it with a tape measure: "
            f"write <session>/baseline.json as "
            f'{{"from_frame": N, "to_frame": M, "distance_m": D}} and re-run.'
        )
    return result
