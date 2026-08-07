#!/usr/bin/env python3
"""Turn ADE20K SceneParsing into a YOLO instance-segmentation dataset of
indoor classes.

Why this exists: the Hailo Model Zoo has no segmentation model that knows
indoor rooms. Every prebuilt hef is COCO-80 (no cabinet, no door, no wall),
Cityscapes (a street) or Pascal VOC (20 classes). Measured on a real room
keyframe, the COCO model's maximum class score across all anchors and all 80
classes was 0.369 with 99.8% of the class head exactly zero — it is not being
thresholded too hard, it simply has no output for the things in the room. So
the model has to be fine-tuned, and this builds the data for it.

Two conversions happen here, and both are choices worth stating:

1. **Semantic -> instance.** ADE20K SceneParsing ships per-pixel class maps,
   not instances. Each class mask is split into connected components, and each
   component becomes one instance. For furniture that is what you want. For
   `wall` and `floor` it produces large irregular instances, which is still
   useful — the robot needs to know where the floor is — but it is not what
   those labels meant originally, so it is recorded rather than implied.

2. **Class merging.** ADE20K distinguishes cabinet / wardrobe / chest of
   drawers; a person describing a room says "шкаф". Merging both matches how
   the output will be used AND fixes a data problem: wardrobe alone appears in
   380 training images, which is thin for a fine-tune, while the merged class
   has ~3800. The mapping is data, not code — see the JSON config.

Class ids come from the config file's order, and the ADE20K indices are looked
up BY NAME in the dataset's own objectInfo150.txt. Hard-coding the indices
would silently produce a dataset labelled with the wrong classes if the file
ever changes.

    python3 ade20k_to_yolo.py --root ~/cv_research/ade20k/ADEChallengeData2016 \
        --classes seg_classes_indoor.json --out ~/cv_research/ade20k_yolo --stats
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# A component smaller than this fraction of the image is not a thing the robot
# can act on, and a two-pixel speck becomes a degenerate polygon that hurts
# training more than the missing label does.
DEFAULT_MIN_AREA_FRAC = 0.0015
# Douglas-Peucker tolerance as a fraction of the contour perimeter. YOLO stores
# polygons as flat vertex lists; an unsimplified mask contour is thousands of
# points per instance and makes the label files enormous for no gain.
POLY_EPS_FRAC = 0.004
MIN_POLYGON_POINTS = 3


def load_ade_index(root: Path) -> dict[str, int]:
    """ADE20K name -> pixel index, from the dataset's own table.

    Names in objectInfo150.txt are comma-separated synonym lists
    ("wardrobe, closet, press"); every synonym is registered so the config can
    use whichever reads best.
    """
    info = root / "objectInfo150.txt"
    mapping: dict[str, int] = {}
    with info.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            idx = int(row["Idx"])
            for synonym in row["Name"].split(","):
                mapping[synonym.strip().lower()] = idx
    return mapping


def build_class_table(config: dict, ade_index: dict[str, int]) -> tuple[list[str], dict[int, int]]:
    """(class names in id order, ADE pixel value -> our class id).

    Refuses on an unknown source name instead of skipping it: a typo in the
    config would otherwise silently produce a class with no data, and a class
    the model never sees is indistinguishable from one it cannot detect.
    """
    names: list[str] = []
    pixel_to_class: dict[int, int] = {}
    for class_id, entry in enumerate(config["classes"]):
        names.append(entry["name"])
        for source in entry["ade20k"]:
            key = source.strip().lower()
            if key not in ade_index:
                raise KeyError(
                    f"class {entry['name']!r} lists ADE20K source {source!r}, which is "
                    f"not in objectInfo150.txt")
            pixel = ade_index[key]
            if pixel in pixel_to_class:
                raise ValueError(
                    f"ADE20K source {source!r} (pixel {pixel}) is claimed by two "
                    f"classes: {names[pixel_to_class[pixel]]!r} and {entry['name']!r}")
            pixel_to_class[pixel] = class_id
    return names, pixel_to_class


def mask_to_polygons(mask: np.ndarray, min_area_px: float) -> list[np.ndarray]:
    """Connected components of a binary mask -> one simplified contour each.

    Only the EXTERNAL contour is kept. YOLO's segment format is a single closed
    polygon per instance and cannot express holes, so a doughnut-shaped mask is
    stored as its filled outline. That is a real (small) loss, and the
    alternative — dropping such instances — loses more.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    polygons: list[np.ndarray] = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < min_area_px:
            continue
        component = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        eps = POLY_EPS_FRAC * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if len(simplified) < MIN_POLYGON_POINTS:
            continue
        polygons.append(simplified)
    return polygons


def convert_one(job: tuple) -> tuple[str, Counter, int]:
    """One annotation PNG -> one YOLO label file. Returns per-class counts."""
    ann_path, out_label, pixel_to_class, min_area_frac = job
    ann = cv2.imread(str(ann_path), cv2.IMREAD_UNCHANGED)
    if ann is None:
        return (ann_path.name, Counter(), 0)
    if ann.ndim == 3:
        ann = ann[:, :, 0]
    height, width = ann.shape[:2]
    min_area_px = max(16.0, min_area_frac * height * width)

    counts: Counter = Counter()
    lines: list[str] = []
    present = np.unique(ann)
    for pixel in present:
        class_id = pixel_to_class.get(int(pixel))
        if class_id is None:
            continue
        for polygon in mask_to_polygons(ann == pixel, min_area_px):
            xs = np.clip(polygon[:, 0] / width, 0.0, 1.0)
            ys = np.clip(polygon[:, 1] / height, 0.0, 1.0)
            coords = " ".join(f"{x:.5f} {y:.5f}" for x, y in zip(xs, ys))
            lines.append(f"{class_id} {coords}")
            counts[class_id] += 1

    if out_label is not None:
        out_label.parent.mkdir(parents=True, exist_ok=True)
        out_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return (ann_path.name, counts, len(lines))


def run_split(
    root: Path, out: Path, split: str, ade_split: str,
    pixel_to_class: dict[int, int], names: list[str],
    min_area_frac: float, workers: int, stats_only: bool, limit: int | None,
) -> Counter:
    ann_dir = root / "annotations" / ade_split
    img_dir = root / "images" / ade_split
    annotations = sorted(ann_dir.glob("*.png"))
    if limit:
        annotations = annotations[:limit]

    jobs = []
    for ann_path in annotations:
        out_label = None if stats_only else out / "labels" / split / f"{ann_path.stem}.txt"
        jobs.append((ann_path, out_label, pixel_to_class, min_area_frac))

    totals: Counter = Counter()
    images_with_any = 0
    with mp.Pool(workers) as pool:
        for i, (_, counts, n_lines) in enumerate(pool.imap_unordered(convert_one, jobs, chunksize=32)):
            totals.update(counts)
            if n_lines:
                images_with_any += 1
            if (i + 1) % 2000 == 0:
                print(f"  {split}: {i + 1}/{len(jobs)}", flush=True)

    if not stats_only:
        # Symlink rather than copy: 20 210 JPEGs are ~1 GB and the originals
        # are not going anywhere.
        link_dir = out / "images" / split
        link_dir.mkdir(parents=True, exist_ok=True)
        for ann_path in annotations:
            src = img_dir / f"{ann_path.stem}.jpg"
            dst = link_dir / src.name
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve())

    print(f"{split}: {len(jobs)} images, {images_with_any} with >=1 instance, "
          f"{sum(totals.values())} instances")
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True,
                        help="ADEChallengeData2016 directory")
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-area-frac", type=float, default=DEFAULT_MIN_AREA_FRAC)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--stats", action="store_true",
                        help="Count instances per class without writing labels")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N images (smoke test)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.classes.read_text(encoding="utf-8"))
    ade_index = load_ade_index(args.root)
    names, pixel_to_class = build_class_table(config, ade_index)
    print(f"{len(names)} classes from {len(pixel_to_class)} ADE20K sources")

    totals: Counter = Counter()
    for split, ade_split in (("train", "training"), ("val", "validation")):
        totals.update(run_split(
            args.root, args.out, split, ade_split, pixel_to_class, names,
            args.min_area_frac, args.workers, args.stats, args.limit,
        ))

    print("\ninstances per class (train+val):")
    for class_id, name in enumerate(names):
        print(f"  {class_id:>3} {name:<20} {totals.get(class_id, 0):>8}")
    thin = [names[c] for c in range(len(names)) if totals.get(c, 0) < 500]
    if thin:
        print(f"\nWARNING thin classes (<500 instances): {thin}")

    if not args.stats:
        yaml_path = args.out / "dataset.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        body = [f"path: {args.out.resolve()}", "train: images/train", "val: images/val",
                "names:"]
        body += [f"  {i}: {n}" for i, n in enumerate(names)]
        yaml_path.write_text("\n".join(body) + "\n", encoding="utf-8")
        print(f"\nwrote {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
