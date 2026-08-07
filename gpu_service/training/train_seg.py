#!/usr/bin/env python3
"""Fine-tune YOLO11-seg on the indoor ADE20K classes.

Model choice, measured rather than assumed (scripts/probe_hef.py on the Pi and
gpu_service/training/head_check):

- The Pi runs `yolov8m_seg_h8` at 39.8 ms (25 fps) against a 2-3 fps target,
  so the NPU has ~10x headroom and the small models are leaving quality on the
  table for nothing.
- YOLO11-seg's head is byte-identical in geometry to YOLOv8-seg's: same
  `Segment` module, reg_max 16 (=> 64 box channels), 32 mask coefficients.
  The Pi's existing pure-numpy decoder therefore works unchanged; only
  `SegArch.num_classes` moves from 80 to 31.
- `yolo11l-seg` has 27.7M parameters against `yolov8m-seg`'s 27.3M — the same
  training cost — for 42.6 vs 40.1 COCO mask mAP. `yolo11x-seg` buys a further
  +0.7 mAP for 2.2x the parameters, which on a 20k-image fine-tune is a bad
  trade: more capacity to memorise, half the inference speed.

Preprocessing note that matters more than it looks: ultralytics letterboxes
(aspect preserved, grey 114 padding). The Pi recorder used to squash 1536x864
straight into 640x640. Measured on 53 recorded keyframes, squash cost 11
detections against letterbox's 16 with the stock COCO hef — so the recorder was
fixed to letterbox rather than training on squashed input, because letterbox is
also what every pretrained weight in the chain expects.

    python3 train_seg.py --data ~/cv_research/ade20k_yolo/dataset.yaml --device 1
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="yolo11l-seg.pt")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="MUST match the hef input the Pi runs (640).")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="1",
                        help="GPU index. Default 1 — GPU 0 carries another "
                             "user's vLLM and must not be touched.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="indoor_yolo11l")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    from ultralytics import YOLO

    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        resume=args.resume,
        # Mosaic helps small-object recall but glues four rooms into one image,
        # which is unlike anything the robot will ever see. Turning it off for
        # the last epochs lets the model settle on realistic full scenes.
        close_mosaic=10,
        # The robot's camera is fixed upright on a chassis that only yaws, so
        # it never sees a mirrored or rotated room in a way that matters, but
        # horizontal flip is still a free doubling of the data.
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        # Indoor lighting varies a lot between rooms and times of day; this is
        # the augmentation most likely to pay off here.
        hsv_v=0.4,
        val=True,
        plots=True,
    )
    print("done:", Path(args.project) / args.name / "weights" / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
