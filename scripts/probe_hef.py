#!/usr/bin/env python3
"""Measure what a hef actually is, before writing a decoder for it.

The existing `seg_postprocess.py` identifies YOLOv8-seg output blobs **by
shape** — protos at `h == 160`, then a per-scale triple keyed by channel count
`{64: boxes, 80: class scores, 32: mask coefficients}`. That works because
somebody once looked at the real tensors. Writing the next decoder from a forum
post instead would be guessing, and the failure mode is not a crash: a decoder
that misreads which blob is which produces plausible-looking garbage.

So this runs first, on the Pi, read-only, and answers three questions with
numbers:

1. **What comes out?** Every input and output vstream, name and shape. That
   classifies the model into one of: raw 10-blob YOLOv8-seg layout (the existing
   decoder can be parameterized), a single on-chip-NMS output (a different,
   smaller decoder), or something else entirely (stop and reconsider).
2. **Is it alive?** One real inference on a recorded keyframe, with per-blob
   min/max/mean. A blob of all zeros, or one whose range says "logits" when you
   expected "probabilities", is caught here rather than three modules later.
3. **What does it cost?** Median and p95 latency. Two variants matter: a fresh
   `InferVStreams` context per call (what `hailo_infer._ConfiguredHef` does
   today) versus one context entered once and reused. If the difference is
   large, that is a free speedup; if HailoRT refuses two models holding
   vstreams under the scheduler, that is a constraint worth knowing before the
   design leans on it.

    ./scripts/run_probe_hef.sh models/yolov8m_seg_h8.hef
    ./scripts/run_probe_hef.sh models/fastsam_s_h8.hef --compare models/clip_resnet_50x4_h8.hef
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "client/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

# YOLOv8-seg with 32 mask prototypes, as `seg_postprocess.order_endnodes`
# expects it: one proto blob plus a (boxes, scores, coeffs) triple per stride.
YOLOV8_SEG_BLOB_COUNT = 10


def describe_streams(hef: Any) -> dict[str, Any]:
    inputs = [{"name": i.name, "shape": list(i.shape)} for i in hef.get_input_vstream_infos()]
    outputs = [{"name": o.name, "shape": list(o.shape)} for o in hef.get_output_vstream_infos()]
    return {"inputs": inputs, "outputs": outputs}


def classify(streams: dict[str, Any]) -> dict[str, Any]:
    """Name the output layout, or say plainly that it is unrecognised.

    Deliberately conservative: an unrecognised layout returns `unknown` with
    the evidence attached, rather than a best guess. A wrong guess here costs
    a decoder rewrite discovered days later.
    """
    outputs = streams["outputs"]
    shapes = [tuple(o["shape"]) for o in outputs]
    channels = [s[-1] for s in shapes if len(s) == 3]
    notes: list[str] = []

    if len(outputs) == 1:
        only = shapes[0]
        # An embedding head is spatially collapsed: (1, 1, D) or (D,). Measured
        # on clip_resnet_50x4_h8: (1, 1, 640). Naming it matters because the
        # probe is also the tool for checking the CLIP hef is healthy, and a
        # model we depend on reported as "unknown" trains the operator to
        # ignore the field that exists to catch real surprises.
        if len(only) == 1 or (len(only) == 3 and only[0] == 1 and only[1] == 1):
            return {"layout": "embedding",
                    "detail": f"single {only[-1]}-d embedding — no decoder needed; "
                              f"normalize and compare by cosine",
                    "dim": only[-1], "evidence": shapes}
        if len(only) == 2:
            return {"layout": "on_chip_nms",
                    "detail": "single 2-d output — NMS baked in; parse boxes/scores "
                              "directly and keep only process_mask",
                    "evidence": shapes}

    if len(outputs) == YOLOV8_SEG_BLOB_COUNT:
        heights = sorted({s[0] for s in shapes if len(s) == 3})
        # Protos are the largest spatial blob; the detection scales are the
        # three below it. For a 640 input that is 160 and (20, 40, 80).
        proto_h = heights[-1] if heights else None
        scale_hs = heights[:-1]
        counts = {c: channels.count(c) for c in set(channels)}
        notes.append(f"channel counts: {counts}")

        # Derive the architecture from the blobs instead of assuming it. Pull
        # the proto blob out FIRST — its channel count IS num_masks, and
        # leaving it in the tally is what makes the coefficient count read as
        # 4 instead of 3. What remains must be nine blobs forming exactly three
        # roles of three; any other multiplicity means two roles share a
        # channel width and shape alone cannot separate them.
        proto_shape = next((s for s in shapes if len(s) == 3 and s[0] == proto_h), None)
        num_masks = proto_shape[-1] if proto_shape else None
        det_channels = list(channels)
        if num_masks is not None:
            det_channels.remove(num_masks)  # exactly one proto blob
        det_counts = {c: det_channels.count(c) for c in set(det_channels)}

        reg_max = None
        num_classes = None
        if sorted(det_counts.values()) != [3, 3, 3]:
            notes.append(
                "AMBIGUOUS: the nine detection blobs do not form three distinct "
                f"roles of three (counts {det_counts}) — two heads share a channel "
                "width, so shape alone cannot tell them apart. Identify them by "
                "value range on a real frame before decoding.")
        else:
            remaining = sorted(det_counts)
            if num_masks in remaining:
                remaining.remove(num_masks)
            else:
                notes.append("AMBIGUOUS: no detection blob matches the proto channel "
                             f"count {num_masks}; mask coefficients not identifiable")
            for candidate in list(remaining):
                # The box head is a DFL distribution: 4 sides x (reg_max+1) bins.
                if candidate % 4 == 0 and candidate >= 4:
                    reg_max = candidate // 4 - 1
                    remaining.remove(candidate)
                    break
            if len(remaining) == 1:
                num_classes = remaining[0]
            else:
                notes.append(f"could not isolate a class head; leftover {remaining}")

        return {"layout": "yolov8_seg_raw",
                "detail": (f"10 blobs — parameterize seg_postprocess.SegArch("
                           f"num_classes={num_classes}, reg_max={reg_max}, "
                           f"num_masks={num_masks})"),
                "proto_h": proto_h, "scale_h": scale_hs,
                "num_classes": num_classes, "reg_max": reg_max, "num_masks": num_masks,
                "channel_counts": counts, "detection_channel_counts": det_counts,
                "evidence": shapes, "notes": notes}

    return {"layout": "unknown",
            "detail": f"{len(outputs)} outputs — does not match either known layout; "
                      f"inspect before writing anything",
            "evidence": shapes}


def infer_stats(model: Any, image: Any) -> dict[str, Any]:
    import numpy as np

    out = model.infer(image)
    stats = {}
    for name, arr in sorted(out.items()):
        arr = np.asarray(arr, dtype=np.float64)
        stats[name] = {
            "shape": list(arr.shape),
            "min": round(float(arr.min()), 5),
            "max": round(float(arr.max()), 5),
            "mean": round(float(arr.mean()), 5),
            "zeros_frac": round(float((arr == 0).mean()), 4),
        }
    return stats


def time_model(model: Any, image: Any, runs: int) -> dict[str, float]:
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        model.infer(image)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {
        "runs": runs,
        "median_ms": round(statistics.median(samples), 2),
        "p95_ms": round(samples[int(len(samples) * 0.95) - 1], 2),
        "min_ms": round(samples[0], 2),
        "max_ms": round(samples[-1], 2),
    }


def time_persistent(model: Any, image: Any, runs: int) -> dict[str, Any]:
    """Same inferences, but entering `InferVStreams` once instead of per call.

    Reaches into `_ConfiguredHef`'s privates on purpose: the point is to measure
    the cost of the wrapper's current choice without changing the wrapper first.
    If this is materially faster, `infer()` should hold the context — but that
    may be impossible with two models resident, because the scheduler has to be
    able to swap them. So a HailoRT refusal here is a RESULT, not a failure, and
    is reported as one.
    """
    import numpy as np

    batch = np.expand_dims(image, axis=0)
    try:
        with model._infer_cls(
            model._network_group, model._input_params, model._output_params
        ) as pipeline:
            pipeline.infer({model.input_name: batch})  # warm up, not measured
            samples = []
            for _ in range(runs):
                start = time.perf_counter()
                pipeline.infer({model.input_name: batch})
                samples.append((time.perf_counter() - start) * 1000.0)
    except Exception as exc:  # noqa: BLE001 - the exception type IS the finding
        return {"supported": False, "error": f"{type(exc).__name__}: {exc}"}
    samples.sort()
    return {
        "supported": True,
        "runs": runs,
        "median_ms": round(statistics.median(samples), 2),
        "p95_ms": round(samples[int(len(samples) * 0.95) - 1], 2),
    }


def load_probe_image(path: Path | None, shape: tuple[int, int, int]) -> Any:
    """A real recorded keyframe if one is given, else a deterministic pattern.

    A real frame matters for the value ranges: a synthetic constant image can
    make a healthy model look dead (every score at the same value) and hide
    exactly the problem this probe exists to catch.
    """
    import numpy as np

    height, width = shape[0], shape[1]
    if path is not None and path.exists():
        import cv2

        bgr = cv2.imread(str(path))
        if bgr is not None:
            resized = cv2.resize(bgr, (width, height))
            return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        print(f"warning: could not read {path}, falling back to a pattern", file=sys.stderr)
    grid = np.indices((height, width)).sum(axis=0) % 256
    return np.stack([grid, (grid * 2) % 256, (grid * 3) % 256], axis=-1).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hef", type=Path)
    parser.add_argument("--compare", type=Path, default=None,
                        help="A second hef to load alongside, so the scheduler's "
                             "cost with two resident models is measured rather "
                             "than assumed.")
    parser.add_argument("--image", type=Path, default=None,
                        help="A recorded keyframe rgb.jpg. Strongly preferred over "
                             "the synthetic fallback — see load_probe_image.")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.hef.exists():
        print(f"No such hef: {args.hef}", file=sys.stderr)
        return 2

    from pi_client.hailo_infer import HailoMultiModel

    report: dict[str, Any] = {"hef": str(args.hef)}
    hailo = HailoMultiModel()
    try:
        model = hailo.load("probe", str(args.hef))
        report["streams"] = describe_streams(model.hef)
        report["classification"] = classify(report["streams"])
        report["input_shape"] = list(model.input_shape)

        image = load_probe_image(args.image, model.input_shape)
        report["probe_image"] = str(args.image) if args.image else "synthetic pattern"
        report["output_stats"] = infer_stats(model, image)
        report["timing_alone"] = time_model(model, image, args.runs)
        report["timing_persistent"] = time_persistent(model, image, args.runs)

        if args.compare is not None and args.compare.exists():
            other = hailo.load("compare", str(args.compare))
            # Re-measure the persistent variant with a second model resident.
            # This is the question that actually matters: holding vstreams open
            # may be fine alone and refused once the scheduler must swap models.
            report["timing_persistent_two_models"] = time_persistent(
                model, image, args.runs
            )
            other_image = load_probe_image(args.image, other.input_shape)
            # Interleave: this is what the recorder actually does, and it is
            # the number that decides whether per-detection CLIP can stay on
            # the Pi at all.
            samples = []
            for _ in range(args.runs):
                start = time.perf_counter()
                model.infer(image)
                other.infer(other_image)
                samples.append((time.perf_counter() - start) * 1000.0)
            samples.sort()
            report["timing_interleaved"] = {
                "with": str(args.compare),
                "runs": args.runs,
                "median_ms": round(statistics.median(samples), 2),
                "p95_ms": round(samples[int(len(samples) * 0.95) - 1], 2),
            }
    finally:
        hailo.close()

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}", file=sys.stderr)

    cls = report["classification"]
    print(f"\nLAYOUT: {cls['layout']} — {cls['detail']}", file=sys.stderr)
    print(f"LATENCY: median {report['timing_alone']['median_ms']} ms, "
          f"p95 {report['timing_alone']['p95_ms']} ms", file=sys.stderr)
    persistent = report.get("timing_persistent", {})
    if persistent.get("supported"):
        saved = report["timing_alone"]["median_ms"] - persistent["median_ms"]
        print(f"PERSISTENT vstreams: median {persistent['median_ms']} ms "
              f"({saved:+.2f} ms vs per-call)", file=sys.stderr)
    else:
        print(f"PERSISTENT vstreams: refused — {persistent.get('error')}",
              file=sys.stderr)
    for note in cls.get("notes", []):
        print(f"NOTE: {note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
