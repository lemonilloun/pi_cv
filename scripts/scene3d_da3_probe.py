#!/usr/bin/env python3
"""Measure whether Depth Anything 3 is viable on this Mac before the
scene3d pipeline is rewritten around it.

DA3 cannot live in the server's environment: it pins ``numpy<2`` while the
server runs numpy 2.x and opencv 5, so installing it there would downgrade
the stack under a running server. It therefore gets its own virtualenv
(``.ven-da3``) and this probe runs *there*:

    KMP_DUPLICATE_LIB_OK=TRUE .ven-da3/bin/python3 scripts/scene3d_da3_probe.py \
        --session session_20260724_144728 --frames 16 --model DA3-BASE

What it answers, all of which the plan treats as gates:
  * does it fit — peak process RSS and MPS allocation for a window of N views
  * how slow — seconds per frame
  * is there a usable confidence channel
  * is the depth metric — compared against the existing DAv2 metric maps
  * are the poses trustworthy — aligned to the COLMAP poses already computed
    for the same session (that run registered 233/235 frames, so it is a
    fair reference) and reported as a similarity-aligned RMS error

Nothing here writes into a session; it only reads and prints.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Detached:
    """Plain numpy copy of the fields we analyse, so the model can be freed
    before the (slow, memory-hungry) comparison work starts."""

    __slots__ = ("depth", "conf", "intrinsics", "extrinsics")

    def __init__(self, pred) -> None:
        def arr(value):
            if value is None:
                return None
            if hasattr(value, "detach"):
                value = value.detach().to("cpu")
            return np.asarray(value)

        self.depth = arr(getattr(pred, "depth", None))
        self.conf = arr(getattr(pred, "conf", None))
        self.intrinsics = arr(getattr(pred, "intrinsics", None))
        self.extrinsics = arr(getattr(pred, "extrinsics", None))


def _detach(pred) -> _Detached:
    return _Detached(pred)


def release(model, torch) -> None:
    """Drop the model and hand the GPU allocation back.

    On a shared-memory Mac a resident model plus MPS's cached blocks is the
    difference between 'fits' and 'the machine swaps' — so this runs even
    when the probe fails partway.
    """
    try:
        del model
    except Exception:
        pass
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass


def purge_model_cache(model_name: str) -> None:
    """Delete a downloaded HF snapshot once we're done measuring it."""
    cache = Path.home() / ".cache/huggingface/hub"
    target = cache / f"models--depth-anything--{model_name}"
    if target.exists():
        size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        shutil.rmtree(target, ignore_errors=True)
        print(f"  purged {target.name} ({size/1e9:.2f} GB)")


def rss_gb() -> float:
    """Peak resident set size. macOS reports ru_maxrss in bytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform mapping src onto dst.

    Poses from two different reconstructions share no gauge — origin,
    orientation and (for a monocular method) scale are all arbitrary. Only
    after removing that 7-DoF freedom is a positional error meaningful.
    """
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / len(src)
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1.0
    rot = u @ s @ vt
    var = (src_c**2).sum() / len(src)
    scale = float(np.trace(np.diag(d) @ s) / var) if var > 1e-12 else 1.0
    trans = dst_mean - scale * rot @ src_mean
    return scale, rot, trans


def extrinsics_to_world_from_cam(ext: np.ndarray) -> np.ndarray:
    """DA3 returns OpenCV/COLMAP world-to-camera; the pipeline stores
    camera-to-world. Same inversion poses_step.py already does."""
    mat = np.asarray(ext, dtype=np.float64)
    rot, trans = mat[:3, :3], mat[:3, 3]
    out = np.eye(4)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="session_20260724_144728")
    ap.add_argument("--frames", type=int, default=16, help="views in one window")
    ap.add_argument("--model", default="DA3-BASE",
                    help="DA3-SMALL | DA3-BASE | DA3-LARGE-1.1 | DA3METRIC-LARGE")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--fp16", action="store_true", help="half precision weights")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--sessions-dir", default="data/scene_sessions")
    ap.add_argument("--purge-model", action="store_true",
                    help="delete the downloaded weights after measuring them")
    args = ap.parse_args()

    import torch
    from depth_anything_3.api import DepthAnything3

    session_dir = REPO_ROOT / args.sessions_dir / args.session
    kf_dir = session_dir / "keyframes"
    if not kf_dir.is_dir():
        print(f"no such session: {session_dir}", file=sys.stderr)
        return 2

    all_kfs = sorted(p for p in kf_dir.iterdir() if (p / "rgb.jpg").exists())
    if not all_kfs:
        print("session has no keyframes", file=sys.stderr)
        return 2
    # A contiguous middle slice: consecutive views are what a real window is.
    start = max(0, len(all_kfs) // 2 - args.frames // 2)
    kfs = all_kfs[start : start + args.frames]
    images = [str(p / "rgb.jpg") for p in kfs]
    indices = [int(p.name) for p in kfs]
    print(f"session {args.session}: {len(all_kfs)} keyframes, probing {len(kfs)} "
          f"({indices[0]}..{indices[-1]})")

    print(f"loading {args.model} on {args.device} (fp16={args.fp16}) ...")
    model = None
    try:
        t0 = time.time()
        model = DepthAnything3.from_pretrained(f"depth-anything/{args.model}")
        model = model.to(device=torch.device(args.device))
        model.eval()
        load_s = time.time() - t0
        params = sum(p.numel() for p in model.parameters())
        print(f"  loaded in {load_s:.1f}s, {params/1e6:.0f}M params, RSS {rss_gb():.2f} GB")

        print(f"running inference on {len(images)} views "
              f"(process_res={args.process_res}) ...")
        t0 = time.time()
        # Half precision has to come from autocast, not model.half(): DA3's
        # own preprocessing hands the network float32 tensors, so casting the
        # weights alone throws "Input type (float) and bias type (c10::Half)".
        with torch.no_grad():
            if args.fp16:
                with torch.autocast(device_type=args.device, dtype=torch.float16):
                    pred = model.inference(images, process_res=args.process_res)
            else:
                pred = model.inference(images, process_res=args.process_res)
        infer_s = time.time() - t0

        mps_gb = 0.0
        if args.device == "mps" and hasattr(torch, "mps"):
            try:
                mps_gb = torch.mps.driver_allocated_memory() / 1e9
            except Exception:
                pass
        # Everything below is numpy on the returned arrays; the weights are
        # dead weight from here on and this box has 8.6 GB to share.
        pred = _detach(pred)
    finally:
        release(model, torch)
        if args.purge_model:
            purge_model_cache(args.model)
    print(f"\n=== COST ===")
    print(f"  {infer_s:.1f}s total, {infer_s/len(images):.2f}s/frame")
    print(f"  peak RSS {rss_gb():.2f} GB, MPS driver {mps_gb:.2f} GB")
    print(f"  extrapolated to 235 frames @ window={len(images)}: "
          f"{infer_s/len(images)*235/60:.1f} min")

    depth = np.asarray(pred.depth)
    print(f"\n=== OUTPUTS ===")
    print(f"  depth {depth.shape} {depth.dtype}  "
          f"median {np.median(depth[depth>0]):.2f}  p99 {np.percentile(depth,99):.2f}")
    conf = getattr(pred, "conf", None)
    if conf is not None:
        conf = np.asarray(conf)
        print(f"  conf  {conf.shape}  range [{conf.min():.3f}, {conf.max():.3f}]  "
              f"median {np.median(conf):.3f}")
    else:
        print("  conf  ABSENT — the filter would lose its confidence channel")
    ixt = np.asarray(pred.intrinsics)
    print(f"  intrinsics fx={ixt[0,0,0]:.1f} fy={ixt[0,1,1]:.1f} "
          f"cx={ixt[0,0,2]:.1f} cy={ixt[0,1,2]:.1f}")

    # --- is the depth metric? compare to the DAv2 metric maps already on disk
    print(f"\n=== METRIC SCALE (vs existing DAv2 metric depth) ===")
    ratios = []
    for slot, idx in enumerate(indices):
        p = session_dir / "derived" / "depth" / f"{idx:06d}.npy"
        if not p.exists():
            continue
        dav2 = np.load(p).astype(np.float32)
        d3 = depth[slot]
        h = min(dav2.shape[0], d3.shape[0])
        w = min(dav2.shape[1], d3.shape[1])
        a = dav2[:h, :w]
        b = d3[:h, :w]
        m = (a > 0.2) & (a < 8.0) & (b > 0.01)
        if m.sum() > 1000:
            ratios.append(float(np.median(a[m] / b[m])))
    if ratios:
        r = float(np.median(ratios))
        print(f"  median(DAv2 / DA3) over {len(ratios)} frames = {r:.3f}")
        print("  -> DA3 output already ~metric" if 0.8 < r < 1.25
              else f"  -> NOT metric as-is; needs a x{r:.3f} scale (or a metric variant)")
    else:
        print("  no overlapping DAv2 depth to compare (run the depth step first)")

    # --- are the poses trustworthy? compare to COLMAP on the same frames
    print(f"\n=== POSES (vs COLMAP, which registered 233/235 on this session) ===")
    poses_path = session_dir / "derived" / "poses.json"
    if not poses_path.exists():
        print("  no poses.json — skip")
        return 0
    colmap = json.loads(poses_path.read_text())["world_from_cam"]
    da3_c, ref_c = [], []
    for slot, idx in enumerate(indices):
        if str(idx) not in colmap:
            continue
        da3_c.append(extrinsics_to_world_from_cam(pred.extrinsics[slot])[:3, 3])
        ref_c.append(np.asarray(colmap[str(idx)], dtype=np.float64)[:3, 3])
    if len(da3_c) < 4:
        print(f"  only {len(da3_c)} shared frames — not enough to align")
        return 0
    da3_c, ref_c = np.array(da3_c), np.array(ref_c)
    scale, rot, trans = umeyama_sim3(da3_c, ref_c)
    aligned = (scale * (rot @ da3_c.T)).T + trans
    err = np.linalg.norm(aligned - ref_c, axis=1)
    span = float(np.linalg.norm(ref_c.max(axis=0) - ref_c.min(axis=0)))
    print(f"  {len(da3_c)} shared frames, COLMAP path span {span:.2f} m")
    print(f"  after Sim3 alignment: RMS {np.sqrt((err**2).mean()):.3f} m, "
          f"max {err.max():.3f} m")
    print(f"  relative to path span: {100*np.sqrt((err**2).mean())/max(span,1e-6):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
