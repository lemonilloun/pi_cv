#!/usr/bin/env python3
"""Write a session's `baseline.json` — the metric scale reference.

Metric scale has to come from somewhere physical. On this rig the two
automatic candidates are both weak: the floor-plane fit needs floor, and at a
13 cm camera height looking slightly up the floor is a sliver at the frame
edge (measured: 15% inliers); the IMU route double-integrates an
accelerometer, which this project has measured at 78x wrong over a walk.

A tape measure has neither problem. Drive the robot in a straight line over a
measured distance, note which two keyframes bracket it, and this records the
pair. `reconstruct_step` then scales the whole reconstruction so those two
camera centres are exactly that far apart.

    python3 scripts/make_baseline.py session_20260803_171339 \\
        --from 4 --to 21 --length-m 1.00

Use `--list` first to see which keyframes exist and when they were captured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session")
    parser.add_argument("--sessions-dir", type=Path, default=REPO_ROOT / "data/scene_sessions")
    parser.add_argument("--from", dest="from_frame", type=int, default=None)
    parser.add_argument("--to", dest="to_frame", type=int, default=None)
    parser.add_argument("--length-m", type=float, default=None)
    parser.add_argument("--list", action="store_true",
                        help="Show the keyframes and their capture times, then exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = args.sessions_dir / args.session
    if not session_dir.is_dir():
        print(f"No such session: {session_dir}", file=sys.stderr)
        return 2

    keyframes = sorted(p for p in (session_dir / "keyframes").glob("*") if p.is_dir())
    if args.list:
        first_ns = None
        print(f"{len(keyframes)} keyframes in {args.session}")
        for path in keyframes:
            meta_path = path / "meta.json"
            stamp = ""
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    ns = int(meta.get("timestamp_ns", 0))
                    first_ns = first_ns if first_ns is not None else ns
                    stamp = f"t+{(ns - first_ns) / 1e9:6.2f}s"
                except (OSError, ValueError):
                    pass
            print(f"  {path.name}  {stamp}  {path / 'rgb.jpg'}")
        print("\nOpen the two images that bracket your measured straight drive, then "
              "re-run with --from N --to M --length-m <tape reading>.")
        return 0

    missing = [n for n, v in (("--from", args.from_frame), ("--to", args.to_frame),
                              ("--length-m", args.length_m)) if v is None]
    if missing:
        print(f"Need {', '.join(missing)} (or --list to browse the keyframes)",
              file=sys.stderr)
        return 2
    if args.length_m <= 0:
        print("--length-m must be positive", file=sys.stderr)
        return 2
    if args.from_frame == args.to_frame:
        print("--from and --to must be different keyframes", file=sys.stderr)
        return 2

    names = {int(p.name) for p in keyframes if p.name.isdigit()}
    for label, value in (("--from", args.from_frame), ("--to", args.to_frame)):
        if names and value not in names:
            print(f"{label} {value} is not a keyframe in this session "
                  f"(range {min(names)}..{max(names)})", file=sys.stderr)
            return 2

    payload = {
        "from_frame": args.from_frame,
        "to_frame": args.to_frame,
        "length_m": args.length_m,
    }
    out = session_dir / "baseline.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}: {json.dumps(payload)}")
    print("Now re-run the reconstruct step with --force to apply it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
