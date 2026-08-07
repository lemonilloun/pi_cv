"""Render a recorded episode as video with its action trace drawn on top.

The single most valuable check in the whole dataset pipeline, and the one no
unit test can replace: **when the operator pressed left, the picture must
turn left**.

Everything upstream can be individually correct and the dataset still wrong.
`mix_drive` deliberately swaps the wheels to compensate this chassis' mirrored
motor leads; the RVC yaw is a compass heading and gets negated at the driver;
the frames are stamped on the Pi and the actions on the laptop, joined through
a measured clock offset. Any one of those conventions inverted, or the join
shifted by a second, produces a dataset that looks perfectly healthy in every
summary statistic and trains a policy to drive the wrong way.

Watching thirty seconds of overlaid video settles all of it at once.

    python3 -m mac_server.vla.replay ep_20260803_180000 --out replay.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

PWM_LIMIT = 255.0


def draw_overlay(cv2, frame: Any, sample: dict[str, Any], task: str, index: int,
                 total: int, np: Any) -> Any:
    """Wheel bars, heading dial and the task text, burned into one frame.

    The wheel bars are drawn on the side of the image the wheel is on, and
    grow up for forward and down for reverse. That is deliberate: it makes a
    left/right swap immediately obvious as "the bar on the wrong side grows",
    rather than something you have to read off a number and reason about.
    """
    height, width = frame.shape[:2]
    canvas = frame.copy()
    left, right = sample["raw_action"]
    state = sample["state"]

    panel_h = 92
    cv2.rectangle(canvas, (0, height - panel_h), (width, height), (18, 18, 18), -1)

    bar_w, bar_max = 46, 34
    mid_y = height - panel_h // 2
    for label, value, cx in (("L", left, 40), ("R", right, width - 40)):
        cv2.line(canvas, (cx - bar_w // 2, mid_y), (cx + bar_w // 2, mid_y), (90, 90, 90), 1)
        extent = int(bar_max * max(-1.0, min(1.0, value / PWM_LIMIT)))
        colour = (70, 200, 70) if value >= 0 else (60, 110, 235)
        if extent:
            cv2.rectangle(canvas, (cx - bar_w // 2, mid_y),
                          (cx + bar_w // 2, mid_y - extent), colour, -1)
        cv2.putText(canvas, f"{label} {value:+4d}", (cx - 34, mid_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)

    # Heading dial from the state vector's sin/cos pair. Relative to the
    # episode's first frame, which is the only heading that means anything
    # across episodes (the RVC datum is per-power-up random).
    import math

    yaw = math.atan2(state[0], state[1])
    dial_cx, dial_r = width // 2, 30
    cv2.circle(canvas, (dial_cx, mid_y), dial_r, (90, 90, 90), 1)
    tip = (int(dial_cx + dial_r * 0.85 * math.sin(yaw)),
           int(mid_y - dial_r * 0.85 * math.cos(yaw)))
    cv2.line(canvas, (dial_cx, mid_y), tip, (90, 200, 255), 2)
    cv2.putText(canvas, f"yaw {math.degrees(yaw):+6.1f}", (dial_cx - 44, mid_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)

    cv2.putText(canvas, f'"{task}"   {index + 1}/{total}', (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
    if left == 0 and right == 0:
        cv2.putText(canvas, "STOPPED", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (60, 110, 235), 2, cv2.LINE_AA)
    return canvas


def turn_direction(sample: dict[str, Any]) -> str:
    """What the action says the robot is doing, in words.

    `mix_drive` returns the wheels swapped for this chassis, so the pair
    (left > right) means a LEFT turn here. Spelling it out in the overlay is
    what turns "watch the video" into a decisive test rather than a vibe.
    """
    left, right = sample["raw_action"]
    if left == 0 and right == 0:
        return "stop"
    if abs(left - right) < 0.15 * PWM_LIMIT:
        return "forward" if left > 0 else "reverse"
    return "LEFT" if left > right else "RIGHT"


def render(episode_dir: Path, out_path: Path, fps: float = 10.0,
           scale: float = 1.0) -> dict[str, Any]:
    import cv2
    import numpy as np

    from mac_server.vla.build_dataset import build_episode

    report = build_episode(episode_dir)
    samples = report["samples"]
    if not samples:
        raise RuntimeError(f"{episode_dir.name} has no labelled frames to render")

    first = cv2.imread(str(samples[0]["frame"]))
    if first is None:
        raise RuntimeError(f"Cannot read {samples[0]['frame']}")
    height, width = first.shape[:2]
    if scale != 1.0:
        width, height = int(width * scale), int(height * scale)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open {out_path} for writing")

    counts: dict[str, int] = {}
    try:
        for index, sample in enumerate(samples):
            frame = cv2.imread(str(sample["frame"]))
            if frame is None:
                continue
            if scale != 1.0:
                frame = cv2.resize(frame, (width, height))
            counts[turn_direction(sample)] = counts.get(turn_direction(sample), 0) + 1
            writer.write(draw_overlay(cv2, frame, sample, report["task"] or "",
                                      index, len(samples), np))
    finally:
        writer.release()

    return {
        "episode_id": report["episode_id"],
        "task": report["task"],
        "frames_rendered": len(samples),
        "seconds": round(len(samples) / fps, 1),
        "action_mix": counts,
        "output": str(out_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episode")
    parser.add_argument("--episodes-dir", type=Path,
                        default=REPO_ROOT / "data/vla_episodes")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    import json

    args = parse_args()
    episode_dir = args.episodes_dir / args.episode
    if not episode_dir.is_dir():
        print(f"No such episode: {episode_dir}", file=sys.stderr)
        return 2
    out = args.out or (episode_dir / "replay.mp4")
    report = render(episode_dir, out, fps=args.fps, scale=args.scale)
    print(json.dumps(report, indent=2))
    print(f"\nWatch it: {out}")
    print("The one thing to check: when a bar grows on the LEFT and shrinks on the "
          "RIGHT, the picture must swing left. If it swings right, a wheel or sign "
          "convention is inverted somewhere and the dataset is unusable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
