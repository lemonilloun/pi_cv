"""Join recorded frames to recorded actions and emit a training dataset.

The frames come from the Pi (`vla_recorder.py`, 10 Hz, stamped with the Pi's
monotonic clock); the actions come from this laptop (`action_log.py`, stamped
with its own). `shared/clock_sync` measured the offset between the two at
episode start, and this module is where that measurement is finally spent.

Three decisions here are worth more than the code that implements them.

**Zero-order hold, and it is exact rather than an approximation.** A drive
command is not a sample of a continuous signal — it is a step that persists
until replaced. `RobocarService._repeat_loop` literally retransmits the held
command every 150 ms, so between two commands the wheels really were being
told the same thing. Interpolating between them would invent commands that
were never issued.

**Gaps are filled with (0, 0), not skipped.** When the operator releases a
key the panel stops posting, and `COMMAND_TTL_S` (0.6 s) expires into a STOP.
Those stretches are real "hold still" actions and a policy must learn them;
dropping the frames instead would teach it that the robot is always moving,
and it would never stop.

**Frames before the first action are dropped, not zero-filled.** At the head
of an episode there is no evidence either way — the operator may have been
holding a key before recording started. Fabricating a stop there would put a
wrong label on real pixels, which is worse than one less frame.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Must match robocar.COMMAND_TTL_S — after this long with no refresh the
# server itself sends STOP, so the recorded reality is (0, 0).
COMMAND_TTL_S = 0.6
PWM_LIMIT = 255.0
OBS_STATE_DIM = 7


@dataclass(frozen=True)
class ActionEvent:
    t_pi_ns: int          # already converted into the Pi's timeline
    left: int
    right: int
    source: str = "panel"


def actions_to_pi_timeline(records: list[dict[str, Any]], offset_ns: int) -> list[ActionEvent]:
    """Convert laptop-stamped drive records into the Pi's monotonic clock.

    `offset_ns` is `pi + offset = mac`, so the inverse is applied here. Only
    `drive` rows are used; heartbeats are diagnostics for verifying the join,
    not labels.
    """
    events = [
        ActionEvent(
            t_pi_ns=int(record["t_mono_ns"]) - offset_ns,
            left=int(record["left"]),
            right=int(record["right"]),
            source=str(record.get("source", "panel")),
        )
        for record in records
        if record.get("kind") == "drive"
    ]
    events.sort(key=lambda e: e.t_pi_ns)
    return events


def resample_actions(
    frame_times_ns: list[int],
    events: list[ActionEvent],
    ttl_s: float = COMMAND_TTL_S,
) -> list[tuple[int, int] | None]:
    """One action per frame by zero-order hold, with TTL expiry.

    Returns None for frames that precede the first command — those have no
    label and must be dropped rather than guessed at.
    """
    out: list[tuple[int, int] | None] = []
    ttl_ns = int(ttl_s * 1e9)
    cursor = 0
    held: ActionEvent | None = None
    for t_ns in frame_times_ns:
        while cursor < len(events) and events[cursor].t_pi_ns <= t_ns:
            held = events[cursor]
            cursor += 1
        if held is None:
            out.append(None)
        elif t_ns - held.t_pi_ns > ttl_ns:
            # The server would have sent STOP by now, so this is a real
            # (0, 0), not a missing label.
            out.append((0, 0))
        else:
            out.append((held.left, held.right))
    return out


def observation_state(
    imu: dict[str, Any] | None,
    yaw0_deg: float | None,
    prev_action: tuple[int, int],
    yaw_rate_dps: float,
) -> list[float]:
    """The 7-vector the policy sees alongside the image.

    `[sin(yaw_rel), cos(yaw_rel), pitch/90, roll/90, yaw_rate/180,
      prev_left/255, prev_right/255]`

    Yaw is RELATIVE to the episode's first sample. The RVC datum is whatever
    the sensor powered up facing, so an absolute heading is a different random
    constant in every episode — a feature the policy would happily memorize
    into noise. sin/cos rather than the angle so there is no wrap
    discontinuity at +/-180.

    `prev_left/right` are here because the ESP ramps toward a commanded PWM at
    8 units per 15 ms, so what the motors are actually doing depends on what
    was commanded a moment ago. Without it the action space has hidden state
    and the policy cannot model its own inertia.
    """
    yaw_rel = 0.0
    pitch = roll = 0.0
    if imu:
        if yaw0_deg is not None and imu.get("yaw_deg") is not None:
            yaw_rel = math.radians(float(imu["yaw_deg"]) - yaw0_deg)
        pitch = float(imu.get("pitch_deg") or 0.0)
        roll = float(imu.get("roll_deg") or 0.0)
    return [
        math.sin(yaw_rel),
        math.cos(yaw_rel),
        pitch / 90.0,
        roll / 90.0,
        max(-1.0, min(1.0, yaw_rate_dps / 180.0)),
        prev_action[0] / PWM_LIMIT,
        prev_action[1] / PWM_LIMIT,
    ]


def yaw_rate_series(frames: list[dict[str, Any]]) -> list[float]:
    """Degrees per second, by differencing the fused yaw between frames.

    RVC exposes no raw gyro, so this is the rig's only proprioceptive sense
    of turning. The yaw the recorder stores is already unwrapped (see
    `imu_rvc.RvcReader.read_orientation`), so a plain difference is safe.
    """
    rates = [0.0]
    for previous, current in zip(frames, frames[1:]):
        p_imu, c_imu = previous.get("imu") or {}, current.get("imu") or {}
        dt = (int(current["t_pi_mono_ns"]) - int(previous["t_pi_mono_ns"])) / 1e9
        if dt <= 0 or p_imu.get("yaw_deg") is None or c_imu.get("yaw_deg") is None:
            rates.append(rates[-1])
            continue
        rates.append((float(c_imu["yaw_deg"]) - float(p_imu["yaw_deg"])) / dt)
    return rates


def mirror_sample(state: list[float], action: tuple[int, int]) -> tuple[list[float], tuple[int, int]]:
    """Left-right mirror of one sample, for use with a horizontally flipped
    image.

    An exact symmetry of a differential drive rather than an approximation:
    reflect the world and swap the wheels and the robot does the mirrored
    thing. That makes it a free and *correct* doubling of the dataset, which
    most manipulation VLA pipelines cannot use because an arm's workspace is
    not symmetric.

    Mirroring negates the heading and the turn rate, swaps the previous wheel
    pair, and leaves pitch alone; roll flips sign with the world.
    """
    sin_yaw, cos_yaw, pitch, roll, yaw_rate, prev_l, prev_r = state
    return (
        [-sin_yaw, cos_yaw, pitch, -roll, -yaw_rate, prev_r, prev_l],
        (action[1], action[0]),
    )


def build_episode(episode_dir: Path) -> dict[str, Any]:
    """Join one episode into per-frame samples, with a quality report."""
    meta = json.loads((episode_dir / "episode_meta.json").read_text(encoding="utf-8"))
    frames = [
        json.loads(line)
        for line in (episode_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frames.sort(key=lambda f: int(f["frame_idx"]))

    actions_path = episode_dir / "actions.jsonl"
    if not actions_path.exists():
        raise FileNotFoundError(
            f"{episode_dir.name} has frames but no actions.jsonl — the recording was "
            f"started without arming the action log, so nothing links the pixels to "
            f"what the robot was told to do. The episode is not usable."
        )
    records = [
        json.loads(line)
        for line in actions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    offset_ns = int(meta["clock"]["start"]["offset_ns"])
    events = actions_to_pi_timeline(records, offset_ns)
    frame_times = [int(f["t_pi_mono_ns"]) for f in frames]
    labels = resample_actions(frame_times, events)
    rates = yaw_rate_series(frames)

    yaw0 = None
    for frame in frames:
        imu = frame.get("imu") or {}
        if imu.get("yaw_deg") is not None:
            yaw0 = float(imu["yaw_deg"])
            break

    samples: list[dict[str, Any]] = []
    prev_action = (0, 0)
    unlabeled = 0
    for frame, action, rate in zip(frames, labels, rates):
        if action is None:
            unlabeled += 1
            continue
        samples.append({
            "frame": episode_dir / "frames" / f"{int(frame['frame_idx']):06d}.jpg",
            "state": observation_state(frame.get("imu"), yaw0, prev_action, rate),
            "action": [action[0] / PWM_LIMIT, action[1] / PWM_LIMIT],
            "raw_action": action,
            "t_pi_mono_ns": int(frame["t_pi_mono_ns"]),
        })
        prev_action = action

    stopped = sum(1 for s in samples if s["raw_action"] == (0, 0))
    return {
        "episode_id": episode_dir.name,
        "task": meta.get("task"),
        "samples": samples,
        "frames_total": len(frames),
        "frames_unlabeled_dropped": unlabeled,
        "stopped_frac": round(stopped / len(samples), 4) if samples else 0.0,
        "action_events": len(events),
        "clock_quality_ms": meta["clock"]["start"].get("quality_ms"),
        "max_pwm": next((r.get("max_pwm") for r in records if r.get("kind") == "drive"), None),
    }


def check_episode(report: dict[str, Any]) -> list[str]:
    """Problems that make an episode unfit for training.

    Deliberately opinionated: a dataset assembled from episodes nobody looked
    at is how a policy ends up learning an artefact of the recording rig.
    """
    problems = []
    if not report["samples"]:
        problems.append("no labelled frames at all")
        return problems
    if report["frames_unlabeled_dropped"] > 0.25 * report["frames_total"]:
        problems.append(
            f"{report['frames_unlabeled_dropped']}/{report['frames_total']} frames had no "
            f"action yet — the action log was armed late relative to the camera"
        )
    quality = report.get("clock_quality_ms")
    if quality is not None and quality > 20.0:
        problems.append(
            f"clock offset error bar {quality:.1f} ms exceeds the 20 ms budget, so "
            f"frames and actions may be misaligned by more than a control step"
        )
    if report["action_events"] < 5:
        problems.append(
            f"only {report['action_events']} drive commands in the whole episode — "
            f"was the robot actually driven?"
        )
    if report["stopped_frac"] > 0.9:
        problems.append(
            f"{report['stopped_frac']:.0%} of frames are (0, 0) — almost nothing happened"
        )
    return problems


# ---------------------------------------------------------------- export
# Writing the joined samples out for training.
#
# **Why a self-describing manifest and not only LeRobot.** The LeRobot
# on-disk format has churned across releases (v2.0 -> v2.1 -> v3.0), and this
# dataset will be collected over weeks before anything trains on it. Pinning
# a release is necessary but not sufficient: the recording is the expensive,
# unrepeatable part, and it must not become unreadable because a library moved
# on. So the export always writes a plain, versioned manifest — JSONL rows
# plus the original frames — and additionally emits a LeRobot dataset when the
# library is present. The manifest is trivially convertible to whatever format
# is current at training time; a half-migrated LeRobot directory is not.

MANIFEST_VERSION = 1


def episode_splits(
    reports: list[dict[str, Any]], holdout_frac: float = 0.2, seed: int = 0
) -> dict[str, list[str]]:
    """Split by whole EPISODE, never by frame.

    Adjacent frames at 10 Hz are near-duplicates, so a frame-level split puts
    almost every validation frame within 100 ms of a training one. The
    resulting curve is beautiful and means nothing.
    """
    import random

    ids = sorted(r["episode_id"] for r in reports)
    rng = random.Random(seed)
    rng.shuffle(ids)
    cut = max(1, int(round(len(ids) * holdout_frac))) if len(ids) > 1 else 0
    return {"val": sorted(ids[:cut]), "train": sorted(ids[cut:])}


def write_manifest(
    reports: list[dict[str, Any]],
    out_dir: Path,
    mirror: bool = True,
) -> dict[str, Any]:
    """Emit the format-independent record: one JSONL row per sample.

    `mirror` adds the horizontally-flipped copy of every sample. For a
    differential drive that reflection is an EXACT symmetry of the dynamics —
    flip the world, swap the wheels, and the robot does the mirrored thing —
    so it is a free and *correct* doubling rather than an approximation. Most
    manipulation pipelines cannot use it because an arm's workspace is not
    symmetric; a two-wheeled base is.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = episode_splits(reports)
    split_of = {eid: name for name, ids in splits.items() for eid in ids}

    rows = 0
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for report in reports:
            for sample in report["samples"]:
                base = {
                    "episode_id": report["episode_id"],
                    "split": split_of.get(report["episode_id"], "train"),
                    "task": report["task"],
                    "frame": str(sample["frame"]),
                    "t_pi_mono_ns": sample["t_pi_mono_ns"],
                    "state": sample["state"],
                    # Derived from raw_action, the same source the mirrored
                    # row uses. Reading one from `action` and the other from
                    # `raw_action` would let a mirrored pair disagree if the
                    # two ever drift apart.
                    "action": [sample["raw_action"][0] / PWM_LIMIT,
                               sample["raw_action"][1] / PWM_LIMIT],
                    "mirrored": False,
                }
                handle.write(json.dumps(base) + "\n")
                rows += 1
                if mirror:
                    state, action = mirror_sample(sample["state"], tuple(sample["raw_action"]))
                    handle.write(json.dumps({
                        **base,
                        "state": state,
                        "action": [action[0] / PWM_LIMIT, action[1] / PWM_LIMIT],
                        "mirrored": True,
                    }) + "\n")
                    rows += 1

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "rows": rows,
        "mirrored": mirror,
        "episodes": [r["episode_id"] for r in reports],
        "splits": splits,
        "state_layout": [
            "sin(yaw_rel)", "cos(yaw_rel)", "pitch/90", "roll/90",
            "yaw_rate/180", "prev_left/255", "prev_right/255",
        ],
        "action_layout": ["left/255", "right/255"],
        "control_hz": 10.0,
        "notes": (
            "Yaw is relative to each episode's first sample: the RVC datum is "
            "whatever the sensor powered up facing, so an absolute heading is a "
            "different random constant per episode. Mirrored rows are an exact "
            "symmetry of a differential drive, not an augmentation heuristic."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_lerobot(reports: list[dict[str, Any]], out_dir: Path,
                  repo_id: str = "pi_cv/robocar") -> dict[str, Any]:
    """Best-effort LeRobot dataset. Never the only output — see the note above.

    Returns a skipped-with-reason record rather than raising when the library
    is absent or its API has moved: the manifest is already written by then,
    and losing an export is not a reason to lose a recording session.
    """
    try:
        import lerobot                       # noqa: F401
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        return {"written": False, "reason": f"lerobot unavailable: {exc}"}

    try:
        features = {
            "observation.images.front": {"dtype": "video", "shape": (360, 640, 3),
                                         "names": ["height", "width", "channel"]},
            "observation.state": {"dtype": "float32", "shape": (OBS_STATE_DIM,),
                                  "names": ["state"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["action"]},
        }
        dataset = LeRobotDataset.create(repo_id=repo_id, fps=10, root=out_dir,
                                        features=features)
        import cv2
        import numpy as np

        for report in reports:
            for sample in report["samples"]:
                image = cv2.imread(str(sample["frame"]))
                if image is None:
                    continue
                dataset.add_frame({
                    "observation.images.front": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    "observation.state": np.asarray(sample["state"], dtype=np.float32),
                    "action": np.asarray(sample["action"], dtype=np.float32),
                }, task=report["task"] or "drive")
            dataset.save_episode()
        return {"written": True, "version": getattr(lerobot, "__version__", "unknown")}
    except Exception as exc:
        return {"written": False, "reason": f"lerobot export failed: {exc}"}


def export(episodes_dir: Path, out_dir: Path, mirror: bool = True,
           skip_bad: bool = True) -> dict[str, Any]:
    """Join every episode and write the dataset.

    Episodes that fail `check_episode` are excluded by default and listed in
    the result. Training on an episode whose clock sync was loose, or whose
    action log armed late, teaches a subtly wrong policy that no later metric
    will attribute to the data.
    """
    reports, rejected = [], []
    for path in sorted(episodes_dir.glob("ep_*")):
        try:
            report = build_episode(path)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            rejected.append({"episode_id": path.name, "problems": [str(exc)]})
            continue
        problems = check_episode(report)
        if problems and skip_bad:
            rejected.append({"episode_id": path.name, "problems": problems})
            continue
        reports.append(report)

    if not reports:
        return {"episodes_used": 0, "rejected": rejected,
                "reason": "no usable episodes"}

    manifest = write_manifest(reports, out_dir, mirror=mirror)
    return {
        "episodes_used": len(reports),
        "rejected": rejected,
        "manifest": manifest,
        "lerobot": write_lerobot(reports, out_dir / "lerobot"),
        "out_dir": str(out_dir),
    }
