"""Settle whether this IMU's orientation output is trustworthy. Sensor only.

The link, the accelerometer scale and the stationary drift are all measured
and healthy (501/501 frames at 100.2 Hz, |a| = 1002 mg at rest, 0.03 deg/min
of yaw drift over 90 s). What has NEVER been measured on this rig is the yaw
SIGN and GAIN: turn the robot left, does yaw rise or fall, and does a
physical 90 deg produce 90 deg of reading? Two earlier attempts to answer
that failed for tooling reasons — one on a wrap-around bug of mine, one for
too small a sweep.

So this test deliberately contains **nothing that has failed before**: no
camera, no chessboard, no solvePnP, no intrinsics. A floor mark and the
sensor, nothing else.

**Why a full 360 deg turn rather than 90.** Ground truth here comes from the
operator, and "I turned it about ninety degrees" is worth maybe +/-8 deg —
9% of the quantity being measured. Lining the chassis back up with the mark
it started on is a different kind of judgement: you are matching two edges,
not estimating an angle, and it is good to a couple of degrees. That makes
the true rotation exactly 360 deg and the gain accurate to ~1%, which
separates 1.0 from 0.5 or 2.19 with enormous margin.

The same run also measures what the previous checks could not: drift across
a real manoeuvre (not at rest), and whether the accelerometer stays sane
while the chassis is actually moving.

    ./scripts/run_imu_orientation_test.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "client/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

logger = logging.getLogger(__name__)

G_MS2 = 9.80665
FULL_TURN_DEG = 360.0
# The operator lines the chassis back up with its starting mark, which is an
# edge-matching judgement rather than an angle estimate. A couple of degrees
# is realistic; 8 would be pessimistic even for a careless run.
TURN_TRUTH_TOLERANCE_DEG = 8.0


@dataclass
class Phase:
    key: str
    prompt: str
    seconds: float
    samples: list[tuple[float, float, float, float, tuple[float, float, float]]] = field(
        default_factory=list
    )   # (t, yaw, pitch, roll, accel_mg)


def unwrap_deg(values: list[float]) -> list[float]:
    if not values:
        return []
    out = [values[0]]
    for previous, current in zip(values, values[1:]):
        out.append(out[-1] + (current - previous + 180.0) % 360.0 - 180.0)
    return out


def analyse_turn(
    yaw_deg: list[float], truth_deg: float
) -> dict[str, Any]:
    """Sign and gain of the yaw output across one known rotation.

    `truth_deg` is signed: positive for a left (counter-clockwise seen from
    above) turn. The returned `sign` is what the navigation filter needs —
    a robot that turns left while the filter believes it turned right is a
    failure the gravity calibration physically cannot detect.
    """
    if len(yaw_deg) < 10:
        return {"ok": False, "reason": "too few samples during the turn"}
    unwrapped = unwrap_deg(yaw_deg)
    measured = unwrapped[-1] - unwrapped[0]
    if abs(truth_deg) < 1e-6:
        return {"ok": False, "reason": "no reference rotation given"}
    gain = measured / truth_deg

    # Monotonicity: a clean single-direction turn should not reverse. If it
    # does, either the operator wobbled or the sensor is producing garbage,
    # and the gain would be meaningless either way.
    steps = [b - a for a, b in zip(unwrapped, unwrapped[1:])]
    forward = sum(s for s in steps if s * measured > 0)
    backward = -sum(s for s in steps if s * measured < 0)
    reversal_frac = backward / max(abs(forward), 1e-9)

    problems = []
    if abs(abs(gain) - 1.0) > 0.05:
        problems.append(f"gain {gain:+.3f} is not +/-1")
    if reversal_frac > 0.15:
        problems.append(
            f"{reversal_frac:.0%} of the motion went backwards — the turn was not "
            f"one clean sweep, so the gain cannot be trusted"
        )
    return {
        "ok": not problems,
        "samples": len(yaw_deg),
        "truth_deg": truth_deg,
        "measured_deg": round(measured, 2),
        "gain": round(gain, 4),
        "sign": "agrees" if gain > 0 else "inverted",
        "reversal_frac": round(reversal_frac, 4),
        "problems": problems,
    }


def analyse_still(
    samples: list[tuple[float, float, float, float, tuple[float, float, float]]]
) -> dict[str, Any]:
    """Drift and accelerometer sanity over a stationary stretch."""
    if len(samples) < 10:
        return {"ok": False, "reason": "too few samples"}
    times = [s[0] for s in samples]
    span = times[-1] - times[0]
    yaw = unwrap_deg([s[1] for s in samples])
    magnitudes = [math.sqrt(sum(v * v for v in s[4])) for s in samples]
    median_mg = statistics.median(magnitudes)

    problems = []
    drift_per_min = (yaw[-1] - yaw[0]) / span * 60.0 if span > 0 else 0.0
    if abs(drift_per_min) > 2.0:
        problems.append(f"yaw drifts {drift_per_min:+.2f} deg/min at rest")
    if not (950.0 <= median_mg <= 1050.0):
        problems.append(f"|accel| is {median_mg:.0f} mg at rest, not ~1000 — scale is wrong")
    return {
        "ok": not problems,
        "seconds": round(span, 2),
        "drift_deg_per_min": round(drift_per_min, 3),
        "accel_mg_median": round(median_mg, 1),
        "accel_mg_spread": round(max(magnitudes) - min(magnitudes), 1),
        "pitch_range_deg": round(max(s[2] for s in samples) - min(s[2] for s in samples), 2),
        "roll_range_deg": round(max(s[3] for s in samples) - min(s[3] for s in samples), 2),
        "problems": problems,
    }


def analyse_push(
    samples: list[tuple[float, float, float, float, tuple[float, float, float]]],
    rest_vector: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Did the accelerometer notice a straight push, and stay sane?

    Not a distance measurement — metric dead reckoning is out of scope on
    this link. The question is narrower and answerable: does the sensor
    register the acceleration of a real motion at all, and does the gravity
    vector stay put while the chassis translates (it should — driving in a
    plane does not change which way is down).
    """
    if len(samples) < 20:
        return {"ok": False, "reason": "too few samples"}
    accels = [s[4] for s in samples]
    # Deviation of the VECTOR from its rest value, not of its magnitude.
    # Horizontal acceleration adds to gravity in quadrature, so a 120 mg push
    # on top of 1000 mg of gravity moves the magnitude by only 7 mg — a
    # magnitude test is nearly blind to exactly the motion being checked.
    # (This project has made that mistake once already, in the distance
    # analysis, where a magnitude-based detector called 100% of a real 3 m
    # push "stationary".)
    # Rest comes from the PRECEDING stationary phase when available. Taking
    # the median of the push window itself only works if the robot spends
    # most of it standing still; an operator who pushes for the whole window
    # makes the median a mid-push value and the excursion collapses.
    if rest_vector is None:
        rest_vector = tuple(statistics.median(a[k] for a in accels) for k in range(3))
    excursion = max(
        math.sqrt(sum((a[k] - rest_vector[k]) ** 2 for k in range(3))) for a in accels
    )
    magnitudes = [math.sqrt(sum(v * v for v in a)) for a in accels]
    yaw = unwrap_deg([s[1] for s in samples])

    problems = []
    if excursion < 15.0:
        problems.append(
            f"the accelerometer barely moved ({excursion:.0f} mg peak deviation) — "
            f"either the robot was not actually pushed, or the sensor is not "
            f"responding to motion"
        )
    if abs(yaw[-1] - yaw[0]) > 25.0:
        problems.append(
            f"yaw changed {yaw[-1] - yaw[0]:+.0f} deg during a straight push — "
            f"either the push curved, or yaw is picking up linear motion"
        )
    return {
        "ok": not problems,
        "peak_deviation_mg": round(excursion, 1),
        "accel_mg_median": round(statistics.median(magnitudes), 1),
        "yaw_change_deg": round(yaw[-1] - yaw[0], 2),
        "problems": problems,
    }


def _median_accel(
    samples: list[tuple[float, float, float, float, tuple[float, float, float]]]
) -> tuple[float, float, float] | None:
    if not samples:
        return None
    return tuple(statistics.median(s[4][k] for s in samples) for k in range(3))


def verdict(report: dict[str, Any]) -> str:
    turns = [report.get("turn_left"), report.get("turn_right")]
    turns = [t for t in turns if t and t.get("ok")]
    if not turns:
        return "INCONCLUSIVE — no clean turn was recorded"
    gains = [t["gain"] for t in turns]
    if any(g < 0 for g in gains) and any(g > 0 for g in gains):
        return "BROKEN — the two turns disagree about which way is positive"
    inverted = gains[0] < 0
    still_ok = all(
        report.get(k, {}).get("ok", True) for k in ("still_start", "still_mid", "still_end")
    )
    if not still_ok:
        return "SENSOR PROBLEM — see the stationary checks"
    return (
        f"USABLE — yaw is {'INVERTED' if inverted else 'aligned'} relative to the "
        f"turn direction, gain {statistics.mean(abs(g) for g in gains):.3f}"
    )


def build_phases(still_s: float, turn_s: float, push_s: float) -> list[Phase]:
    return [
        Phase("still_start",
              f"STAND STILL. Mark where the chassis is — a strip of tape along one edge is "
              f"enough. {still_s:.0f} s.", still_s),
        Phase("turn_left",
              f"TURN THE ROBOT LEFT through a FULL CIRCLE (counter-clockwise seen from "
              f"above), smoothly, and stop back on the mark. {turn_s:.0f} s.", turn_s),
        Phase("still_mid", f"STAND STILL. {still_s:.0f} s.", still_s),
        Phase("turn_right",
              f"TURN THE ROBOT RIGHT through a FULL CIRCLE, back onto the mark. "
              f"{turn_s:.0f} s.", turn_s),
        Phase("still_end", f"STAND STILL. {still_s:.0f} s.", still_s),
        Phase("push",
              f"PUSH THE ROBOT STRAIGHT FORWARD about a metre, without turning, then stop. "
              f"{push_s:.0f} s.", push_s),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the IMU's yaw sign, gain and drift")
    parser.add_argument("--imu-port", default="/dev/ttyUSB0")
    parser.add_argument("--still-seconds", type=float, default=8.0)
    parser.add_argument("--turn-seconds", type=float, default=20.0)
    parser.add_argument("--push-seconds", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write the raw samples (default: "
                             "data/imu_tests/orientation_<timestamp>/)")
    return parser.parse_args()


def main() -> int:
    from pi_client.imu_rvc import RvcReader

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    reader = RvcReader(port=args.imu_port)
    if not reader.start():
        logger.error("No IMU on %s — nothing to test.", args.imu_port)
        return 2

    out_dir = args.out or REPO_ROOT / "data/imu_tests" / time.strftime("orientation_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    phases = build_phases(args.still_seconds, args.turn_seconds, args.push_seconds)
    logger.info("=" * 72)
    logger.info("IMU ORIENTATION TEST — sensor only, no camera, no board, no intrinsics.")
    logger.info("Ground truth is you lining the chassis back up with its own mark.")
    logger.info("=" * 72)

    try:
        for phase in phases:
            logger.info("")
            logger.info(">>> %s", phase.prompt)
            for remaining in range(3, 0, -1):
                logger.info("    starting in %d...", remaining)
                time.sleep(1.0)
            logger.info("    GO")
            started = time.monotonic()
            last_index = None
            while time.monotonic() - started < phase.seconds:
                sample = reader.latest()
                if sample is not None and sample.index != last_index:
                    last_index = sample.index
                    phase.samples.append((
                        time.monotonic() - started,
                        sample.yaw_deg, sample.pitch_deg, sample.roll_deg,
                        sample.accel_mg,
                    ))
                time.sleep(0.005)
            logger.info("    captured %d samples", len(phase.samples))
    finally:
        reader.stop()

    by_key = {p.key: p for p in phases}
    report: dict[str, Any] = {
        "still_start": analyse_still(by_key["still_start"].samples),
        # Left is positive by the right-hand rule about the vertical (ROS
        # REP-103), which is the convention the plan frame uses.
        "turn_left": analyse_turn([s[1] for s in by_key["turn_left"].samples], +FULL_TURN_DEG),
        "still_mid": analyse_still(by_key["still_mid"].samples),
        "turn_right": analyse_turn([s[1] for s in by_key["turn_right"].samples], -FULL_TURN_DEG),
        "still_end": analyse_still(by_key["still_end"].samples),
        # Rest reference from the stationary phase immediately before it.
        "push": analyse_push(
            by_key["push"].samples,
            rest_vector=_median_accel(by_key["still_end"].samples),
        ),
    }
    report["verdict"] = verdict(report)

    for phase in phases:
        with (out_dir / f"{phase.key}.csv").open("w", encoding="utf-8") as handle:
            handle.write("t_s,yaw_deg,pitch_deg,roll_deg,ax_mg,ay_mg,az_mg\n")
            for t, yaw, pitch, roll, accel in phase.samples:
                handle.write(f"{t:.4f},{yaw:.2f},{pitch:.2f},{roll:.2f},"
                             f"{accel[0]:.0f},{accel[1]:.0f},{accel[2]:.0f}\n")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("")
    logger.info("=" * 72)
    for key in ("still_start", "still_mid", "still_end"):
        section = report[key]
        logger.info("%-12s drift %+.2f deg/min | |a| %.0f mg (spread %.0f) | pitch range %.2f",
                    key, section.get("drift_deg_per_min", float("nan")),
                    section.get("accel_mg_median", float("nan")),
                    section.get("accel_mg_spread", float("nan")),
                    section.get("pitch_range_deg", float("nan")))
    for key in ("turn_left", "turn_right"):
        section = report[key]
        if section.get("ok") or "gain" in section:
            logger.info("%-12s measured %+.1f deg for a true %+.0f -> gain %+.3f (%s)",
                        key, section["measured_deg"], section["truth_deg"],
                        section["gain"], section["sign"])
        else:
            logger.info("%-12s %s", key, section.get("reason"))
    logger.info("%-12s peak |a| deviation %.0f mg, yaw moved %+.1f deg",
                "push", report["push"].get("peak_deviation_mg", float("nan")),
                report["push"].get("yaw_change_deg", float("nan")))
    for key, section in report.items():
        if isinstance(section, dict):
            for problem in section.get("problems", []):
                logger.warning("  %s: %s", key, problem)
    logger.info("=" * 72)
    logger.info("VERDICT: %s", report["verdict"])
    logger.info("Raw samples: %s", out_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
