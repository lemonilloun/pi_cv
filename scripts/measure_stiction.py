#!/usr/bin/env python3
"""Measure where this chassis actually starts moving, and write the profile.

`drive_profile.DriveProfile` ships with placeholder thresholds, and a
placeholder that looks authoritative is how a guess turns into a fact. This
replaces them with numbers from the robot in front of you.

**Why it must be re-measured, not measured once.** The threshold is not a
property of the motors alone — it moves with battery voltage, floor surface
and how much the chassis is carrying. Re-run it after changing the battery,
the payload, or the room.

**BLOCKED — no motion sensor reaches the server any more.** Motion was
detected from IMU yaw arriving in `nav_query`, and that message went with
the metric localization stack. The measurement procedure below is kept
because it is the right procedure, not because it currently runs; it needs
a heading (or any motion) feed restored first, and `status["heading_deg"]`
no longer exists, so it refuses immediately rather than reporting a
threshold it did not measure.

What it measured, and why that is the useful quantity: the TURNING
threshold. Turning in place scrubs both wheels sideways and needs far more
than driving straight. The straight threshold was estimated from it and
flagged as an estimate.

    ./scripts/run_server.sh                      # laptop, in another shell
    python3 scripts/measure_stiction.py          # refuses: no heading source
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

# Turning needs more than driving straight because both wheels scrub
# sideways rather than rolling. This ratio is a rule of thumb used only to
# ESTIMATE the straight threshold from the measured turning one; the result
# is marked as an estimate so nobody mistakes it for a measurement.
STRAIGHT_FROM_TURN_RATIO = 0.65


def panel_get(base: str, path: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(f"{base}{path}", timeout=timeout) as response:
        return json.loads(response.read())


def panel_post(base: str, path: str, payload: dict, timeout: float = 5.0) -> dict:
    request = urllib.request.Request(
        f"{base}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", default="http://127.0.0.1:8080")
    parser.add_argument("--from-request", type=int, default=60)
    parser.add_argument("--to-request", type=int, default=255)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--pulse-s", type=float, default=0.6,
                        help="How long each trial command is held. Short on "
                             "purpose: a stalled motor is drawing its peak "
                             "current the whole time.")
    parser.add_argument("--settle-s", type=float, default=1.5)
    parser.add_argument("--moved-deg", type=float, default=3.0,
                        help="Yaw change that counts as 'it moved'. Well above "
                             "the sensor's 0.03 deg/min drift and its 0.05 deg "
                             "jitter, so a positive is never noise.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "config/drive_profile.json")
    return parser.parse_args()


def main() -> int:
    from mac_server.drive_profile import DriveProfile, request_to_motor_pwm

    args = parse_args()
    try:
        status = panel_get(args.panel, "/api/robot/status")
    except OSError as exc:
        print(f"Cannot reach the panel at {args.panel}: {exc}\n"
              f"Start the server first: ./scripts/run_server.sh", file=sys.stderr)
        return 2
    if not status.get("connected"):
        print("No robot connected — power the ESP and wait for it to find the server.",
              file=sys.stderr)
        return 2
    if status.get("heading_deg") is None:
        print("No motion feed reaching the server — this script is currently "
              "BLOCKED (see the module docstring). It detected motion from IMU "
              "yaw carried by nav_query, which no longer exists.", file=sys.stderr)
        return 2

    max_pwm = int(status.get("max_pwm", 150))
    print(f"Robot at {status['address']}, MAX_PWM {max_pwm}")
    print("Put it on the floor it will actually drive on, with the real payload.\n")

    trials = []
    threshold_request = None
    for request in range(args.from_request, args.to_request + 1, args.step):
        before = panel_get(args.panel, "/api/robot/status").get("heading_deg")
        # Spin in place: the wheels fight each other, which is the hardest
        # case and the one the localization survey repeats.
        panel_post(args.panel, "/api/robot/drive",
                   {"throttle": 0.0, "steer": 1.0, "speed": request})
        time.sleep(args.pulse_s)
        panel_post(args.panel, "/api/robot/drive", {"throttle": 0.0, "steer": 0.0})
        time.sleep(args.settle_s)
        after = panel_get(args.panel, "/api/robot/status").get("heading_deg")

        turned = None
        if before is not None and after is not None:
            turned = abs((after - before + 180.0) % 360.0 - 180.0)
        motor = request_to_motor_pwm(request, max_pwm=max_pwm)
        moved = turned is not None and turned >= args.moved_deg
        trials.append({"request": request, "motor_pwm": motor,
                       "turned_deg": None if turned is None else round(turned, 2),
                       "moved": moved})
        print(f"  request {request:3d} -> motor {motor:3d}  turned "
              f"{'--' if turned is None else f'{turned:5.1f}'} deg  "
              f"{'MOVED' if moved else 'stalled'}")
        if moved and threshold_request is None:
            threshold_request = request
            break

    panel_post(args.panel, "/api/robot/command", {"command": "STOP"})

    if threshold_request is None:
        print("\nThe chassis never moved across the whole range. Either the battery is "
              "flat, or something is mechanically stuck. Nothing written.", file=sys.stderr)
        return 1

    turn_motor = request_to_motor_pwm(threshold_request, max_pwm=max_pwm)
    straight_motor = max(int(round(turn_motor * STRAIGHT_FROM_TURN_RATIO)),
                         DriveProfile().min_pwm)
    profile = DriveProfile(
        straight_motor_pwm_min=straight_motor,
        turn_motor_pwm_min=turn_motor,
        max_pwm=max_pwm,
        measured=True,
        notes=(f"turn threshold measured {time.strftime('%Y-%m-%d %H:%M')}; "
               f"straight ESTIMATED at {STRAIGHT_FROM_TURN_RATIO:.2f} of it. "
               f"Re-measure after changing battery, payload or floor."),
    )
    profile.save(args.out)
    print(f"\nTurning threshold: request {threshold_request} -> motor PWM {turn_motor}")
    print(f"Straight (estimated): motor PWM {straight_motor}")
    print(f"Wrote {args.out}")
    (args.out.parent / "drive_profile_trials.json").write_text(
        json.dumps(trials, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
