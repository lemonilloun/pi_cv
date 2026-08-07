#!/usr/bin/env python3
"""CLI over pi_client.imu_distance_analysis: analyse known-distance IMU
logs recorded by imu_raw_log.py (or by the guided imu_calibration.py
wizard, which stores the same CSV format per run).

Usage:
    python3 scripts/analyze_imu_distance.py data/imu_tests/dist_3m_a.csv --truth-m 3.0
    python3 scripts/analyze_imu_distance.py data/imu_tests/cal_*/run_*.csv --truth-m 3.0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "client/src"))

from pi_client.imu_distance_analysis import analyze_run, summarize  # noqa: E402


def load_csv(path: Path):
    t, gyro, accel = [], [], []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            t.append(int(row["t_ns"]) * 1e-9)
            gyro.append([float(row["gx"]), float(row["gy"]), float(row["gz"])])
            accel.append([float(row["ax"]), float(row["ay"]), float(row["az"])])
    return np.asarray(t), np.asarray(gyro), np.asarray(accel)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, nargs="+")
    ap.add_argument("--truth-m", type=float, default=None, help="Tape-measured travelled distance")
    ap.add_argument("--head-s", type=float, default=None,
                    help="Known still seconds at the start (the wizard's settle phase minus its "
                         "margin). Beats auto-detection, which cannot tell rolling from parked.")
    ap.add_argument("--tail-s", type=float, default=None, help="Known still seconds at the end")
    ap.add_argument("--calibration", type=Path,
                    default=REPO_ROOT / "config/imu_calibration_shtp.json",
                    help="Stored calibration whose up_imu_reference the live integrator uses")
    args = ap.parse_args()

    reference_up = None
    if args.calibration.exists():
        try:
            reference_up = json.loads(args.calibration.read_text(encoding="utf-8")).get("up_imu_reference")
        except (OSError, ValueError):
            pass

    results = []
    for path in args.csv:
        t, gyro, accel = load_csv(path)
        res = analyze_run(t, gyro, accel, truth_m=args.truth_m, reference_up=reference_up,
                          head_s=args.head_s, tail_s=args.tail_s)
        results.append(res)
        print(f"\n=== {path.name} ===")
        print(f"{res.samples} samples, {res.duration_s}s, {res.rate_hz} Hz effective"
              + (f" | {res.rejected_samples} corrupt samples dropped ({100*res.rejected_frac:.1f}%)"
                 if res.rejected_samples else ""))
        print(f"static head {res.static_head_s}s / tail {res.static_tail_s}s, "
              f"motion {res.motion_s}s")
        if not res.ok:
            print(f"UNUSABLE: {res.reason}")
            print("Re-record: stay perfectly still ~3s, push the measured distance, "
                  "then hold still ~3s before stopping the log.")
            continue
        print(f"gravity |g|={res.gravity_mag} m/s², gyro bias {res.gyro_bias_dps} deg/s")
        print(f"noise floor while still: {res.noise_ms2} m/s²")
        if res.tilt_error_deg:
            print(f"vs stored calibration reference: {res.tilt_error_deg}° tilt "
                  f"-> {res.leak_ms2} m/s² of gravity leaking into the horizontal plane")
        else:
            print("no stored gravity reference — tilt error against the live "
                  "integrator not measured (pass --calibration)")
        print(f"UNCORRECTED: end speed {res.raw_end_speed_ms} m/s (true 0), "
              f"distance {res.raw_distance_m} m")
        print(f"implied constant bias: {res.bias_ms2} m/s² (|{res.bias_mag_ms2}|)")
        print(f"CORRECTED:   distance {res.corrected_distance_m} m")
        if res.error_m is not None:
            print(f"ground truth {res.truth_m} m -> error {res.error_m:+.3f} m "
                  f"({100 * res.error_frac:+.1f}%)")
            print(f"-> {res.verdict}")

    if len(results) > 1:
        summary = summarize(results)
        print("\n=== SUMMARY ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
    return 0 if any(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
