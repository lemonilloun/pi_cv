"""Log raw timestamped IMU samples to CSV for offline motion-pattern analysis
(e.g. comparing forward walking vs turning) - no camera involved, so this is
much lighter/faster to run than euroc_capture.py when the goal is purely
characterizing what the sensor stream looks like during a specific motion.

Output: <out>.csv with columns
    t_ns,gx,gy,gz,ax,ay,az
where t_ns is the RECONSTRUCTED parse time (monotonic - age_s), matching
euroc_capture.py's dedup convention - see that module's docstring for why
sample.monotonic itself is the wrong thing to log (it's call-time, not
parse-time).

Usage (run on the Pi):
    ./scripts/run_imu_raw_log.sh --out data/imu_tests/forward.csv --duration 20 --label forward
    ./scripts/run_imu_raw_log.sh --out data/imu_tests/turn_left.csv --duration 15 --label turn_left
    ./scripts/run_imu_raw_log.sh --out data/imu_tests/turn_right.csv --duration 15 --label turn_right
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def record(out_path: Path, duration_s: float, imu_port: str, imu_baud: int, poll_interval_s: float = 0.002) -> dict:
    from pi_client.imu_shtp_uart import ShtpUartReader

    reader = ShtpUartReader(port=imu_port, baudrate=imu_baud)
    if not reader.start():
        raise RuntimeError(f"IMU failed to start on {imu_port} - check wiring/jumpers (docs/imu_shtp_uart.md)")
    time.sleep(1.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, float, float, float, float, float, float]] = []
    last_accel_parse_t: float | None = None
    last_gyro_parse_t: float | None = None

    print(f"Recording {duration_s:.0f}s to {out_path} ... move the sensor now.")
    t_end = time.monotonic() + duration_s
    try:
        while time.monotonic() < t_end:
            sample = reader.latest()
            if sample is not None:
                accel_parse_t = sample.monotonic - sample.accel_age_s
                gyro_parse_t = sample.monotonic - sample.gyro_age_s
                if accel_parse_t != last_accel_parse_t or gyro_parse_t != last_gyro_parse_t:
                    last_accel_parse_t = accel_parse_t
                    last_gyro_parse_t = gyro_parse_t
                    t_ns = int(max(accel_parse_t, gyro_parse_t) * 1e9)
                    gx, gy, gz = sample.gyro_rads
                    ax, ay, az = sample.accel_ms2
                    rows.append((t_ns, gx, gy, gz, ax, ay, az))
            time.sleep(poll_interval_s)
    finally:
        reader.stop()

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_ns", "gx", "gy", "gz", "ax", "ay", "az"])
        w.writerows(rows)

    summary = {
        "samples": len(rows),
        "duration_s": duration_s,
        "hz_actual": len(rows) / duration_s if duration_s > 0 else 0.0,
        "stats": reader.stats(),
    }
    print(f"Recorded {summary['samples']} samples ({summary['hz_actual']:.1f} Hz actual) -> {out_path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output CSV path (relative to repo root)")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--imu-port", default="/dev/ttyUSB0")
    parser.add_argument("--imu-baud", type=int, default=3_000_000)
    parser.add_argument("--label", default="", help="optional note, printed only, not stored")
    args = parser.parse_args()

    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if args.label:
        print(f"Label: {args.label}")
    record(out_path, args.duration, args.imu_port, args.imu_baud)
    return 0


if __name__ == "__main__":
    sys.exit(main())
