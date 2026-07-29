"""Manual smoke test for ShtpUartReader - run on the Pi against the real
sensor. See docs/imu_shtp_uart.md for expected output and what the numbers
mean; run via scripts/run_imu_shtp_uart_test.sh.

Every printed line shows BOTH the instantaneous reading at print time and
the peak |value| seen since the previous line, so a quick motion between
two print ticks isn't averaged away - useful for a live "move the sensor
and watch it react" test.
"""

from __future__ import annotations

import argparse
import sys
import time

from pi_client.imu_shtp_uart import ShtpUartReader

FRESH_THRESHOLD_S = 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=15.0, help="seconds to run")
    parser.add_argument("--print-interval", type=float, default=0.5, help="seconds between lines")
    args = parser.parse_args()

    reader = ShtpUartReader(port="/dev/ttyUSB0", baudrate=3_000_000, report_hz=100.0)
    if not reader.start():
        print("start() failed - check wiring/port (docs/imu_shtp_uart.md)", file=sys.stderr)
        return 1

    time.sleep(0.5)
    start = time.monotonic()
    samples = 0
    fresh = 0
    last_print = 0.0
    peak_accel = 0.0
    peak_gyro = 0.0
    while time.monotonic() - start < args.duration:
        sample = reader.latest()
        if sample is not None:
            samples += 1
            is_fresh = (
                sample.accel_age_s < FRESH_THRESHOLD_S and sample.gyro_age_s < FRESH_THRESHOLD_S
            )
            if is_fresh:
                fresh += 1
                accel_dev = max(abs(v) for v in sample.accel_ms2) - 9.8  # deviation from gravity
                gyro_mag = max(abs(v) for v in sample.gyro_rads)
                peak_accel = max(peak_accel, abs(accel_dev))
                peak_gyro = max(peak_gyro, gyro_mag)
            now = time.monotonic()
            if now - last_print > args.print_interval:
                last_print = now
                flag = "" if is_fresh else "  <-- STALE"
                print(
                    "t=%5.1fs  accel %7.2f %7.2f %7.2f  gyro %7.3f %7.3f %7.3f  "
                    "peak(|a-g|=%.2f |gyro|=%.2f)%s"
                    % (
                        now - start,
                        *sample.accel_ms2,
                        *sample.gyro_rads,
                        peak_accel,
                        peak_gyro,
                        flag,
                    )
                )
                peak_accel = 0.0
                peak_gyro = 0.0
        time.sleep(0.02)

    elapsed = time.monotonic() - start
    stats = reader.stats()
    reader.stop()

    print(f"\n--- {elapsed:.1f}s summary ---")
    print(f"latest() calls with data: {samples}, fresh (<{FRESH_THRESHOLD_S}s): {fresh}")
    print(f"stats: {stats}")
    if stats["packets_ok"]:
        print(f"effective valid sensor-packet rate: {stats['packets_ok'] / elapsed:.1f} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
