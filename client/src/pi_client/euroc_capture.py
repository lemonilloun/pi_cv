"""Record a short ASL/EuRoC-format sequence (camera + IMU) from THIS
project's own hardware, for testing ORB-SLAM3 against real data instead
of an external dataset mirror (the standard EuRoC servers were down when
this was written - see the ORB-SLAM3 setup notes).

Output layout matches what ORB-SLAM3's mono_euroc / mono_inertial_euroc
examples expect:

    <out_dir>/mav0/cam0/data/<timestamp_ns>.png
    <out_dir>/cam0_timestamps.txt      (one timestamp per line - the
                                        "times file" argument mono_euroc wants)
    <out_dir>/mav0/imu0/data.csv       (#timestamp[ns],w_x,w_y,w_z,a_x,a_y,a_z
                                        - ASL/EuRoC column order: GYRO
                                        first, then ACCEL, both already
                                        calibrated, matching
                                        imu_shtp_uart.ImuSample's units:
                                        rad/s and m/s^2)

Camera: manual-focus continuous capture via PicameraStreamSource, at the
SAME lens_position config/scene_intrinsics.json was calibrated at (a
different focus distance means the intrinsics don't apply).
IMU: imu_shtp_uart.ShtpUartReader (SHTP over UART - see
docs/imu_shtp_uart.md for why UART rather than SPI, and its measured
~50-90 Hz ceiling).

Both streams timestamp off the SAME time.monotonic() clock, converted to
integer nanoseconds - ORB-SLAM3 only needs correctly-scaled, strictly
increasing relative time, not real wall-clock/epoch alignment.

**Why a separate polling thread for IMU, not one read per camera tick**:
`ShtpUartReader.latest()` returns only the single most recent sample; at
~20 fps camera vs ~50-90 Hz IMU, reading it once per camera frame would
silently drop most IMU samples (2-4x more IMU updates happen between
frames than that). A dedicated thread polls `latest()` frequently and
logs each new sample as soon as it appears - lightweight since `latest()`
itself is just a lock + tuple copy, not a read from the port.

**Why dedup on the reconstructed parse time, not `sample.monotonic`**:
`ShtpUartReader.latest()` stamps `.monotonic` with the time *you called
it*, not the time the reading was actually parsed off the wire (that's
the whole point of its `.accel_age_s`/`.gyro_age_s` fields - see its own
docstring). A first version of this poller compared `.monotonic` between
polls to decide "is this new data", which is always true when polling
faster than the data actually arrives - it logged the SAME stale reading
repeatedly with fabricated, evenly-spaced timestamps (caught because the
result was a suspicious ~485 Hz, far above this driver's documented
~50-90 Hz ceiling; the real distinct-sample count matched that ceiling
exactly). The correct dedup key is the reconstructed parse time
(`sample.monotonic - sample.*_age_s`), which only changes when the
underlying value genuinely updates.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _imu_poll_loop(
    imu: Any,
    rows: list[tuple[int, float, float, float, float, float, float]],
    stop: threading.Event,
    poll_interval_s: float = 0.002,
) -> None:
    last_accel_parse_t: float | None = None
    last_gyro_parse_t: float | None = None
    while not stop.is_set():
        sample = imu.latest()
        if sample is not None:
            # Reconstruct when each axis was ACTUALLY parsed (not "now",
            # which is what sample.monotonic is - see module docstring for
            # why that distinction matters here).
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


def record(
    out_dir: Path,
    duration_s: float,
    fps: float,
    width: int,
    height: int,
    lens_position: float,
    imu_port: str,
    imu_baud: int,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    from pi_client.camera import CameraFocusOptions
    from pi_client.camera_session import CaptureSettings, PicameraStreamSource
    from pi_client.imu_shtp_uart import ShtpUartReader

    cam_dir = out_dir / "mav0" / "cam0" / "data"
    imu_dir = out_dir / "mav0" / "imu0"
    cam_dir.mkdir(parents=True, exist_ok=True)
    imu_dir.mkdir(parents=True, exist_ok=True)

    imu = ShtpUartReader(port=imu_port, baudrate=imu_baud)
    if not imu.start():
        raise RuntimeError(
            f"IMU failed to start on {imu_port} - check wiring/jumpers "
            "(docs/imu_shtp_uart.md) before recording."
        )
    # Let the reset+enable sequence settle and the first reports arrive
    # before the camera loop starts consuming wall-clock time.
    time.sleep(1.0)

    focus = CameraFocusOptions(autofocus_mode="manual", lens_position=lens_position)
    source = PicameraStreamSource(
        CaptureSettings(width=width, height=height, fps=fps, focus_options=focus)
    )
    source.start()

    imu_rows: list[tuple[int, float, float, float, float, float, float]] = []
    imu_stop = threading.Event()
    imu_thread = threading.Thread(
        target=_imu_poll_loop, args=(imu, imu_rows, imu_stop), daemon=True
    )
    imu_thread.start()

    cam_timestamps: list[int] = []
    logger.info("Recording %.0fs of camera+IMU to %s ...", duration_s, out_dir)
    t_end = time.monotonic() + duration_s
    try:
        while time.monotonic() < t_end:
            data = source.next_stream_jpeg(timeout=2.0)
            t_ns = int(time.monotonic() * 1e9)
            arr = np.frombuffer(data, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            cv2.imwrite(str(cam_dir / f"{t_ns}.png"), image)
            cam_timestamps.append(t_ns)
    finally:
        imu_stop.set()
        imu_thread.join(timeout=2.0)
        source.stop()
        imu.stop()

    (out_dir / "cam0_timestamps.txt").write_text(
        "\n".join(str(t) for t in cam_timestamps) + "\n", encoding="utf-8"
    )
    with (imu_dir / "data.csv").open("w", encoding="utf-8") as f:
        f.write(
            "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
            "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],"
            "a_RS_S_z [m s^-2]\n"
        )
        for t_ns, gx, gy, gz, ax, ay, az in imu_rows:
            f.write(f"{t_ns},{gx},{gy},{gz},{ax},{ay},{az}\n")

    summary = {
        "frames": len(cam_timestamps),
        "imu_samples": len(imu_rows),
        "duration_s": duration_s,
        "camera_fps_actual": (
            len(cam_timestamps) / duration_s if duration_s > 0 else 0.0
        ),
        "imu_hz_actual": len(imu_rows) / duration_s if duration_s > 0 else 0.0,
        "imu_stats": imu.stats(),
    }
    logger.info(
        "Recorded %d frames (%.1f fps actual), %d IMU samples (%.1f Hz actual)",
        summary["frames"], summary["camera_fps_actual"],
        summary["imu_samples"], summary["imu_hz_actual"],
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/orb_slam3_capture")
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=864)
    parser.add_argument(
        "--lens-position", type=float, default=1.0,
        help="MUST match config/scene_intrinsics.json's lens_position or the "
             "calibrated intrinsics won't be valid for these frames.",
    )
    parser.add_argument("--imu-port", default="/dev/ttyUSB0")
    parser.add_argument("--imu-baud", type=int, default=3_000_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    summary = record(
        out_dir=args.out,
        duration_s=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        lens_position=args.lens_position,
        imu_port=args.imu_port,
        imu_baud=args.imu_baud,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
