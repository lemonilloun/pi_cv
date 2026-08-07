"""Record behaviour-cloning episodes: camera + IMU at a fixed 10 Hz.

The observations half of a VLA dataset. The actions half is logged on the
laptop by `mac_server/vla/action_log.py` (every wheel command funnels through
`RobocarService.drive()`), and `mac_server/vla/build_dataset.py` joins the two
afterwards using the clock offset this recorder measures at episode start.

**Why this is not `scene_recorder.py` with different settings.** That recorder
selects keyframes by parallax and sharpness, which is exactly right for
reconstruction: it wants views that add geometric information and it does not
care when they arrived. A policy learns a mapping from an observation to the
action taken at that moment, at a regular control interval — so a time axis
that stretches and compresses with how fast the robot happened to be moving
is actively harmful. Here every frame counts, including the boring ones where
nothing moves; "the robot sat still for two seconds" is a fact the policy must
learn, and a parallax gate would delete precisely those frames.

**Capture never blocks on the network.** The capture loop hands frames to a
bounded queue and a sender thread drains it. A Wi-Fi stall must not skew the
sampling interval, because an irregular time axis is the one defect that
cannot be repaired afterwards — whereas a dropped frame is merely a gap, and
a counted one at that. If the queue ever fills, the drop is recorded in the
episode metadata rather than passing silently: a dataset with an unexplained
hole is worse than a smaller one.

    ./scripts/run_vla_recorder.sh --task "go to the chair" --seconds 30
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "client/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from pi_client.camera import CameraFocusOptions
from pi_client.camera_session import CaptureSettings, make_capture_source
from pi_client.network import ClientConnectionError, PiClient
from pi_client.protocol import (
    make_time_sync_message,
    make_vla_frame_message,
    make_vla_session_end_message,
    make_vla_session_start_message,
)
from shared.clock_sync import DEFAULT_SAMPLES, SyncSample, drift_ns_per_s, estimate_offset
from shared.config import load_config

logger = logging.getLogger(__name__)

# 10 Hz is the control rate the policy will run at, and the action chunks it
# emits are expressed in these steps. Recording at the same rate means no
# resampling of the actions is ever needed.
CONTROL_HZ = 10.0

# Small enough to keep the round trip inside the 100 ms budget on Wi-Fi, big
# enough for the policy's 256x256 input to be a downscale rather than an
# upscale. 640x360 keeps the camera's 16:9 framing so nothing is cropped away
# that the robot could see.
FRAME_WIDTH, FRAME_HEIGHT, JPEG_QUALITY = 640, 360, 80

# A 30 s episode is 300 frames; at ~40 KB each the whole episode is ~12 MB, so
# this queue only ever fills if the network has stopped entirely.
QUEUE_DEPTH = 600


def measure_clock_offset(client: PiClient, samples: int = DEFAULT_SAMPLES) -> Any:
    """Burst of `time_sync` round trips -> the Pi->Mac monotonic offset.

    See shared/clock_sync.py: the estimate comes from the single fastest
    round trip, not the average, because queueing delay is one-sided.
    """
    collected: list[SyncSample] = []
    for seq in range(samples):
        t0 = time.monotonic_ns()
        response, _ = client.request(make_time_sync_message("pi", t0, seq=seq))
        t1 = time.monotonic_ns()
        if response.type != "time_sync_reply":
            continue
        mac_ns = response.payload.get("mac_monotonic_ns")
        if mac_ns is None:
            continue
        collected.append(SyncSample(t0_pi_ns=t0, t1_pi_ns=t1, mac_ns=int(mac_ns)))
    return estimate_offset(collected)


class FrameSender(threading.Thread):
    """Drains the capture queue to the server. Never touched by the timer."""

    def __init__(self, client: PiClient, device_id: str, episode_id: str) -> None:
        super().__init__(name="vla-sender", daemon=True)
        self.client = client
        self.device_id = device_id
        self.episode_id = episode_id
        self.queue: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
        self.sent = 0
        self.failed = 0
        self._stop = threading.Event()

    def submit(self, frame_idx: int, t_ns: int, jpeg: bytes, imu: dict | None) -> bool:
        try:
            self.queue.put_nowait((frame_idx, t_ns, jpeg, imu))
            return True
        except queue.Full:
            return False

    def run(self) -> None:
        while not self._stop.is_set() or not self.queue.empty():
            try:
                frame_idx, t_ns, jpeg, imu = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                message = make_vla_frame_message(
                    self.device_id, self.episode_id, frame_idx, t_ns, len(jpeg), imu
                )
                response, _ = self.client.request(message, jpeg)
                if response.type == "error":
                    raise ClientConnectionError(str(response.payload.get("error")))
                self.sent += 1
            except (ClientConnectionError, OSError) as exc:
                self.failed += 1
                if self.failed == 1:
                    logger.warning("Frame upload failing: %s", exc)

    def finish(self, timeout: float = 30.0) -> bool:
        """Drain the queue and report whether it finished.

        The return value matters: this thread and the main thread share one
        `PiClient`, and only one of them may be writing to that socket at a
        time. The main thread's end-of-episode clock sync must therefore wait
        for this to be genuinely done, not merely asked to stop. A silent
        timeout followed by concurrent writes interleaves two messages'
        framing on the wire and corrupts both.
        """
        self._stop.set()
        self.join(timeout=timeout)
        return not self.is_alive()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a VLA behaviour-cloning episode")
    parser.add_argument("--task", required=True,
                        help="What the operator is demonstrating, e.g. 'go to the chair'. "
                             "Prefer object-referential phrasing over directional: 'go to "
                             "the blue chair' grounds on something visible and can "
                             "generalize, 'turn left then forward' grounds on nothing and "
                             "memorizes the room.")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--imu-port", default="/dev/ttyUSB0")
    parser.add_argument("--imu-calibration", type=Path,
                        default=REPO_ROOT / "config/imu_calibration.json")
    parser.add_argument("--lens-position", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    import cv2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("pi_client.network").setLevel(logging.WARNING)
    args = parse_args()

    config = load_config(REPO_ROOT / "config/default.json", REPO_ROOT / ".env")
    host = args.host or config.get("client", {}).get("server_host", "127.0.0.1")
    port = args.port or int(config.get("server", {}).get("port", 8765))
    device_id = str(config.get("client", {}).get("device_id", "raspberry_pi_01"))
    episode_id = args.episode_id or time.strftime("ep_%Y%m%d_%H%M%S")

    lens_position = args.lens_position
    intrinsics_path = REPO_ROOT / "config/scene_intrinsics.json"
    if lens_position is None and intrinsics_path.exists():
        import json

        try:
            lens_position = json.loads(intrinsics_path.read_text()).get("lens_position")
        except (OSError, ValueError):
            pass

    imu = None
    if not args.no_imu:
        from pi_client.imu_rvc import RvcReader

        imu = RvcReader(port=args.imu_port, calibration_path=args.imu_calibration)
        if not imu.start():
            logger.warning("IMU unavailable — recording vision-only; observation.state "
                           "will be missing its attitude channels")
            imu = None

    try:
        client = PiClient(host, port, timeout_seconds=6.0)
        client.connect()
    except (ClientConnectionError, OSError) as exc:
        logger.error("Cannot reach the server at %s:%s (%s). Nothing recorded — an "
                     "episode without its action log is not usable data.", host, port, exc)
        if imu is not None:
            imu.stop()
        return 2

    logger.info("Measuring the clock offset (%d round trips)...", DEFAULT_SAMPLES)
    offset_start = measure_clock_offset(client)
    if offset_start is None:
        logger.error("Clock sync failed — refusing to record. Without a shared "
                     "timeline the frames cannot be joined to the actions.")
        client.close()
        if imu is not None:
            imu.stop()
        return 2
    # Presented in hours, and explained, because the raw number looks like a
    # fault and is not: each machine's monotonic clock counts from its own
    # boot, so the offset between two boxes is naturally hours. What matters
    # is the error bar, not the magnitude.
    logger.info("Clock offset %+.3f h between the two boot clocks (normal — monotonic "
                "counts from boot on each machine)", offset_start.offset_ns / 3.6e12)
    logger.info("  error bar +-%.2f ms from the fastest of %d round trips "
                "(budget 20 ms) -> %s",
                offset_start.quality_ms, offset_start.samples,
                "good" if offset_start.quality_ms < 20.0 else "TOO LOOSE to join frames to actions")

    source = make_capture_source(
        "camera",
        CaptureSettings(
            width=args.width, height=args.height, fps=max(args.hz, 1.0),
            jpeg_quality=args.jpeg_quality,
            focus_options=CameraFocusOptions(
                autofocus_mode="manual" if lens_position is not None else "continuous",
                autofocus_range="normal", autofocus_speed="normal",
                lens_position=lens_position,
            ),
        ),
        REPO_ROOT / "data/cat.jpg",
    )
    source.start()

    settings = {
        "hz": args.hz, "width": args.width, "height": args.height,
        "jpeg_quality": args.jpeg_quality, "lens_position": lens_position,
        "imu": imu is not None,
    }
    client.request(make_vla_session_start_message(
        device_id, episode_id, args.task, settings,
        clock={"start": offset_start.to_dict()},
    ))

    sender = FrameSender(client, device_id, episode_id)
    sender.start()

    stop = {"flag": False}

    def handle_signal(_signum: int, _frame: Any) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    interval = 1.0 / max(args.hz, 1.0)
    frame_idx = 0
    dropped = 0
    intervals: list[float] = []
    logger.info("RECORDING '%s' as %s for %.0f s at %.1f Hz — drive now (Ctrl+C to stop)",
                args.task, episode_id, args.seconds, args.hz)

    started = time.monotonic()
    # Absolute schedule, not sleep(interval): sleeping a fixed amount after
    # variable work makes the period drift, and the interval is the one thing
    # that cannot be fixed after the fact.
    next_tick = started
    last_tick = None
    try:
        while not stop["flag"] and time.monotonic() - started < args.seconds:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.02))
                continue
            next_tick += interval

            bgr = source.capture_bgr()
            if bgr is None:
                continue
            t_ns = time.monotonic_ns()
            ok, encoded = cv2.imencode(".jpg", bgr,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            if not ok:
                continue
            orientation = imu.read_orientation() if imu is not None else None
            frame_idx += 1
            if not sender.submit(frame_idx, t_ns, encoded.tobytes(), orientation):
                dropped += 1
            if last_tick is not None:
                intervals.append(t_ns / 1e9 - last_tick)
            last_tick = t_ns / 1e9
            if frame_idx % 50 == 0:
                logger.info("  %d frames, %d sent, %d dropped", frame_idx, sender.sent, dropped)
    finally:
        try:
            source.stop()
        except Exception:
            pass
        drained = sender.finish()
        if not drained:
            logger.error(
                "Frame sender did not drain in time — %d frames still queued. "
                "Skipping the closing clock sync: the sender still owns the "
                "socket, and writing from two threads would corrupt both "
                "messages.", sender.queue.qsize(),
            )

        offset_end = measure_clock_offset(client) if drained else None
        drift = None
        if offset_end is not None:
            drift = drift_ns_per_s(offset_start, offset_end, max(time.monotonic() - started, 1e-6))

        # The interval statistics are the dataset's own quality report: a
        # policy trained on frames that are supposed to be 100 ms apart but
        # are not will learn the wrong dynamics, and nothing downstream can
        # detect that unless it was measured here.
        intervals_ms = sorted(v * 1000.0 for v in intervals)
        episode_meta = {
            "task": args.task,
            "frames_captured": frame_idx,
            "frames_sent": sender.sent,
            "frames_dropped_queue_full": dropped,
            "frames_failed_upload": sender.failed,
            "frames_left_queued": sender.queue.qsize(),
            "sender_drained": drained,
            "duration_s": round(time.monotonic() - started, 2),
            "target_hz": args.hz,
            "interval_ms_median": round(intervals_ms[len(intervals_ms) // 2], 2) if intervals_ms else None,
            "interval_ms_p95": round(intervals_ms[int(len(intervals_ms) * 0.95)], 2) if intervals_ms else None,
            "interval_ms_max": round(intervals_ms[-1], 2) if intervals_ms else None,
            "clock": {
                "start": offset_start.to_dict(),
                "end": offset_end.to_dict() if offset_end else None,
                "drift_ns_per_s": round(drift, 1) if drift is not None else None,
            },
            "imu_stats": imu.stats() if imu is not None else None,
        }
        try:
            client.request(make_vla_session_end_message(device_id, episode_id, episode_meta))
        except (ClientConnectionError, OSError) as exc:
            logger.error("Could not close the episode on the server: %s", exc)
        client.close()
        if imu is not None:
            imu.stop()

    logger.info("Episode %s: %d frames captured, %d sent, %d dropped, %d failed",
                episode_id, frame_idx, sender.sent, dropped, sender.failed)
    if intervals_ms:
        logger.info("  frame interval: median %.1f ms, p95 %.1f ms, worst %.1f ms (target %.1f)",
                    episode_meta["interval_ms_median"], episode_meta["interval_ms_p95"],
                    episode_meta["interval_ms_max"], 1000.0 / args.hz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
