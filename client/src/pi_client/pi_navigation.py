"""Live localization client: where is the Pi right now?

Loop (~1-2 Hz): grab a frame -> full-frame CLIP embedding on the Hailo NPU
(same RN50x4 the scene recorder uses) -> optionally fast_depth on the same
NPU (forward distance) -> nav_query to the Mac. The Mac matches the
embedding against the place indexes of all reconstructed sessions
(pipeline step `navindex`) and answers which room this looks like, the
closest recorded viewpoint, and the position/heading on that room's floor
plan (now also in raw metres via `position_m`, not just the display
`plan_frac` — see navindex.py's `_plan_position_m`).

Between visual fixes, attitude comes from the BNO08x on the UART-RVC link
(pi_client.imu_rvc) and is fused by `ekf_localization.Ekf2DHeading`. RVC
reports a fused, ABSOLUTE yaw at 100 Hz — measured on this device it
drifts 0.03 deg/min stationary — so the filter uses it as an absolute
heading observation offset by an unknown datum `b`, and identifies `b`
from the visual fixes. After a few fixes the IMU alone gives absolute map
heading between them, not merely "you turned by this much".

Position is deliberately NOT dead-reckoned. RVC exposes no raw gyro, this
rig has no wheel odometry, and the accelerometer route was measured at 78x
too large (667 m over a walk VGGT measured at ~8.5 m). The filter
propagates "moved an unknown amount" with uncertainty growing at a
realistic platform speed and lets position come from the visual fix. That
is the validated behaviour, and metric dead reckoning is explicitly out of
scope for this loop — orientation is what navigation needs from the IMU.

This is loosely-coupled visual-inertial fusion, not SLAM: no
feature tracking, no bundle adjustment, no map-building here — the map
(place index) already exists from the scene3d pipeline. The fused state
stays a coarse localization prior, same as the raw visual fix always was,
just smoother between the ~1-2 Hz visual updates and self-aware of its
own growing uncertainty when only IMU has been available for a while (see
ekf_localization.Ekf2DHeading.predict's process-noise growth).

    ./scripts/run_pi_navigation.sh          # Ctrl+C to stop
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "client/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import numpy as np

from pi_client.camera import CameraFocusOptions
from pi_client.camera_session import CaptureSettings, make_capture_source
from pi_client.ekf_localization import Ekf2DHeading
from pi_client.hailo_infer import HailoMultiModel
from pi_client.network import ClientConnectionError, PiClient
from pi_client.protocol import make_nav_query_message
from shared.config import load_config


logger = logging.getLogger(__name__)


def _visual_covariance(
    similarity: float,
    heading_spread_deg: float | None,
    pos_std_floor: float,
    pos_std_scale: float,
    theta_std_floor_deg: float,
) -> np.ndarray:
    """Turn navindex's confidence signals into an update_visual covariance.

    Linear in (1 - similarity) for position — not precisely calibrated,
    just a monotonic, tunable mapping from "the panel already shows this
    as a confidence number" to "the filter should trust it proportionally
    less". heading_spread_deg is already close to a real std-dev proxy (0
    = the k nearest neighbours agree exactly on facing direction), used
    almost directly.
    """
    pos_std = pos_std_floor + pos_std_scale * max(0.0, 1.0 - similarity)
    # Explicit None check, NOT `or`. A spread of exactly 0.0 means the
    # neighbouring keyframes agree perfectly on facing direction — the BEST
    # possible case — and it happens routinely: a single surviving neighbour
    # gives a resultant length of exactly 1.0, and any tight cluster rounds
    # to 0.0 at one decimal. `heading_spread_deg or 30.0` treated every one
    # of those as the no-information default, so the most confident fixes
    # were handed the worst covariance. Exactly backwards.
    spread = 30.0 if heading_spread_deg is None else heading_spread_deg
    theta_std_deg = theta_std_floor_deg + spread
    return np.diag([pos_std ** 2, pos_std ** 2, math.radians(theta_std_deg) ** 2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localize the Pi against recorded rooms")
    parser.add_argument("--clip-hef", type=Path, default=REPO_ROOT / "models/clip_resnet_50x4_h8.hef")
    parser.add_argument("--depth-hef", type=Path, default=REPO_ROOT / "models/fast_depth_h8.hef",
                        help="Optional monocular depth on the NPU (forward distance)")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--source", choices=["camera", "synthetic"], default="camera")
    parser.add_argument("--synthetic-image", type=Path, default=REPO_ROOT / "data/cat.jpg")
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=864)
    parser.add_argument("--hz", type=float, default=1.5, help="Query rate")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--lens-position", type=float, default=None)
    parser.add_argument("--no-imu", action="store_true",
                        help="Vision-only: skip the EKF, log just the raw visual fix each tick "
                             "(the old behavior)")
    parser.add_argument("--imu-port", default="/dev/ttyUSB0")
    parser.add_argument("--imu-calibration", type=Path,
                        default=REPO_ROOT / "config/imu_calibration.json")
    # 0.2 deg, from acceptance test B on this sensor: yaw increment noise
    # 0.00025 deg and drift 0.003 deg/min over 10 minutes.
    #
    # NOT 0.00025. That figure is how quietly the fused output moves between
    # consecutive samples, which is a statement about smoothness, not about
    # how well the heading matches the room. Feeding it as the measurement
    # sigma would make the filter treat IMU yaw as ground truth and stop
    # visual fixes from ever correcting the heading. 0.2 deg is the guide's
    # own figure for relative yaw over a short window, and it is the honest
    # one to trust.
    parser.add_argument("--imu-yaw-sigma-deg", type=float, default=0.2,
                        help="Measurement noise on the RVC fused yaw. The bench figure is far "
                             "tighter (0.03 deg/min drift, +-0.05 deg jitter, stationary); 1.0 "
                             "leaves room for the sensor-to-chassis mounting not being perfectly "
                             "rigid while driving.")
    parser.add_argument("--imu-unknown-speed-ms", type=float, default=0.6,
                        help="The filter does not dead-reckon distance (RVC has no raw gyro and "
                             "the accelerometer route measured 78x too large), so position "
                             "uncertainty grows at roughly the platform's realistic top speed "
                             "between visual fixes instead.")
    parser.add_argument("--ekf-pos-std-floor", type=float, default=0.15,
                        help="Metres of position std-dev applied at similarity=1.0")
    parser.add_argument("--ekf-pos-std-scale", type=float, default=2.0,
                        help="Extra metres of std-dev added at similarity=0.0")
    parser.add_argument("--ekf-theta-std-floor-deg", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    import cv2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    config = load_config(REPO_ROOT / "config/default.json", REPO_ROOT / ".env")
    host = args.host or config.get("client", {}).get("server_host", "127.0.0.1")
    port = args.port or int(config.get("server", {}).get("port", 8765))
    device_id = str(config.get("client", {}).get("device_id", "raspberry_pi_01"))

    lens_position = args.lens_position
    intrinsics_path = REPO_ROOT / "config/scene_intrinsics.json"
    if lens_position is None and intrinsics_path.exists():
        import json

        try:
            lens_position = json.loads(intrinsics_path.read_text()).get("lens_position")
        except (OSError, ValueError):
            pass

    source = make_capture_source(
        args.source,
        CaptureSettings(
            width=args.width, height=args.height, fps=max(args.hz, 1.0),
            jpeg_quality=85,
            focus_options=CameraFocusOptions(
                autofocus_mode="manual" if lens_position is not None else "continuous",
                autofocus_range="normal", autofocus_speed="normal",
                lens_position=lens_position,
            ),
        ),
        args.synthetic_image,
    )

    hailo = HailoMultiModel()
    clip = hailo.load("clip", str(args.clip_hef))
    depth = None
    if not args.no_depth and args.depth_hef.exists():
        try:
            depth = hailo.load("depth", str(args.depth_hef))
        except Exception as exc:
            logger.warning("Depth hef unavailable (%s) — continuing without", exc)

    imu = None
    if not args.no_imu:
        from pi_client.imu_rvc import RvcReader

        imu = RvcReader(port=args.imu_port, calibration_path=args.imu_calibration)
        if not imu.start():
            logger.warning("IMU failed to start — continuing vision-only (no fusion, no predict)")
            imu = None

    ekf: Ekf2DHeading | None = None
    last_predict_at: float | None = None
    last_yaw_deg: float | None = None
    last_session_id: str | None = None

    client: PiClient | None = None
    stop = {"flag": False}

    def handle_signal(signum: int, _frame: Any) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    source.start()
    interval = 1.0 / max(args.hz, 0.2)
    logger.info("Navigation: querying %s:%s at %.1f Hz (Ctrl+C to stop, IMU %s)",
                host, port, args.hz, "on" if imu else "off")
    try:
        while not stop["flag"]:
            tick = time.monotonic()
            try:
                bgr = source.capture_bgr()

                in_h, in_w = clip.input_shape[:2]
                rgb = cv2.cvtColor(cv2.resize(bgr, (in_w, in_h)), cv2.COLOR_BGR2RGB)
                out = clip.infer(rgb)
                vec = np.asarray(next(iter(out.values())), dtype=np.float32).reshape(-1)
                vec = vec / (float(np.linalg.norm(vec)) or 1.0)

                depth_center = None
                if depth is not None:
                    dh, dw = depth.input_shape[:2]
                    drgb = cv2.cvtColor(cv2.resize(bgr, (dw, dh)), cv2.COLOR_BGR2RGB)
                    dmap = np.asarray(next(iter(depth.infer(drgb).values()))).squeeze()
                    ch, cw = dmap.shape[0] // 4, dmap.shape[1] // 4
                    depth_center = float(np.median(
                        dmap[dmap.shape[0] // 2 - ch:dmap.shape[0] // 2 + ch,
                             dmap.shape[1] // 2 - cw:dmap.shape[1] // 2 + cw]
                    ))

                # ---- IMU: grow uncertainty for the elapsed time, then feed
                # the RVC fused yaw in as an ABSOLUTE heading observation.
                # It is deliberately not used in predict() as well — one
                # measurement counted twice would make the filter
                # overconfident about the very thing it is best at.
                orientation = imu.read_orientation() if imu is not None else None
                # How far the IMU says we actually turned since the last
                # tick. Used to SIZE the heading process noise, not to move
                # the state — the state still moves only through
                # update_imu_yaw. Without it the filter assumes the robot
                # could have swung 57 deg/s at every tick and throws its
                # prior away, which is what made the fused marker jump.
                turned_rad = None
                if orientation is not None:
                    if last_yaw_deg is not None:
                        turned_rad = math.radians(orientation["yaw_deg"] - last_yaw_deg)
                    last_yaw_deg = orientation["yaw_deg"]

                # NO accelerometer-derived displacement here. This was tried
                # and reverted the same day: RVC gives fused attitude plus raw
                # accelerometer, and turning that into position needs DOUBLE
                # integration, where a tilt error of 0.5 deg becomes 0.086 m/s^2
                # of phantom acceleration and 0.43 m of phantom travel in 10 s.
                # The fused marker flew off the plan. No filter fixes it —
                # the error is in the input, not the estimator.
                #
                # The right displacement source for this chassis is the wheel
                # encoders, and the right heading source is yaw. A differential
                # drive is non-holonomic: it can only move along its own X
                # axis, so the direction of travel IS the heading and there is
                # nothing to integrate. Until encoder counts reach this loop,
                # position moves only on a visual fix and `speed_ms` states the
                # honest "it could be anywhere within this radius".
                if ekf is not None:
                    now = time.monotonic()
                    dt_s = (now - last_predict_at) if last_predict_at else interval
                    last_predict_at = now
                    ekf.predict(dt_s=dt_s, speed_ms=args.imu_unknown_speed_ms,
                                turned_rad=turned_rad)
                    if orientation is not None:
                        ekf.update_imu_yaw(
                            math.radians(orientation["yaw_deg"]),
                            sigma_deg=args.imu_yaw_sigma_deg,
                        )
                if imu is not None and orientation is None:
                    # Stale or dead link. read_orientation() returns None
                    # rather than the last value precisely so this is
                    # visible instead of looking like a perfectly steady
                    # heading — the failure mode that hid the SHTP link's
                    # collapse for a whole session.
                    logger.warning("IMU stale — heading unaided this tick (%s)", imu.stats())

                if client is None:
                    client = PiClient(host, port, timeout_seconds=4.0)
                    client.connect()
                # Snapshot the filter BEFORE this tick's visual update: what
                # the EKF believes right now, having only predicted since the
                # last fix. That is the honest "live" pose to draw.
                fused_payload = None
                if ekf is not None:
                    fx, fy = ekf.position
                    fused_payload = {
                        "position_m": [round(fx, 4), round(fy, 4)],
                        "heading_deg": round(math.degrees(ekf.heading_rad), 1),
                        "heading_std_deg": round(ekf.heading_std_deg, 2),
                        "std_m": round(max(ekf.position_std_m), 3),
                        "yaw_bias_deg": round(math.degrees(ekf.yaw_bias_rad), 2),
                        "yaw_bias_std_deg": round(ekf.yaw_bias_std_deg, 2),
                        "filter": "heading_rvc",
                    }
                    if orientation is not None:
                        # Raw, uncorrected RVC yaw. The panel shows it next
                        # to the drive keys so the sensor's turn convention
                        # is readable by eye: press left, watch which way it
                        # moves. That sign cannot be recovered from the
                        # gravity calibration, and getting it wrong makes
                        # the filter steer the estimate the wrong way.
                        fused_payload["imu_yaw_deg"] = round(orientation["yaw_deg"], 2)
                        fused_payload["imu_pitch_deg"] = orientation["pitch_deg"]
                        fused_payload["imu_roll_deg"] = orientation["roll_deg"]

                # Raw attitude travels at top level, independent of whether a
                # filter exists yet — see make_nav_query_message's docstring
                # for the deadlock that nesting it caused.
                imu_payload = None
                if orientation is not None:
                    imu_payload = {
                        "yaw_deg": round(orientation["yaw_deg"], 2),
                        "pitch_deg": orientation["pitch_deg"],
                        "roll_deg": orientation["roll_deg"],
                        "age_s": orientation["age_s"],
                    }
                response, _ = client.request(
                    make_nav_query_message(
                        device_id, [round(float(v), 5) for v in vec], depth_center,
                        fused=fused_payload, imu=imu_payload,
                    )
                )
                payload = response.payload
                if payload.get("located"):
                    best = payload["best"]
                    room = best.get("name") or best["session_id"].replace("session_", "")
                    heading = best.get("heading_deg")
                    position_m = best.get("position_m")
                    similarity = float(best.get("similarity", 0.0))

                    if position_m is not None and heading is not None:
                        theta = math.radians(heading)
                        if ekf is None or last_session_id != best["session_id"]:
                            # (Re)initialize on the first fix, or whenever
                            # the winning room changes — a different
                            # session's plan frame has a different origin/
                            # axes, so the old state means nothing there.
                            # The yaw bias is re-learned too: it is defined
                            # against THAT plan frame's heading zero.
                            ekf = Ekf2DHeading.initialize(
                                position_m[0], position_m[1], theta,
                                pos_std=args.ekf_pos_std_floor,
                                theta_std_deg=args.ekf_theta_std_floor_deg,
                            )
                            last_predict_at = time.monotonic()
                            last_yaw_deg = (
                                orientation["yaw_deg"] if orientation is not None else None
                            )
                            if orientation is not None:
                                # Seed b from this fix so the next tick
                                # already has absolute heading instead of
                                # waiting for convergence.
                                ekf.update_imu_yaw(
                                    math.radians(orientation["yaw_deg"]),
                                    sigma_deg=args.imu_yaw_sigma_deg,
                                )
                            last_session_id = best["session_id"]
                            logger.info("EKF initialized in room %s at (%.2f, %.2f)",
                                        room, position_m[0], position_m[1])
                        else:
                            R = _visual_covariance(
                                similarity, best.get("heading_spread_deg"),
                                args.ekf_pos_std_floor, args.ekf_pos_std_scale,
                                args.ekf_theta_std_floor_deg,
                            )
                            ekf.update_visual(position_m[0], position_m[1], theta, R)

                    parts = [
                        f"room: {room}",
                        f"kf {best['keyframe']:06d}",
                        f"sim {similarity:.2f}",
                    ]
                    if heading is not None:
                        spread = best.get("heading_spread_deg")
                        trust = f" ±{spread:.0f}°" if spread is not None else ""
                        parts.append(f"heading {heading:+.0f}°{trust}")
                    if depth_center is not None:
                        parts.append(f"ahead {depth_center:.1f}m")
                    if ekf is not None:
                        fx, fy = ekf.position
                        fstd = ekf.position_std_m
                        parts.append(
                            f"fused ({fx:.2f},{fy:.2f}) ±{max(fstd):.2f}m "
                            f"hdg {math.degrees(ekf.heading_rad):+.0f}°"
                            f"±{ekf.heading_std_deg:.1f}°"
                        )
                        # Watching this std fall is how you tell the filter
                        # is fusing rather than just echoing the last fix:
                        # once it is small, IMU yaw alone gives absolute map
                        # heading between fixes.
                        parts.append(
                            f"yawbias {math.degrees(ekf.yaw_bias_rad):+.1f}°"
                            f"±{ekf.yaw_bias_std_deg:.1f}°"
                        )
                    logger.info("  ".join(parts))
                else:
                    logger.info("not located: %s", payload.get("reason") or payload.get("error"))
            except (ClientConnectionError, OSError) as exc:
                logger.warning("Server unreachable (%s) — retrying", exc)
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass
                    client = None
                time.sleep(3.0)
            except Exception as exc:
                logger.error("Tick failed: %s", exc)

            elapsed = time.monotonic() - tick
            if elapsed < interval:
                time.sleep(interval - elapsed)
    finally:
        try:
            source.stop()
        except Exception:
            pass
        hailo.close()
        if imu is not None:
            imu.stop()
        if client is not None:
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
