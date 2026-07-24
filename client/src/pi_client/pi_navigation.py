"""Live localization client: where is the Pi right now?

Loop (~1-2 Hz): grab a frame -> full-frame CLIP embedding on the Hailo NPU
(same RN50x4 the scene recorder uses) -> optionally fast_depth on the same
NPU (forward distance) -> nav_query to the Mac. The Mac matches the
embedding against the place indexes of all reconstructed sessions
(pipeline step `navindex`) and answers which room this looks like, the
closest recorded viewpoint, and the position/heading on that room's floor
plan. The panel's Scene tab draws the fix on the plan (GET /api/nav/last).

    ./scripts/run_pi_navigation.sh          # Ctrl+C to stop

This is retrieval-based localization (place recognition), not SLAM: it
answers "which room, roughly where, roughly which way" — exactly what the
future navigation stack needs as its coarse prior.
"""

from __future__ import annotations

import argparse
import logging
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
from pi_client.hailo_infer import HailoMultiModel
from pi_client.network import ClientConnectionError, PiClient
from pi_client.protocol import make_nav_query_message
from shared.config import load_config


logger = logging.getLogger(__name__)


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

    client: PiClient | None = None
    stop = {"flag": False}

    def handle_signal(signum: int, _frame: Any) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    source.start()
    interval = 1.0 / max(args.hz, 0.2)
    logger.info("Navigation: querying %s:%s at %.1f Hz (Ctrl+C to stop)", host, port, args.hz)
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

                if client is None:
                    client = PiClient(host, port, timeout_seconds=4.0)
                    client.connect()
                response, _ = client.request(
                    make_nav_query_message(
                        device_id, [round(float(v), 5) for v in vec], depth_center
                    )
                )
                payload = response.payload
                if payload.get("located"):
                    best = payload["best"]
                    room = best.get("name") or best["session_id"].replace("session_", "")
                    heading = best.get("heading_deg")
                    parts = [
                        f"room: {room}",
                        f"kf {best['keyframe']:06d}",
                        f"sim {best['similarity']:.2f}",
                    ]
                    if heading is not None:
                        spread = best.get("heading_spread_deg")
                        trust = f" ±{spread:.0f}°" if spread is not None else ""
                        parts.append(f"heading {heading:+.0f}°{trust}")
                    if depth_center is not None:
                        parts.append(f"ahead {depth_center:.1f}m")
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
        if client is not None:
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
