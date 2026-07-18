"""CLI driver for the Scene3D pipeline (same code path as the web tab).

    ./scripts/run_scene_pipeline.sh <session_id> [step ...]
    ./scripts/run_scene_pipeline.sh --list
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "server/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from mac_server.scene3d.pipeline import STEP_ORDER, ScenePipeline
from shared.config import load_config


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run the Scene3D reconstruction pipeline")
    parser.add_argument("session_id", nargs="?", help="session_YYYYMMDD_HHMMSS")
    parser.add_argument("steps", nargs="*", help=f"subset of {STEP_ORDER} (default: all)")
    parser.add_argument("--list", action="store_true", help="List sessions and exit")
    parser.add_argument("--force", action="store_true", help="Re-run steps already done")
    args = parser.parse_args()

    config = load_config(REPO_ROOT / "config/default.json", REPO_ROOT / ".env").get("scene3d", {})
    sessions_dir = Path(config.get("sessions_dir", "data/scene_sessions"))
    if not sessions_dir.is_absolute():
        sessions_dir = REPO_ROOT / sessions_dir
    pipeline = ScenePipeline(sessions_dir, config, REPO_ROOT)

    if args.list or not args.session_id:
        for s in pipeline.sessions():
            steps = ",".join(
                f"{k}:{v.get('status')}" for k, v in (s.get("steps") or {}).items()
            ) or "-"
            print(f"{s['session_id']}  keyframes={s['keyframes']}  calibrated={s['calibrated']}  {steps}")
        return 0

    pipeline.run(args.session_id, steps=args.steps or None, force=args.force)
    while True:
        status = pipeline.status()
        if not status.get("running"):
            break
        step = status.get("current_step") or "?"
        print(f"\r[{step}] {status.get('progress') or ''}    ", end="", flush=True)
        time.sleep(1.0)
    print()
    status = pipeline.status()
    for step, state in status.get("steps", {}).items():
        print(f"{step:8s} {state.get('status'):8s} {state.get('report') or state.get('error') or ''}")
    return 1 if status.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
