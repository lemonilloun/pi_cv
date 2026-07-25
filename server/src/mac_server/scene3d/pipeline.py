"""Scene3D pipeline orchestrator: runs the 5 steps for a session on a
background thread, with per-step status/log persisted to
<session>/derived/pipeline_state.json (survives server restarts) and a
live view for the panel.

One job at a time — every step is heavy (MPS model or SfM), running two
sessions concurrently on an 8 GB Mac would thrash.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from mac_server.scene3d.session_io import SceneSession, list_sessions


logger = logging.getLogger(__name__)

STEP_ORDER = ["depth", "poses", "tsdf", "objects", "graph", "navindex"]


def _step_functions() -> dict[str, Callable]:
    # Lazy imports: each step pulls heavy deps (torch/open3d/pycolmap).
    from mac_server.scene3d.depth_step import run_depth_step
    from mac_server.scene3d.poses_step import run_poses_step
    from mac_server.scene3d.tsdf_step import run_tsdf_step
    from mac_server.scene3d.objects_step import run_objects_step
    from mac_server.scene3d.graph_step import run_graph_step
    from mac_server.scene3d.navindex_step import run_navindex_step

    return {
        "depth": run_depth_step,
        "poses": run_poses_step,
        "tsdf": run_tsdf_step,
        "objects": run_objects_step,
        "graph": run_graph_step,
        "navindex": run_navindex_step,
    }


class ScenePipeline:
    def __init__(self, sessions_dir: Path, config: dict[str, Any], repo_root: Path) -> None:
        self.sessions_dir = sessions_dir
        self.config = config
        self.repo_root = repo_root
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._job: dict[str, Any] | None = None

    # ---------------------------------------------------------------- API

    def sessions(self) -> list[dict[str, Any]]:
        return list_sessions(self.sessions_dir)

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._job is None:
                return {"running": False}
            return {
                "running": self._thread is not None and self._thread.is_alive(),
                "session_id": self._job["session_id"],
                "steps": self._job["steps"],
                "current_step": self._job.get("current_step"),
                "progress": self._job.get("progress"),
                "error": self._job.get("error"),
                "started_at": self._job.get("started_at"),
            }

    def _resolve(self, session_id: str) -> SceneSession:
        safe = "".join(c for c in session_id if c.isalnum() or c == "_")
        session = SceneSession(self.sessions_dir / safe)
        if not safe or not session.exists():
            raise ValueError(f"Unknown session: {session_id}")
        return session

    def delete(self, session_id: str) -> dict[str, Any]:
        import shutil

        session = self._resolve(session_id)
        with self._lock:
            if (
                self._thread is not None
                and self._thread.is_alive()
                and self._job is not None
                and self._job.get("session_id") == session.session_id
            ):
                raise RuntimeError("A pipeline run is active on this session")
        shutil.rmtree(session.root)
        logger.info("Deleted scene session %s", session.session_id)
        return {"deleted": session.session_id}

    def rename(self, session_id: str, name: str) -> dict[str, Any]:
        """Set the display name (kept in session_meta.json — the directory
        name stays stable so artifacts and running jobs never break)."""
        session = self._resolve(session_id)
        name = name.strip()
        if not name:
            raise ValueError("Name must not be empty")
        meta_path = session.root / "session_meta.json"
        meta = session.session_meta()
        meta["name"] = name
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"session_id": session.session_id, "name": name}

    def run(self, session_id: str, steps: list[str] | None = None, force: bool = False) -> dict[str, Any]:
        session = SceneSession(self.sessions_dir / session_id)
        if not session.exists():
            raise ValueError(f"Unknown session: {session_id}")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("A pipeline run is already active")
            selected = [s for s in STEP_ORDER if steps is None or s in steps]
            if not selected:
                raise ValueError("No valid steps selected")
            self._job = {
                "session_id": session_id,
                "steps": self._load_state(session).get("steps", {}),
                "selected": selected,
                "force": force,
                "started_at": time.time(),
            }
            self._thread = threading.Thread(
                target=self._run_job, args=(session,), daemon=True, name="scene3d"
            )
            self._thread.start()
        return {"session_id": session_id, "steps": selected}

    # ------------------------------------------------------------- worker

    def _run_job(self, session: SceneSession) -> None:
        job = self._job
        assert job is not None
        functions = _step_functions()
        session.derived.mkdir(parents=True, exist_ok=True)

        for step in job["selected"]:
            state = job["steps"].get(step, {})
            if state.get("status") == "done" and not job["force"]:
                continue
            with self._lock:
                job["current_step"] = step
                job["progress"] = "starting"
                job["steps"][step] = {"status": "running", "started_at": time.time()}
            self._save_state(session, job)

            def progress(msg: str, _step=step) -> None:
                with self._lock:
                    job["progress"] = msg

            try:
                started = time.time()
                # Steps cache their own per-frame artifacts; they need to know
                # a re-run was explicitly requested, otherwise --force only
                # re-enters the step and every cached file short-circuits it.
                step_config = {**self.config, "force": bool(job["force"])}
                report = functions[step](session, step_config, self.repo_root, progress)
                with self._lock:
                    job["steps"][step] = {
                        "status": "done",
                        "seconds": round(time.time() - started, 1),
                        "report": report,
                    }
            except Exception as exc:
                logger.error("Scene3D step %s failed: %s\n%s", step, exc, traceback.format_exc())
                with self._lock:
                    job["steps"][step] = {"status": "failed", "error": str(exc)}
                    job["error"] = f"{step}: {exc}"
                self._save_state(session, job)
                break
            self._save_state(session, job)

        with self._lock:
            job["current_step"] = None
            job["progress"] = None
        self._save_state(session, job)
        logger.info("Scene3D pipeline finished for %s", session.session_id)

    # -------------------------------------------------------------- state

    def _load_state(self, session: SceneSession) -> dict[str, Any]:
        if session.state_path().exists():
            try:
                return json.loads(session.state_path().read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        return {}

    def _save_state(self, session: SceneSession, job: dict[str, Any]) -> None:
        try:
            session.state_path().write_text(
                json.dumps({"steps": job["steps"], "updated_at": time.time()}),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not persist pipeline state: %s", exc)
