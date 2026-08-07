"""Monitor controller: consumes the enriched objects stream and drives the
v2 pipeline — tracker → presence → relations → scene graph — plus snapshots,
the apfel digester (with key moments) and Q&A.

One thread per active monitoring session (pattern: mapping.ScanController).
Timestamps are Mac wall clock (`computed_at` from cv_worker); `pi_frame_index`
is used only for deduplication.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from mac_server.monitoring.events import (
    RelationConfig,
    RelationEngine,
    RelationTransition,
)
from mac_server.monitoring.graph import SceneGraph
from mac_server.monitoring.presence import PresenceConfig, PresenceManager, PresentEntity
from mac_server.monitoring.scenes import SceneStore
from mac_server.monitoring.store import MonitoringStore
from mac_server.monitoring.tracker import GreedyTracker, Track, TrackerConfig


logger = logging.getLogger(__name__)

STALLED_AFTER_S = 5.0
SWEEP_INTERVAL_S = 3600.0


def _maybe_start_service(
    service_config: dict[str, Any],
    binary_name: str,
    build_args: Callable[[str, int], list[str]],
    log_path: Path,
    default_port: int,
    build_env: Callable[[str, int], dict[str, str]] | None = None,
) -> subprocess.Popen | None:
    """Launch a local AI helper SERVICE (apfel / Ollama) if it isn't already
    listening on its configured port.

    The service processes are tiny at idle (apfel ~8 MB, ollama similar) —
    the heavy MODELS are never held resident: gemma loads per caption call
    and unloads (vision keep_alive="0s"), apfel's model is managed by macOS.
    A process that was already listening (started by the user) is left alone
    and never terminated by us — only copies we spawned are ours to stop."""
    if not service_config.get("enabled", False):
        return None
    if not service_config.get("auto_start", True):
        return None

    base_url = str(service_config.get("base_url", f"http://127.0.0.1:{default_port}"))
    parsed = urlparse(base_url)
    host, port = parsed.hostname or "127.0.0.1", parsed.port or default_port

    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.settimeout(0.5)
        if probe.connect_ex((host, port)) == 0:
            logger.info("%s already listening on %s:%s — leaving it as is", binary_name, host, port)
            return None

    if shutil.which(binary_name) is None:
        logger.warning(
            "%s binary not found in PATH; this monitoring feature stays unavailable",
            binary_name,
        )
        return None

    env = None
    if build_env is not None:
        env = dict(os.environ)
        env.update(build_env(host, port))
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "ab")
        process = subprocess.Popen(
            build_args(host, port),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        logger.info(
            "Started %s (pid=%s) on %s:%s, logging to %s",
            binary_name, process.pid, host, port, log_path,
        )
        return process
    except OSError as exc:
        logger.warning("Failed to start %s: %s", binary_name, exc)
        return None


def _terminate_service(process: subprocess.Popen | None, name: str) -> None:
    if process is None:
        return
    logger.info("Stopping %s (pid=%s) — monitoring session ended", name, process.pid)
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()


class MonitorError(RuntimeError):
    """Raised when monitoring cannot start or stop."""


class MonitorController:
    def __init__(
        self,
        cv_worker: Any,
        frame_hub: Any,
        registry: Any,
        scene_store: SceneStore,
        store: MonitoringStore,
        config: dict[str, Any],
        scan_controller: Any = None,
        agent: Any = None,
        vision: Any = None,
    ) -> None:
        self._cv_worker = cv_worker
        self._frame_hub = frame_hub
        self._registry = registry
        self.scene_store = scene_store
        self.store = store
        self._config = config
        self._scan_controller = scan_controller
        self.agent = agent  # ApfelClient | None
        self.vision = vision  # OllamaVisionClient | None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._digester_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._session: dict[str, Any] | None = None
        self._signature_provider = None  # set lazily (needs cv2/numpy/torch)
        self._apfel_process: subprocess.Popen | None = None
        self._ollama_process: subprocess.Popen | None = None

    # ---------------------------------------------------------------- API

    def status(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
            agent_base_url = str(
                self._config.get("agent", {}).get("base_url", "http://127.0.0.1:11500")
            )
            if self.agent is not None:
                agent_status = {
                    "enabled": True,
                    "healthy": self.agent.healthy,
                    "base_url": self.agent.base_url,
                }
            elif self._config.get("agent", {}).get("enabled"):
                agent_status = {"enabled": True, "healthy": False, "base_url": agent_base_url}
            else:
                agent_status = {"enabled": False, "healthy": False, "base_url": agent_base_url}
            if session is None:
                return {"active": False, "agent": agent_status}
            presence: PresenceManager = session["presence"]
            relations: RelationEngine = session["relations"]
            graph: SceneGraph = session["graph"]
            labels = {e.entity_key: presence.label_of(e) for e in presence.alive_entities()}
            return {
                "active": session["active"],
                "scene_id": session["scene_id"],
                "since": session["started_at"],
                "stalled": session["stalled"],
                "tick_hz": session["tick_hz"],
                "entities": graph.now_state(
                    presence.alive_entities(), labels, relations.open_relations
                ),
                "open_edges": len(relations.open_relations),
                "events_total": session["events_total"],
                "agent": agent_status,
            }

    def start(self, scene_id: str) -> dict[str, Any]:
        scene = self.scene_store.load_scene(scene_id)
        if scene is None:
            raise ValueError(f"Unknown scene: {scene_id}")
        cv_state = self._cv_worker.status().get("state")
        if cv_state != "running":
            raise MonitorError(f"Server CV worker is not running (state: {cv_state})")
        if self._scan_controller is not None:
            scan = self._scan_controller.status()
            if scan.get("active"):
                raise MonitorError("A room scan is active; stop it before monitoring")

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise MonitorError("Monitoring is already active")

            frame_w = 1280.0
            frame_h = 720.0
            tracker_config = TrackerConfig(
                min_confidence=float(self._config.get("min_confidence", 0.35)),
                iou_gate=float(self._config.get("iou_gate", 0.1)),
                centroid_gate_frac=float(self._config.get("centroid_gate_frac", 0.15)),
                confirm_hits=int(self._config.get("confirm_hits", 3)),
                lost_after_s=float(self._config.get("lost_after_s", 2.0)),
                end_after_s=float(self._config.get("end_after_s", 10.0)),
                frame_width=frame_w,
                frame_height=frame_h,
            )
            presence_cfg = self._config.get("presence", {})
            presence_config = PresenceConfig(
                exit_edge_frac=float(presence_cfg.get("exit_edge_frac", 0.10)),
                exit_absent_s=float(presence_cfg.get("exit_absent_s", 25.0)),
                presence_timeout_s=float(presence_cfg.get("presence_timeout_s", 300.0)),
                frame_width=frame_w,
                frame_height=frame_h,
            )
            relations_cfg = self._config.get("relations", {})
            relation_config = RelationConfig(
                open_hold_s=float(self._config.get("event_open_hold_s", 2.0)),
                close_hold_s=float(self._config.get("event_close_hold_s", 3.0)),
                overlap_on_ratio=float(self._config.get("overlap_on_ratio", 0.3)),
                depth_agree_m=float(self._config.get("depth_agree_m", 0.7)),
                near_gap_frac=float(relations_cfg.get("near_gap_frac", 0.05)),
                near_depth_agree_m=float(relations_cfg.get("near_depth_agree_m", 1.0)),
                move_min_frac=float(relations_cfg.get("move_min_frac", 0.15)),
                frame_width=frame_w,
                frame_height=frame_h,
            )
            provider = self._ensure_signature_provider()
            self._session = {
                "active": True,
                "scene_id": scene["scene_id"],
                "scene": scene,
                "tracker": GreedyTracker(
                    tracker_config,
                    excluded_classes=set(self._config.get("anchor_classes", [])),
                ),
                "presence": PresenceManager(
                    presence_config,
                    similarity=provider.similarity if provider is not None else None,
                ),
                "relations": RelationEngine(relation_config),
                "graph": SceneGraph(self.store, scene["scene_id"]),
                "started_at": time.time(),
                "stalled": False,
                "tick_hz": None,
                "tick_times": deque(maxlen=40),
                "events_total": 0,
                "last_sweep": time.monotonic(),
            }
            self._session["graph"].ensure_anchor_nodes(
                scene.get("anchors", []), time.time()
            )
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="monitor")
            self._thread.start()
            if self.agent is not None:
                self._digester_thread = threading.Thread(
                    target=self._digester_run, daemon=True, name="monitor-digester"
                )
                self._digester_thread.start()

        # Monitoring consumes YOLO detections — switch the Pi best-effort.
        try:
            self._registry.send_command(device_id=None, action="set_mode", mode="yolo")
        except Exception as exc:
            logger.info("Could not auto-switch Pi to yolo mode: %s", exc)

        # AI helper services live exactly as long as the monitoring session:
        # started here, stopped in stop()/shutdown().
        self._start_ai_services()

        return {"scene_id": scene["scene_id"]}

    def _start_ai_services(self) -> None:
        # "managed"/backend="ollama" mean a local binary this process might
        # need to spawn (and must remember to kill). The default agent/
        # vision backend now points at the already-running Qwen vLLM
        # instance on pdfserver — external infrastructure this process
        # must never spawn OR kill, so both auto-start calls are skipped
        # entirely rather than let _maybe_start_service's "already
        # listening" probe merely happen to no-op correctly (it would,
        # since Qwen is reachable — but skipping is the honest way to say
        # "not ours to manage", and keeps the logs from claiming "apfel"/
        # "ollama" is what's actually listening there).
        if self._config.get("agent", {}).get("managed", False):
            self._apfel_process = _maybe_start_service(
                self._config.get("agent", {}),
                binary_name="apfel",
                build_args=lambda host, port: ["apfel", "--serve", "--host", host, "--port", str(port)],
                log_path=self.store.data_dir / "apfel.log",
                default_port=11500,
            )
        # Ollama takes no host/port CLI flags — it reads OLLAMA_HOST from env.
        if str(self._config.get("vision", {}).get("backend", "qwen")).lower() == "ollama":
            self._ollama_process = _maybe_start_service(
                self._config.get("vision", {}),
                binary_name="ollama",
                build_args=lambda host, port: ["ollama", "serve"],
                build_env=lambda host, port: {"OLLAMA_HOST": f"{host}:{port}"},
                log_path=self.store.data_dir / "ollama.log",
                default_port=11434,
            )

    def _stop_ai_services(self) -> None:
        _terminate_service(self._apfel_process, "apfel")
        self._apfel_process = None
        _terminate_service(self._ollama_process, "ollama")
        self._ollama_process = None

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._session is None or self._thread is None:
                raise MonitorError("Monitoring is not active")
            session = self._session
        self._stop_event.set()
        self._thread.join(timeout=5)
        if self._digester_thread is not None:
            self._digester_thread.join(timeout=5)
            self._digester_thread = None
        with self._lock:
            self._thread = None
            session["active"] = False

        now = time.time()
        closed = 0
        try:
            presence: PresenceManager = session["presence"]
            for event in presence.force_leave_all(now):
                self._persist_exit(session, event)
                closed += 1
            relations: RelationEngine = session["relations"]
            graph: SceneGraph = session["graph"]
            for transition in relations.force_close_all(now):
                graph.record_transition(transition)
                closed += 1
            presence.prune_left()
        except Exception as exc:
            # A persistence failure here must never leave the controller
            # stuck (active=False but _session non-None, refusing new
            # start() calls or reporting stale status forever).
            logger.error("Failed to close out monitoring session cleanly: %s", exc)
        finally:
            with self._lock:
                self._session = None
            self._stop_ai_services()
        return {"scene_id": session["scene_id"], "events_closed": closed}

    def shutdown(self) -> None:
        self._stop_event.set()
        for thread in (self._thread, self._digester_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2)
        self._stop_ai_services()
        self.store.close()

    def freeze_anchors(self, scene_id: str | None) -> list[dict[str, Any]]:
        target = scene_id
        if target is None:
            with self._lock:
                if self._session is None:
                    raise MonitorError("No active session and no scene_id given")
                target = self._session["scene_id"]
        _, latest = self._cv_worker.objects_store.latest()
        if latest is None or not latest.get("objects"):
            raise MonitorError(
                "No detected objects available — is the Pi in yolo mode with furniture in view?"
            )
        anchors = self.scene_store.set_anchors(
            target,
            latest["objects"],
            anchor_classes=set(self._config.get("anchor_classes", [])),
        )
        with self._lock:
            if self._session is not None and self._session["scene_id"] == target:
                self._session["scene"]["anchors"] = anchors
                self._session["graph"].ensure_anchor_nodes(anchors, time.time())
        return anchors

    # ----------------------------------------------------------------- Q&A

    def ask(self, question: str, hours: float = 3.0) -> dict[str, Any]:
        """Answer a free-form question over the scene graph slice."""
        if self.agent is None:
            raise MonitorError("Agent is not configured")
        with self._lock:
            session = self._session
            if session is None:
                raise MonitorError("Monitoring is not active")
            scene_id = session["scene_id"]
            scene_name = session["scene"].get("name", scene_id)
            presence: PresenceManager = session["presence"]
            relations: RelationEngine = session["relations"]
            graph: SceneGraph = session["graph"]
            labels = {e.entity_key: presence.label_of(e) for e in presence.alive_entities()}
            now_state = graph.now_state(
                presence.alive_entities(), labels, relations.open_relations
            )

        from mac_server.monitoring.agent import format_now_state

        now = time.time()
        window_start = now - hours * 3600
        edges = self.store.edges_between(scene_id, window_start, now, limit=400)
        digests = self.store.digests_between(scene_id, window_start, now)
        rolling = self.store.get_rolling_summary(scene_id)
        answer = self.agent.ask(
            scene_name,
            question,
            format_now_state(now_state),
            edges,
            digests,
            rolling_summary=rolling["text"] if rolling else None,
        )
        if answer is None:
            raise MonitorError("Agent is unavailable (is apfel running?)")
        key_moments = self.store.edges_between(
            scene_id, window_start, now, limit=20, key_only=True
        )
        return {"answer": answer, "key_moments": key_moments}

    # ------------------------------------------------------------- worker

    def _run(self) -> None:
        session = self._session
        assert session is not None
        tracker: GreedyTracker = session["tracker"]
        sequence = -1
        last_frame_index: Any = None
        last_data_at = time.monotonic()

        logger.info("Monitoring started: scene=%s", session["scene_id"])

        while not self._stop_event.is_set():
            sequence, batch = self._cv_worker.objects_store.wait_for_next(sequence, timeout=1.0)
            now = time.time()

            if batch is not None and batch.get("pi_frame_index") != last_frame_index:
                last_frame_index = batch.get("pi_frame_index")
                last_data_at = time.monotonic()
                session["stalled"] = False
                objects = batch.get("objects", [])
                tick_time = float(batch.get("computed_at", now))
                session["tick_times"].append(time.monotonic())
            else:
                # Timeout or duplicate: age tracks with an empty tick so the
                # presence layer keeps its exit clocks running when the Pi
                # stops sending — the panel shows `stalled` to disambiguate.
                objects = []
                tick_time = now
                if time.monotonic() - last_data_at > STALLED_AFTER_S:
                    session["stalled"] = True

            try:
                self._tick(session, tracker, objects, tick_time)
            except Exception as exc:
                logger.error("Monitoring tick failed: %s", exc)

            ticks = session["tick_times"]
            if len(ticks) >= 2 and ticks[-1] > ticks[0]:
                session["tick_hz"] = round((len(ticks) - 1) / (ticks[-1] - ticks[0]), 1)

            if time.monotonic() - session["last_sweep"] > SWEEP_INTERVAL_S:
                session["last_sweep"] = time.monotonic()
                try:
                    self.store.sweep(
                        retention_days=float(self._config.get("retention_days", 14)),
                        max_snapshot_mb=float(self._config.get("max_snapshot_mb", 200)),
                        snapshots_root=self.store.data_dir,
                        key_snapshot_retention_h=float(
                            self._config.get("qa", {}).get("key_snapshot_retention_h", 24)
                        ),
                    )
                except Exception as exc:
                    logger.warning("Retention sweep failed: %s", exc)

        logger.info("Monitoring stopped: scene=%s", session["scene_id"])

    def _tick(
        self,
        session: dict[str, Any],
        tracker: GreedyTracker,
        objects: list[dict[str, Any]],
        tick_time: float,
    ) -> None:
        presence: PresenceManager = session["presence"]
        relations: RelationEngine = session["relations"]
        graph: SceneGraph = session["graph"]

        updates = tracker.update(objects, tick_time)
        for track in updates.confirmed_new:
            self._compute_signature(track, tick_time)
            entered = presence.on_track_confirmed(track, tick_time)
            if entered is not None:
                node_id = graph.bind_entity(entered.entity, tick_time)
                edge_id = graph.record_presence_event(entered, node_id)
                session["events_total"] += 1
                if "entered" in self._config.get("snapshot_events", []):
                    path = self._save_crop(session["scene_id"], entered.entity.last_bbox)
                    if path is not None:
                        self.store.mark_edge_snapshot(edge_id, path)
        for track in updates.ended:
            presence.on_track_ended(track, tick_time)

        # Presence tick: sync positions, expire occluded -> exited.
        live_tracks = tracker.confirmed_tracks()
        for event in presence.update(live_tracks, tick_time):
            self._persist_exit(session, event)
        presence.prune_left()

        present = presence.present_entities()
        alive = presence.alive_entities()
        labels = {e.entity_key: presence.label_of(e) for e in alive}
        occluded_keys = {e.entity_key for e in alive if e.state == "occluded"}
        anchors = session["scene"].get("anchors", [])
        for transition in relations.evaluate(
            present, labels, anchors, tick_time, occluded_keys
        ):
            edge_id = graph.record_transition(transition)
            if transition.action in {"open", "point"}:
                session["events_total"] += 1
            if transition.action == "open" and edge_id is not None:
                if transition.relation in self._config.get("snapshot_events", []):
                    subject = next(
                        (e for e in present if e.entity_key == transition.subject_key), None
                    )
                    if subject is not None:
                        path = self._save_crop(session["scene_id"], subject.last_bbox)
                        if path is not None:
                            self.store.mark_edge_snapshot(edge_id, path)
                if transition.relation in self._config.get("vision", {}).get(
                    "caption_events", []
                ):
                    self._request_caption(session, transition, edge_id)

    def _persist_exit(self, session: dict[str, Any], event: Any) -> None:
        relations: RelationEngine = session["relations"]
        graph: SceneGraph = session["graph"]
        entity: PresentEntity = event.entity
        for transition in relations.entity_left(entity.entity_key, event.t):
            graph.record_transition(transition)
        node_id = graph.node_of(entity.entity_key)
        if node_id is not None:
            graph.record_presence_event(event, node_id)
            session["events_total"] += 1
        graph.release_entity(entity, event.t, visible_s=entity.total_visible_s)

    # ---------------------------------------------------------- signatures

    def _compute_signature(self, track: Track, now: float) -> None:
        provider = self._ensure_signature_provider()
        if provider is None:
            return
        try:
            _, frame = self._frame_hub.get("pi").latest()
            if frame is None:
                return
            signature = provider.compute_from_jpeg(frame.data, track.bbox)
            if signature is not None:
                track.signature = signature
                track.signature_updated_at = now
        except Exception as exc:
            logger.debug("Signature computation failed: %s", exc)

    def _ensure_signature_provider(self):
        if self._signature_provider is not None:
            return self._signature_provider
        embeddings_cfg = self._config.get("embeddings", {})
        if embeddings_cfg.get("enabled", True):
            try:
                from mac_server.monitoring.embeddings import ClipSignatureProvider

                self._signature_provider = ClipSignatureProvider(
                    model_name=str(embeddings_cfg.get("model", "MobileCLIP-S1")),
                    device=str(embeddings_cfg.get("device", "mps")),
                )
                logger.info("Signature provider: CLIP embeddings (%s)",
                            embeddings_cfg.get("model", "MobileCLIP-S1"))
                return self._signature_provider
            except Exception as exc:
                logger.warning("CLIP embeddings unavailable (%s); falling back to HSV", exc)
        try:
            from mac_server.monitoring.signatures import HistogramSignatureProvider

            self._signature_provider = HistogramSignatureProvider()
        except Exception as exc:
            logger.debug("No signature provider available: %s", exc)
        return self._signature_provider

    # ----------------------------------------------------------- snapshots

    def _save_crop(
        self,
        scene_id: str,
        bbox: tuple[float, float, float, float],
        margin_frac: float = 0.15,
    ) -> str | None:
        """Crop the current Pi frame around a bbox and save it; returns the
        path (or None). Used for entered/on snapshots and key moments."""
        try:
            import cv2
            import numpy as np

            _, frame = self._frame_hub.get("pi").latest()
            if frame is None:
                return None
            image = cv2.imdecode(np.frombuffer(frame.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return None
            h, w = image.shape[:2]
            x0, y0, x1, y1 = bbox
            margin_x = (x1 - x0) * margin_frac
            margin_y = (y1 - y0) * margin_frac
            ix0 = max(0, int(x0 - margin_x))
            iy0 = max(0, int(y0 - margin_y))
            ix1 = min(w, int(x1 + margin_x))
            iy1 = min(h, int(y1 + margin_y))
            crop = image[iy0:iy1, ix0:ix1]
            if crop.size == 0:
                return None
            snap_dir = self.store.data_dir / scene_id / "snapshots"
            snap_dir.mkdir(parents=True, exist_ok=True)
            path = snap_dir / f"{int(time.time() * 1000)}.jpg"
            ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                path.write_bytes(encoded.tobytes())
                return str(path)
        except Exception as exc:
            logger.debug("Snapshot failed: %s", exc)
        return None

    # --------------------------------------------------- vision captioning

    def _request_caption(
        self, session: dict[str, Any], transition: RelationTransition, edge_id: int
    ) -> None:
        """Fire-and-forget: caption this edge's crop with the vision model
        on a short-lived thread. Never blocks the tick loop — with
        keep_alive=0 a call can take several seconds (full model load), and
        the caption is expected to land in the feed a moment after the edge
        itself, not synchronously with it."""
        if self.vision is None:
            return
        presence: PresenceManager = session["presence"]
        subject = presence.entities.get(transition.subject_key)
        if subject is None:
            return
        anchor_bbox = None
        for anchor in session["scene"].get("anchors", []):
            if anchor.get("anchor_id") == transition.object_label:
                anchor_bbox = anchor.get("bbox_xyxy")
                break
        thread = threading.Thread(
            target=self._caption_worker,
            args=(edge_id, tuple(subject.last_bbox), anchor_bbox),
            daemon=True,
            name="monitor-caption",
        )
        thread.start()

    def _caption_worker(
        self,
        edge_id: int,
        subject_bbox: tuple[float, float, float, float],
        anchor_bbox: list[float] | None,
    ) -> None:
        try:
            import cv2
            import numpy as np

            from mac_server.monitoring.vision import union_bbox_with_margin

            _, frame = self._frame_hub.get("pi").latest()
            if frame is None:
                return
            image = cv2.imdecode(np.frombuffer(frame.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return
            h, w = image.shape[:2]
            box_b = tuple(anchor_bbox) if anchor_bbox else subject_bbox
            margin = float(self._config.get("vision", {}).get("crop_margin_frac", 0.15))
            x0, y0, x1, y1 = union_bbox_with_margin(subject_bbox, box_b, margin, w, h)
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                return
            ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return
            caption = self.vision.caption(encoded.tobytes())
            if caption:
                self.store.set_edge_caption(edge_id, caption)
        except Exception as exc:
            logger.debug("Vision caption failed: %s", exc)

    # ------------------------------------------------------------ digester

    def _digester_run(self) -> None:
        agent = self.agent
        session = self._session
        if agent is None or session is None:
            return
        interval = float(self._config.get("agent", {}).get("digest_interval_s", 300))
        max_moments = int(self._config.get("qa", {}).get("key_moments_per_digest", 3))
        window_start = time.time()
        while not self._stop_event.wait(timeout=min(interval, 10.0)):
            if time.time() - window_start < interval:
                continue
            window_end = time.time()
            scene_id = session["scene_id"]
            scene_name = session["scene"].get("name", scene_id)
            try:
                edges = self.store.edges_between(scene_id, window_start, window_end)
                if edges:
                    text = agent.digest(scene_name, edges)
                    if text:
                        self.store.insert_digest(scene_id, window_start, window_end, text, agent.model)

                    # Key moments: let the model flag the notable edges of
                    # this cycle; they get participant snapshots (kept 24h).
                    for moment in agent.pick_key_moments(edges, max_moments=max_moments):
                        self._mark_key_moment(session, moment)

                    # Efficient context: fold this window into one running
                    # summary instead of re-merging a growing pile of
                    # digests on every question.
                    previous = self.store.get_rolling_summary(scene_id)
                    updated_text = agent.update_rolling_summary(
                        scene_name, previous["text"] if previous else None, edges
                    )
                    if updated_text:
                        total = (previous["event_count"] if previous else 0) + len(edges)
                        self.store.upsert_rolling_summary(scene_id, updated_text, total, window_end)
            except Exception as exc:
                logger.warning("Digest cycle failed: %s", exc)
            window_start = window_end

    def _mark_key_moment(self, session: dict[str, Any], edge: dict[str, Any]) -> None:
        """Flag an edge as a key moment; if it has no snapshot yet and its
        subject is still around, grab one now (best effort — for a subject
        that already left, the mark still lands, just without a photo)."""
        snapshot_path = edge.get("snapshot_path")
        if snapshot_path is None:
            presence: PresenceManager = session["presence"]
            subject = next(
                (
                    e
                    for e in presence.alive_entities()
                    if presence.label_of(e) == edge.get("subject_label")
                    or e.class_name == edge.get("subject_label")
                ),
                None,
            )
            if subject is not None:
                snapshot_path = self._save_crop(session["scene_id"], subject.last_bbox)
        self.store.mark_edge_key_moment(edge["id"], snapshot_path)
