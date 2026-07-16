"""SQLite persistence for the monitoring system (stdlib sqlite3, WAL).

Threading contract: MonitoringStore owns one long-lived connection, opened
with `check_same_thread=False` because it is legitimately used from more
than one background thread (the monitor tick thread and the digester
thread), not just its constructor's thread. `check_same_thread=False` only
lifts Python's same-thread check — it does NOT make concurrent use safe by
itself, so every access to `self._conn` is serialized through `self._lock`.
HTTP handler threads instead read through short-lived per-request
connections (`read_connection()`), which WAL makes safe against the
lock-protected writer without contending for the same lock.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS entities (
  id INTEGER PRIMARY KEY,
  scene_id TEXT NOT NULL,
  class TEXT NOT NULL,
  label TEXT NOT NULL,
  signature TEXT NOT NULL,
  first_seen REAL NOT NULL,
  last_seen REAL NOT NULL,
  total_visible_s REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tracks (
  id INTEGER PRIMARY KEY,
  scene_id TEXT NOT NULL,
  entity_id INTEGER REFERENCES entities(id),
  class TEXT NOT NULL,
  t_start REAL NOT NULL,
  t_end REAL,
  frames INTEGER NOT NULL DEFAULT 0,
  meta TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  scene_id TEXT NOT NULL,
  type TEXT NOT NULL,
  track_id INTEGER,
  entity_id INTEGER,
  subject_label TEXT NOT NULL,
  object_label TEXT,
  t_start REAL NOT NULL,
  t_end REAL,
  details TEXT,
  snapshot_path TEXT
);
CREATE TABLE IF NOT EXISTS digests (
  id INTEGER PRIMARY KEY,
  scene_id TEXT NOT NULL,
  t_start REAL NOT NULL,
  t_end REAL NOT NULL,
  text TEXT NOT NULL,
  model TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rolling_summaries (
  scene_id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  event_count INTEGER NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_scene_t ON events(scene_id, t_start);
CREATE INDEX IF NOT EXISTS idx_digests_scene_t ON digests(scene_id, t_start);
CREATE INDEX IF NOT EXISTS idx_entities_scene ON entities(scene_id, class, last_seen);
"""


class MonitoringStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.db_path = data_dir / "monitoring.sqlite3"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.close()
        except Exception:
            pass

    def read_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=2.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------ writes

    def insert_track(self, scene_id: str, class_name: str, t_start: float) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO tracks (scene_id, class, t_start) VALUES (?, ?, ?)",
                (scene_id, class_name, t_start),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def finish_track(
        self,
        db_id: int,
        t_end: float,
        frames: int,
        entity_id: int | None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE tracks SET t_end = ?, frames = ?, entity_id = ?, meta = ? WHERE id = ?",
                (t_end, frames, entity_id, json.dumps(meta or {}), db_id),
            )
            self._conn.commit()

    def insert_event(
        self,
        scene_id: str,
        event_type: str,
        subject_label: str,
        t_start: float,
        t_end: float | None,
        track_db_id: int | None = None,
        entity_id: int | None = None,
        object_label: str | None = None,
        details: dict[str, Any] | None = None,
        snapshot_path: str | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (scene_id, type, track_id, entity_id, subject_label,"
                " object_label, t_start, t_end, details, snapshot_path)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    scene_id,
                    event_type,
                    track_db_id,
                    entity_id,
                    subject_label,
                    object_label,
                    t_start,
                    t_end,
                    json.dumps(details or {}),
                    snapshot_path,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def close_event(self, event_db_id: int, t_end: float, details: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE events SET t_end = ?, details = ? WHERE id = ?",
                (t_end, json.dumps(details), event_db_id),
            )
            self._conn.commit()

    def set_event_snapshot(self, event_db_id: int, snapshot_path: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE events SET snapshot_path = ? WHERE id = ?", (snapshot_path, event_db_id)
            )
            self._conn.commit()

    def set_event_caption(self, event_db_id: int, caption: str) -> None:
        """Merge a vision-model caption into an event's details JSON. Read
        modify-write (not SQLite's json_set) to avoid depending on the
        JSON1 extension being compiled into the local sqlite3 build."""
        with self._lock:
            row = self._conn.execute(
                "SELECT details FROM events WHERE id = ?", (event_db_id,)
            ).fetchone()
            if row is None:
                return
            details = json.loads(row[0]) if row[0] else {}
            details["caption"] = caption
            self._conn.execute(
                "UPDATE events SET details = ? WHERE id = ?",
                (json.dumps(details), event_db_id),
            )
            self._conn.commit()

    def update_event_labels(self, track_db_id: int, entity_id: int, label: str) -> None:
        """Backfill entity info onto events created before resolution."""
        with self._lock:
            self._conn.execute(
                "UPDATE events SET entity_id = ?, subject_label = ? WHERE track_id = ?",
                (entity_id, label, track_db_id),
            )
            self._conn.commit()

    # ---------------------------------------------------------- entities

    def insert_entity(
        self, scene_id: str, class_name: str, label: str, signature: dict[str, Any], now: float
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO entities (scene_id, class, label, signature, first_seen, last_seen)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (scene_id, class_name, label, json.dumps(signature), now, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def update_entity(
        self,
        entity_id: int,
        signature: dict[str, Any] | None,
        last_seen: float,
        visible_s: float,
    ) -> None:
        with self._lock:
            if signature is not None:
                self._conn.execute(
                    "UPDATE entities SET signature = ?, last_seen = ?,"
                    " total_visible_s = total_visible_s + ? WHERE id = ?",
                    (json.dumps(signature), last_seen, visible_s, entity_id),
                )
            else:
                self._conn.execute(
                    "UPDATE entities SET last_seen = ?, total_visible_s = total_visible_s + ?"
                    " WHERE id = ?",
                    (last_seen, visible_s, entity_id),
                )
            self._conn.commit()

    def candidate_entities(
        self, scene_id: str, class_name: str, since: float
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, label, signature, first_seen, last_seen FROM entities"
                " WHERE scene_id = ? AND class = ? AND last_seen >= ?",
                (scene_id, class_name, since),
            ).fetchall()
        return [
            {
                "id": row[0],
                "label": row[1],
                "signature": json.loads(row[2]),
                "first_seen": row[3],
                "last_seen": row[4],
            }
            for row in rows
        ]

    def entity_count(self, scene_id: str, class_name: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM entities WHERE scene_id = ? AND class = ?",
                (scene_id, class_name),
            ).fetchone()
        return int(row[0])

    # ---------------------------------------------------------- digests

    def insert_digest(
        self, scene_id: str, t_start: float, t_end: float, text: str, model: str
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO digests (scene_id, t_start, t_end, text, model, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (scene_id, t_start, t_end, text, model, time.time()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    # ----------------------------------------------------------- queries
    # (touch the shared connection under self._lock; HTTP handlers that
    # don't need the controller's live view use read_connection() instead,
    # which never contends for this lock)

    def events_between(
        self, scene_id: str, t_start: float, t_end: float, limit: int = 500
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, type, subject_label, object_label, t_start, t_end, details"
                " FROM events WHERE scene_id = ? AND t_start >= ? AND t_start < ?"
                " ORDER BY t_start LIMIT ?",
                (scene_id, t_start, t_end, limit),
            ).fetchall()
        return [
            {
                "id": row[0],
                "type": row[1],
                "subject_label": row[2],
                "object_label": row[3],
                "t_start": row[4],
                "t_end": row[5],
                "details": json.loads(row[6]) if row[6] else {},
            }
            for row in rows
        ]

    def digests_between(self, scene_id: str, t_start: float, t_end: float) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT t_start, t_end, text FROM digests"
                " WHERE scene_id = ? AND t_end > ? AND t_start < ? ORDER BY t_start",
                (scene_id, t_start, t_end),
            ).fetchall()
        return [{"t_start": row[0], "t_end": row[1], "text": row[2]} for row in rows]

    # ---------------------------------------------------- rolling summary

    def get_rolling_summary(self, scene_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT text, event_count, updated_at FROM rolling_summaries WHERE scene_id = ?",
                (scene_id,),
            ).fetchone()
        if row is None:
            return None
        return {"text": row[0], "event_count": row[1], "updated_at": row[2]}

    def upsert_rolling_summary(
        self, scene_id: str, text: str, event_count: int, now: float
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO rolling_summaries (scene_id, text, event_count, updated_at)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(scene_id) DO UPDATE SET"
                " text = excluded.text, event_count = excluded.event_count,"
                " updated_at = excluded.updated_at",
                (scene_id, text, event_count, now),
            )
            self._conn.commit()

    # --------------------------------------------------------- retention

    def sweep(self, retention_days: float, max_snapshot_mb: float, snapshots_root: Path) -> None:
        cutoff = time.time() - retention_days * 86400
        with self._lock:
            old_snapshots = self._conn.execute(
                "SELECT snapshot_path FROM events WHERE t_start < ? AND snapshot_path IS NOT NULL",
                (cutoff,),
            ).fetchall()
            self._conn.execute("DELETE FROM events WHERE t_start < ?", (cutoff,))
            self._conn.execute("DELETE FROM tracks WHERE t_start < ?", (cutoff,))
            self._conn.execute("DELETE FROM digests WHERE t_start < ?", (cutoff,))
            self._conn.commit()
        for (path_str,) in old_snapshots:
            try:
                Path(path_str).unlink(missing_ok=True)
            except OSError:
                pass

        # Enforce the snapshot size cap, oldest first.
        if snapshots_root.exists():
            files = sorted(
                (p for p in snapshots_root.rglob("*.jpg")),
                key=lambda p: p.stat().st_mtime,
            )
            total = sum(p.stat().st_size for p in files)
            budget = max_snapshot_mb * 1024 * 1024
            for path in files:
                if total <= budget:
                    break
                try:
                    size = path.stat().st_size
                    path.unlink()
                    total -= size
                except OSError:
                    break
