"""Scene knowledge graph: durable nodes + episodic edges over SQLite.

Nodes are the scene's long-term memory — one per anchor (couch_1) and one
per entity identity (person, laptop; a concurrent second one becomes
person_2). Node labels are stable across sessions: a returning object binds
to the existing free node of its class, so the node's edge history is its
life story in this scene ("where has the laptop been?").

Edges are episodes produced by the relation engine and the presence layer:
entered/exited (unary points), on/near (durational, t_end NULL while open),
moved (point). The digest, the feed and Q&A all read edges — the old
events/entities tables are legacy.

SceneGraph itself is thin: node binding/acquisition plus a serializable
"now" view. All heavy lifting (hysteresis, exit rules) happens upstream.
"""

from __future__ import annotations

import logging
from typing import Any

from mac_server.monitoring.events import ActiveRelation, RelationTransition
from mac_server.monitoring.presence import PresenceEvent, PresentEntity
from mac_server.monitoring.store import MonitoringStore


logger = logging.getLogger(__name__)


class SceneGraph:
    def __init__(self, store: MonitoringStore, scene_id: str) -> None:
        self.store = store
        self.scene_id = scene_id
        self._anchor_nodes: dict[str, int] = {}  # anchor_id -> node_id
        self._entity_nodes: dict[int, int] = {}  # entity_key -> node_id
        # Any edge left open by a previous crash is stale by definition.
        self.store.close_stale_open_edges(scene_id, t_end=0.0)

    # -------------------------------------------------------------- nodes

    def ensure_anchor_nodes(self, anchors: list[dict[str, Any]], now: float) -> None:
        existing = {
            n["label"]: n["id"] for n in self.store.nodes_for_scene(self.scene_id, "anchor")
        }
        for anchor in anchors:
            anchor_id = str(anchor["anchor_id"])
            if anchor_id in self._anchor_nodes:
                continue
            if anchor_id in existing:
                self._anchor_nodes[anchor_id] = existing[anchor_id]
            else:
                self._anchor_nodes[anchor_id] = self.store.insert_node(
                    self.scene_id, "anchor", str(anchor.get("class", "?")), anchor_id, now
                )

    def bind_entity(self, entity: PresentEntity, now: float) -> int:
        """Bind a newly-entered entity to a durable node: reuse the free
        node of its class with the plainest label, else mint a new one
        (`person`, then `person_2`, ...)."""
        nodes = [
            n
            for n in self.store.nodes_for_scene(self.scene_id, "entity")
            if n["class"] == entity.class_name
        ]
        bound = set(self._entity_nodes.values())
        free = sorted(
            (n for n in nodes if n["id"] not in bound),
            key=lambda n: (len(n["label"]), n["label"]),
        )
        if free:
            node_id = free[0]["id"]
            self.store.touch_node(node_id, last_seen=now)
        else:
            label = (
                entity.class_name
                if not nodes
                else f"{entity.class_name}_{len(nodes) + 1}"
            )
            node_id = self.store.insert_node(
                self.scene_id, "entity", entity.class_name, label, now,
                embedding=entity.signature,
            )
        self._entity_nodes[entity.entity_key] = node_id
        entity.node_db_id = node_id
        return node_id

    def release_entity(self, entity: PresentEntity, now: float, visible_s: float) -> None:
        node_id = self._entity_nodes.pop(entity.entity_key, None)
        if node_id is not None:
            self.store.touch_node(
                node_id, last_seen=now, visible_s=visible_s, embedding=entity.signature
            )

    def node_of(self, entity_key: int) -> int | None:
        return self._entity_nodes.get(entity_key)

    def label_of_node(self, node_id: int) -> str | None:
        for node in self.store.nodes_for_scene(self.scene_id):
            if node["id"] == node_id:
                return node["label"]
        return None

    # -------------------------------------------------------------- edges

    def record_presence_event(self, event: PresenceEvent, node_id: int) -> int:
        return self.store.insert_edge(
            self.scene_id,
            src_node=node_id,
            dst_node=None,
            relation=event.event_type,
            t_start=event.t,
            t_end=event.t,
            details=event.details,
        )

    def record_transition(self, transition: RelationTransition) -> int | None:
        """Persist a relation transition; returns the edge row id for
        `open`/`point` (stashed on the ActiveRelation for later close)."""
        src = self._entity_nodes.get(transition.subject_key)
        if src is None:
            logger.debug("Transition for unbound entity %s dropped", transition.subject_key)
            return None
        dst: int | None = None
        if transition.object_key is not None:
            dst = self._entity_nodes.get(transition.object_key)
        elif transition.object_label is not None:
            dst = self._anchor_nodes.get(str(transition.object_label))

        if transition.action == "open":
            edge_id = self.store.insert_edge(
                self.scene_id, src, dst, transition.relation,
                transition.t_start, None, transition.details,
            )
            if transition.active is not None:
                transition.active.db_id = edge_id
            return edge_id
        if transition.action == "close":
            active: ActiveRelation | None = transition.active
            if active is not None and active.db_id is not None:
                self.store.close_edge(active.db_id, transition.t_end or transition.t_start, transition.details)
                return active.db_id
            return None
        # point
        return self.store.insert_edge(
            self.scene_id, src, dst, transition.relation,
            transition.t_start, transition.t_end, transition.details,
        )

    # ---------------------------------------------------------------- now

    def now_state(
        self,
        entities: list[PresentEntity],
        labels: dict[int, str],
        open_relations: list[ActiveRelation],
    ) -> list[dict[str, Any]]:
        """Live per-entity view: where is everything right now."""
        rel_by_subject: dict[int, list[str]] = {}
        for rel in open_relations:
            phrase = rel.relation
            if rel.object_label is not None:
                phrase = f"{rel.relation} {rel.object_label}"
            elif rel.object_key is not None:
                phrase = f"{rel.relation} {labels.get(rel.object_key, '?')}"
            rel_by_subject.setdefault(rel.subject_key, []).append(phrase)
            # near is symmetric — show it from both ends.
            if rel.relation == "near" and rel.object_key is not None:
                rel_by_subject.setdefault(rel.object_key, []).append(
                    f"near {labels.get(rel.subject_key, '?')}"
                )

        state = []
        for entity in entities:
            state.append(
                {
                    "label": labels.get(entity.entity_key, entity.class_name),
                    "class": entity.class_name,
                    "state": entity.state,
                    "relations": rel_by_subject.get(entity.entity_key, []),
                    "depth_m": round(entity.depth_m, 2) if entity.depth_m is not None else None,
                    "last_seen": entity.last_seen,
                }
            )
        return state

    def to_payload(self, t_start: float, t_end: float, limit: int = 500) -> dict[str, Any]:
        """Serializable graph slice for GET /api/graph."""
        return {
            "scene_id": self.scene_id,
            "nodes": [
                {k: v for k, v in node.items() if k != "embedding"}
                for node in self.store.nodes_for_scene(self.scene_id)
            ],
            "edges": self.store.edges_between(self.scene_id, t_start, t_end, limit=limit),
        }
