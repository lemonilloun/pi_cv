# Smart Monitoring v2: data formats and behavior

Plain-language reference for what the monitoring system stores, where, and
what the scene graph actually contains. See `docs/architecture.md` for the
code-level data-flow summary.

## The v2 pipeline in one paragraph

Detections (yolo26n) flow tracker → **presence** → **relations** → **scene
graph**. The tracker does frame-to-frame association and dies fast; the
presence layer is what the user sees: a class-keyed entity ("person",
"laptop" — no more Person#20) that survives detection flicker and only
*leaves* when it plausibly walked out of the frame. Relations (`on`, `near`,
`moved`) become **edges of a per-scene knowledge graph** stored in SQLite;
the apfel agent digests edges, flags key moments (with snapshots), and
answers free-form questions over the graph.

## Where everything lives (all on the Mac, under the repo)

```
data/monitoring/
  monitoring.sqlite3          # graph_nodes, graph_edges, digests (see below)
  monitoring.sqlite3-wal      # SQLite WAL file (normal, not a bug)
  scenes/
    <scene_id>.json           # one file per scene: name, notes, frozen anchors
  <scene_id>/
    snapshots/
      <unix_ms>.jpg           # bbox crops for some edges (see below)
```

**`apfel --serve` (and `ollama serve`, when vision is enabled) are tied to
the monitoring session**: `POST /api/monitor/start` launches them, `stop`
terminates them (`monitoring.agent.auto_start` / `monitoring.vision
.auto_start`, default true). A service already listening on its port (you
started it yourself) is used as-is and never touched. Logs:
`data/monitoring/apfel.log`, `.../ollama.log`.

`data/monitoring/` is gitignored — it never leaves your Mac.

Check the graph at any time:

```bash
sqlite3 -header -column data/monitoring/monitoring.sqlite3 \
  "SELECT e.id, e.relation, s.label AS subj, d.label AS obj, \
          datetime(e.t_start,'unixepoch','localtime') AS start, e.key_moment \
   FROM graph_edges e JOIN graph_nodes s ON s.id=e.src_node \
   LEFT JOIN graph_nodes d ON d.id=e.dst_node \
   ORDER BY e.t_start DESC LIMIT 20;"
```

## Scenes and anchors

A **scene** is a named camera placement (`data/monitoring/scenes/<id>.json`).
An **anchor** is furniture whose position you've explicitly frozen (couch_1,
bed_1) via **Freeze anchors** in the panel — click it once **with the room
empty**. Live furniture bboxes flicker under occlusion exactly when you need
them stable, hence the explicit freeze. Without anchors, `on`/`near
furniture` relations never fire; entered/left/moved work regardless.

## Entities: no numbering, presence over flicker

- The only entity of a class is just **`person`** / **`laptop`**. Suffixes
  (`person_2`) appear only while two same-class objects are in view at
  once, and are dropped again after.
- **Detection dropout ≠ departure.** A track dying mid-frame makes the
  entity `occluded`, not gone; when a same-class track confirms again it
  re-binds to that entity (embedding + position matched) with **no**
  entered/exited events. This kills the "appeared/disappeared 5× in 2
  minutes" spam by construction.
- An entity **leaves** only when its last bbox was near a frame border
  (`presence.exit_edge_frac`, default 10%) AND it has been unseen for
  `presence.exit_absent_s` (default 25 s). Fallback: unseen anywhere for
  `presence.presence_timeout_s` (default 5 min) ⇒ left.
- Appearance matching uses **CLIP embeddings** (MobileCLIP via open_clip on
  MPS, `monitoring.embeddings`); if open_clip isn't installed it silently
  falls back to HSV histograms.

## The scene graph

**Nodes** are durable per-scene identities: every anchor and every entity
class (`person`, `laptop`, plus `person_2` when concurrency ever happened).
A returning object binds to its existing node — the node's edge history is
its life story in the scene, which is what "where has the laptop been?"
queries read. Nodes are never deleted by retention.

**Edges** are episodes with time intervals (`t_end` NULL while open):

| relation | kind | meaning / details worth knowing |
|---|---|---|
| `entered` / `exited` | point | crossed into / out of the frame; `exited.details.visible_s`, `reason` (left_frame / timeout / session_stop) |
| `on` | durational | overlaps a frozen anchor at agreeing depth; `details.posture` (sitting/lying, majority-voted) |
| `near` | durational | small bbox gap + agreeing depth, entity↔anchor or entity↔entity; `details.distance_m`, `arrangement` (left-of/…) — the feed phrases open as "approached", close as "moved away" |
| `moved` | point | sustained centroid shift > `relations.move_min_frac` of the frame diagonal; `details.from_zone/to_zone`, `nearest_anchor` |

Hysteresis (2 s open / 3 s close) still applies, and an **occluded subject
pauses the close countdown** — someone briefly hidden on the couch keeps one
continuous `on couch_1` edge instead of close/reopen churn. Micro-jitter
never fires `moved`.

Every edge's `details` carries geometry (depth, gap, mutual bbox
arrangement) so the LLM can describe interactions concretely.

## Key moments and snapshots

Each digest cycle (5 min) apfel also picks 0-3 notable edges
(`monitoring.qa.key_moments_per_digest`); they get `key_moment=1`, a ⭐ in
the feed, and a participant snapshot if one can still be taken. Snapshots
are also saved on `entered`/`on` opens (`monitoring.snapshot_events`).
Because the Pi already draws boxes onto yolo frames, crops show the thin
green box — that's expected. Fetch: `GET /monitor/snapshot.jpg?id=<edge_id>`.

## Retention

Hourly sweep: closed edges older than `retention_days` (14) are deleted
with their snapshots; **key-moment photos live only 24 h**
(`qa.key_snapshot_retention_h`) — the ⭐ mark stays; remaining snapshots are
trimmed oldest-first over `max_snapshot_mb` (200). Nodes and digests follow
the old rules (nodes: never).

## The apfel agent: digests, rolling summary, Q&A

Every 5 minutes with edges in the window, apfel writes an aggregate digest
("A person spent 25 minutes sitting on the couch; a laptop sat nearby and
was moved twice") — no numbering, no speculation. A rolling summary is
folded incrementally each cycle as internal context.

The panel's **Ask** box (replacing the old Summarize button) posts to
`POST /api/monitor/ask {question, hours}`: apfel gets the current graph
state ("person: on couch_1 (2.9 m)"), recent edges, digests and the rolling
summary within its 4096-token budget, answers directly ("where is the
laptop?" → last known position relative to anchors), and the response also
lists the window's key moments with snapshot links. If apfel is down,
monitoring keeps logging exactly as before — only the AI layer degrades.

## Vision captioning (Gemma via Ollama) — still DISABLED by default

Unchanged from v1: `monitoring.vision.enabled: false` because the 6 GB
model load froze the 8 GB Mac. The code path now captions graph edges
(`caption_events: ["on"]`) and is otherwise identical. Do not enable
without more RAM or a lighter model.

## API

`GET/POST /api/scenes` · `POST /api/monitor/start|stop` ·
`POST /api/monitor/anchors/freeze` · `GET /api/monitor/status` (live
entities + relations) · `GET /api/monitor/events` (edge feed) ·
`GET /api/monitor/entities` (graph nodes) · `GET /api/graph?hours=H`
(nodes + edges slice) · `POST /api/monitor/ask` ·
`GET /api/monitor/summary` (rolling summary, no LLM call) ·
`GET /monitor/snapshot.jpg?id=`.

## SQL schema

```sql
graph_nodes(id, scene_id, kind /*entity|anchor*/, class, label, embedding,
            first_seen, last_seen, total_visible_s, best_snapshot_path)
graph_edges(id, scene_id, src_node, dst_node, relation, t_start, t_end,
            details, caption, key_moment, snapshot_path)
digests(id, scene_id, t_start, t_end, text, model, created_at)
rolling_summaries(scene_id, text, event_count, updated_at)
-- entities/tracks/events remain from v1 but are no longer written.
```

`details` and `embedding` are JSON text columns — use `json_extract` or
`json.loads()` to read them.
