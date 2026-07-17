# Smart Monitoring: data formats and behavior

Plain-language reference for what the monitoring system stores, where, and
what an "event" actually is. See `docs/architecture.md` for the code-level
data-flow summary.

## Where everything lives (all on the Mac, under the repo)

```
data/monitoring/
  monitoring.sqlite3          # entities, tracks, events, digests (see below)
  monitoring.sqlite3-wal      # SQLite WAL file (normal, not a bug)
  scenes/
    <scene_id>.json           # one file per scene: name, notes, frozen anchors
  <scene_id>/
    snapshots/
      <event_id>_<unix_ms>.jpg   # bbox crop saved for some event types (see below)
```

**`apfel --serve` and `ollama serve` are tied to the monitoring session**:
`POST /api/monitor/start` (the panel's Start monitoring button) launches
them, `stop` terminates them (`monitoring.agent.auto_start` /
`monitoring.vision.auto_start`, default true). You never launch them by hand
— and they don't linger when monitoring is off. Key distinction that makes
this safe on 8 GB: what gets auto-started are the tiny **service** processes
(apfel idles at ~8 MB, ollama idle is similarly small) — the heavy
**models** are never held resident: gemma (6.1 GB) loads only for the
duration of a caption call (`keep_alive: "0s"`), and apfel's model is
managed by macOS itself. If a service is already listening on its port (you
started it yourself), monitoring uses it, leaves it alone, and won't
terminate it on stop — it only stops copies it spawned. Logs:
`data/monitoring/apfel.log`, `.../ollama.log`.

One timing note: the first vision caption can fire seconds after monitoring
starts, while `ollama serve` is still booting — that one caption is skipped
(the client backs off 60s) and the next attempt works. Digests are
unaffected (first call comes 5 minutes in).

`data/monitoring/` is gitignored — it never leaves your Mac.

Check what's actually in the database at any time with:

```bash
sqlite3 -header -column data/monitoring/monitoring.sqlite3 \
  "SELECT id, type, subject_label, object_label, datetime(t_start,'unixepoch','localtime') AS start, \
          datetime(t_end,'unixepoch','localtime') AS end FROM events ORDER BY t_start DESC LIMIT 20;"
```

## Scenes and anchors

A **scene** is a named camera placement (`data/monitoring/scenes/<id>.json`):
`scene_id`, `name`, `notes`, `anchors` (list), `created_at`.

An **anchor** is a piece of furniture whose position you've explicitly told
the system to remember — `couch_1`, `bed_1`, etc. Anchors are **not**
detected automatically frame-by-frame; they only change when you click
**Freeze anchors** in the panel, which snapshots whatever furniture YOLO
currently sees (classes: couch, bed, chair, dining table, tv) into the scene
file. This is deliberate: a live furniture bbox flickers or shrinks the
moment someone sits on it or walks past it — exactly when you'd want the
anchor to stay put. Freeze anchors **once, with the room empty** (before or
during monitoring — the running session picks up new anchors immediately).
Without frozen anchors, `on_furniture` events (e.g. "sitting on the couch")
never fire; `entered`/`exited`/`stationary`/`moving` work regardless.

## What a "track" is

A track is one continuously-observed physical object. There's no hand-picked
allowlist of trackable classes — **every** class YOLO tags gets tracked
(person, cat, laptop, backpack, whatever it detects) **except** the frozen
furniture/anchor classes (`monitoring.anchor_classes`: couch, bed, chair,
dining table, tv by default), which are handled as fixed reference points
instead (see above) rather than as moving subjects. This means a moved
object — a laptop that ends up closer to the couch, say — is tracked and can
trigger `moving`/`stationary`/`on_furniture` exactly like a person would, with
no extra configuration. A track goes through states: `tentative` (just
detected, unconfirmed) → `confirmed` (seen 3 times in a row) → `lost`
(missing up to 2s, e.g. brief occlusion) → `ended` (missing more than 10s, or
monitoring stopped). A single flickered YOLO detection that never repeats is
dropped silently and never becomes a track — this is what keeps tracking
every class from turning into noise: a one-off misdetection of some random
object never survives to `confirmed`, let alone generates an event.

## What an "event" is

An event is a semantic, timestamped fact **derived** from tracks + anchors —
never a raw frame. Four types ship today:

| type | when it fires | fields worth knowing |
|---|---|---|
| `entered` | a track becomes `confirmed` | point-in-time |
| `exited` | a track becomes `ended` | `details.visible_s` — how long it was around |
| `stationary` / `moving` | the track's on-screen position has(n't) moved over the last 5s | opens/closes as motion state changes |
| `on_furniture` | a person/cat/dog's box overlaps a **frozen anchor** by ≥30% of its own area, AND their depth estimates agree within 0.7m | `details.posture` — `sitting`/`lying`/`unclear` from the box's height÷width ratio, decided by majority vote over the whole event |

Events use **hysteresis**: a condition must hold true for 2 seconds before an
event opens, and must hold false for 3 seconds before it closes. This is why
a person shifting on the couch for a moment doesn't create ten separate
"sat down" events — brief flicker in either direction is absorbed.

**Honest current limitation**: only `person`/`cat`/`dog` are tracked, so an
object like a laptop moving around the scene produces zero events today —
see the note on generalizing this below.

## Snapshots (bbox crops)

Only two event types save an image crop by default (`monitoring.snapshot_events`
config): `entered` and `on_furniture`. The crop comes from the bbox + 15%
margin, JPEG quality 80, saved to `data/monitoring/<scene>/snapshots/`.
**Important**: because the Pi's Hailo YOLO already draws boxes/labels onto
the frame before sending it (that's the `yolo` view you see live), the
snapshot crop shows that same annotated image, not a clean photo — you'll
see a thin green box and a confidence label baked into the crop. Fetch one
via `GET /monitor/snapshot.jpg?id=<event_id>` (also linked as a 📷 icon next
to events with a snapshot in the panel feed).

## Retention

Once an hour, the controller deletes events/tracks/digests older than
`monitoring.retention_days` (default 14) and their snapshot files, then trims
remaining snapshots oldest-first if they exceed `monitoring.max_snapshot_mb`
(default 200 MB). Entities (see below) are never deleted by retention — they
are your identity memory and are tiny.

## Persistent identities ("Person#1", "Cat#1")

This is **not** a neural network and does **not** run on the Pi — it's plain
Python/OpenCV code running in-process on the Mac, inside the same monitoring
controller thread. When a track is confirmed, it computes an HSV color
histogram of the bbox crop (`monitoring/signatures.py`) and compares it
against recently-seen entities of the same class in this scene (histogram
intersection + size + recency, weighted). A match is only accepted if it
clearly beats the runner-up — on any real doubt, a **new** entity is created
rather than merging two different people. This means:

- The same person leaving and returning in the same clothes is correctly
  re-recognized as `Person#1`.
- A clothing change (different shirt, jacket on/off) will very likely create
  `Person#2` — the system has no way to know it's the same person from color
  alone. This is a known, deliberate trade-off (see the panel/code comments).
- Query current entities: `GET /api/monitor/entities`.

## Vision captioning (optional, Gemma 4 E4B via Ollama) — DISABLED by default

**`monitoring.vision.enabled` is `false` by default**: on the 8 GB M2, the
6 GB model load froze the entire machine for its duration — unusable in
practice, even with all the guards below. The feature is fully implemented
and verified; flip the flag to re-enable when there's more RAM or with a
lighter model (`gemma4:e2b-it-qat`, 4.3 GB, untested). Everything below
describes how it behaves when enabled.

Separately from apfel (text-only), `monitoring.vision` can attach a one-sentence
natural-language caption to events, generated by a local multimodal model
(`gemma4:e4b-it-qat`, ~6.1 GB, via Ollama) looking at an actual crop of the
frame — not just the geometric rule that fired. Captioning is deliberately
**sparse and asynchronous**: it only runs on event **open** for the types
listed in `monitoring.vision.caption_events` (default: `["on_furniture"]`),
in a background thread that never blocks the tracker tick loop, and the crop
is the union of the subject's and anchor's boxes with a margin — so the
caption describes the actual relationship ("a cat is lying on the couch"),
not just an isolated bbox.

**Latency is real and by design**: `monitoring.vision.keep_alive: "0s"`
unloads the 6.1 GB model from memory immediately after every call, trading
speed for not permanently occupying a third of an 8 GB Mac's RAM. In
practice this means each caption costs **~40-50 seconds** (mostly loading
the model from disk), so a caption typically appears in the event feed well
after the event itself opened — never assume it's instant. `timeout_s` is
120s (2 minutes) to give a cold model load real headroom. If you have RAM
to spare, raising `keep_alive` (e.g. `"5m"`) trades memory for much faster
repeat captions.

**Only one caption call runs at a time.** If several captionable events open
close together, the extra requests are dropped (logged, not queued) rather
than firing overlapping 6 GB model loads — on an 8 GB Mac, two concurrent
loads would thrash memory and could hang the machine, which is exactly what
this guards against.

**Known gap**: if an event closes before its caption finishes (rare — only
for very short-lived events), the caption is still written to the database
but the already-rendered panel feed won't pick it up without a page reload,
since the feed only re-polls events that are still open or newer than its
last-seen cursor.

Ollama's default port (11434) is what apfel used to occupy too; the server
defaults apfel to port 11500 instead so both can run side by side without
you having to reconfigure anything.

## The apfel agent (digests + summaries)

Both services start automatically with `POST /api/monitor/start` and stop
with monitoring (see the top of this document) — no manual launching, and
nothing AI-related runs while monitoring is off.

Every 5 minutes (`monitoring.agent.digest_interval_s`) with events in the
window, the controller asks apfel (on-device, 4096-token context, text-only)
to turn the compact event log into 1-3 plain sentences, stored in the
`digests` table. Clicking **Summarize** in the panel asks apfel to merge the
relevant digests into a paragraph (`GET /api/monitor/summary?hours=H`); if
the digest history is too large for one call, it merges in two passes
(chunk → merge → merge-of-merges). If apfel isn't reachable, monitoring
keeps working exactly as before — you just see the raw digests instead of a
merged paragraph, and the panel shows "agent down".

## SQL schema (for anyone who wants to query directly)

```sql
entities(id, scene_id, class, label, signature, first_seen, last_seen, total_visible_s)
tracks(id, scene_id, entity_id, class, t_start, t_end, frames, meta)
events(id, scene_id, type, track_id, entity_id, subject_label, object_label,
       t_start, t_end, details, snapshot_path)   -- t_end IS NULL while open
digests(id, scene_id, t_start, t_end, text, model, created_at)
```

`details` and `signature` are JSON text columns — use `json_extract` or just
`json.loads()` in Python to read them.
