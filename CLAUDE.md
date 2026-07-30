# pi_cv Project Summary

## Purpose

`pi_cv` is a distributed computer-vision project built around:

- Raspberry Pi 5 as an edge sensor/CV node.
- MacBook as the processing server, storage endpoint, and visualization machine.

The current system already supports TCP communication over Wi-Fi, headless Raspberry Pi usage over SSH, camera capture, MJPEG preview streaming, and early CV experiment packaging for Depth Anything / YOLO workflows.

## Repository Shape

```text
client/   Raspberry Pi client code
server/   MacBook server code
shared/   JSON message schema, config loader, TCP framing
scripts/  Run/setup scripts
config/   Default JSON config
docs/     Architecture notes
data/     Local test data and received artifacts
research/ Local research notes, ignored by Git
models/   Local model weights, ignored by Git
external/ Local cloned model repos, ignored by Git
```

Important ignored paths:

- `.env`
- `research/`
- `models/`
- `external/`
- `data/received/`
- `data/camera_local/`
- `*.pt`, `*.pth`, `*_ncnn_model/`

## Protocol

TCP is used directly, without heavy frameworks.

Packets are length-prefixed:

```text
[4 bytes JSON header length][8 bytes binary payload length][JSON header][binary payload]
```

Main message types:

- `status` - hello/status message.
- `telemetry` - device/platform/python info from Raspberry Pi.
- `image` - static JPEG file transfer.
- `camera_frame` - one captured camera frame.
- `camera_stream_frame` - one frame in MJPEG-like stream mode; session frames carry `view` (camera/depth/yolo/combined), `mode`, and `inference_ms`.
- `cv_result` - zip package containing CV experiment artifacts and metadata.

Persistent session types (used by `scripts/run_pi_session.sh`):

- `session_hello` - client→server registration; only this makes a connection command-capable.
- `command` - server→client push (`set_mode`, `ping`), sent from the web panel via `POST /api/mode`.
- `command_result` - client→server outcome of a command.
- `system_telemetry` - 1 Hz CPU/temperature/memory sample from `/proc` and `/sys`.

## Configuration

Default config:

```text
config/default.json
```

Local overrides:

```text
.env
.env.example
```

Typical Raspberry Pi `.env`:

```env
PI_CV_SERVER_HOST=192.168.1.114
PI_CV_SERVER_PORT=8765
PI_CV_DEVICE_ID=raspberry_pi_01
```

Server listens on TCP `8765` by default. HTTP MJPEG preview runs on `127.0.0.1:8080`.

## Core Commands

Start MacBook server:

```bash
cd /Users/lehacho/Desktop/works/cv_pojects/pi_cv
./scripts/run_server.sh
```

Run basic Raspberry Pi client:

```bash
cd ~/pi_cv
source .venv/bin/activate
./scripts/run_client.sh --telemetry
```

Send test image:

```bash
./scripts/run_client.sh --telemetry --image data/cat.jpg
```

One-shot camera capture:

```bash
./scripts/run_client.sh --telemetry --camera-shot
```

Start camera stream from Raspberry Pi:

```bash
./scripts/run_client.sh --telemetry --camera-stream
```

Open stream preview on MacBook:

```text
http://127.0.0.1:8080/
```

## Persistent Session + Web Control Panel

One long-running client process on the Raspberry Pi, controlled from the web
panel at `http://127.0.0.1:8080/` (mode buttons + live Pi monitoring with
per-core CPU, temperature, and memory charts).

Start on the Raspberry Pi (server must be running on the MacBook):

```bash
cd ~/pi_cv
source .venv/bin/activate
./scripts/run_pi_session.sh
```

Modes (switched from the web panel, no restarts needed):

- `idle` - connected, telemetry only, camera off.
- `stream` - plain camera MJPEG at ~30 fps.
- `depth` - 1 fps depth heatmap (Depth Anything V2, model stays warm).
- `yolo` - 1 fps YOLO annotated frames (NCNN, model stays warm).
- `pipeline` - YOLO + depth combined frame (heavy on CPU-only; wait for AI HAT).

Known issue: `yolo26n-seg_ncnn_model` (segmentation) crashes ultralytics'
segment postprocess with `IndexError: index 1 is out of bounds for dimension 0
with size 1`. Root cause: the NCNN export only produced one output blob
instead of the two segmentation needs (detections + mask protos), so
`AutoBackend.forward()` collapses to a single tensor and `preds[1]` fails in
`ultralytics/models/yolo/segment/predict.py`. Until the seg export is fixed
(re-export with a newer ultralytics/pnnx, or file an upstream issue), session
default is `models/yolo26n_ncnn_model` with `yolo_task: "detect"` (no masks,
bounding boxes only — sufficient for the depth-per-object use case). The
client already recovers gracefully from CV inference errors (falls back to
`idle`, no crash) if this resurfaces.

Defaults (model paths, fps, mode) live in the `session` section of
`config/default.json` and can be overridden with `PI_CV_SESSION_*` env vars or
CLI flags. The client reconnects automatically with backoff and restores the
last requested mode; loaded models survive mode switches and reconnects.

### YOLO backend: ncnn yolo26n default, Hailo optional

**Session default is `yolo_backend: "ncnn"` + `models/yolo26n_ncnn_model`**
— the Model Zoo `yolov8s` hef detected too poorly in real use, and detection
quality matters more than fps for event monitoring. `yolo` mode ticks at
`session.yolo_fps` (25) as a best-effort cap; on the Pi CPU the real rate is
~1-3 fps.

Hailo (AI HAT+, `--yolo-backend hailo`) remains available with two head
formats selected by `session.hailo_arch`:

- `"yolov8"` — Model Zoo hefs with on-chip NMS via `picamera2.devices.Hailo`
  (`WarmHailoYolo`), e.g. `/usr/share/hailo-models/yolov8s_h8l.hef`.
- `"yolo26"` — a custom-compiled yolo26 hef (raw outputs, host-side decode
  via the `external/yolo26_hailo` project; `WarmHailoYolo26`). Compile it
  with `scripts/compile_yolo26_hef/` (needs x86_64 Linux — Docker
  `--platform linux/amd64` on the M2 or any x86 box; DFC wheel from the
  Hailo Developer Zone) and clone the decoder on the Pi with
  `./scripts/setup_yolo26_hailo.sh`. This is the endgame: yolo26n quality
  at NPU speed (~86 fps on 8L per upstream benchmarks).

depth/pipeline modes still tick at `cv_fps` (CPU-bound).

### Objects as data (not just pixels)

In `yolo` and `pipeline` modes every CV frame header carries an `objects`
list (`class`, `confidence`, `bbox_xyxy`, and in pipeline mode
`depth_median`). The Mac CV worker runs metric depth on those same frames and
attaches per-object `depth_median_m` (median of the bbox's inner 50%);
the enriched list is exposed at `GET /api/status` → `server_cv.objects` for
downstream geometry/mapping on the Mac. The depth heatmap view is just
debug visualization — the data path never needs it.

HTTP API used by the panel: `GET /api/status`, `GET /api/telemetry?seconds=120`,
`POST /api/mode` with `{"mode": "depth"}` (409 when no session client is
connected).

## Server-Side CV (Metric Depth at ~15 fps) + Room Mapping

The Mac server can run Depth Anything V2 **metric** (indoor, meters) on MPS
from the incoming 30 fps camera stream — this is how depth reaches 13-17 fps
(Pi CPU stays at ~1 fps for headless use). One-time Mac setup:

```bash
python3 -m pip install -r server/requirements-server-cv.txt   # torch/cv2/numpy
./scripts/setup_server_cv.sh   # clones DA2 repo, downloads metric checkpoint, verifies MPS
```

Config sections: `server_cv` (device mps/cpu, input size 392, target_fps) and
`mapping` (grid resolution, height band, camera FOV/height defaults) in
`config/default.json`. `--no-cv` server flag disables everything CV; the TCP
core stays stdlib-only.

Panel views (pills on the video card): `Pi` (whatever the Pi sends), `Depth`
(server-computed heatmap, fixed 0-8 m color scale), `Map` (live room map),
`Radar` (live top-down object scatter, see below).
Streams: `/stream.mjpg?view=pi|depth|map`, `/latest.jpg?view=...`.

Room mapping (no IMU — constrained-motion convention): camera faces one of
north/east/south/west and moves only forward/backward along the viewing axis;
ego-position is anchored per frame by the metric distance to the facing wall
(median of central patch, EMA + jump rejection), so drift never accumulates.
Scanning from all 4 sides auto-estimates room dimensions. Camera default
height 0.3 m (future cart platform), configurable per room.

**Known limitation**: real-room testing showed the wall-scan map is not
accurate enough to be useful yet — monocular metric depth has real-world
scale error, and the anchor-to-wall convention has no way to correct for it
without an external reference. Room mapping is paused pending either a
TF-Luna (or similar) ToF sensor for calibration/ground truth, or further
tuning of `mapping.depth_scale_correction`. The Radar view (below) was built
specifically to not depend on this — it's a single-frame snapshot, so it has
no ego-motion or drift to get wrong.

Mapping API: `GET/POST /api/rooms`, `POST /api/scan/start`
`{room_id, direction}` (auto-switches the Pi to stream), `POST /api/scan/stop`,
`GET /api/scan/status`. Scan grids and the fused map persist in
`data/rooms/<room_id>/` (`scan_<dir>.npz`, `map.png`, `map_grid.npz`,
`state.json`). Mapping math lives in `server/src/mac_server/mapping/`
(geometry.py is pure numpy, unit-tested).

Deferred (after room mapping is trustworthy again): YOLO object landmarks on
the map (ultralytics on the Mac).

### Live Radar (spatial awareness without scanning)

In `yolo`/`pipeline` modes, every detected object's `server_cv.objects` entry
(see "Objects as data" above) additionally carries `bearing_deg` and
Cartesian camera-frame `lateral_m`/`forward_m` (`server/src/mac_server/
mapping/geometry.py`: `bbox_bearing_deg`, `bbox_camera_xy`). **Important**:
`lateral_m` is `tan(bearing) * forward_m`, not `sin(bearing) * forward_m` —
`forward_m` is Z-depth (distance along the optical axis), and a polar
(bearing, depth) plot is a materially different, wrong point except at
bearing=0 (they diverge ~35%+ at this camera's wide 102° FOV edges). The
radar view and any future consumer of these fields must use the Cartesian
pair, not re-derive a polar plot from `bearing_deg` + `depth_median_m`.

`GET /api/objects` (separate from `/api/status`, polled by the panel at
~5 Hz only while the Radar view is open) returns the latest enriched object
list. The panel renders it as a top-down scatter: camera at bottom, forward
up, FOV wedge bounded by `server_cv.hfov_deg`, range rings out to
`server_cv.display_max_m`, points colored by class, short fading trail,
hover tooltip. No scanning motion, no wall anchor, no drift — a live
snapshot of what's around the camera right now.

### Smart Monitoring v2 (fixed scene → presence → scene graph → Q&A)

For a fixed camera placement, the Mac turns detections into a per-scene
**knowledge graph** instead of pixels. Pipeline: tracker → presence →
relations → graph. Package `server/src/mac_server/monitoring/` (pure-logic
modules unit-tested in `server/tests/`, 134 tests):

- `tracker.py` — greedy IoU+centroid tracker (unchanged from v1),
  tentative→confirmed(3 hits)→lost(2s)→ended(10s); single-tick tentatives
  dropped. Tracks every class except `monitoring.anchor_classes` (no
  allowlist). Tracks are short-lived plumbing — identity lives above.
- `presence.py` — **the v2 core**: class-keyed entities with NO permanent
  numbering ("person", "laptop"; `person_2` only while ≥2 same-class objects
  are simultaneously in view). Track death mid-frame = `occluded`, not gone;
  a re-confirming same-class track re-binds (CLIP embedding + position) with
  no entered/exited events — appearance/disappearance flicker produces zero
  events by construction. An entity `left`s only when its last bbox was near
  a frame border (exit_edge_frac 0.10) AND absent ≥ exit_absent_s (25 s);
  fallback presence_timeout_s (5 min).
- `events.py` — `RelationEngine`: hysteresis (2s open / 3s close) graph
  edges instead of the old event zoo. `on(entity, anchor)` (ex-on_furniture,
  frozen anchors + posture vote), `near(A,B)` (bbox gap + depth agree;
  entity↔anchor AND entity↔entity; feed phrases open/close as
  "approached"/"moved away"), `moved` (point, sustained ≥15% frame-diag
  displacement — micro-movements never fire). stationary/moving are GONE.
  An occluded subject PAUSES close countdowns (no close/reopen churn).
  Edge details carry geometry (depth, distance_m, left-of/overlapping) for
  the LLM.
- `graph.py` — `SceneGraph`: durable `graph_nodes` (anchors + entity
  classes; a returning laptop binds to the same node — its edge history is
  its life story in the scene) + episodic `graph_edges` (t_end NULL while
  open). `GET /api/graph?hours=H` serves the slice; stale open edges are
  closed on session start.
- `embeddings.py` — `ClipSignatureProvider` (MobileCLIP-S1 via
  open_clip_torch on MPS, in requirements-server-cv.txt), cosine similarity;
  drives presence re-binding and node embeddings. Falls back to the HSV
  `signatures.py` provider when open_clip is missing.
- `store.py` — SQLite, same threading contract as v1 (shared conn +
  `check_same_thread=False` + lock; HTTP reads via `read_connection()`).
  New tables graph_nodes/graph_edges; v1 tables (entities/tracks/events)
  remain but are no longer written. Retention sweep hourly: closed edges >
  14 days, key-moment photos > 24 h (mark stays), 200 MB snapshot budget;
  graph nodes never deleted.
- `agent.py` — apfel (Apple Intelligence, 4096 tokens, text-only, port
  11500): aggregate digests without numbering ("A person spent 25 minutes
  on the couch; a laptop sat nearby"), per-cycle **key-moment picking**
  (0-3 notable edges get ⭐ + participant snapshots), incremental rolling
  summary (internal context), and **`ask()` — free-form Q&A** over current
  graph state + edges + digests (`POST /api/monitor/ask {question, hours}`).
  The one-button Summarize is gone from the UI. Degrades gracefully;
  **service lifecycle = the monitoring session** (start spawns
  `apfel --serve`, stop kills only copies it spawned).
- `controller.py` — per-session thread consuming `cv_worker.objects_store`;
  Mac wall clock, `pi_frame_index` dedup, `stalled` flag; mutually exclusive
  with room scans; auto-switches the Pi to `yolo`.
- `vision.py` — optional Gemma captioning, still **DISABLED by default**
  (6 GB load froze the 8 GB Mac); now captions graph edges
  (`caption_events: ["on"]`), otherwise unchanged.

API: `GET/POST /api/scenes`, `POST /api/monitor/start|stop`,
`POST /api/monitor/anchors/freeze`, `GET /api/monitor/status|events|entities`,
`GET /api/graph?hours=H`, `POST /api/monitor/ask`,
`GET /api/monitor/summary` (rolling text only), `GET /monitor/snapshot.jpg?id=`.
Panel: "Monitoring" card — scene select, start/stop, freeze anchors, a
"Now" line (current relations per entity), edge feed in plain phrases
("laptop approached couch_1"), and an **Ask** box with key moments +
snapshots under the answer. Config: `monitoring` section
(`presence`/`relations`/`embeddings`/`qa` subsections) in
`config/default.json`, `PI_CV_MONITORING_*` env overrides. Full
data-format walkthrough: `docs/monitoring.md`.

### Scene3D (semantic room reconstruction — the current main project)

One slow walk with the Pi camera → metric room mesh + 3D objects with
open-vocab labels + scene graph. Full walkthrough: `docs/scene3d.md`.

- **Pi records** (`scripts/run_scene_recorder.sh`, ~2-3 fps): yolov8s_seg
  hef + CLIP RN50x4 hef share the Hailo-8 via the HailoRT scheduler
  (`pi_client/hailo_infer.py`); seg raw outputs decoded on host by the
  pure-numpy port in `pi_client/seg_postprocess.py` (from MIT hailo-apps;
  no scipy/cython needed). Keyframes are **streamed live to the Mac** over
  the project TCP (scene_session_start/scene_keyframe/scene_session_end
  messages; server materializes `data/scene_sessions/<id>/` per the
  keyframes contract: rgb.jpg + uint16 masks.png + meta.json with track_id
  and 640-d clip_emb per detection) — the microSD stays clean; local disk
  is only the Wi-Fi-drop fallback (sync those keyframes with `git pull` on
  the current branch — no manual rsync/scp).
  An annotated live preview also streams to the panel's Live tab
  (mode `scene_rec`). `--offline` restores pure local recording.
  Calibrate once with `scripts/run_scene_calibrate.sh` (fixed
  LensPosition!).
- **Mac reconstructs** (`server/src/mac_server/scene3d/`, steps
  depth→poses→tsdf→objects→graph): DAv2 metric on MPS; pycolmap SfM
  (`poses.matching_mode`: `auto` uses exhaustive matching ≤600 frames so
  loops close — the biggest quality lever; splits into >1 sub-reconstruction
  are surfaced/warned, not silently dropped) + metric scale from
  median(DAv2/COLMAP-sparse) with MAD rejection; Open3D TSDF
  (`room_mesh.ply`, `floor_plan.png`); DINOv2
  association into an object bank; CLIP text RN50x4/openai (MUST match the
  Pi hef) for open-vocab labels + near/on OBB edges (`scene_graph.json`).
- **Intrinsics are one artifact with the poses.** When a session has no
  calibration, COLMAP refines the focal length, so the poses only mean
  something relative to the *refined* camera. `poses_step` saves it to
  `derived/intrinsics_refined.json` + inside `poses.json`; tsdf/objects read
  it via `SceneSession.active_intrinsics()`. Never call
  `session.intrinsics()` downstream of `poses` — that returns the
  as-recorded guess and silently reconstructs a different camera (measured:
  fx 622 vs 989, a 12×15 m mesh for a 2×4 m walk). `fallback_hfov_deg` is
  **75°** (COLMAP and DA3 independently measured 75.7°/74.4°); the old 102°
  was the Camera Module 3 *Wide* figure and does not match this camera.
- **Depth hygiene is mandatory before fusion** (`depth_filter.py`): flying
  pixels (edge-on surfaces — the radial streaks) + multi-view consistency
  (≥2 neighbouring keyframes must confirm each point). Cached once in
  `derived/depth_filtered/` as float16 and shared by tsdf AND objects, so
  the mesh and the objects are built from the same geometry.
- **The IMU is a metric ruler, not just a plumb line.** Every keyframe
  carries `imu.segment` — motion preintegrated since the previous keyframe
  (`distance_m`, `speed_ms`, `peak_linear_accel_ms2`, `zupts`). Inertial
  position diverges only when uncorrected; over one keyframe interval, with
  gravity known to ~0.2°, the error is millimetres and the next frame resets
  it. `poses_step.imu_scale_samples()` turns that into a **network-independent
  metric scale** (IMU metres / COLMAP chord), reported alongside the DAv2
  scale as `scale_imu`/`scale_depth`; `poses.scale_source` default `auto`
  prefers the IMU when it has ≥20 intervals and a tight spread. Velocity
  carries across keyframes (a steady roll has ~zero acceleration) and is
  pinned by a **visual ZUPT** — the recorder's own inter-frame image shift,
  since the accelerometer cannot tell stopped from rolling steadily.
- **The floor plan is a carved occupancy grid** (`occupancy.py`), not a
  vertex histogram: FREE between camera and surface, OCCUPIED at it, UNKNOWN
  beyond. Gravity comes from a RANSAC floor-plane fit, so a tilted camera no
  longer tilts the plan. `estimate_gravity(imu_up=...)` is the **IMU seam**:
  the Pi may put `gravity` (camera frame) in a keyframe's meta.json,
  `SceneSession.imu_up_vector()` rotates it into world space, and it wins
  over the floor fit. Absent field = old behaviour, no migration.
- **IMU is fitted and driven by `pi_client/imu_rvc.py`**: BNO08x in
  **UART-RVC** (not SHTP) on a CH340 at `/dev/ttyUSB0` 115200, 19-byte
  frames at 100 Hz, checksum `sum(bytes 2..17) & 0xFF`. Gravity is taken
  from the **accelerometer**, not the fused pitch/roll — the two disagree by
  an axis permutation (same ~1.4° tilt, different frame), so the Euler tilt
  angle is used only as a convention-free sanity check.
- **The robot drives in the X-Y plane and never tilts — so gravity in the
  camera frame is a CONSTANT** (yaw turns about the gravity axis, leaving it
  unchanged; driving around yields the same pair forever and cannot
  constrain rotation about gravity). `pi_client/imu_calibrate.py` therefore
  measures that constant once against a chessboard taped **plumb to a
  vertical wall** — not lying on the floor, where a 40 cm-high pitched-up
  camera sees it at a grazing angle and solvePnP is ill-conditioned.
  Modes in `config/imu_calibration.json`: `reference` (the constant + a
  tilt-tolerance check that withholds gravity if the rig leaves that pose)
  and optional `full` (a real rotation, needs the rig propped on a book for
  ~8° of attitude spread). No calibration ⇒ the reader returns None rather
  than a guessed axis mapping. Details: `docs/scene3d.md`.
- **SHTP-over-UART fallback (2026-07-28)**: `docs/imu_shtp_setup.md`'s SPI
  plan is blocked — the AI HAT+ physically occupies the GPIO header no
  header extender available, confirmed hard blocker, not "not done yet."
  `pi_client/imu_shtp_uart.py` (`ShtpUartReader`) is a from-scratch SHTP
  driver over the *existing* UART wiring instead (just PS1/PS0 flipped
  from RVC to SHTP mode, fixed 3 Mbaud) — written after `adafruit_bno08x`
  turned up three real correctness bugs on this lossy link (batch parser
  aborts whole packets on unrecognized report IDs, leaks stale slices
  across batches, returns silently-stale readings). Measured ~50-90 Hz
  effective valid accel+gyro rate, self-healing, every sample carries a
  freshness age — well short of real-time VIO's ≥200 Hz floor, but usable
  for lower-rate consumers. **RVC and SHTP-UART are mutually exclusive on
  one sensor** (mode pins) — check the physical jumpers before assuming
  `imu_rvc.RvcReader` is receiving data. Not yet integrated into
  `scene_recorder.py` (no calibration/motion-integration layer built on
  top yet). Motion-validated live (user physically rotated/lifted/rolled
  the sensor for ~30s): gyro Z (the board's own yaw axis) was consistently
  the most active component during rotation-dominant handling, matching
  the board's silkscreen labeling — axis assignment checks out, not just
  the decode arithmetic. Full account, measurements, and next steps:
  `docs/imu_shtp_uart.md`.
- Measured effect of the above on session_20260724_144728 (poses unchanged,
  99.1% registered): mesh bbox 12.08×4.85×15.08 m → **8.78×2.72×8.84 m**;
  connected components 37 097 → **186** (largest 80.9%); PLY 141 MB → 16 MB;
  depth truncation 5.0 m fixed → **4.19 m** adaptive; floor plan from a
  starburst of rays → walls around 16 001 free cells.
- **Panel `Scene` tab**: sessions table, Run pipeline with per-step
  progress, labeled floor plan, three.js mesh viewer (vendored in
  static/vendor/), objects table. CLI:
  `./scripts/run_scene_pipeline.sh <session_id> [steps] [--force]`.
- Config: `scene3d` section in config/default.json. Deps (Mac, system
  python3 — no venv, matches every other launch script):
  `python3 -m pip install -r server/requirements-server-cv.txt -r server/requirements-scene3d.txt`
  (open3d, pycolmap, pillow, certifi, torchvision; the colmap CLI via
  brew is optional and currently blocked by a qtsvg link conflict —
  pycolmap does everything).
- Physics note: `poses` NEEDS camera motion (parallax); a static-tripod
  session fails with "COLMAP registered nothing" by design.

### Reconstruction quality ceiling on the M2 — migration to a GPU server planned (2026-07-30)

Real room-walk testing this session (two sessions, `session_20260730_103727`
and `session_20260730_105059`, 181 and 267 keyframes) surfaced two concrete,
diagnosed failure modes in the current M2/pycolmap/DAv2 stack — **not** a
"too many keyframes" problem (a prior session with more frames,
`session_20260724_144728`, registered 99.1%; the count itself isn't the
lever):

1. **Incremental SfM can drop a large contiguous chunk of the walk.** In
   `session_20260730_105059`, frames 160-259 (100 frames straight, ~65s of a
   174s walk) never registered into the main reconstruction — confirmed via
   `poses.json`'s `world_from_cam` keys (one contiguous gap, not scattered
   noise). Root cause, confirmed by inspecting the actual keyframes (not
   guessed): the low-cart camera height (~0.3 m) put the camera very close to
   furniture (chair casters, nightstand) for that whole stretch, in front of
   a large repetitive-pattern poster — extreme parallax + ambiguous repeated
   texture, not motion blur (checked: Laplacian-variance sharpness was
   *higher* in the dropped range, 374 vs 350, ruling out blur as the cause).
   Since most of the room's furniture was visible in exactly that dropped
   stretch, this directly explains the resulting floor plan looking nothing
   like the room and only 2 objects surviving to the graph.
2. **Depth-scale vs IMU-scale disagreement of 220%** (`scale_depth: 0.38` vs
   `scale_imu: 1.22`) on the same session, with the IMU-side estimate's own
   internal spread (`IQR/median: 0.935` across the 35 usable intervals)
   too large to trust either — plausibly the same root cause as the
   `imu_shtp_motion.py` ZUPT-starved-drift finding below (zero ZUPT events
   fired across either 110s or 174s walk when the rig never paused), not
   necessarily "the depth network is wrong" as the pipeline's generic
   auto-message assumes.

A follow-up test with denser keyframes (`--keyframe-shift-px 4
--keyframe-min-interval 0.1`, aiming for 300-400 keyframes over a similar
walk) made results *worse across the board*, not better — keyframe density
was never the actual lever for either diagnosed failure mode above, and
increasing it did not address either one.

**Decision:** the M2 (16GB unified memory, MPS backend, no dedicated VRAM)
is being treated as a ceiling for the model weight this reconstruction
problem plausibly needs (heavier SfM/dense-reconstruction transformers —
e.g. DROID-SLAM/DPVO-class dense methods, VGGT/DUSt3R/MASt3R-class
feed-forward 3D transformers — are all far too heavy for MPS at usable
speed/quality). Plan: move the Mac-server role (currently run from this
M2) to a **different laptop** that has SSH access to a GPU host (`ssh
pdfserver` from *that* laptop — 2x A6000, 48GB VRAM each, 96GB total) and
explore heavier reconstruction methods there, likely via a small API/VLM
service on the GPU host that the new laptop's server process calls into.
`plan.md` (repo root) is the original M0-M7 implementation plan this
Scene3D pipeline was built from — reference it before designing the new
GPU-backed reconstruction step so the existing data contract (session
directory layout, `meta.json` schema) doesn't need to change on the Pi
side.

**Pi connection** (unchanged by this migration — the Pi keeps recording the
same way regardless of which laptop is "the server"):
- Host: `cv-pi.local`, user `first`, repo at `~/Desktop/work/pi_cv` (NOT
  `~/pi_cv` — a generic path in this doc's own Core Commands section above
  doesn't match this specific Pi's actual checkout location).
- `~/Desktop/work/pi_cv/.env` on the Pi currently has
  `PI_CV_SERVER_HOST=vedro.local` (this M2's hostname) — **must be updated**
  to the new laptop's hostname/IP once it takes over the server role, or the
  Pi will keep trying to stream to the old machine.
- SSH key auth already works from this M2 (`~/.ssh/id_rsa_morozov`,
  `Host cv-pi.local` in `~/.ssh/config`); the Pi's `~/.ssh/authorized_keys`
  currently has 2 keys. Setting up the new laptop needs its own key added —
  see `docs/new_laptop_setup.md` for the exact steps and
  `scripts/check_lan_ip.sh` for verifying the new laptop's LAN IP before
  updating the Pi's `.env`.
- The GPU host (`pdfserver`) is reachable via SSH **only from the new
  laptop**, not from this Mac or from the Pi — do not attempt to connect to
  it from either.

Loopback test on the MacBook without a camera:

```bash
./scripts/run_server.sh
./scripts/run_pi_session.sh --host 127.0.0.1 --source synthetic --depth-backend synthetic
curl -X POST -H 'Content-Type: application/json' -d '{"mode":"depth"}' http://127.0.0.1:8080/api/mode
```

Monitoring loopback needs detectable objects: ultralytics is installed on the
Mac, so use the ncnn backend with the bundled test image
(`--yolo-backend ncnn --yolo-model yolo11n.pt --initial-mode yolo
--synthetic-image <site-packages>/ultralytics/assets/bus.jpg`). bus.jpg
contains several people — a free check that concurrent same-class entities
get `person_1`/`person_2` suffixes while a lone one is plain `person`.

Note: one-shot camera clients cannot use the camera while the session holds it —
switch the session to `idle` first (idle closes the camera).

## Raspberry Pi Camera Setup

Direct camera checks on Raspberry Pi:

```bash
rpicam-hello --list-cameras
rpicam-hello --timeout 0
rpicam-still -o ~/camera_test.jpg
rpicam-jpeg -o ~/camera_test.jpg --timeout 2000
```

Recommended venv:

```bash
sudo apt update
sudo apt install python3-venv python3-picamera2 --no-install-recommends
cd ~/pi_cv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -c "from picamera2 import Picamera2; print('picamera2 ok')"
```

Camera Module 3 autofocus examples:

```bash
./scripts/run_client.sh --camera-stream \
  --camera-autofocus-mode continuous \
  --camera-autofocus-range macro \
  --camera-autofocus-speed fast
```

Manual focus example:

```bash
./scripts/run_client.sh --camera-stream \
  --camera-autofocus-mode manual \
  --camera-lens-position 2.0
```

## CV Experiments

CV experiments use:

```bash
./scripts/run_cv_client.sh
```

Modes:

- `depth`
- `yolo`
- `pipeline`
- `depth-check`

The server receives `cv_result` zip packages and extracts them to:

```text
data/received/cv/<run_id>/
```

Expected artifacts:

- `metadata.json`
- `original.jpg`
- `depth_heatmap.jpg`
- `depth_raw.npz`
- `yolo_annotated.jpg`
- `combined.jpg`
- `result.zip`

Not every mode creates every artifact.

## Depth Anything V2

Real depth-model setup for Raspberry Pi:

```bash
cd ~/pi_cv
source .venv/bin/activate
./scripts/setup_depth_anything_v2.sh small
```

This clones the official Depth-Anything-V2 repo into `external/` and downloads:

```text
models/depth_anything_v2_vits.pth
```

Check environment:

```bash
./scripts/run_cv_client.sh depth-check
```

Run real depth from camera:

```bash
./scripts/run_cv_client.sh depth \
  --source camera \
  --depth-backend depth-anything-v2 \
  --depth-model-path models/depth_anything_v2_vits.pth \
  --depth-encoder vits \
  --depth-input-size 392 \
  --telemetry
```

Depth Anything V2 Small (`vits`) is the first target for Raspberry Pi 5 CPU. Base/large are heavier and should be tested only after small works acceptably.

Important: standard Depth Anything V2 checkpoints output relative depth, not calibrated meters. Use `--depth-is-metric` only with a metric depth model.

## YOLO

Install YOLO dependencies:

```bash
pip install -r client/requirements-yolo.txt
```

Prepare NCNN model:

```bash
yolo detect predict model=yolo11n.pt
yolo export model=yolo11n.pt format=ncnn
```

Run YOLO detection from camera:

```bash
./scripts/run_cv_client.sh yolo \
  --source camera \
  --yolo-model models/yolo11n_ncnn_model \
  --yolo-task detect \
  --yolo-confidence 0.5 \
  --telemetry
```

Run YOLO segmentation:

```bash
./scripts/run_cv_client.sh yolo \
  --source camera \
  --yolo-model models/yolo11n-seg_ncnn_model \
  --yolo-task segment \
  --yolo-confidence 0.5
```

## Combined Pipeline

Combined experiment:

1. Capture one camera image.
2. Run YOLO detection/segmentation.
3. Run depth model.
4. Attach depth statistics to each object bounding box.
5. Send combined artifacts to MacBook.

Command:

```bash
./scripts/run_cv_client.sh pipeline \
  --source camera \
  --yolo-model models/yolo11n_ncnn_model \
  --yolo-task detect \
  --yolo-confidence 0.5 \
  --depth-backend depth-anything-v2 \
  --depth-model-path models/depth_anything_v2_vits.pth \
  --depth-encoder vits \
  --depth-input-size 392 \
  --telemetry
```

## Current Engineering Constraints

- Raspberry Pi storage is currently a 64 GB microSD, so avoid unnecessary large dependencies and persistent frame dumps.
- Use `--system-site-packages` venv so `python3-picamera2` installed through `apt` is visible.
- Keep `research/`, models, and cloned external repos out of Git.
- Prefer explicit one-shot image/CV tests before long-running video or ML loops.
- Streaming currently uses MJPEG over the project TCP protocol plus MacBook HTTP preview. H.264 via `rpicam-vid` is a future higher-quality/lower-bandwidth direction.

## Verification Commands

Syntax check:

```bash
python3 -m compileall shared client/src server/src
```

Mapping unit tests:

```bash
python3 -m unittest discover server/tests
```

Basic transport:

```bash
./scripts/run_server.sh --once --no-preview
./scripts/run_client.sh --host 127.0.0.1 --telemetry --image data/cat.jpg
```

Synthetic depth transport smoke test:

```bash
./scripts/run_cv_client.sh depth \
  --host 127.0.0.1 \
  --source image \
  --image data/cat.jpg \
  --depth-backend synthetic \
  --telemetry
```

Real depth requires `./scripts/setup_depth_anything_v2.sh small` first.
