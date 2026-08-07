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
cd ~/Desktop/work/pi_cv     # this Pi's actual checkout; no venv, system python3
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
cd ~/Desktop/work/pi_cv
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

### RoboCar drive link (the two-wheel chassis the Pi rides on)

`robot/robocar_esp.ino` (ESP8266) + `server/src/mac_server/robocar.py`
(`RobocarService`, started by `server.py` unless `--no-robot`). The
standalone `robot/robocar_server.py` remains as the reference/CLI, but its
own HTTP server is NOT used — it listened on **8080, the panel's port**.

- Discovery: ESP broadcasts `ROBOCAR?<SECRET>` on UDP 5001, we answer
  `ROBOCAR!<SECRET>:5000` and beacon the same every 2 s. The token is a
  "my server" label, **not security** — plaintext, any LAN host can drive.
- Commands: newline ASCII on TCP 5000. `M <l> <r>` (±255), `STOP`,
  `BRAKE`, `MAX <pwm>`, `PING`, `STATE`, … The sketch stops the motors
  after `FAILSAFE_MS` (400 ms) of silence, so a held key is retransmitted
  every 150 ms.
- **Two safety additions over the original script.** (1) `repeat_cmd` now
  expires after `COMMAND_TTL_S` (0.6 s): the sketch's failsafe only covers
  the *network* dropping, so a crashed browser tab used to leave the server
  cheerfully retransmitting full throttle forever. (2) `ALLOWED_COMMANDS`
  blocks `FORGET` (wipes the ESP's Wi-Fi credentials) and `REBOOT` from
  being one stray fetch away in a browser.
- API: `GET /api/robot/status`, `POST /api/robot/drive`
  (`{throttle, steer, speed}` mixed server-side, or explicit
  `{left, right}`), `POST /api/robot/command {command}`,
  `POST /api/robot/spin {speed?, target_deg?}` / `{cancel: true}`.
- **`mix_drive` returns the wheels SWAPPED.** The chassis' motor leads are
  wired mirrored relative to `M <l> <r>`, so A steered right and D steered
  left while forward/back looked fine (a swapped pair cancels when both
  wheels get the same sign — which is why it survived as legacy for so
  long). Fixed in `mix_drive`, the single place that turns *intent* into
  wheels, so the panel and the spin both inherit it; explicit `left`/
  `right` passed to `drive()` stay literal so the low-level path remains
  testable against the hardware.
- **`MAX_PWM` is raised to 150 on every connect** (`DEFAULT_MAX_PWM`). The
  sketch boots at 80, which was tuned on a bare chassis and barely moves
  one carrying a Pi 5 + camera + battery. It lives in the ESP's RAM, so a
  reset silently reverts it — hence re-sending it per connection, not once.
- **Localization spin** (`start_spin`, panel button "Spin to localize"):
  one in-place revolution so the place index gets a look at the whole
  room, because a robot set down facing a blank wall matches nothing.
  It stops on measured rotation when a heading is available —
  `note_heading()` is fed from the `nav_query` handler (~1.5 Hz,
  unwrapped), `SPIN_TIMEOUT_S` 45 s is a backstop — and falls back to a
  timed `SPIN_BLIND_S` 12 s spin when it is not. Any manual drive or STOP
  aborts it instantly (`drive(..., _internal=True)` is how the spin's own
  commands avoid cancelling themselves).
  **Two real bugs found on the first live run, both worth remembering:**
  - `SPIN_SPEED` 110 did not move the chassis *at all* while manual
    driving was fine. Turning in place fights far more friction than
    driving straight, and the sketch maps a request onto MIN_PWM(40)..
    MAX_PWM(150) — so 110 arrived as ~86, under the static-friction floor.
    Now **160** (~108 at the motor).
  - The spin is **pulsed** (`SPIN_PULSE_S` 0.40 driving, `SPIN_SETTLE_S`
    1.20 stopped), not continuous. Two independent reasons: the place
    index is matched on a CLIP embedding of ONE frame, and a frame grabbed
    mid-rotation is motion-blurred into something that matches nothing in
    an index built from a slow, sharp walk; and the nav loop only ticks at
    ~1.5 Hz, so continuous rotation samples yaw tens of degrees apart while
    the chassis is still moving. The settle deliberately outlasts one nav
    tick so no step goes unphotographed. Blind mode runs
    `SPIN_BLIND_STEPS` 14 pulses.
  - Requiring an IMU heading before spinning was a **deadlock**: the spin
    exists to GET localized, but `note_heading` was called inside
    `if ... result.get("located")` AND `pi_navigation` only built its
    `fused` payload once an EKF existed, i.e. after the first successful
    fix. The button returned 409 "no IMU heading yet" forever. Fixed at
    all three levels — raw attitude now travels as a **top-level `imu`
    field** in `nav_query` (not nested in `fused`), the handler forwards
    it regardless of the fix outcome, and the spin degrades to a blind
    timed spin instead of refusing. General lesson: raw sensor telemetry
    must not be gated behind a downstream success condition.
  Deliberately NOT auto-triggered on nav start: a robot that begins
  spinning by itself when a process connects is a bad surprise if it is
  sitting on a table.
- Panel: a **Drive** card in the Scene tab, next to the floor plan on
  purpose — you steer while watching the live fix move. Keyboard capture is
  **opt-in** via a toggle (the page has text inputs; a global WASD handler
  would eat them), and blur/visibilitychange release the keys.
- `pi_navigation` now also reports raw `imu_yaw_deg`/`pitch`/`roll` in its
  `fused` payload, shown in that card. **That readout is how the sensor's
  turn convention gets established**: hold left, see which way yaw moves.
  The gravity calibration physically cannot determine it (rotation about
  gravity leaves the gravity vector unchanged) — only this or the yaw
  check's `gain` sign can.
- **The marker arrow shows CAMERA facing, not chassis facing.** Both the
  red (raw fix) and blue (fused) arrows are rotated by `heading_deg`, which
  traces back to `navindex_step`'s `forwards.append(wfc[:3, 2])` — the
  camera's optical axis in the recorded keyframe poses. If the camera is
  bolted on facing away from the driving direction, both arrows sit 180°
  from where the robot goes **while the localization itself is correct**.
  Reported live: both markers agreed with each other and both opposed the
  robot, which is the signature of a mounting offset rather than a maths
  error. The panel's Drive card has a **cam→body** selector (0/±90/180,
  persisted in localStorage) applied at DISPLAY time only — the EKF's theta
  is camera heading throughout and is self-consistent, so rotating it
  inside the filter would be fixing the wrong thing. Ten-second check:
  press W and watch the preview; if the scene recedes, the camera faces
  backwards.
- **IMU mounting, 2026-08-01**: the user reports the module remounted as
  x-forward, y-left, "z down". The **measurement disagrees about z**: at
  rest the accelerometer reads +1002 mg on Z, and an accelerometer at rest
  measures the normal force pointing UP, so +Z points up — as the fresh
  calibration independently agrees (`up_imu_reference` ≈ [-0.028, 0.002,
  0.9996], i.e. +Z). x-forward/y-left/z-**up** is also the right-handed
  (ROS REP-103) triple; x-forward/y-left/z-down is left-handed. Nothing in
  the code assumes an axis triple — `up_imu_reference` is measured and the
  yaw sign is measured — so this does not currently break anything, but the
  yaw sign is still UNVERIFIED pending a good `--yaw-check` run.

### VLA dataset pipeline (behaviour cloning for the two-wheel base)

Observations come from the Pi, actions from the laptop, and the two are
joined offline. `docs/` not written yet; the modules carry the reasoning.

- `client/src/pi_client/vla_recorder.py` — **fixed 10 Hz, 640x360, NOT
  parallax-gated.** `scene_recorder`'s keyframe selection is exactly wrong
  here: a policy learns observation->action at a regular control interval, so
  a time axis that stretches with the robot's speed is harmful, and the
  "nothing happened" frames are facts it must learn. Capture never blocks on
  the network (bounded queue + sender thread); an irregular interval cannot be
  repaired afterwards, a counted dropped frame can. Measured live: median
  interval 100.0 ms, p95 100.0, worst 100.2 over 200 frames, 0 dropped.
- `shared/clock_sync.py` — Cristian's algorithm over the existing TCP link,
  **offset from the minimum-RTT sample, never the mean** (queueing delay is
  one-sided). Measured live: +/-1.39 ms against a 20 ms budget; crystal drift
  -9.2 us/s, i.e. 0.28 ms over a 30 s episode.
- `mac_server/vla/action_log.py` — hooks `RobocarService.drive()`, the single
  funnel every wheel command passes through. Logs post-`mix_drive` wheels AND
  pre-mix intent (the wheel swap would make a wheels-only log silently wrong
  if the wiring is ever fixed), plus the ESP heartbeats, which report applied
  post-ramp PWM and are an independent check on the timestamp join.
- **Armed automatically by `vla_session_start`**, not by a panel button: the
  recorder is the authority on when an episode begins, which removes the
  "recorded frames, forgot the actions" failure the first smoke episode hit.
- `mac_server/vla/build_dataset.py` — zero-order hold (exact, not an
  approximation: `_repeat_loop` really does retransmit every 150 ms),
  `(0,0)` synthesized in TTL gaps (real stop actions), frames before the
  first command dropped rather than zero-filled. `observation.state` is 7-dim
  with yaw RELATIVE to episode start — the RVC datum is per-power-up random,
  so an absolute heading is a per-episode random feature.
- Export writes a **self-describing manifest** always and a LeRobot dataset
  when the library is present. The LeRobot on-disk format has churned across
  releases and the recording is the unrepeatable part; a pinned release is
  necessary but not sufficient.
- `mac_server/vla/replay.py` — video with the action trace burned in. The one
  check no unit test replaces: press left, the picture must turn left. Wheel
  bars are drawn on the side the wheel is on so a swap reads as "the wrong bar
  grew".
- `mac_server/drive_profile.py` — **a stalled DC motor draws its STALL
  current**, the largest it ever draws, all of it heat. A command just below
  the moving threshold therefore costs more battery than driving. Requests
  below threshold are rounded UP; pulsing (which would creep slower) is
  opt-in and OFF for recording, because a pulsed command is an alternation
  the action log cannot represent, and a wrong action label is worse than a
  lost creep. Thresholds from `scripts/measure_stiction.py`; `measured: false`
  until then so a placeholder is never mistaken for a measurement.

### The detector is a fine-tune now, not COCO (2026-08-05)

`models/indoor_yolo11l_seg.hef` — yolo11l-seg fine-tuned on 31 indoor ADE20K
classes, replacing `yolov8m_seg_h8`. Measured on the same 53 recorded
keyframes, same NPU, threshold 0.4:

| | COCO yolov8m | indoor yolo11l |
|---|---|---|
| detections | 16 | **112** |
| non-empty frames | 15/53 | **46/53** |
| ms/frame (infer+decode) | 72 | 91 |

COCO found `sports ball ×3, surfboard ×2, teddy bear ×2`; the fine-tune finds
`wall ×50, painting ×20, window ×9, lamp/chair/floor ×6, ceiling/door ×4`.
On a fresh camera snapshot the COCO hef returned **zero** detections and the
fine-tune returned nine. Validation over 2000 held-out images: box mAP50-95
**0.480**, mask **0.398** (best epoch 39 of 50; early-stopped).

Things worth knowing before touching this:

- **There is no off-the-shelf indoor model for Hailo-8.** All three Model Zoo
  lists were checked: instance segmentation is 18 COCO models (**FastSAM and
  SAM are NOT there** — an earlier plan in this repo claimed otherwise and was
  wrong), semantic segmentation is Cityscapes/Pascal-VOC/Oxford-Pets. Nothing
  knows `wall`, `cabinet` or `door`. Fine-tuning was the only route.
- **YOLO11's head is geometrically identical to YOLOv8's** — same `Segment`
  module, reg_max 16 (=> 64 box channels), 32 mask coefficients — so the Pi's
  pure-numpy decoder needed no changes beyond `SegArch(num_classes=31)`.
  Checked on the ultralytics model AND on the exported ONNX before training.
- **Our hef emits class LOGITS, the Model Zoo's emit probabilities.** Cutting
  the graph at `/model.23/cv3.x/cv3.x.2/Conv` leaves YOLO's sigmoid outside,
  so the blobs range -45.7..1.48 instead of 0..1. `SegArch.class_logits`
  applies it host-side. This is silent when wrong: the decode compares -34.5
  against a 0.4 threshold and finds nothing, forever, with no error. Caught by
  `probe_hef.py`'s per-blob min/max, which is exactly why that tool exists.
- **The DFC needs `normalization1 = normalization([0,0,0], [255,255,255])`.**
  The Pi feeds uint8 0-255 while the net trained on 0-1, and without an
  on-chip normalization layer quantisation fails outright on the softmax
  inside YOLO11's C2PSA attention blocks (`NegativeSlopeExponentNonFixable`).
  Normalising the calibration data instead would work for the compiler and
  then silently mismatch the Pi's input format.
- **Quantise on CPU.** `CUDA_VISIBLE_DEVICES=""` in `compile_hef.sh`: GPU 0 is
  another user's vLLM holding all 48 GB (cuDNN init fails with a misleading
  "No DNN in stream executor"), and pinning to GPU 1 died mid-quantisation
  with `CUDA_ERROR_INVALID_HANDLE` — the DFC's bundled TensorFlow disagreeing
  with this box's driver 545.
- **The model does not fit one Hailo-8 context**; the DFC split it into 5,
  after a 35-minute partition search. That is the whole of the 72 -> 91 ms
  difference and is still ~11 fps against a 2-3 fps target.
- **DFC 3.34 builds load fine on HailoRT 4.23** (the documented pairing is
  3.33; verified by running, not assumed).
- **Weak classes, from the per-class validation:** `door` 0.288 despite 6666
  training images (flat, blends into walls), `wall` 0.258, `shelf` 0.144,
  `box` 0.128, `stairs` 0.113. `wall` being weak matters because it is 50 of
  the 112 detections above. Strong: refrigerator 0.603 (the *thinnest* class
  at 478 images — a fridge is a big distinctive box), toilet 0.588,
  fireplace 0.585, painting 0.567, ceiling 0.515, curtain 0.484, bed/sofa
  0.451, cabinet 0.436.
- Rebuild path: `gpu_service/training/ade20k_to_yolo.py` (semantic ->
  instance via connected components, class merges in
  `config/seg_classes_indoor.json`), `train_seg.py`, `compile_hef.sh`.
  **Never reorder `seg_classes_indoor.json`** — its order is the class ids
  baked into the weights; `load_class_names` refuses on a count mismatch
  because a short list silently relabels everything past the difference.

### Metric scale is the whole ballgame (2026-08-07)

Two sessions in a row produced a "floor plan" that was a starburst of rays with
no walls. Perception was never the problem — the door was detected 16 times in
16 keyframes at 0.73. The scale was:

| | before | after |
|---|---|---|
| scale / source | 25.62 (imu) | 1.515 (camera_height) |
| depth surviving the 5 m truncation | 21% | **90%** |
| mesh bbox | 24.2 x 5.5 x 31.2 m | **4.09 x 2.48 x 3.58 m** |
| objects | 2 | **12** |

The room really is ~4 x 3.6 m, so the second column is right and the first was
out by a factor of ~17.

- **`reconstruction.camera_height_m` IS the scale** when the floor-fit source
  is used — the two are proportional. It was 0.13 m from an older rig while the
  camera actually sat at 0.095 m. Re-measure it with a tape whenever the mount
  changes; nothing else in the pipeline notices.
- **The IMU distance is not trustworthy on this rig.** It reported **87.9 m of
  path** for one 153 s pass around a small room, with single keyframe intervals
  of 5.6 m and a peak "speed" of 2.5 m/s. It double-integrates acceleration, so
  a gravity-subtraction error grows quadratically — and stays perfectly
  self-consistent while doing it. Its reported spread (IQR 6%) describes its
  repeatability, not its accuracy. Note **zero keyframes carried a `gravity`
  field** in that session: the tilt-tolerance check had withheld it after the
  tripod changed, which is the calibration behaving correctly and the
  preintegration not noticing.
- **`scale_source: auto` now ranks by evidence, not by a fixed list.** The old
  order was `known_baseline -> imu -> camera_height`, with camera_height last
  because on the *previous* rig (13 cm, level) the floor was a sliver giving
  15% plane inliers. After the tripod was raised the same fit reported **95.6%
  inliers over 48 frames, IQR 3.9%** — and the fixed order still took the IMU's
  number. Selection now puts camera_height ahead of the IMU when
  `floor_inlier_frac >= floor_inlier_trust` (0.6). A tape-measured
  `baseline.json` still outranks both.
- **`object_roles.scale_plausibility` compares object extents to a table of
  typical sizes**, and its tolerance is deliberately ASYMMETRIC: 1.5x up, 0.35x
  down. A reconstruction only ever contains the surfaces the camera saw — the
  front of a cabinet, one side of a table — so extents are biased downward and
  a median ratio of 0.6 is normal. The first, symmetric version flagged the
  known-good 4.09 m reconstruction as broken.

### Object identity and description (2026-08-05)

The reconstruction now answers "what is in this room" with one entry per real
thing, each carrying a place, a picture and a sentence — the form a VLA policy
can actually consume. Three changes, each fixing something measured:

- **Class is evidence, not a gate.** `ObjectBank._compatible` required an exact
  class match. On a real snapshot one box came back as `cabinet 0.339` AND
  `door 0.339`, which that rule turns into two objects standing in the same
  place. Now geometry and DINOv2 decide identity and the name is voted from
  every observation, weighted by confidence (`object_roles.vote_label`).
  Confusable pairs are forgiven via `CONFUSABLE_GROUPS` (large flat vertical
  surfaces; seat furniture), unrelated classes still block, so appearance
  alone cannot weld a chair to a fridge. Each object reports
  `label_agreement` — the winning name's share of the evidence — and a low
  value is passed to the VLM as "treat that label as a guess" instead of being
  hidden behind a confident-looking word.
- **`wall`/`floor`/`ceiling` are regions, not objects.** `role` lives in
  `config/seg_classes_indoor.json` next to the class ids. On the 53-frame
  session the fine-tune returned **50 walls out of 112 detections**; instancing
  them would bury the furniture on the floor plan while adding nothing that
  the occupancy grid does not already hold. `floor` is the one AGGREGATE class
  — one record with a nominal centre, because the robot drives on it. `door`
  and `window` stay instanced: "the white door 15° to the left" is the whole
  point.
- **The VLM arbitrates borderline merges.** `ObjectBank(arbiter=...)` is asked
  only when DINOv2 cosine lands inside `arbiter_band` (0.15) BELOW the accept
  threshold and geometry already agrees — above it the embeddings have
  decided, far below they have decided the other way, so the band is the only
  place a second opinion changes anything. **A None verdict must never read as
  "different objects"**: an outage would then silently double every object in
  the room, so it falls through to the embedding decision.
  `describe_step.make_vlm_arbiter` builds one from `QwenVisionClient`.

`describe_step.py` (pipeline step `describe`, between `objects` and `graph`)
writes `derived/objects_described.json` + `derived/object_crops/`:

- Up to 3 views per object, **ranked** by mask area and halved for touching a
  frame border — the first observation is usually the worst one (object
  entering frame, half cut off), and describing that half as the whole thing
  is the error to avoid.
- The instance is **outlined in green before cropping**, so the contour follows
  the true mask. Without it a VLM describes the room rather than the object.
- Replies are parsed defensively: the live Qwen wraps its JSON in ```json
  fences and adds chatter, so `parse_json_reply` takes the first balanced
  object and returns None on anything else. Verified live — the fine-tune's
  uncertain `cabinet 0.339` was described as *"Black Filing Cabinet — a tall
  black filing cabinet with a wood top"*, `label_ok: true`.
- Descriptions cache under `derived/descriptions/`; a missing or unreachable
  vLLM yields crops + coordinates + null descriptions, which is a usable scene.
- `QwenVisionClient.ask(images, prompt)` is the multi-image generalisation;
  `caption()` is now a one-image call to it. **That vLLM belongs to another
  user of the GPU box** — it queues rather than cold-loads, so overlapping
  requests are fine, but the 60 s cooldown on failure stays.

Compute placement, unchanged and worth restating: DINOv2, CLIP image and CLIP
text already run on `pdfserver` via `scene3d/gpu_client.py`; the Pi does
segmentation on the NPU; the laptop only orchestrates.

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
- **The Hailo-8 budget, measured (`scripts/probe_hef.py`, 2026-08-04).**
  Run it before writing any decoder or leaning on any latency assumption:
  it prints every vstream shape, per-blob min/max/mean on a real keyframe,
  and timings. On this Pi, `yolov8m_seg_h8` **39.8 ms** median (p95 40.1),
  `clip_resnet_50x4_h8` **74.3 ms**, both interleaved **126.0 ms** — so
  scheduler switching costs ~12 ms over the 114 ms sum. Two consequences:
  - **Per-detection CLIP cannot stay on the Pi once proposals go
    class-agnostic.** 74 ms x 20-40 proposals is 1.5-3.0 s per keyframe,
    against a 2-3 fps target. This is why `objects_step` does that CLIP on
    the Mac; it is a hard constraint, not an optimization.
  - **Holding `InferVStreams` open is a dead lever.** `_ConfiguredHef.infer`
    opens a fresh context per call, which looked like obvious overhead;
    measured, it costs **0.2 ms of 39.8 (0.5%)**, and is unchanged with two
    models resident. Do not restructure `hailo_infer` for it.
  The probe also classifies the output layout from shapes alone (raw
  10-blob YOLOv8-seg / on-chip-NMS / embedding / unknown) and derives
  `num_classes`, `reg_max`, `num_masks` **without hard-coding any of them**
  — that derivation is what tells a 1-class FastSAM head from an 80-class
  one. It refuses with the evidence attached rather than guessing when two
  heads share a channel width. Pure-logic part unit-tested in
  `client/tests/test_probe_hef.py` against shapes measured from real hefs.
- **The COCO detector's own tensors say it has nothing to report in this
  room.** On keyframe 20 of `session_20260803_171339`, the maximum class
  score across all three scales, all anchors, all 80 classes is **0.369** —
  under the 0.4 threshold — and ≥99.79% of every class blob is exactly
  zero. That is independent, source-level confirmation of the
  14-detections/53-keyframes finding: the model is not being thresholded
  too hard, it genuinely does not fire on cabinets, doors and walls.
- **The recorder must LETTERBOX, not squash** (`seg_postprocess.letterbox` +
  `Letterbox.box_to_frame`/`crop_masks`). It used to `cv2.resize` a 1536x864
  frame straight into the hef's 640x640, flattening 16:9 into a square —
  a shape no YOLO checkpoint in this chain has ever seen, since the Model Zoo
  hefs, the ultralytics weights and the fine-tune are all trained and
  validated on aspect-preserved input with grey-114 padding. Measured on the
  53 recorded keyframes with the stock COCO hef: **11 detections squashed vs
  16 letterboxed**, and on keyframe 20 the same object moved 0.369 → 0.494,
  from under the 0.4 threshold to over it. Boxes are mapped back through
  `box_to_frame`, and masks are cropped to the content band before storage so
  the stored grid stays proportional to the frame and every reader can keep
  resizing naively (`mask_shape` becomes [360, 640] for a 16:9 frame).
- **Near-threshold detections are also genuinely unstable, independently of
  that.** Re-encoding the same keyframe at JPEG q95 and re-running moved a
  score **0.369 → 0.478 → 0.502 → 0.522** with the box fixed to 0.6 px, on a
  perturbation invisible to the eye. (Every score is an exact multiple of
  1/255 — honest dequantized INT8, not a resolution artifact; a 0.15 swing is
  ~38 quantization steps.) NOTE: an earlier version of this file blamed the
  0.369-vs-0.494 gap on that JPEG sensitivity. That was wrong — the cause was
  the squash/letterbox mismatch above, which is systematic and directional,
  whereas the JPEG effect has no consistent sign. Two consequences:
  - The recorded "14 detections at 0.408-0.655" are **coin flips at the
    threshold**, not 14 findings. This is a second, independent reason the
    COCO path is unusable here, on top of the classes simply being absent.
  - **Never write an acceptance criterion that requires reproducing a
    detection COUNT across a re-run.** Replay from saved JPEGs is exact at
    the decode stage — old and new decoders were verified bit-identical on
    the same raw NPU outputs at thresholds 0.4/0.1/0.01 — but the NPU stage
    is not, because the pixels are not. Compare decode against decode.
- **`seg_postprocess` is parameterized by `SegArch`** (name, num_classes,
  reg_max, num_masks, input_shape, strides) with `YOLOV8_SEG` and
  `FASTSAM_S`; `order_endnodes(outputs, arch)` and
  `yolov8_seg_postprocess(..., arch=)` default to `YOLOV8_SEG` so the
  existing path is unchanged (asserted by a test). What used to be literals
  and is now derived: prototypes are the `num_masks`-channel blob at a grid
  finer than any detection scale (was `h == 160`), the role map comes from
  the arch (was `{64, 80, 32}`), and the scale heights come from
  `input_shape / strides` (was `[20, 40, 80]`). That last one was a live
  trap: the recorder reads the real hef input shape but the decoder assumed
  640x640, so a differently-sized hef decoded into the wrong coordinate
  space with no error. `scene_recorder` now builds the arch with
  `dataclasses.replace(..., input_shape=self.seg.input_shape[:2])`.
  A `SegArch` whose three channel widths are not all distinct refuses at
  construction — blobs are told apart by channel count and nothing else, so
  a collision is genuinely undecodable. `--seg-arch` records into
  `session_meta.json` **and every keyframe's `meta.json`** (absent = legacy
  `yolov8_seg`). With a 1-class arch the recorder emits class `"object"`,
  never a COCO name — indexing COCO80 with class 0 would label every
  proposal "person".
- **`--source replay --replay-session <dir>` re-feeds a recorded session's
  keyframes through the whole Pi pipeline.** This is the tool that makes
  every later perception change measurable: fixed pixels, no robot, no
  battery, no room. Measured: the 53-keyframe session replays in **6 s**
  against 174 s of walking, and it does not sleep to the capture fps (a
  replay is not rate-limited by a sensor). `ReplaySource.exhausted` ends the
  run instead of looping — a wrap-around would look like continuing
  progress. Fixture on the Pi: `data/replay/session_20260803_171339`.
  Reproduced 11 detections against the original 14, which is the expected
  near-threshold spread, not a defect — see the reproducibility note above.
- **Masks are `stack_v1`**: per-instance binary masks stacked vertically at
  INFERENCE resolution, still `masks.png`, with `masks_format`/`mask_shape`/
  `mask_count`/`frame_shape` in each keyframe's `meta.json`. Read through
  `session_io.Keyframe.instance_mask(instance_id, out_shape)`, which also
  serves the legacy format (absent field = `labels_v0`), so no session needs
  migrating. The old uint16 label image **cannot represent overlap** —
  `instance_map[mask] = id` means the last write wins every shared pixel —
  which is fine for a handful of disjoint COCO objects and wrong for
  class-agnostic proposals, where nesting (cabinet ⊃ drawer, table ⊃ objects
  on it) is the design. A unit test asserts the loss concretely: a 20×20
  mask containing a 10×10 one reads back as 300 px under `labels_v0` and 400
  under `stack_v1`. Size was a worry and is not one: **53 keyframes of masks
  total under 0.1 MB** (median 67 bytes) — PNG compresses binary masks
  almost to nothing.
- **`GreedyTracker` now returns `updates.assignments`**, a track id per
  INPUT detection positionally aligned to the list passed in (`None` for
  ones it declined). It replaces the recorder's reverse lookup keyed on the
  rounded bbox, which silently merged two detections whose boxes rounded the
  same — harmless at 14 detections a session, routine once proposals are
  dense and nested. Purely additive; nothing else reads it.
- **Proposal gates** (`gate_proposals`, pure and unit-tested):
  `--min-mask-area-frac` / `--max-mask-area-frac` / `--reject-edge-count`,
  applied BEFORE tracking and before CLIP so a rejected proposal costs
  neither an NPU call nor a track id. **All three default to inert**
  (0.0 / 1.0 / 0 = off) and every rejection is counted by reason into
  `session_meta.json:proposal_gates` — a gate that quietly drops half the
  proposals is otherwise indistinguishable from a detector that never found
  them, and the two call for opposite fixes. Area is measured on the MASK
  (a box can be a rectangle around a thin diagonal sliver; the mask is what
  gets reconstructed), edge contact on the BOX. Note 2 edges is legitimate —
  a cabinet in a corner touches two.
- **`--track-min-confidence` is split from `--confidence`.** One value drove
  both, which is wrong in principle (a proposal worth reconstructing can be
  far weaker than one worth asserting an identity about across frames) and
  breaks outright with a class-agnostic arch, where every proposal scores
  ~1.0 and a shared value stops gating anything.
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
- **Reverted to UART-RVC (2026-08-01) — RVC is now the default everywhere.**
  The sensor was rewired: pins straight to the CH340 adapter, straight to
  USB, PS1/PS0 back to RVC. Same path `/dev/ttyUSB0` (stable alias
  `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0`), 115200. Measured
  after rewiring: **501 frames in 5.0 s (100.2 Hz), zero checksum resyncs,
  zero gaps in the frame index**, |accel| median 1002 mg — against the
  SHTP link's ~25% well-formed frames at the end. Stationary yaw drift
  **0.03°/min** over 90 s (9011 samples). `--imu-driver`/`--driver` in
  `scene_recorder.py` and `imu_calibrate.py` now default to `rvc`; the
  SHTP modules stay in the tree for the case the jumpers are flipped back.
  Consequences that matter:
  - **No raw gyro on this link.** RVC gives fused yaw/pitch/roll +
    accelerometer only. That is fine for the robot's actual rotational DOF
    (it drives in-plane and only yaws), and the on-chip fusion beats
    host-side integration anyway — but metric dead reckoning is now
    explicitly out of scope for live navigation, by the user's own call.
    `Ekf2DVio` and `imu_distance_analysis.py` remain for the scene3d
    metric-scale experiment; nothing live consumes them.
  - **`Ekf2DHeading` (ekf_localization.py) is the navigation filter.**
    4-state [x, y, θ, b] where `b` is the offset between RVC's arbitrary
    yaw datum and the room's plan-frame heading zero. RVC yaw enters as an
    ABSOLUTE observation (`z = θ - b`), never in `predict` — using it in
    both would count one measurement twice. Visual fixes make `b`
    observable; once converged, IMU yaw alone gives absolute map heading
    between fixes. **The filter only works because `turn_rate_rad_s`
    (1.0) and `bias_walk_per_s` (0.0002) are far apart** — θ and b enter
    only as their difference, so the split of a yaw change between "robot
    turned" and "datum drifted" is set by their variance ratio. A draft
    using 0.02 rad/s for heading charged half of a 90° turn to bias.
  - **`imu_rvc.RvcReader` gained real link health**: `frames_bad` used to
    be a dead counter that was never incremented (`iter_frames` now
    returns a resync count), `frames_dropped` catches loss the checksum
    cannot see (gaps in the RVC frame index), `start()` waits for one
    valid frame instead of declaring success on a port that merely opened,
    and `read_orientation()` returns **None** when stale rather than the
    last value — a frozen heading is indistinguishable from a steady one,
    which is how the SHTP collapse hid for a session.
  - **Yaw validation: `run_imu_calibrate.sh --yaw-check 60`** (or
    `--yaw-check-only` against the calibration already on disk). Swing the
    robot left/right in front of the same plumb wall board; the board's PnP
    pose is sub-degree ground truth for heading, so `fit_yaw_relation`
    reports gain (must be +/-1 — an inverted yaw convention means the robot
    turns left while the filter thinks right, which the gravity calibration
    physically cannot detect), per-sample RMS error (→ set
    `pi_navigation --imu-yaw-sigma-deg` from it), and **drift under motion**
    (→ `Ekf2DHeading.predict`'s `bias_walk_per_s`, previously a guess; the
    0.03 deg/min figure was measured stationary). Deliberately NOT an input
    to the solve: rotating about gravity leaves the gravity vector in the
    camera frame unchanged, so no amount of turning can refine
    `up_camera_reference`. Result is stored as `yaw_check` in the
    calibration file. Yaw is measured about the board-derived vertical, not
    the camera's own axis, so the rig's camera pitch does not smear into it
    (unit-tested). Raw (t, camera_yaw, imu_yaw, distance) pairs are always
    dumped to `data/imu_tests/yaw_check_<ts>.csv` so a failed check can be
    diagnosed instead of re-run blind.
    **The first real run failed on a bug in this check, not the sensor**
    (gain -2.19, offset_deg -174.4): the board's outward normal points back
    at the camera, so the measured angle sat on the +/-180 `atan2`
    discontinuity and `unwrap_deg` banked every noise flip as real
    rotation — smooth, confident, meaningless. The normal is now negated so
    the operating point is 0, `fit_yaw_relation` refuses outright when >20%
    of samples land beyond 150 deg, and the synthetic test fixture (which
    had the board facing AWAY from the camera, which is why it passed) is
    physical. Also hardened after that run: MIN_YAW_SPAN_DEG 25 (it swept
    only 15.8), a `MIN_CORNER_PITCH_PX` 12 gate (intrinsics-FREE — distance
    comes out of solvePnP, which is the thing under suspicion), a robust
    outlier pass for planar-PnP pose flips, and a reported `gain_std_err`
    so an imprecise gain reads as "inconclusive run" instead of "IMU yaw
    axis is not vertical". Working distance is ~0.7-1.0 m; an earlier
    "must be >= 0.8 m or there is nowhere to turn" rule was wrong reasoning
    (this camera has a wide field — even at 0.5 m there is +/-26 deg of
    turn room, far more than the 25 deg required).
- **Focal length settled by tape measure (2026-08-03).** `run_focal_check.sh
  --distance-m 0.50` puts the board on a wall, measures lens-to-board with a
  tape, and inverts `d_measured = d_true * fx_assumed / fx_true`. Measured:
  solvePnP said 0.567 m for a tape 0.500 m → **fx ≈ 968, HFOV 76.9°**,
  spread 0.02% over 20 frames. That agrees with VGGT (991) and COLMAP/DA3
  (1001) and refutes the 2026-08-03 chessboard solve (1098, 69.9°).
  **Why the chessboard solve was wrong, and it is not what you would guess:**
  it had RMS 0.484 px, 30 boards, full 3×3 frame coverage and three distance
  buckets — every usual quality metric passed. The board was taped to a wall
  and photographed by a robot that only drives on the floor and yaws, so it
  was **fronto-parallel in every view**. Head-on, a change in focal length
  and a change in distance produce almost the same image, so the solve slides
  along that degenerate pair and still reports a low RMS. `scene_calibrate.py`
  now computes `board_tilt_spread_deg()` from the solved rvecs and rejects a
  capture whose orientation spread is under `MIN_BOARD_TILT_SPREAD_DEG` (25°),
  telling the operator to **hold the board by hand and tilt it**. Distance
  variety is not a substitute. Working distance for the tape check is
  0.4–0.6 m: the binding constraint is corner size in pixels (53 px/square at
  0.5 m vs 26 at 1.0 m), not tape precision (1 cm = 2% at 0.5 m against a
  ~13% effect).
- **`config/scene_intrinsics.json` was wrong and everything PnP-based
  inherited it (2026-08-01; the 110° figure below is superseded by the
  measurement above, but the lesson stands).** It claims fx 534.457 / cy 500 at
  1536x864 — a **110 deg** horizontal field with the principal point 68 px
  off centre. Three independent sources say ~75 deg: VGGT's refined
  intrinsics on session_20260730_161007 (75.6 deg, fx→991 scaled to 1536),
  and the COLMAP/DA3 measurements already recorded above (75.7/74.4 deg,
  fx ~1001). Almost certainly calibrated at a different capture
  resolution. Consequences, none of them cosmetic: reported PnP distances
  are ~1.85x too small (the yaw check's "0.22-0.35 m" was really
  ~0.41-0.65 m), and for a planar target focal error couples directly into
  the recovered out-of-plane rotation — so this biases the yaw check AND
  `up_camera_reference` in the gravity calibration itself. `imu_calibrate.py`
  now runs `check_intrinsics_sanity()` at startup and logs both problems as
  errors, but **the fix is to re-run `./scripts/run_scene_calibrate.sh` at
  the resolution the pipeline actually captures at**; until then treat every
  PnP-derived number from this rig as provisional.
  - **The Jul-27 `config/imu_calibration.json` (mode `full`) survives the
    rewiring but is marginal**: measured 3.03° from `up_imu_reference`
    against a 4.0° `tilt_tolerance_deg`, and live pitch/roll (-0.49/0.50)
    sit ~3° outside where `tilt_model` was fitted (2.50/0.91). Gravity is
    still being reported, but one more nudge silently stops it. Re-run
    `./scripts/run_imu_calibrate.sh` against the plumb wall board.
- **SHTP-over-UART (2026-07-28, superseded by the above)**:
  `docs/imu_shtp_setup.md`'s SPI
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

**This Pi has NO venv, on purpose (2026-08-05).** Every `scripts/run_*.sh`
already execs bare `python3`, never an activated environment, and the system
Python 3.13.5 has everything the project needs — numpy, cv2, picamera2 and
`hailo_platform` all come from apt (`hailo-all`), which is exactly the case a
venv makes harder rather than easier.

There WAS a `.venv` there, 6.6 GB of it, and it was never usable: its
`pyvenv.cfg` read `home = /Library/Frameworks/Python.framework/Versions/3.12`
and `command = ... /Users/aleksey.morozov/Desktop/hlam/pi_cv/.venv` — a venv
built on the **MacBook** and copied over by some rsync/scp that ignored
`.gitignore` (where `.venv/` has always been listed). It contained two
half-populated package trees (3.12 and 3.13), no interpreter at all, and
2.9 GB of `nvidia` CUDA libraries plus 652 MB of `triton` on a machine with no
NVIDIA GPU. Deleting it plus the 2.8 GB pip cache took the card from 22 GB
used to 13 GB, and the probe + a 53-keyframe replay both still ran unchanged.

If a venv is ever genuinely needed, it must be created ON the Pi and inherit
the apt packages — otherwise picamera2 and hailo_platform vanish:

```bash
sudo apt update
sudo apt install python3-venv python3-picamera2 --no-install-recommends
cd ~/Desktop/work/pi_cv
python3 -m venv --system-site-packages .venv   # --system-site-packages is not optional
source .venv/bin/activate
python3 -c "from picamera2 import Picamera2; print('picamera2 ok')"
```

**Never rsync a laptop venv to the Pi.** Add `--exclude .venv` to anything that
syncs the tree; wheels are per-OS and per-architecture, so the copy is dead
weight at best.

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
cd ~/Desktop/work/pi_cv
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
