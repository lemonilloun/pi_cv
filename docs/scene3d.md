# Scene3D: semantic room reconstruction (Pi records → Mac reconstructs)

One slow walk around the room with the Pi camera → a metric 3D mesh of the
room, per-object 3D segments with open-vocabulary labels, and a scene
graph. Offline by design: the Pi writes a session to disk, the Mac
processes it — no real-time constraint (2-3 fps recording is the target).

## Division of labor

**Pi 5 + Hailo-8 (all NPU, CPU stays light):**
- `yolov8s_seg` hef — instance segmentation, 640×640, host-side decode
  (pure-numpy port of hailo-apps' postprocess in
  `client/src/pi_client/seg_postprocess.py`; ~48 ms infer + ~40 ms decode)
- CLIP RN50x4 hef — a 640-d image embedding per detection crop (~77 ms),
  shares the NPU with seg via the HailoRT model scheduler
  (`client/src/pi_client/hailo_infer.py`)
- greedy tracker (reused `mac_server.monitoring.tracker`) for session-wide
  `track_id`s; keyframe selector (min interval 0.5 s + blur gate)

**Mac M2 (offline pipeline, `server/src/mac_server/scene3d/`):**
1. `depth` — Depth Anything V2 **metric** (hypersim, vits) on MPS →
   `depth/<kf>.npy` in meters (~250 ms/frame)
2. `poses` — pycolmap: SIFT → sequential matching → incremental mapping
   (OPENCV camera model with the calibrated K + distortion), then **scale
   alignment**: per-frame median(DAv2 depth / COLMAP sparse depth) →
   global median with MAD rejection → metric `poses.json`
3. `tsdf` — Open3D ScalableTSDFVolume (voxel 2 cm) → `room_mesh.ply` +
   top-down `floor_plan.png` (gravity ≈ mean camera-down)
4. `objects` — per keyframe: mask + depth + pose → world points → DBSCAN;
   association into an object bank: Pi `track_id` prior, then DINOv2
   cosine > 0.6 AND centroid < 0.5 m → `objects.json` + per-object `.ply`.
   Two safety caps keep one bad detection from stalling the whole step for
   minutes: a mask covering more than `max_mask_area_frac` (0.35) of the
   frame is skipped outright (almost always a mis-segmented wall/floor,
   not a bounded object), and any point cloud is randomly subsampled to
   `dbscan_max_points` (8000) before clustering — DBSCAN's cost isn't
   linear, so an oversized cloud can otherwise turn one detection into a
   multi-minute stall that looks identical to a hang from the outside.
5. `graph` — CLIP text encoder (**RN50x4/openai — must match the Pi hef**)
   vs ~70-word indoor vocabulary on the averaged Pi CLIP embeddings, COCO
   vote as prior; `near`/`on` edges from OBB geometry →
   `scene_graph.json` + `floor_plan_labeled.png`

## The data contract (session directory)

```
data/scene_sessions/session_YYYYMMDD_HHMMSS/
├── intrinsics.json          # from scene_calibrate.py (optional but wanted)
├── session_meta.json        # written on recorder stop
├── keyframes/000001/
│   ├── rgb.jpg              # full res (1536x864), q95
│   ├── masks.png            # uint16, 0=background, else instance_id
│   └── meta.json            # detections: instance_id, track_id,
│                            #   class_coco, confidence, bbox_xyxy,
│                            #   clip_emb (640 floats, L2-normalized)
└── derived/                 # everything the Mac pipeline produces
    ├── depth/  colmap/  poses.json  scale_report.json
    ├── room_mesh.ply  floor_plan.png  floor_plan_labeled.png
    ├── objects.json  objects_pcd/  scene_graph.json
    └── pipeline_state.json  # per-step status for the panel
```

`masks.png` is at rgb.jpg resolution (nearest-neighbor upscale from the
640×640 inference); `instance_id` links pixels to `meta.json` entries;
`track_id` is session-wide. Everything the Mac needs is this directory.

## Workflow (who runs what)

**Once — calibrate on the Pi** (print a 9×6 chessboard, 25 mm squares):

```bash
./scripts/run_scene_calibrate.sh --lens-position 2.0
```

Target RMS < 0.5 px. The LensPosition is saved into intrinsics.json and
reused by the recorder — do not change focus between calibration and
recording. Without calibration the pipeline still runs (FOV-estimated K,
`estimated: true`) but metric quality suffers.

**Record on the Pi** (walk slowly, lots of view overlap, come back to
where you started — loop closure helps COLMAP). Start the Mac server
first — keyframes are **streamed to the Mac live** (nothing accumulates on
the microSD), and the panel's Live tab shows an annotated preview of what
the recorder sees while you walk:

```bash
./scripts/run_scene_recorder.sh          # Ctrl+C to stop
# useful flags: --fps 3 --keyframe-interval 0.5 --no-clip --no-preview
# --offline reverts to local-disk recording (old behavior)
```

The server host comes from `.env` (`PI_CV_SERVER_HOST=vedro.local`). If
Wi-Fi drops mid-walk, affected keyframes are written locally under
`data/scene_sessions/<id>/` and the recorder prints the exact rsync command
to merge them afterwards (run it **from the Mac** — the Mac has no SSH
server, so Pi→Mac rsync needs Remote Login enabled; Mac→Pi always works).

**Process** — panel `Scene` tab (select session → Run pipeline) or CLI:

```bash
./scripts/run_scene_pipeline.sh --list
./scripts/run_scene_pipeline.sh session_YYYYMMDD_HHMMSS          # all steps
./scripts/run_scene_pipeline.sh session_YYYYMMDD_HHMMSS poses tsdf --force
```

## Panel (Scene tab)

Sessions table (keyframes, calibrated?, per-step chips) · Run pipeline
(with re-run-done toggle) · live per-step progress · results: labeled
floor plan, interactive 3D mesh (three.js, vendored — no CDN), objects
table with CLIP top-3 + `near`/`on` relations.

API: `GET /api/scene3d/sessions`, `POST /api/scene3d/run
{session_id, steps?, force?}`, `GET /api/scene3d/status`,
`GET /scene3d/artifact?session_id=&name=` (whitelisted names).

## Localization (pi_navigation)

After a session is reconstructed, the pipeline's 6th step (`navindex`)
builds a **place index**: a full-frame CLIP embedding + camera pose for
every registered keyframe. The live client then answers "which room am I
in and which way am I looking":

```bash
./scripts/run_pi_navigation.sh        # Pi; Mac server must be running
```

Per query (~1.5 Hz): frame -> CLIP RN50x4 on the NPU (the same model
family the index was built with) -> `nav_query` over TCP -> the Mac
matches against every session's index and returns room (session name),
nearest recorded viewpoint, similarity, floor-plan position + heading.
`fast_depth` (also on the NPU) adds the free-space distance ahead. The
Scene tab shows the live fix (📍 line) and draws an oriented arrow on the
floor plan when the located room is selected. This is retrieval-based
place recognition — a coarse prior for the future navigation stack, not
SLAM.

**Why heading is the weak link, and what calibrated `scene_intrinsics.json`
does and doesn't fix for it**: intrinsics feed pycolmap during the
`poses` step, so a real calibration makes the *stored* pose of every
keyframe in the index more accurate — but `pi_navigation` never computes a
pose from the live frame at all. It's pure retrieval: the reported
heading is borrowed from whichever indexed keyframe(s) had the most
similar CLIP embedding. CLIP is deliberately somewhat viewpoint-invariant
(good for "which room", bad for "which way") — the single-nearest-neighbor
version of this borrowed the heading of just one match, so it could easily
be wrong even in a well-calibrated, well-reconstructed session.
`navindex.query()` now averages the top `intra_k` (default 7) most similar
keyframes *within the winning session* with a proper circular mean
(weighted by similarity), and reports `heading_spread_deg` — how much
those neighbors actually agree. Low spread (a few degrees) means trust the
heading; high spread (tens of degrees) means the match is genuinely
ambiguous (e.g. a symmetric-looking spot, or the walkthrough never
recorded that facing direction from nearby) and no amount of averaging
fixes that — the real fix is recording with more angular variety (glance
around while walking, not just straight ahead) so the index has more
headings to draw from at every position. A true fix (feature-matching +
PnP against the COLMAP sparse model, i.e. visual relocalization) would
compute pose from the live frame directly instead of borrowing it, but
that's a separate, heavier feature — not implemented yet.

## Session management

Sessions can be renamed (display name, stored in session_meta.json — the
directory stays stable) and deleted from the Scene tab (✎ / 🗑 buttons),
or via `POST /api/scene3d/rename|delete`.

## Model choices

- Pi segmentation: `yolov8m_seg` (40.1 mask mAP) is the recorder default;
  pass `--seg-hef models/yolov8s_seg_h8.hef` if you ever need more fps.
- Mac depth: DAv2 metric **vitb** (scene3d.depth in config); the live
  stream keeps vits for real-time.
- CLIP: `RN50x4-quickgelu`/openai on the Mac — the quickgelu variant is
  the exact match for the Pi hef's weights.

## Known constraints

- **SfM needs parallax**: a static-tripod session fails `poses` with
  "COLMAP registered nothing" — that's physics, not a bug. Walk the camera.
- The world frame is COLMAP's (gravity unknown); the floor plan uses mean
  camera-down as up, which assumes a roughly level camera during recording.
- Monocular metric depth has scale error; the COLMAP+DAv2 scale alignment
  averages it out (check `scale_report.json`: IQR/median < 0.15 is good).
- The recorder and the live session client can't share the camera — stop
  the session client (or switch it to `idle`) before recording.
- One pipeline run at a time (8 GB RAM; every step is heavy).
