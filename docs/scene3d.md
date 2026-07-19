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
   cosine > 0.6 AND centroid < 0.5 m → `objects.json` + per-object `.ply`
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
