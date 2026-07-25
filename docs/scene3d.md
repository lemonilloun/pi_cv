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
2. `poses` — pycolmap: SIFT → matching → incremental mapping (OPENCV camera
   model with the calibrated K + distortion), then **scale alignment**:
   per-frame median(DAv2 depth / COLMAP sparse depth) → global median with
   MAD rejection → metric `poses.json`.
   **Matching mode** (`poses.matching_mode`, default `auto`) is the single
   biggest quality lever. Sequential matching only compares frames that are
   close *in time*, so walking a loop and returning never gets matched
   across the loop — the trajectory drifts and splits into separate
   sub-reconstructions (the classic "one side of the room reconstructs
   great, the other side smears into the far wall or vanishes"). Exhaustive
   matching compares every pair, so any revisit closes the loop; it's O(N²)
   but cheap for a few hundred frames. `auto` = exhaustive up to
   `max_exhaustive_frames` (600), else `sequential+loop`. Options:
   `exhaustive` | `sequential` | `sequential+loop` (needs a COLMAP vocab
   tree at `poses.vocab_tree_path`; falls back to plain sequential with a
   warning if absent). COLMAP can still return >1 sub-reconstruction — the
   step keeps the largest but reports all sizes in `sub_reconstructions` and
   **warns loudly** when frames were dropped, instead of silently hiding the
   unstitched part of the room.
   COLMAP runs **single-threaded** (`poses.colmap_num_threads`, default 1):
   torch and pycolmap each ship their own `libomp.dylib`, the server sets
   `KMP_DUPLICATE_LIB_OK=TRUE` so both load, and multi-threaded bundle
   adjustment then deadlocks against torch's OpenMP runtime (the process
   parks at ~0% CPU mid-`poses`, no output). Single-threaded is slower but
   doesn't hang; only raise it in an environment with a single shared libomp.
   **The camera bundle adjustment converges on is saved** to
   `derived/intrinsics_refined.json` and embedded in `poses.json`, and every
   later step reads it via `SceneSession.active_intrinsics()`. This matters
   more than it sounds: when the session has no calibration, COLMAP is
   allowed to refine the focal length, so the poses are expressed in terms of
   the *refined* camera. Unprojecting depth with the original guess then
   reconstructs a different camera than the poses describe. On
   session_20260724_144728 the gap was fx 622 (the old 102° fallback) versus
   fx 989 (what BA measured, HFOV 75.7° — DA3 independently estimated 74.4°),
   and the mesh came out 12 × 15 m for a room the camera crossed in 2 × 4 m.
   The step reports `hfov_deg` / `hfov_deg_assumed` and warns when BA had to
   move the focal length more than 10%.
3. `tsdf` — depth hygiene, then Open3D ScalableTSDFVolume (voxel 2 cm) →
   `room_mesh.ply` + carved `floor_plan.png` + `occupancy.npz`:
   - **filtering** (`depth_filter.py`) writes `derived/depth_filtered/` once,
     shared with the objects step so mesh and objects come from the same
     geometry. Two filters: *flying pixels* (surfaces seen too edge-on to be
     real — these are the radial streaks the old plan was made of) and
     *multi-view consistency* (a point must be confirmed by ≥ 2 neighbouring
     keyframes' own depth). Measured on a real session: 84.9% of pixels kept,
     1.9% flying, 12.6% inconsistent.
   - **adaptive truncation** replaces the fixed 5 m cutoff with the 98th
     percentile of surviving depth (4.19 m on that session). A 5 m cutoff
     around a 3.9 m camera path permits a ~14 m mesh by construction.
   - **mesh cleanup**: components below `min_component_triangles` (800) are
     dropped, then optional decimation to `decimate_triangles`.
   - **gravity** from a RANSAC floor-plane fit (`occupancy.py`), not the
     average camera-down axis — so a tilted camera no longer tilts the plan.
     Reported as `gravity_source`: `imu` > `floor_fit` > `camera_average`.
   - **the plan is carved, not histogrammed**: each depth ray marks the space
     between camera and surface FREE, the surface OCCUPIED, beyond it
     UNKNOWN. That is what produces walls and traversable floor instead of a
     density cloud, and it's the form a navigation consumer actually wants.
4. `objects` — per keyframe: mask + **filtered** depth + pose → world points →
   DBSCAN; association into an object bank: Pi `track_id` prior (now
   geometry-checked, because the Pi reuses ids), same-class gate, and a
   centroid gate that **grows with the object** (a 2 m sofa's two views are
   further apart than any fixed 0.5 m threshold). Boxes are **yaw-only,
   gravity-aligned** — free 3-DoF PCA on a partially-seen object was how a
   bed came out 0.09 m thick. Each object carries a `size_verdict`
   (`ok`/`too_large`/`too_small`) from a per-class plausibility table;
   implausible ones are flagged, not deleted, and skipped when building graph
   edges. → `objects.json` + per-object `.ply`.
   Two safety caps keep one bad detection from stalling the whole step for
   minutes: a mask covering more than `max_mask_area_frac` (0.35) of the
   frame is skipped outright (almost always a mis-segmented wall/floor,
   not a bounded object), and any point cloud is randomly subsampled to
   `dbscan_max_points` (8000) before clustering — DBSCAN's cost isn't
   linear, so an oversized cloud can otherwise turn one detection into a
   multi-minute stall that looks identical to a hang from the outside.
5. `graph` — CLIP text encoder (**RN50x4/openai — must match the Pi hef**)
   vs ~70-word indoor vocabulary on the averaged Pi CLIP embeddings, COCO
   vote as prior; `near` edges now measure the gap between OBB **surfaces**
   rather than between centroids (a lamp touching a 2 m sofa is 1 m from its
   centre — the old rule produced 4 edges for 10 objects), `on` uses the
   box's vertical axis → `scene_graph.json` + `floor_plan_labeled.png`

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
│                            # optional: gravity [x,y,z] in the CAMERA frame
└── derived/                 # everything the Mac pipeline produces
    ├── depth/  depth_filtered/  colmap/  poses.json  scale_report.json
    ├── intrinsics_refined.json  # the camera COLMAP's BA converged on
    ├── room_mesh.ply  floor_plan.png  floor_plan_labeled.png
    ├── occupancy.npz        # grid (0 unknown / 1 free / 2 occupied) + walls
    ├── plan_frame.json      # up, axes, origin, resolution, gravity_source
    ├── objects.json  objects_pcd/  scene_graph.json
    └── pipeline_state.json  # per-step status for the panel
```

`masks.png` is at rgb.jpg resolution (nearest-neighbor upscale from the
640×640 inference); `instance_id` links pixels to `meta.json` entries;
`track_id` is session-wide. Everything the Mac needs is this directory.

`meta.json`'s **`gravity`** field is the IMU seam. It is absent today and
absent-by-design in every session recorded before the sensor is fitted; when
present it is the accelerometer vector in the camera frame, and
`SceneSession.imu_up_vector()` rotates it into world space for
`occupancy.estimate_gravity()`, which prefers it over the floor fit. Fitting
the IMU means setting `SceneRecorder.gravity_source` to an object with a
`.read()` — no pipeline change.

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
# useful flags: --fps 3 --no-clip --no-preview
# keyframe selection: --keyframe-shift-px 12 --keyframe-min-interval 0.2
#                     --keyframe-max-interval 3.0 --blur-threshold 100
# --offline reverts to local-disk recording (old behavior)
```

The server host comes from `.env` (`PI_CV_SERVER_HOST=vedro.local`). If
Wi-Fi drops mid-walk, affected keyframes are written locally under
`data/scene_sessions/<id>/`; sync them to the Mac with a `git pull` on the
current branch (do **not** rsync/scp by hand).

### How to record one room correctly

The reconstruction is only as good as the camera path. COLMAP recovers pose
from **parallax** (the apparent shift of features as the camera *translates*)
and stitches the room by **matching the same place across time**. Every rule
below serves one of those two needs.

1. **Translate, don't pivot — especially in corners.** Pure rotation in place
   gives zero parallax and poses degenerate; corners are where this bites
   most. Take corners as a smooth *arc* (move forward while turning), never a
   spin-on-the-spot. On the robot: prefer curved paths over rotate-then-go.
2. **Close the loop, deliberately.** Walk the room as a loop and **return to
   the exact starting spot facing the same way**, holding there a second or
   two. That gives the matcher near-identical start/end frames to lock onto —
   this is what stops "the far side of the room drifting into the opposite
   wall." With `matching_mode=auto`/`exhaustive` the pipeline will find that
   revisit; your job is to make sure it physically exists in the footage.
3. **Overlap generously.** Consecutive keyframes must share a lot of view —
   move slowly enough that neighbouring frames look ~70%+ the same. Rushing a
   stretch breaks the chain and splits the reconstruction.
4. **Slow down through turns and add frames there.** Turns are the weak point,
   so give them the most coverage — glance around while arcing so several
   keyframes see each new heading, rather than one blurry frame swinging past.
5. **Vary your heading while moving** (glance left/right as you go). The
   localization index can only report a facing direction it actually
   recorded; a straight-ahead-only walk leaves heading ambiguous everywhere.
6. **Tilt the camera up 15–25°.** On a low robot chassis this is the single
   highest-value change available, and it is worth more than any model swap:
   a floor-level camera fills the frame with floor and skirting boards, which
   is both featureless for SIFT and far outside the distribution metric depth
   was trained on (human eye height, furniture in view). Tilting up puts the
   room and its furniture in frame instead. Height being fixed on the mount is
   fine — tilt is the degree of freedom that matters. Roll should stay near
   zero, but pitch no longer has to: the pipeline now fits gravity from the
   floor plane (and will use the IMU when fitted) instead of assuming the
   camera is level.
7. **Fight motion blur.** The robot's jerky moves + low fps = smeared frames
   that SIFT can't match on bare walls. Move as smoothly as the chassis
   allows; if frames come out blurry, slow the traverse rather than raising
   fps.
8. **Give SIFT something to match.** Blank walls have no features. It's fine —
   even helpful — for furniture, posters, clutter to be in frame; a room
   that's all bare white walls is the hardest case.
9. **Don't touch focus after calibration.** LensPosition is baked into the
   intrinsics; changing it invalidates the calibrated K.

After the run, check the `poses` result: `registered_frac` should be high
(≥ ~0.8) and `sub_reconstructions` should be `[N]` — a single number. If you
see `[80, 54]` (more than one piece), the loop didn't close: re-record with a
cleaner return to the start, and confirm `matching_mode` is `auto`/`exhaustive`.

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
