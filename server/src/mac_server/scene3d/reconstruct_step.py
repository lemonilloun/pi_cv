"""Reconstruction step: replaces depth_step.py + poses_step.py with one
call to VGGT (facebook/VGGT-1B, feed-forward, run on the GPU service on
pdfserver over an SSH tunnel — see gpu_service/vggt_backend.py).

Why: COLMAP's incremental SfM could silently drop a contiguous chunk of
frames under low texture/close furniture (docs/scene3d.md), and
reconciling two independently-scaled sources (COLMAP up-to-scale poses vs
Depth Anything V2 metric depth) produced a 220% scale disagreement on a
real session. VGGT solves pose and depth *jointly* in one forward pass, so
there is no incremental registration to fail midway, and only one
(consistently up-to-scale) geometry to convert to metres — not two to
reconcile.

Metric scale: the camera's known height above the floor
(scene3d.reconstruction.camera_height_m, kept in sync with
mapping.camera_height_m — same physical rig, duplicated here because the
scene3d pipeline only ever receives the `scene3d` config section, see
server.py:_build_scene3d). Fit the floor plane on the raw (unscaled) point
cloud with the exact same "lowest slab by coarse up, then RANSAC" approach
occupancy.estimate_gravity uses, then scale = known_height /
median(per-frame camera-to-floor distance). IMU distance is kept as an
independent diagnostic (unchanged logic from poses_step.py).

The old COLMAP+DAv2 backend is NOT deleted — set
scene3d.reconstruction.backend to "colmap_dav2" to fall back to it; see
_run_colmap_dav2_fallback below.

Outputs (same contract the rest of the pipeline already expects):
    derived/depth/<idx>.npy         scaled to metres
    derived/poses.json              {"world_from_cam": ..., "scale": ..., "intrinsics": ...}
    derived/scale_report.json       scale estimates + diagnostics
    derived/intrinsics_refined.json VGGT's own estimated camera (median across frames)
"""

from __future__ import annotations

import json
import logging
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from mac_server.scene3d.session_io import SceneSession


logger = logging.getLogger(__name__)

_MIN_FRAMES_BEFORE_GIVING_UP = 20


class GpuServiceError(RuntimeError):
    """gpu_service returned a non-OOM error (bad request, unreachable,
    tunnel down, etc). Not retried — surfaced to the pipeline as-is."""


class GpuOutOfMemory(RuntimeError):
    """gpu_service returned 507: the requested frame batch didn't fit on
    GPU1. Triggers the subsampling fallback in _reconstruct_with_fallback."""


def run_reconstruct_step(
    session: SceneSession,
    config: dict[str, Any],
    repo_root: Path,
    progress: Callable[[str], None] = lambda msg: None,
) -> dict[str, Any]:
    recon_cfg = config.get("reconstruction", {})
    backend = str(recon_cfg.get("backend", "vggt")).lower()
    if backend == "colmap_dav2":
        return _run_colmap_dav2_fallback(session, config, repo_root, progress)
    return _run_vggt(session, recon_cfg, repo_root, progress)


def _run_colmap_dav2_fallback(
    session: SceneSession,
    config: dict[str, Any],
    repo_root: Path,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    """The original two-step pipeline, kept importable and selectable via
    scene3d.reconstruction.backend="colmap_dav2" — not the default, not
    getting further engineering investment, but not deleted either."""
    from mac_server.scene3d.depth_step import run_depth_step
    from mac_server.scene3d.poses_step import run_poses_step

    depth_report = run_depth_step(session, config, repo_root, progress)
    poses_report = run_poses_step(session, config, repo_root, progress)
    return {"backend": "colmap_dav2", "depth": depth_report, "poses": poses_report}


# ------------------------------------------------------------------ VGGT


def _run_vggt(
    session: SceneSession,
    recon_cfg: dict[str, Any],
    repo_root: Path,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    import numpy as np

    frames = session.keyframes()
    if not frames:
        raise RuntimeError(f"Session {session.session_id} has no keyframes")
    frame_indices = [kf.index for kf in frames]

    rsync_host = str(recon_cfg.get("rsync_host", "pdfserver"))
    remote_sessions_dir = str(recon_cfg.get("remote_sessions_dir", "~/cv_research/sessions"))
    gpu_service_url = str(recon_cfg.get("gpu_service_url", "http://127.0.0.1:8700"))
    timeout_s = float(recon_cfg.get("request_timeout_s", 1800))
    camera_height_m = float(recon_cfg.get("camera_height_m", 0.3))

    progress(f"vggt: syncing {len(frame_indices)} frames to {rsync_host}")
    _sync_frames_to_gpu(session, rsync_host, remote_sessions_dir)

    result, used_indices = _reconstruct_with_fallback(
        gpu_service_url, session.session_id, frame_indices, timeout_s, progress
    )

    progress("vggt: syncing raw depth back")
    staging_dir = session.derived / "_gpu_raw"
    _sync_depth_from_gpu(session, rsync_host, remote_sessions_dir, staging_dir)

    world_from_cam_unscaled = {
        int(k): np.asarray(v, dtype=np.float64) for k, v in result["world_from_cam"].items()
    }
    intrinsic_by_frame = {
        int(k): np.asarray(v, dtype=np.float64) for k, v in result["intrinsic"].items()
    }
    input_h, input_w = result["input_hw"]

    depth_by_frame: dict[int, Any] = {}
    for idx in used_indices:
        depth_path = staging_dir / f"{idx:06d}.npy"
        if depth_path.exists():
            depth_by_frame[idx] = np.load(depth_path)

    progress("vggt: fitting metric scale from camera height")
    imu_up = session.imu_up_vector(world_from_cam_unscaled)
    height_scale, height_diag = _camera_height_scale(
        world_from_cam_unscaled, depth_by_frame, intrinsic_by_frame, imu_up, camera_height_m
    )

    centers = {k: v[:3, 3] for k, v in world_from_cam_unscaled.items()}
    from mac_server.scene3d.poses_step import imu_scale_samples, median_scale

    imu_samples = imu_scale_samples(session.imu_segments(), centers)
    imu_scale = imu_iqr = None
    if len(imu_samples) >= 3:
        imu_scale, imu_iqr, _ = median_scale(imu_samples)

    baseline_cfg = None
    baseline_path = session.root / "baseline.json"
    if baseline_path.exists():
        try:
            baseline_cfg = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable baseline.json, ignoring: %s", exc)
    base_scale, base_diag = baseline_scale(centers, baseline_cfg)

    # `auto` prefers a tape-measured baseline, then the IMU, then the floor
    # fit. The order is not the historical one: at this rig's 13 cm camera
    # height the floor is a sliver at the frame edge and its plane fit is
    # ill-conditioned (measured on session_20260803_171339: 15% inliers), so
    # camera height is now the LAST resort rather than the first.
    #
    # This used to default to `known_baseline` with no fallback, on the
    # reasoning that a silent substitution is what let a 5-6x scale error
    # survive a whole round of testing. That reasoning was right about the
    # danger and wrong about the remedy: it turned every session recorded
    # without a baseline into a hard failure with no way forward. The danger
    # was never the fallback, it was that nothing announced it — so the
    # fallback is loud and recorded now instead of fatal.
    preference = str(recon_cfg.get("scale_source", "auto"))
    # Above this the floor plane fit is describing a real floor rather than
    # scraping a sliver at the frame edge. 15% was measured with the camera at
    # 13 cm looking level; 95.6% with it raised and tilted down.
    FLOOR_INLIER_TRUST = float(recon_cfg.get("floor_inlier_trust", 0.6))
    candidates = {
        "known_baseline": base_scale,
        "camera_height": height_scale,
        "imu": imu_scale,
    }
    if preference == "auto":
        # A tape measure always wins. Between the two estimates, prefer the one
        # whose own diagnostics say it is well conditioned, instead of a fixed
        # ranking — the right order depends on the rig, and the rig changes.
        #
        # Measured on session_20260807_171938, after the tripod was raised so
        # the floor came back into view:
        #     camera_height 2.07  (95.6% floor inliers, 48 frames, IQR 3.9%)
        #     imu          25.62  (IQR 6.1%, but 87.9 m of path in one room)
        # The old fixed order put IMU first and took the 12x-larger number.
        # The IMU's tight spread describes its repeatability, not its accuracy:
        # it double-integrates acceleration, so a gravity-subtraction error
        # grows quadratically and stays perfectly self-consistent while doing
        # it. The floor fit has an inlier fraction, which is an actual
        # statement about the data it fitted.
        order = ["known_baseline"]
        floor_ok = float(height_diag.get("floor_inlier_frac") or 0.0) >= FLOOR_INLIER_TRUST
        order += ["camera_height", "imu"] if floor_ok else ["imu", "camera_height"]
    elif preference in candidates:
        # An explicit choice is honoured strictly. Silently falling back to a
        # method the operator deliberately rejected is how the 5-6x scale
        # error survived a whole round of testing.
        order = [preference]
    else:
        raise RuntimeError(
            f"Unknown scene3d.reconstruction.scale_source {preference!r}; "
            f"expected one of auto/{'/'.join(candidates)}"
        )

    chosen = next((name for name in order if candidates[name] is not None), None)
    if chosen is None:
        raise RuntimeError(
            f"Scale alignment failed for scale_source={preference!r}. "
            f"baseline: {base_diag.get('reason', 'ok')}; "
            f"camera height: {height_diag.get('reason', 'ok')}; "
            f"IMU: {len(imu_samples)} usable intervals. "
            "For a known baseline, drive a tape-measured straight line and write "
            "<session>/baseline.json as {\"from_frame\": N, \"to_frame\": M, "
            "\"length_m\": 1.0}."
        )
    scale = candidates[chosen]
    logger.info("Using the %s scale: %.4f", chosen, scale)
    if chosen != "known_baseline":
        # Loud on purpose. Every metric number downstream — the mesh, the
        # floor plan, every position the navigation reports — is this factor
        # times something, and the two fallbacks are both known to be weak on
        # this rig. A one-line INFO would not be proportionate.
        logger.error(
            "SCALE IS PROVISIONAL: no baseline.json, so the %s estimate was used. "
            "Drive a tape-measured straight line, note the two keyframe indices, and "
            "run: python3 scripts/make_baseline.py %s --from N --to M --length-m 1.00 "
            "then re-run this step with --force.",
            chosen, session.session_id,
        )
    for name, value in candidates.items():
        if name != chosen and value is not None:
            logger.info("  cross-check %s scale: %.4f (%+.1f%%)",
                        name, value, 100.0 * (value - scale) / scale)

    disagreement = None
    if height_scale and imu_scale:
        disagreement = abs(height_scale - imu_scale) / max(height_scale, 1e-9)

    progress("vggt: writing scaled depth + poses")
    out_dir = session.depth_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    for idx, depth in depth_by_frame.items():
        scaled = (depth * scale).astype(np.float32)
        np.save(session.depth_path(idx), scaled)
        vis = np.clip(scaled / 8.0, 0, 1)
        vis = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(out_dir / f"{idx:06d}_vis.jpg"), cv2.resize(vis, (384, 216)))

    world_from_cam = {}
    for idx, wfc in world_from_cam_unscaled.items():
        if idx not in depth_by_frame:
            continue  # dropped by the OOM-subsampling fallback
        scaled_wfc = wfc.copy()
        scaled_wfc[:3, 3] *= scale
        world_from_cam[idx] = scaled_wfc

    median_intrinsic = np.median(
        np.stack([intrinsic_by_frame[i] for i in world_from_cam]), axis=0
    )
    refined_intrinsics = {
        "width": int(input_w),
        "height": int(input_h),
        "fx": float(median_intrinsic[0, 0]),
        "fy": float(median_intrinsic[1, 1]),
        "cx": float(median_intrinsic[0, 2]),
        "cy": float(median_intrinsic[1, 2]),
        "dist": [0.0, 0.0, 0.0, 0.0],
        "source": "vggt",
    }
    session.derived.mkdir(parents=True, exist_ok=True)
    session.refined_intrinsics_path().write_text(
        json.dumps(refined_intrinsics, indent=1), encoding="utf-8"
    )
    session.poses_path().write_text(
        json.dumps(
            {
                "world_from_cam": {
                    str(k): v.tolist() for k, v in sorted(world_from_cam.items())
                },
                "registered": len(world_from_cam),
                "total": len(frames),
                "scale": scale,
                "intrinsics": refined_intrinsics,
            }
        ),
        encoding="utf-8",
    )
    session.scale_report_path().write_text(
        json.dumps(
            {
                "global_scale": scale,
                "scale_source": chosen,
                "scale_source_requested": preference,
                # All three are always reported, whichever one was used. Two
                # methods disagreeing is the cheapest possible warning that
                # one of them is broken — and the previous round shipped a
                # 5-6x scale error precisely because nothing compared them.
                "scale_known_baseline": None if base_scale is None else round(base_scale, 4),
                "scale_camera_height": None if height_scale is None else round(height_scale, 4),
                "scale_imu": None if imu_scale is None else round(imu_scale, 4),
                # The IMU estimate's own spread. A tight cluster across many
                # keyframe intervals means the estimate is real; a wide one
                # means the intervals disagree and the median is arbitrary.
                # Without this the number looks equally confident either way.
                "scale_imu_iqr_frac": (
                    None if (imu_iqr is None or not imu_scale)
                    else round(imu_iqr / abs(imu_scale), 4)
                ),
                "baseline_diagnostics": base_diag,
                "camera_height_diagnostics": height_diag,
                "imu_intervals": len(imu_samples),
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    report: dict[str, Any] = {
        "backend": "vggt",
        "checkpoint": result.get("checkpoint"),
        "registered": len(world_from_cam),
        "total": len(frames),
        "frames_sent_to_gpu": len(frame_indices),
        "frames_used_after_oom_fallback": len(used_indices),
        "scale": round(scale, 4),
        "scale_source": chosen,
    }
    if chosen != "known_baseline":
        report["scale_warning"] = (
            f"provisional scale from {chosen} — no baseline.json in this session"
        )
    if imu_scale and imu_iqr and imu_iqr / abs(imu_scale) > 0.30:
        report["imu_scale_warning"] = (
            f"IMU scale intervals disagree by {imu_iqr / abs(imu_scale):.0%} of the "
            f"median — treat that estimate as weak"
        )
    if disagreement is not None and disagreement > 0.20:
        report["warning"] = (
            f"Camera-height scale and IMU scale disagree by {disagreement:.0%} "
            f"(height {height_scale:.3f} vs IMU {imu_scale:.3f}); using the {chosen} one."
        )
    if len(used_indices) < len(frame_indices):
        report["warning"] = (
            report.get("warning", "")
            + f" GPU OOM forced subsampling from {len(frame_indices)} to "
            f"{len(used_indices)} frames — coverage is reduced."
        ).strip()
    return report


def _sync_frames_to_gpu(session: SceneSession, rsync_host: str, remote_sessions_dir: str) -> None:
    """rsync just rgb.jpg per keyframe (VGGT doesn't need instance masks)
    to the GPU host, mirroring this session's own keyframes/<idx>/ layout
    so gpu_service can read it with zero path translation."""
    remote_root = f"{remote_sessions_dir}/{session.session_id}"
    subprocess.run(
        ["ssh", rsync_host, f"mkdir -p {remote_root}/keyframes"],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        [
            "rsync", "-a",
            "--include=*/", "--include=rgb.jpg", "--exclude=*",
            f"{session.keyframes_dir}/", f"{rsync_host}:{remote_root}/keyframes/",
        ],
        check=True, capture_output=True, text=True,
    )


def _sync_depth_from_gpu(
    session: SceneSession, rsync_host: str, remote_sessions_dir: str, staging_dir: Path
) -> None:
    remote_root = f"{remote_sessions_dir}/{session.session_id}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync", "-a",
            f"{rsync_host}:{remote_root}/derived_gpu/depth_raw/", f"{staging_dir}/",
        ],
        check=True, capture_output=True, text=True,
    )


def _call_reconstruct(
    gpu_service_url: str, session_id: str, frame_indices: list[int], timeout_s: float
) -> dict[str, Any]:
    body = json.dumps({"session_id": session_id, "frame_indices": frame_indices}).encode("utf-8")
    req = urllib.request.Request(
        f"{gpu_service_url}/reconstruct",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 507:
            raise GpuOutOfMemory(detail) from exc
        raise GpuServiceError(f"gpu_service /reconstruct failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise GpuServiceError(
            f"Could not reach gpu_service at {gpu_service_url} — is the SSH tunnel "
            f"(com.pi_cv.gpu-tunnel launchd agent) up? ({exc})"
        ) from exc


def _reconstruct_with_fallback(
    gpu_service_url: str,
    session_id: str,
    frame_indices: list[int],
    timeout_s: float,
    progress: Callable[[str], None],
) -> tuple[dict[str, Any], list[int]]:
    """Try the whole session in one VGGT pass first (empirically, this GPU
    fits ~300-400 frames — see the plan). On an actual OOM, subsample by
    half and retry rather than building full overlapping-window alignment,
    which is real algorithmic work not yet justified by anything we've hit
    in practice; every fallback is logged/reported, never silent."""
    attempt = list(frame_indices)
    while True:
        progress(f"vggt: reconstructing {len(attempt)} frames")
        try:
            return _call_reconstruct(gpu_service_url, session_id, attempt, timeout_s), attempt
        except GpuOutOfMemory:
            if len(attempt) <= _MIN_FRAMES_BEFORE_GIVING_UP:
                raise RuntimeError(
                    f"VGGT ran out of GPU memory even at {len(attempt)} frames — "
                    "check that nothing else is using GPU1 (nvidia-smi on pdfserver)."
                )
            attempt = attempt[::2]
            logger.warning(
                "VGGT OOM on GPU1 — retrying with %d frames (every-2nd subsample)",
                len(attempt),
            )


def baseline_scale(
    centers: dict[int, Any], baseline: dict[str, Any] | None
) -> tuple[float | None, dict[str, Any]]:
    """Metric scale from a tape-measured straight drive between two keyframes.

    `baseline` is `{"from_frame": int, "to_frame": int, "length_m": float}`,
    read from `<session>/baseline.json`.

    **Why this exists and why it is preferred.** The camera-height method
    (`_camera_height_scale`) measures the distance from the camera to a fitted
    floor plane. That is well conditioned when the camera looks down from
    ~30 cm and the floor fills a good part of the frame. On this rig the
    camera sits at 10-12 cm tilted UP, the floor occupies a sliver at the
    bottom edge viewed nearly edge-on, in the most distorted part of the lens
    — the plane fit is noisy and biased there. Worse, the parameter was left
    at 0.3 m against an actual 0.05-0.06 m for the whole previous round, so
    every map built was ~5-6x the wrong size and nothing flagged it.

    A driven baseline has none of those failure modes: it does not care about
    camera height, floor visibility, tilt or intrinsics. It is one direct
    measurement against a tape measure, and a wrong one is obvious rather
    than plausible.
    """
    import numpy as np

    if not baseline:
        return None, {"reason": "no baseline.json in the session"}
    try:
        a = int(baseline["from_frame"])
        b = int(baseline["to_frame"])
        length_m = float(baseline["length_m"])
    except (KeyError, TypeError, ValueError) as exc:
        return None, {"reason": f"malformed baseline.json: {exc}"}
    if length_m <= 0:
        return None, {"reason": f"baseline length must be positive, got {length_m}"}
    missing = [f for f in (a, b) if f not in centers]
    if missing:
        return None, {"reason": f"baseline frames {missing} were not registered"}

    chord = float(np.linalg.norm(np.asarray(centers[b]) - np.asarray(centers[a])))
    if chord < 1e-6:
        return None, {"reason": f"frames {a} and {b} reconstructed to the same point"}
    return length_m / chord, {
        "from_frame": a, "to_frame": b, "length_m": length_m,
        "reconstructed_chord": round(chord, 6),
    }


def _camera_height_scale(
    world_from_cam: dict[int, Any],
    depth_by_frame: dict[int, Any],
    intrinsic_by_frame: dict[int, Any],
    imu_up,
    camera_height_m: float,
    point_stride: int = 8,
    frame_stride: int = 3,
) -> tuple[float | None, dict[str, Any]]:
    """Metric scale from the camera's known real height above the floor.

    Mirrors occupancy.estimate_gravity's "lowest slab by coarse up, then
    RANSAC" floor detection (not reused directly — that function returns
    only the up vector, and this also needs the plane's offset to measure
    per-frame camera-to-floor distance).
    """
    import numpy as np

    from mac_server.scene3d.occupancy import camera_average_up, fit_plane_ransac
    from mac_server.scene3d.poses_step import median_scale

    frame_indices = sorted(world_from_cam)[::frame_stride]
    clouds = []
    for idx in frame_indices:
        depth = depth_by_frame.get(idx)
        k = intrinsic_by_frame.get(idx)
        if depth is None or k is None:
            continue
        h, w = depth.shape[:2]
        uu, vv = np.meshgrid(
            np.arange(0, w, point_stride), np.arange(0, h, point_stride)
        )
        z = depth[vv, uu]
        valid = z > 0
        if not np.any(valid):
            continue
        x = (uu[valid] - k[0, 2]) / k[0, 0] * z[valid]
        y = (vv[valid] - k[1, 2]) / k[1, 1] * z[valid]
        cam_pts = np.stack([x, y, z[valid]], axis=-1)
        wfc = world_from_cam[idx]
        clouds.append(cam_pts @ wfc[:3, :3].T + wfc[:3, 3])
    if not clouds:
        return None, {"reason": "no valid depth points to build a point cloud"}
    points = np.concatenate(clouds, axis=0)

    coarse_up = np.asarray(
        imu_up if imu_up is not None else camera_average_up(world_from_cam), dtype=np.float64
    )
    norm = np.linalg.norm(coarse_up)
    if norm < 1e-9:
        return None, {"reason": "coarse up vector degenerate"}
    coarse_up /= norm

    heights = points @ coarse_up
    cutoff = np.quantile(heights, 0.25)
    slab = points[heights <= cutoff]
    if len(slab) < 100:
        return None, {"reason": f"floor slab too small ({len(slab)} points)"}

    # fit_plane_ransac's threshold_m default (0.03) assumes real metres —
    # meaningless here, since `points` is still in VGGT's own unscaled
    # units (that's the whole point: scale isn't known yet). Derive a
    # threshold from the point cloud's own extent instead of the metric
    # default, or a decent fraction of the slab can end up "inlier" by
    # coincidence and blow up the refit's inlier count (see the SVD
    # full_matrices note in occupancy.py — the real bug that made this
    # step hang on a real session, not just an inaccurate fit).
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    threshold = max(extent * 0.003, 1e-6)
    normal, offset, inliers = fit_plane_ransac(slab, threshold_m=threshold)
    if normal @ coarse_up < 0:
        normal, offset = -normal, -offset
    inlier_frac = float(inliers.sum() / len(slab))
    if inlier_frac < 0.3:
        return None, {"reason": f"floor plane fit had too few inliers ({inlier_frac:.0%})"}

    per_frame_implied_scale = {}
    for idx in frame_indices:
        if idx not in world_from_cam:
            continue
        center = world_from_cam[idx][:3, 3]
        dist = abs(float(normal @ center + offset))
        if dist > 1e-6:
            per_frame_implied_scale[idx] = camera_height_m / dist

    if len(per_frame_implied_scale) < 3:
        return None, {"reason": "too few frames with a valid floor distance"}

    scale, iqr, rejected = median_scale(per_frame_implied_scale)
    return scale, {
        "iqr_over_median": round(iqr, 4),
        "rejected_frames": rejected,
        "frames_used": len(per_frame_implied_scale),
        "floor_inlier_frac": round(inlier_frac, 3),
    }
