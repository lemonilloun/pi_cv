"""Does the accelerometer actually recover travelled distance, and is the
error a constant bias we can estimate away?

Analyses one known-distance run: the cart starts still, is pushed a
tape-measured distance, and ends still. Those two boundary conditions make
the accelerometer bias observable WITHOUT any visual input — the true end
velocity is zero by construction, so whatever velocity double integration
reports there is pure accumulated error, and dividing it by the motion
duration gives the constant horizontal acceleration that produced it.

That is the same quantity `ekf_localization.Ekf2DVio` estimates online from
visual fixes. Measuring it in isolation answers two questions the online
estimate cannot: whether it is converging to something physically real, and
whether the residual AFTER removing it is small enough that displacement is
worth using at all. If a constant bias explains the whole error, distance
is recoverable; if a large error survives, the noise is time-varying and no
bias estimator will fix it.

Pure numpy, no I/O — `scripts/analyze_imu_distance.py` is the CLI over it
and `imu_calibration.py` calls it live between runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunAnalysis:
    ok: bool
    reason: str = ""
    samples: int = 0
    duration_s: float = 0.0
    rate_hz: float = 0.0
    rejected_samples: int = 0
    rejected_frac: float = 0.0
    static_head_s: float = 0.0
    static_tail_s: float = 0.0
    motion_s: float = 0.0
    gravity_mag: float = 0.0
    gyro_bias_dps: list[float] = field(default_factory=list)
    noise_ms2: float = 0.0
    leak_ms2: float = 0.0
    tilt_error_deg: float = 0.0
    raw_end_speed_ms: float = 0.0
    raw_distance_m: float = 0.0
    bias_ms2: list[float] = field(default_factory=list)
    bias_mag_ms2: float = 0.0
    corrected_distance_m: float = 0.0
    truth_m: float | None = None
    error_m: float | None = None
    error_frac: float | None = None
    verdict: str = ""

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def rolling_static_mask(
    t: np.ndarray, gyro: np.ndarray, accel: np.ndarray, window_s: float = 0.4
) -> np.ndarray:
    """Stationary = the accelerometer components and the gyro are both as
    quiet as this log's own noise floor.

    Two decisions here were made from measurements, not intuition:

    DEVIATION FROM THE LOG'S OWN GRAVITY VECTOR, not magnitude and not
    short-window variance. Magnitude fails because gravity dominates the
    vector: |(0.3, 0, 9.81)| = 9.814 against 9.810 at rest, a 0.04% change
    that disappears into the noise — verified on synthetic runs, where a
    magnitude test marked 100% of a 3 m push as still. Short-window
    variance fails too, for a subtler reason: a smooth push is nearly
    constant inside a 0.4 s window, so its variance is also near zero
    (measured: it caught 3% of a real push). What does separate them is
    the LEVEL — how far the accelerometer sits from the log's median
    (gravity) vector, which is ~0.3 m/s² while pushing and pure noise
    while parked. A tilted-but-still sensor stays correctly classified as
    still, since the tilt is baked into that median.

    ADAPTIVE thresholds, not fixed ones. Measured on this rig's real logs,
    the quietest windows sit at ~0.17 m/s² per-axis and ~0.039 rad/s — both
    ABOVE the fixed thresholds that looked reasonable a priori, so hard
    constants silently classified everything as motion. Scaling from each
    log's own 5th percentile adapts to the sensor and to how the run was
    recorded.

    This can only ever mark the ENDS reliably: a cart rolling at constant
    speed reads much like a stationary one (measured: 82% of samples during
    forward motion pass a stillness test), which is why an inertial ZUPT is
    unusable here and why the protocol leans on operator-enforced stops
    instead — see `analyze_run`'s head_s/tail_s, which let a caller that
    KNOWS when it told the operator to stop skip this heuristic entirely.
    """
    n = len(t)
    dt_med = float(np.median(np.diff(t))) if n > 1 else 0.01
    half = max(1, int(window_s / max(dt_med, 1e-4) / 2))
    gmag = np.linalg.norm(gyro, axis=1)
    dev = np.linalg.norm(accel - np.median(accel, axis=0), axis=1)
    a_dev = np.empty(n)
    g_mean = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        a_dev[i] = dev[lo:hi].mean()
        g_mean[i] = gmag[lo:hi].mean()
    a_thr = max(2.5 * float(np.percentile(a_dev, 5)), 0.05)
    g_thr = max(2.5 * float(np.percentile(g_mean, 5)), 0.02)
    return (a_dev < a_thr) & (g_mean < g_thr)


def reject_corrupt(
    t: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    max_rate_dps: float = 300.0,
    accel_tol_ms2: float = 3.0,
) -> np.ndarray:
    """Mask of samples that are physically possible for a pushed cart.

    The SHTP-over-UART link is lossy (docs/imu_shtp_uart.md) and corrupts
    individual samples. Measured on this rig's own logs: 2.4-4.1% of gyro
    samples in today's runs exceed 500 deg/s, peaking at 1948 deg/s, on a
    cart that physically cannot spin faster than a person can turn it —
    and they appear as isolated single samples between clean neighbours,
    which is packet corruption, not motion.

    One such sample is not a small error. At 30 Hz, a spurious 1890 deg/s
    reading rotates the integrated attitude by 63 degrees instantly, which
    tips the gravity vector into the horizontal plane and injects several
    m/s² of phantom acceleration. Measured effect on run_2 of
    cal_20260731_143730: dropping them took the integrated distance from
    40.4 m to 27.2 m, and combining that with a median gravity reference
    and bias correction brought it to 6.2 m (from 165 m).
    """
    ok = np.ones(len(t), dtype=bool)
    ok &= np.linalg.norm(gyro, axis=1) <= math.radians(max_rate_dps)
    mag = np.linalg.norm(accel, axis=1)
    ok &= np.abs(mag - float(np.median(mag))) <= accel_tol_ms2
    return ok


def _integrate(
    t: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    gyro_bias: np.ndarray,
    up: np.ndarray,
    g_mag: float,
    accel_bias_world: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Strapdown integration in a gravity-aligned world frame.

    The frame is defined by the initial static gravity vector, so gravity
    cancels EXACTLY at t=0 by construction — any residual horizontal
    acceleration afterwards is attitude error accumulating through the
    gyro, not a mis-set reference. Yaw is unobservable from gravity alone
    and left arbitrary; harmless here, since the quantity of interest is
    the LENGTH of the path, which no yaw choice changes.
    """
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ up)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e0 = seed - float(seed @ up) * up
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(up, e0)
    R = np.stack([e0, e1, up])  # world_from_body at t=0

    g_world = np.array([0.0, 0.0, g_mag])
    bias = np.zeros(3) if accel_bias_world is None else np.asarray(accel_bias_world)
    vel = np.zeros(3)
    pos = np.zeros(3)
    traj = np.zeros((len(t), 3))
    vels = np.zeros((len(t), 3))

    for k in range(1, len(t)):
        dt = float(t[k] - t[k - 1])
        if not (0.0 < dt < 0.5):
            traj[k], vels[k] = pos, vel
            continue
        omega = gyro[k] - gyro_bias
        ang = float(np.linalg.norm(omega)) * dt
        if ang > 1e-9:
            axis = omega / np.linalg.norm(omega)
            K = np.array(
                [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
            )
            R = R @ (np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K))
        lin = R @ accel[k] - g_world - bias
        vel = vel + lin * dt
        pos = pos + vel * dt
        traj[k], vels[k] = pos, vel
    return traj, vels


def analyze_run(
    t: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    truth_m: float | None = None,
    min_static_s: float = 1.0,
    accept_frac: float = 0.15,
    head_s: float | None = None,
    tail_s: float | None = None,
    reference_up: list[float] | None = None,
    use_gyro: bool = False,
) -> RunAnalysis:
    """`head_s`/`tail_s`: known-still durations at each end. The guided
    wizard passes its own phase timings here, which is strictly better than
    re-deriving them — it is the thing that TOLD the operator to stop, so it
    knows exactly when the still phases were, while the detector has to
    infer it from a signal that barely distinguishes rolling from parked.

    `use_gyro` defaults to FALSE, which is not a simplification but a
    measurement. On this rig the gyro accumulates 61 degrees of phantom
    rotation over 60 seconds of a provably STATIONARY sensor (~1 deg/s),
    and subtracting a constant bias does not reduce it at all (61.1 ->
    61.1 deg) — it is a random walk driven by residual corruption on the
    lossy SHTP-UART link, not an offset. Feeding that into attitude
    integration tilts the gravity vector by ~12 degrees over a 12 s run,
    injecting ~2 m/s2 of phantom horizontal acceleration and turning a 3 m
    push into 90+ metres. Holding attitude fixed at the still head's
    gravity frame is strictly better for a STRAIGHT push, where the true
    attitude barely changes anyway: the same run then integrates to 6.2 m
    instead of 89 m. Turn it on only once the link is fixed, or for a run
    that genuinely rotates and whose gyro you trust."""
    t = np.asarray(t, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    accel = np.asarray(accel, dtype=np.float64)
    if len(t) < 50:
        return RunAnalysis(ok=False, reason=f"only {len(t)} samples", samples=len(t))

    duration = float(t[-1] - t[0])
    res = RunAnalysis(
        ok=False,
        samples=len(t),
        duration_s=round(duration, 2),
        rate_hz=round((len(t) - 1) / max(duration, 1e-6), 1),
        truth_m=truth_m,
    )

    # Drop link-corrupted samples before anything else touches them — a
    # single bad gyro reading is worth tens of metres of integrated error.
    clean = reject_corrupt(t, gyro, accel)
    res.rejected_samples = int((~clean).sum())
    res.rejected_frac = round(float((~clean).mean()), 4)
    if res.rejected_frac > 0.25:
        res.reason = (
            f"{100 * res.rejected_frac:.0f}% of samples are physically impossible — "
            f"the IMU link is too corrupt to measure anything"
        )
        return res
    t, gyro, accel = t[clean], gyro[clean], accel[clean]

    if head_s is not None and tail_s is not None:
        head = slice(0, int(np.searchsorted(t, t[0] + head_s)))
        tail = slice(int(np.searchsorted(t, t[-1] - tail_s)), len(t))
        if head.stop <= 5 or tail.start >= len(t) - 5:
            res.reason = "declared still phases are shorter than the sample rate allows"
            return res
        # Declaring the phase boundaries skips motion DETECTION, so an
        # operator who forgot to push would otherwise sail through: the run
        # would report ~0 m against a 3 m truth and the summary would call
        # that a real finding ("bias does not explain the error — heading
        # only"), when in fact nothing was measured. Confirm the push phase
        # actually contains motion, against the still head's own noise.
        push = accel[head.stop : tail.start]
        rest_ref = np.median(accel[head], axis=0)
        rest_noise = float(np.linalg.norm(accel[head] - rest_ref, axis=1).mean())
        push_dev = float(np.linalg.norm(push - rest_ref, axis=1).mean())
        if push_dev < max(3.0 * rest_noise, 0.05):
            res.reason = (
                f"no motion during the push phase (deviation {push_dev:.3f} m/s² vs "
                f"noise {rest_noise:.3f}) — was the cart actually pushed?"
            )
            return res
    else:
        static = rolling_static_mask(t, gyro, accel)
        moving = np.flatnonzero(~static)
        if moving.size == 0:
            res.reason = "no motion detected — the cart never moved"
            return res
        head = slice(0, int(moving[0]))
        tail = slice(int(moving[-1]) + 1, len(t))
    res.static_head_s = round(float(t[head.stop - 1] - t[0]) if head.stop > 0 else 0.0, 2)
    res.static_tail_s = round(float(t[-1] - t[tail.start]) if tail.start < len(t) else 0.0, 2)
    res.motion_s = round(duration - res.static_head_s - res.static_tail_s, 2)

    if res.static_head_s < min_static_s or res.static_tail_s < min_static_s:
        # Not a soft warning: without a still head there is no gravity
        # reference, and without a still tail the end velocity is not known
        # to be zero — both boundary conditions the method rests on.
        res.reason = (
            f"need >={min_static_s:.0f}s of stillness at BOTH ends "
            f"(got {res.static_head_s:.1f}s / {res.static_tail_s:.1f}s)"
        )
        return res
    if res.motion_s <= 0.5:
        res.reason = "motion segment too short to fit a bias"
        return res

    # Median, not mean: one corrupt sample that survives the gate still
    # drags a mean badly. Measured on cal_20260731_143730/static.csv, a
    # single bad row moved |g| from 9.879 (median) to 9.452 (mean) — a 4%
    # scale error injected into every metre that follows.
    g_ref = np.median(accel[head], axis=0)
    gyro_bias = np.median(gyro[head], axis=0)
    g_mag = float(np.linalg.norm(g_ref))
    up = g_ref / g_mag
    res.gravity_mag = round(g_mag, 4)
    res.gyro_bias_dps = [round(float(v), 4) for v in np.degrees(gyro_bias)]

    # Noise floor: spread of the still head around its own mean. This is
    # NOT the gravity leak — measuring the residual against a reference
    # derived from the same samples is near-zero by construction.
    res.noise_ms2 = round(float(np.linalg.norm(accel[head] - g_ref, axis=1).mean()), 4)

    # The leak that actually breaks live dead-reckoning is the angle
    # between this run's true gravity and the STORED calibration vector the
    # live pipeline integrates against (imu_calibrate.py's reference). Only
    # measurable when that reference is supplied; without it, this run says
    # nothing about the live system's tilt error, so report zero rather
    # than a self-referential number that always looks perfect.
    if reference_up is not None:
        ref = np.asarray(reference_up, dtype=np.float64)
        ref = ref / max(float(np.linalg.norm(ref)), 1e-9)
        # abs(): the stored reference and the raw accelerometer vector use
        # opposite sign conventions on this rig (accel reads -9.8 on the
        # gravity axis while up_imu_reference is +0.999), so a signed
        # comparison reports a 2.7 deg tilt as 177 deg. What matters is the
        # angle between the AXES, not which end of them each side names up.
        cos = abs(float(np.clip(up @ ref, -1.0, 1.0)))
        res.tilt_error_deg = round(math.degrees(math.acos(cos)), 2)
        res.leak_ms2 = round(g_mag * math.sin(math.radians(res.tilt_error_deg)), 4)

    end_i = tail.start if tail.start < len(t) else len(t) - 1
    gyro_used = gyro if use_gyro else np.zeros_like(gyro)
    gyro_bias_used = gyro_bias if use_gyro else np.zeros(3)
    _, vels = _integrate(t, gyro_used, accel, gyro_bias_used, up, g_mag)
    res.raw_end_speed_ms = round(float(np.linalg.norm(vels[end_i])), 3)
    traj, _ = _integrate(t, gyro_used, accel, gyro_bias_used, up, g_mag)
    res.raw_distance_m = round(float(np.linalg.norm(traj[end_i][:2])), 2)

    bias_world = vels[end_i] / res.motion_s
    bias_world[2] = 0.0  # the cart drives in the horizontal plane
    res.bias_ms2 = [round(float(v), 4) for v in bias_world[:2]]
    res.bias_mag_ms2 = round(float(np.linalg.norm(bias_world)), 4)

    traj_c, vels_c = _integrate(
        t, gyro_used, accel, gyro_bias_used, up, g_mag, accel_bias_world=bias_world
    )
    res.corrected_distance_m = round(float(np.linalg.norm(traj_c[end_i][:2])), 3)
    res.ok = True

    if truth_m:
        err = res.corrected_distance_m - truth_m
        res.error_m = round(err, 3)
        res.error_frac = round(err / truth_m, 4)
        if abs(res.error_frac) < accept_frac:
            res.verdict = "constant bias explains the error — displacement IS recoverable"
        else:
            res.verdict = "residual is time-varying — bias estimation alone will not fix it"
    return res


def summarize(runs: list[RunAnalysis], accept_frac: float = 0.15) -> dict:
    """Aggregate several runs. One run cannot separate a systematic bias
    from a lucky fit; the SPREAD across repeats is what says whether the
    bias is a stable property of the sensor (usable as a prior) or just
    whatever that particular push happened to produce."""
    good = [r for r in runs if r.ok]
    if not good:
        return {"ok": False, "reason": "no usable runs", "runs": len(runs)}

    bias = np.array([r.bias_ms2 for r in good], dtype=np.float64)
    mags = np.array([r.bias_mag_ms2 for r in good])
    out: dict = {
        "ok": True,
        "runs_total": len(runs),
        "runs_usable": len(good),
        "bias_mean_ms2": [round(float(v), 4) for v in bias.mean(axis=0)],
        "bias_std_ms2": [round(float(v), 4) for v in bias.std(axis=0)],
        "bias_mag_mean_ms2": round(float(mags.mean()), 4),
        "noise_mean_ms2": round(float(np.mean([r.noise_ms2 for r in good])), 4),
        "leak_mean_ms2": round(float(np.mean([r.leak_ms2 for r in good])), 4),
        "tilt_error_mean_deg": round(float(np.mean([r.tilt_error_deg for r in good])), 2),
    }
    scored = [r for r in good if r.error_frac is not None]
    if scored:
        errs = np.array([abs(r.error_frac) for r in scored])
        out["error_frac_mean"] = round(float(errs.mean()), 4)
        out["error_frac_worst"] = round(float(errs.max()), 4)
        # Stability matters as much as accuracy: a bias that changes run to
        # run is not something a slowly-adapting online estimator can track.
        stable = float(mags.std()) < 0.25 * max(float(mags.mean()), 1e-6)
        out["bias_stable"] = stable
        if errs.mean() < accept_frac and stable:
            out["verdict"] = "USE_DISPLACEMENT"
            out["verdict_text"] = (
                "A stable constant bias explains the error. Ekf2DVio has something "
                "real to converge to — enable --imu-displacement and seed it with "
                "bias_mean_ms2."
            )
        elif errs.mean() < accept_frac:
            out["verdict"] = "USE_WITH_ONLINE_ESTIMATION"
            out["verdict_text"] = (
                "Distance is recoverable but the bias drifts between runs, so it "
                "cannot be baked in as a constant — keep estimating it online."
            )
        else:
            out["verdict"] = "HEADING_ONLY"
            out["verdict_text"] = (
                "A constant bias does NOT explain the error. Keep the heading-only "
                "filter for position; the gyro is unaffected and still gives "
                "direction and turns."
            )
    return out
