"""Synthetic-ground-truth tests for the known-distance IMU analysis.

Every fixture here is generated from a KNOWN distance and a KNOWN injected
tilt, so the assertions check real recovery rather than self-consistency:
if the analysis reports 3 m for a run built to travel 3 m, the strapdown
integration, the gravity-frame construction and the zero-end-velocity bias
fit all have to be right simultaneously.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi_client.imu_distance_analysis import (  # noqa: E402
    analyze_run,
    rolling_static_mask,
    summarize,
)

RATE = 90.0
GRAVITY = 9.81


def make_run(
    truth_m: float,
    tilt_deg: float = 4.7,
    still_s: float = 4.0,
    move_s: float = 6.0,
    accel_noise: float = 0.02,
    seed: int = 0,
):
    """Still -> smooth push covering exactly `truth_m` -> still, observed by
    a sensor tilted `tilt_deg` from level (which leaks gravity into the
    horizontal plane exactly the way the real rig does)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / RATE
    n_still, n_move = int(still_s * RATE), int(move_s * RATE)
    t = np.arange(2 * n_still + n_move) * dt

    # A full sine of acceleration: net velocity change zero (starts and ends
    # at rest), net displacement positive. Scaled to land on truth_m.
    prof = np.sin(2 * math.pi * np.linspace(0, 1, n_move))
    disp = np.cumsum(np.cumsum(prof) * dt) * dt
    a_move = prof * (truth_m / disp[-1])
    a_fwd = np.concatenate([np.zeros(n_still), a_move, np.zeros(n_still)])

    th = math.radians(tilt_deg)
    rot = np.array(
        [[math.cos(th), 0.0, math.sin(th)], [0.0, 1.0, 0.0], [-math.sin(th), 0.0, math.cos(th)]]
    )
    accel = np.array([rot @ np.array([a, 0.0, GRAVITY]) for a in a_fwd])
    accel += rng.normal(0.0, accel_noise, accel.shape)
    gyro = rng.normal(0.0, 0.0015, accel.shape)
    return t, gyro, accel, still_s


class StaticMaskTests(unittest.TestCase):
    def test_horizontal_push_is_not_mistaken_for_stillness(self):
        """The regression this detector was rewritten for: gravity dominates
        the accelerometer magnitude, so a horizontal push leaves it almost
        unchanged and a magnitude-based test called the whole run still."""
        t, gyro, accel, still_s = make_run(3.0)
        mask = rolling_static_mask(t, gyro, accel)
        n_still = int(still_s * RATE)
        moving_middle = ~mask[n_still + 30 : -n_still - 30]
        self.assertGreater(moving_middle.mean(), 0.8)

    def test_tilted_but_stationary_reads_as_still(self):
        rng = np.random.default_rng(1)
        n = int(5 * RATE)
        t = np.arange(n) / RATE
        accel = np.tile([0.8, 0.0, 9.78], (n, 1)) + rng.normal(0, 0.02, (n, 3))
        gyro = rng.normal(0, 0.0015, (n, 3))
        self.assertGreater(rolling_static_mask(t, gyro, accel).mean(), 0.95)


class AnalyzeRunTests(unittest.TestCase):
    def test_recovers_known_distance_with_declared_phases(self):
        t, gyro, accel, still_s = make_run(3.0)
        res = analyze_run(t, gyro, accel, truth_m=3.0, head_s=still_s - 1, tail_s=still_s - 1)
        self.assertTrue(res.ok, res.reason)
        self.assertAlmostEqual(res.corrected_distance_m, 3.0, delta=0.45)
        self.assertLess(abs(res.error_frac), 0.15)
        self.assertIn("recoverable", res.verdict)

    def test_recovers_known_distance_by_auto_detection(self):
        t, gyro, accel, _ = make_run(3.0)
        res = analyze_run(t, gyro, accel, truth_m=3.0)
        self.assertTrue(res.ok, res.reason)
        self.assertAlmostEqual(res.corrected_distance_m, 3.0, delta=0.6)

    def test_measures_tilt_against_a_stored_reference(self):
        """The leak is a disagreement between this run's true gravity and
        the STORED calibration vector the live pipeline integrates against.
        A 4.7 deg disagreement must read as 9.81*sin(4.7 deg) = 0.80 m/s2 —
        the figure measured on the real rig."""
        t, gyro, accel, still_s = make_run(3.0, tilt_deg=4.7)
        res = analyze_run(
            t, gyro, accel, head_s=still_s - 1, tail_s=still_s - 1,
            reference_up=[0.0, 0.0, 1.0],
        )
        self.assertAlmostEqual(res.tilt_error_deg, 4.7, delta=0.6)
        self.assertAlmostEqual(res.leak_ms2, GRAVITY * math.sin(math.radians(4.7)), delta=0.1)

    def test_no_reference_reports_no_tilt_rather_than_a_flattering_zero(self):
        """Without a stored reference this run cannot say anything about the
        live system's tilt error. It must not manufacture one by measuring
        the still head against a mean taken from that same head — that is
        near-zero by construction and would look like a perfect result."""
        t, gyro, accel, still_s = make_run(3.0, tilt_deg=4.7)
        res = analyze_run(t, gyro, accel, head_s=still_s - 1, tail_s=still_s - 1)
        self.assertEqual(res.tilt_error_deg, 0.0)
        self.assertEqual(res.leak_ms2, 0.0)
        self.assertGreater(res.noise_ms2, 0.0)

    def test_self_referenced_gravity_does_not_diverge(self):
        """A finding worth pinning down, not just an assertion: because this
        analysis re-derives gravity from each run's own still head, it is
        immune to the 0.81 m/s2 leak that wrecks the live integrator. The
        implication is that re-referencing gravity at every stop is itself a
        fix for the live pipeline, independent of bias estimation."""
        t, gyro, accel, still_s = make_run(3.0, tilt_deg=4.7)
        res = analyze_run(t, gyro, accel, head_s=still_s - 1, tail_s=still_s - 1)
        self.assertLess(res.raw_distance_m, 6.0)
        self.assertLess(res.raw_end_speed_ms, 1.0)

    def test_stale_gravity_reference_blows_the_distance_up(self):
        """The failure mode the live pipeline actually hit, reproduced: hand
        the integrator a reference tilted 4.7 deg from truth and the same 3 m
        push integrates to tens of metres."""
        from pi_client.imu_distance_analysis import _integrate

        t, gyro, accel, _ = make_run(3.0, tilt_deg=4.7)
        stale_up = np.array([0.0, 0.0, 1.0])  # believes it is level; it is not
        traj, vels = _integrate(
            t, gyro, accel, np.zeros(3), stale_up, GRAVITY
        )
        self.assertGreater(float(np.linalg.norm(traj[-1][:2])), 20.0)
        self.assertGreater(float(np.linalg.norm(vels[-1])), 5.0)

    def test_rejects_a_log_with_no_still_ends(self):
        t, gyro, accel, _ = make_run(3.0, still_s=0.2)
        res = analyze_run(t, gyro, accel, truth_m=3.0)
        self.assertFalse(res.ok)
        self.assertIn("stillness", res.reason)

    def test_declared_phases_still_reject_a_run_that_never_moved(self):
        """Declaring phase boundaries skips motion detection, which would
        otherwise let a forgotten push through as a valid measurement: ~0 m
        against a 3 m truth, summarised as the real finding "bias does not
        explain the error". A botched run must be rejected, not concluded
        from."""
        rng = np.random.default_rng(3)
        n = int(10 * RATE)
        t = np.arange(n) / RATE
        accel = np.tile([0.8, 0.0, 9.78], (n, 1)) + rng.normal(0, 0.02, (n, 3))
        gyro = rng.normal(0, 0.0015, (n, 3))
        res = analyze_run(t, gyro, accel, truth_m=3.0, head_s=3.0, tail_s=3.0)
        self.assertFalse(res.ok)
        self.assertIn("no motion during the push phase", res.reason)

    def test_rejects_a_log_that_never_moved(self):
        rng = np.random.default_rng(2)
        n = int(6 * RATE)
        t = np.arange(n) / RATE
        accel = np.tile([0.0, 0.0, GRAVITY], (n, 1)) + rng.normal(0, 0.02, (n, 3))
        gyro = rng.normal(0, 0.0015, (n, 3))
        res = analyze_run(t, gyro, accel, truth_m=3.0)
        self.assertFalse(res.ok)
        self.assertIn("never moved", res.reason)


class SummarizeTests(unittest.TestCase):
    def _runs(self, **kw):
        out = []
        for seed in range(3):
            t, gyro, accel, still_s = make_run(3.0, seed=seed, **kw)
            out.append(
                analyze_run(t, gyro, accel, truth_m=3.0, head_s=still_s - 1, tail_s=still_s - 1)
            )
        return out

    def test_consistent_runs_recommend_displacement(self):
        summary = summarize(self._runs())
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["runs_usable"], 3)
        self.assertIn(summary["verdict"], {"USE_DISPLACEMENT", "USE_WITH_ONLINE_ESTIMATION"})

    def test_no_usable_runs_is_reported_not_crashed(self):
        t, gyro, accel, _ = make_run(3.0, still_s=0.2)
        summary = summarize([analyze_run(t, gyro, accel, truth_m=3.0)])
        self.assertFalse(summary["ok"])


if __name__ == "__main__":
    unittest.main()


class ProtocolTests(unittest.TestCase):
    """The protocol shape itself, after a real session was destroyed by it:
    three 3 m runs in one direction needed 9 m of clear floor, and there was
    no unrecorded gap in which to reposition the cart."""

    def test_runs_alternate_direction_so_no_repositioning_is_needed(self):
        from pi_client.imu_calibration import build_phases

        titles = []
        for run in (1, 2, 3):
            move = [p for p in build_phases(run, 3, 3.0, 4.0, 6.0, 15.0) if p.key == "move"]
            self.assertEqual(len(move), 1)
            titles.append(move[0].title)
        self.assertIn("ВПЕРЁД", titles[0])
        self.assertIn("ОБРАТНО", titles[1])
        self.assertIn("ВПЕРЁД", titles[2])

    def test_every_run_starts_with_an_unrecorded_ready_phase(self):
        from pi_client.imu_calibration import build_phases

        phases = build_phases(2, 3, 3.0, 4.0, 6.0, 15.0)
        self.assertEqual(phases[0].key, "ready")
        self.assertFalse(phases[0].recording)
        self.assertGreaterEqual(phases[0].seconds, 10.0)
        # Everything after it must record, or the run has no data.
        self.assertTrue(all(p.recording for p in phases[1:]))


class GyroRejectionTests(unittest.TestCase):
    def test_impossible_gyro_samples_are_dropped(self):
        from pi_client.imu_distance_analysis import reject_corrupt

        rng = np.random.default_rng(7)
        n = 300
        t = np.arange(n) / RATE
        gyro = rng.normal(0, 0.002, (n, 3))
        accel = np.tile([0.0, 0.0, GRAVITY], (n, 1)) + rng.normal(0, 0.02, (n, 3))
        gyro[50] = [math.radians(1890.0), 0.0, 0.0]  # the measured worst case
        gyro[120] = [0.0, math.radians(-1200.0), 0.0]
        ok = reject_corrupt(t, gyro, accel)
        self.assertFalse(ok[50])
        self.assertFalse(ok[120])
        self.assertEqual(int((~ok).sum()), 2)

    def test_gyro_is_ignored_by_default(self):
        """Not a simplification: a gyro that random-walks 61 deg over 60 s of
        provable stillness makes attitude integration actively harmful."""
        t, gyro, accel, still_s = make_run(3.0)
        # A RANDOM WALK, not a constant offset — the distinction is the whole
        # point. Measured on the real sensor, subtracting a constant bias left
        # the 61 deg of accumulated phantom rotation completely unchanged
        # (61.1 -> 61.1), because the corruption is not an offset. A fixture
        # with a constant offset would be silently removed by the analysis's
        # own bias subtraction and prove nothing.
        rng = np.random.default_rng(11)
        drifting = gyro + np.cumsum(rng.normal(0.0, math.radians(2.0), gyro.shape), axis=0) / len(gyro) ** 0.5
        with_gyro = analyze_run(
            t, drifting, accel, truth_m=3.0, head_s=still_s - 1, tail_s=still_s - 1, use_gyro=True
        )
        without = analyze_run(
            t, drifting, accel, truth_m=3.0, head_s=still_s - 1, tail_s=still_s - 1
        )
        self.assertLess(abs(without.error_m), abs(with_gyro.error_m))
