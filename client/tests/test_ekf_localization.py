"""Unit tests for pi_client.ekf_localization: the live position/heading
fusion filter (IMU predict + visual/landmark/range-vs-occupancy updates).

Run directly (no discover wiring exists for client/tests yet):
    python3 -m unittest client.tests.test_ekf_localization -v
from the repo root, with client/src on PYTHONPATH.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(REPO_ROOT / "client/src"),):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import numpy as np  # noqa: E402

from pi_client.ekf_localization import Ekf2D, _wrap_angle  # noqa: E402


class WrapAngleTests(unittest.TestCase):
    def test_wraps_into_pi_range(self):
        self.assertAlmostEqual(_wrap_angle(3 * math.pi), -math.pi, places=6)
        self.assertAlmostEqual(_wrap_angle(-3 * math.pi), -math.pi, places=6)
        self.assertAlmostEqual(_wrap_angle(0.1), 0.1, places=6)


class PredictTests(unittest.TestCase):
    def test_pure_forward_motion_moves_along_current_heading(self):
        ekf = Ekf2D.initialize(0.0, 0.0, 0.0)
        ekf.predict(forward_m=1.0, lateral_m=0.0, dtheta_rad=0.0, dt_s=0.5)
        x, y = ekf.position
        self.assertAlmostEqual(x, 1.0, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)

    def test_forward_motion_at_90_degrees_moves_along_y(self):
        ekf = Ekf2D.initialize(0.0, 0.0, math.pi / 2)
        ekf.predict(forward_m=1.0, lateral_m=0.0, dtheta_rad=0.0, dt_s=0.5)
        x, y = ekf.position
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 1.0, places=6)

    def test_midpoint_heading_used_for_turning_motion(self):
        # Turning 90deg while moving forward 1m: displacement should use the
        # AVERAGE heading (45deg), not the start or end heading exactly -
        # this is the whole point of the midpoint-heading odometry model.
        ekf = Ekf2D.initialize(0.0, 0.0, 0.0)
        ekf.predict(forward_m=1.0, lateral_m=0.0, dtheta_rad=math.pi / 2, dt_s=0.5)
        x, y = ekf.position
        self.assertAlmostEqual(x, math.cos(math.pi / 4), places=6)
        self.assertAlmostEqual(y, math.sin(math.pi / 4), places=6)
        self.assertAlmostEqual(ekf.heading_rad, math.pi / 2, places=6)

    def test_covariance_grows_with_motion(self):
        ekf = Ekf2D.initialize(0.0, 0.0, 0.0, pos_std=0.01, theta_std_deg=1.0)
        p0 = ekf.P.copy()
        ekf.predict(forward_m=2.0, lateral_m=0.0, dtheta_rad=0.1, dt_s=1.0)
        self.assertGreater(ekf.P[0, 0], p0[0, 0])
        self.assertGreater(ekf.P[1, 1], p0[1, 1])

    def test_heading_only_mode_turns_without_translating(self):
        # pi_navigation's default: gyro supplies the turn, displacement is
        # deliberately withheld (the accelerometer's is 78x wrong on this
        # rig). Heading must still track exactly; position must not move,
        # but must become LESS certain, since "unknown displacement" is not
        # the same claim as "did not move".
        ekf = Ekf2D.initialize(2.0, 3.0, 0.0, pos_std=0.1, theta_std_deg=5.0)
        p0 = ekf.P[0, 0]
        ekf.predict(
            forward_m=0.0, lateral_m=0.0, dtheta_rad=math.radians(30.0),
            dt_s=0.7, time_noise_per_s=0.6,
        )
        self.assertAlmostEqual(ekf.position[0], 2.0, places=9)
        self.assertAlmostEqual(ekf.position[1], 3.0, places=9)
        self.assertAlmostEqual(math.degrees(ekf.heading_rad), 30.0, places=6)
        self.assertGreater(ekf.P[0, 0], p0)

    def test_covariance_grows_with_time_even_when_stationary(self):
        # The known hazard (78 m/s drift over 215s on a real session):
        # near-zero displacement must NOT mean near-zero added uncertainty.
        ekf = Ekf2D.initialize(0.0, 0.0, 0.0, pos_std=0.01, theta_std_deg=1.0)
        p0 = ekf.P.copy()
        ekf.predict(forward_m=0.0, lateral_m=0.0, dtheta_rad=0.0, dt_s=5.0)
        self.assertGreater(ekf.P[0, 0], p0[0, 0])


class VisualUpdateTests(unittest.TestCase):
    def test_exact_measurement_snaps_state_and_shrinks_covariance(self):
        ekf = Ekf2D.initialize(0.0, 0.0, 0.0, pos_std=1.0, theta_std_deg=45.0)
        ekf.predict(forward_m=3.0, lateral_m=0.5, dtheta_rad=0.2, dt_s=1.0)
        p_before = ekf.P.copy()
        R = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(5.0) ** 2])
        true_x, true_y, true_theta = ekf.position[0], ekf.position[1], ekf.heading_rad
        ekf.update_visual(true_x, true_y, true_theta, R)
        self.assertAlmostEqual(ekf.position[0], true_x, places=3)
        self.assertLess(ekf.P[0, 0], p_before[0, 0])

    def test_heading_wraps_across_pi_boundary(self):
        ekf = Ekf2D.initialize(0.0, 0.0, math.pi - 0.05)
        R = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(5.0) ** 2])
        # Measurement just past the wrap (-pi + 0.05) is actually CLOSE to
        # the current heading, not ~2*pi away - the update must treat it
        # that way or it'll spin the estimate the wrong way around.
        ekf.update_visual(0.0, 0.0, -math.pi + 0.05, R)
        self.assertLess(abs(_wrap_angle(ekf.heading_rad - (math.pi - 0.025))), 0.1)

    def test_low_confidence_fix_moves_state_less_than_high_confidence(self):
        r_loose = np.diag([1.0 ** 2, 1.0 ** 2, math.radians(60.0) ** 2])
        r_tight = np.diag([0.02 ** 2, 0.02 ** 2, math.radians(2.0) ** 2])

        ekf_loose = Ekf2D.initialize(0.0, 0.0, 0.0, pos_std=0.1, theta_std_deg=10.0)
        ekf_loose.update_visual(5.0, 0.0, 0.0, r_loose)

        ekf_tight = Ekf2D.initialize(0.0, 0.0, 0.0, pos_std=0.1, theta_std_deg=10.0)
        ekf_tight.update_visual(5.0, 0.0, 0.0, r_tight)

        self.assertLess(ekf_loose.position[0], ekf_tight.position[0])


class LandmarkUpdateTests(unittest.TestCase):
    def test_converges_toward_true_pose_given_consistent_observation(self):
        # True pose (1, 0, theta=0) means facing +x (this module's
        # convention: bearing/heading 0 = +x axis). A landmark straight
        # ahead at range 3 is therefore at (4, 0), NOT above the robot.
        landmark = (4.0, 0.0)
        ekf = Ekf2D.initialize(0.8, 0.1, math.radians(5.0), pos_std=0.5, theta_std_deg=20.0)
        R = np.diag([0.05 ** 2, math.radians(3.0) ** 2])
        before = np.linalg.norm(np.array(ekf.position) - np.array([1.0, 0.0]))
        for _ in range(5):
            ekf.update_landmark(landmark[0], landmark[1], bearing_meas=0.0, range_meas=3.0, R=R)
        after = np.linalg.norm(np.array(ekf.position) - np.array([1.0, 0.0]))
        self.assertLess(after, before)

    def test_degenerate_zero_range_is_ignored_not_crashed(self):
        ekf = Ekf2D.initialize(1.0, 3.0, 0.0)
        R = np.diag([0.05 ** 2, math.radians(3.0) ** 2])
        ekf.update_landmark(1.0, 3.0, bearing_meas=0.0, range_meas=0.0, R=R)  # robot AT the landmark
        self.assertTrue(np.all(np.isfinite(ekf.x)))


class RangeUpdateTests(unittest.TestCase):
    def test_converges_toward_a_wall_along_x(self):
        # Synthetic "occupancy grid": a wall at x=5, ray straight along +x
        # (theta=0) from (x,0). Predicted range = 5 - x.
        def predict_range(x, y, theta):
            # Tolerate the small perturbation update_range's numeric
            # Jacobian probes with (a real grid ray-march would too, up to
            # its own angular resolution) - only reject a genuinely
            # different heading.
            if abs(_wrap_angle(theta)) > 0.01:
                return None
            return max(5.0 - x, 0.0)

        ekf = Ekf2D.initialize(0.0, 0.0, 0.0, pos_std=1.0, theta_std_deg=1.0)
        before = abs(ekf.position[0] - 1.0)
        for _ in range(5):
            ekf.update_range(predict_range, range_meas=4.0, R=0.05 ** 2)  # true x=1
        after = abs(ekf.position[0] - 1.0)
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()


class VioBiasEstimationTests(unittest.TestCase):
    """The point of Ekf2DVio: recover the accelerometer bias that made raw
    delta_p_m 78x wrong on the real rig, using visual fixes as the only
    external observation."""

    def _run(self, true_bias, steps=43, dt=0.7, speed=0.4, fix_every=2):
        from pi_client.ekf_localization import Ekf2DVio

        rng = np.random.default_rng(0)
        ekf = Ekf2DVio.initialize(0.0, 0.0, 0.0, pos_std=0.3, theta_std_deg=5.0)
        R = np.diag([0.2 ** 2, 0.2 ** 2, math.radians(8.0) ** 2])
        for i in range(steps):
            true_x = speed * i * dt
            # Constant velocity => true delta_v is 0; the sensor reports
            # only its bias. Exactly the real failure mode.
            ekf.predict((true_bias[0] * dt, true_bias[1] * dt), 0.0, dt)
            if i % fix_every == 0:
                ekf.update_visual(
                    true_x + rng.normal(0, 0.1), rng.normal(0, 0.1), 0.0, R
                )
        return ekf, speed * (steps - 1) * dt

    def test_bias_converges_to_truth(self):
        true_bias = (0.6, 0.55)
        ekf, _ = self._run(true_bias)
        err = math.hypot(ekf.bias[0] - true_bias[0], ekf.bias[1] - true_bias[1])
        start = math.hypot(*true_bias)
        self.assertLess(err, 0.1, "bias should converge")
        self.assertLess(err, start / 5, "and be a large improvement on assuming zero")

    def test_position_tracks_despite_biased_accelerometer(self):
        ekf, true_x = self._run((0.6, 0.55))
        self.assertLess(abs(ekf.position[0] - true_x), 0.5)

    def test_zero_velocity_update_pulls_velocity_down(self):
        from pi_client.ekf_localization import Ekf2DVio

        ekf = Ekf2DVio.initialize(0.0, 0.0, 0.0)
        ekf.predict((1.4, 0.0), 0.0, 0.7)  # ~2 m/s^2 for 0.7s
        moving = abs(ekf.velocity[0])
        self.assertGreater(moving, 0.5)
        ekf.update_zero_velocity()
        self.assertLess(abs(ekf.velocity[0]), moving / 2)

    def test_heading_still_comes_from_gyro(self):
        from pi_client.ekf_localization import Ekf2DVio

        ekf = Ekf2DVio.initialize(0.0, 0.0, 0.0)
        ekf.predict((0.0, 0.0), math.radians(25.0), 0.5)
        self.assertAlmostEqual(math.degrees(ekf.heading_rad), 25.0, places=6)


class Ekf2DHeadingTests(unittest.TestCase):
    """Ekf2DHeading's whole reason to exist: the RVC yaw datum is arbitrary,
    but constant. Estimating that constant turns a relative heading sensor
    into an absolute one between visual fixes."""

    @staticmethod
    def _filter(theta0: float = 0.0):
        from pi_client.ekf_localization import Ekf2DHeading

        return Ekf2DHeading.initialize(0.0, 0.0, theta0, pos_std=0.2, theta_std_deg=10.0)

    def test_bias_converges_to_the_true_datum_offset(self):
        """Visual fixes agree on heading; the IMU reads it offset by a fixed
        amount. The filter must learn that amount, not fight it."""
        true_theta = math.radians(30.0)
        true_bias = math.radians(-110.0)  # IMU powered up facing elsewhere
        ekf = self._filter(true_theta)
        R = np.diag([0.2 ** 2, 0.2 ** 2, math.radians(5.0) ** 2])
        for _ in range(30):
            ekf.predict(dt_s=0.5)
            ekf.update_imu_yaw(true_theta - true_bias, sigma_deg=1.0)
            ekf.update_visual(0.0, 0.0, true_theta, R)
        self.assertAlmostEqual(math.degrees(ekf.yaw_bias_rad), -110.0, delta=1.5)
        self.assertLess(ekf.yaw_bias_std_deg, 2.0)

    def test_converged_bias_makes_imu_yaw_absolute_between_fixes(self):
        """After convergence, a turn seen ONLY by the IMU must move the
        fused heading by that turn. This is the capability the old
        increment-only path could not offer with the same confidence."""
        true_theta = math.radians(10.0)
        true_bias = math.radians(45.0)
        ekf = self._filter(true_theta)
        R = np.diag([0.2 ** 2, 0.2 ** 2, math.radians(3.0) ** 2])
        for _ in range(40):
            ekf.predict(dt_s=0.5)
            ekf.update_imu_yaw(true_theta - true_bias, sigma_deg=1.0)
            ekf.update_visual(0.0, 0.0, true_theta, R)

        # Vision drops out; the robot turns 90 deg, IMU only.
        turned = true_theta + math.radians(90.0)
        for _ in range(10):
            ekf.predict(dt_s=0.2)
            ekf.update_imu_yaw(turned - true_bias, sigma_deg=1.0)
        self.assertAlmostEqual(math.degrees(ekf.heading_rad), 100.0, delta=3.0)

    def test_position_uncertainty_grows_without_a_fix(self):
        """No displacement source means the honest answer is 'could be
        anywhere within top-speed * elapsed'."""
        ekf = self._filter()
        before = max(ekf.position_std_m)
        for _ in range(10):
            ekf.predict(dt_s=1.0, speed_ms=0.6)
        after = max(ekf.position_std_m)
        self.assertGreater(after, before + 1.0)

    def test_imu_yaw_alone_never_invents_a_heading(self):
        """With no visual fix, IMU yaw cannot resolve theta from b — the
        pair is unobservable. Heading confidence must not improve."""
        ekf = self._filter()
        before = ekf.heading_std_deg
        for _ in range(50):
            ekf.predict(dt_s=0.1)
            ekf.update_imu_yaw(math.radians(77.0), sigma_deg=1.0)
        self.assertGreaterEqual(ekf.heading_std_deg, before - 0.5)

    def test_wraps_across_the_180_boundary(self):
        """RVC yaw wraps at +/-180 and plan-frame headings can sit right on
        it; a filter that subtracts naively would see a 360 deg innovation."""
        true_theta = math.radians(179.0)
        true_bias = math.radians(-179.0)  # so IMU reads ~ -2 deg
        ekf = self._filter(true_theta)
        R = np.diag([0.2 ** 2, 0.2 ** 2, math.radians(5.0) ** 2])
        for _ in range(30):
            ekf.predict(dt_s=0.5)
            ekf.update_imu_yaw(_wrap_angle(true_theta - true_bias), sigma_deg=1.0)
            ekf.update_visual(0.0, 0.0, true_theta, R)
        self.assertAlmostEqual(math.degrees(_wrap_angle(ekf.heading_rad)), 179.0, delta=2.0)
        self.assertLess(ekf.yaw_bias_std_deg, 3.0)

    def test_position_only_update_leaves_a_converged_bias_alone(self):
        ekf = self._filter(0.0)
        R3 = np.diag([0.2 ** 2, 0.2 ** 2, math.radians(3.0) ** 2])
        for _ in range(30):
            ekf.predict(dt_s=0.5)
            ekf.update_imu_yaw(math.radians(20.0), sigma_deg=1.0)
            ekf.update_visual(0.0, 0.0, 0.0, R3)
        bias_before = ekf.yaw_bias_rad
        ekf.update_position_only(1.0, 2.0, np.diag([0.1 ** 2, 0.1 ** 2]))
        self.assertAlmostEqual(ekf.yaw_bias_rad, bias_before, places=6)
        # Position moves most of the way to the measurement; the exact gain
        # is set by prior-vs-R and is not what this test is about.
        self.assertGreater(ekf.position[0], 0.5)


class AdaptiveHeadingNoiseTests(unittest.TestCase):
    """Uncertainty must track what the robot DID, not what it could have.

    The blanket 1.0 rad/s envelope grows heading variance by (38 deg)^2 per
    tick at the 1.5 Hz nav rate, deleting the prior every time. The estimate
    then follows whichever measurement arrived last, and the visual heading
    carries tens of degrees of spread — which is exactly why the fused marker
    was seen jumping on the first real run.
    """

    @staticmethod
    def _filter():
        from pi_client.ekf_localization import Ekf2DHeading

        return Ekf2DHeading.initialize(0.0, 0.0, 0.0, pos_std=0.2, theta_std_deg=5.0)

    def test_a_stationary_robot_keeps_a_tight_heading(self):
        ekf = self._filter()
        for _ in range(20):
            ekf.predict(dt_s=0.67, turned_rad=0.0)
        self.assertLess(ekf.heading_std_deg, 8.0)

    def test_the_old_blanket_envelope_destroys_the_prior_in_one_tick(self):
        """Pin the defect so nobody reinstates it: one tick of the fallback
        is worth more than 30 deg of heading uncertainty."""
        ekf = self._filter()
        before = ekf.heading_std_deg
        ekf.predict(dt_s=0.67)          # no turned_rad -> blanket envelope
        self.assertGreater(ekf.heading_std_deg, 30.0)
        self.assertGreater(ekf.heading_std_deg, 4 * before)

    def test_a_real_turn_still_admits_uncertainty(self):
        """The fix must not make the filter overconfident through a turn —
        it has to let theta move when the robot actually turns."""
        still = self._filter()
        turning = self._filter()
        still.predict(dt_s=0.67, turned_rad=0.0)
        turning.predict(dt_s=0.67, turned_rad=math.radians(90.0))
        self.assertGreater(turning.heading_std_deg, still.heading_std_deg)

    def test_noise_scales_with_the_size_of_the_turn(self):
        """Compare the variance ADDED, not the total: the initial prior
        dominates the total and would mask the effect being tested."""
        base = self._filter().P[2, 2]
        small, large = self._filter(), self._filter()
        small.predict(dt_s=0.5, turned_rad=math.radians(10.0))
        large.predict(dt_s=0.5, turned_rad=math.radians(90.0))
        added_small = small.P[2, 2] - base
        added_large = large.P[2, 2] - base
        self.assertGreater(added_large, 50 * added_small)   # (90/10)^2 = 81x

    def test_the_floor_stays_far_from_the_bias_walk(self):
        """theta and b enter only as their difference, so the split of a yaw
        change between 'turned' and 'datum drifted' is set by their variance
        ratio. Collapse the gap and the filter charges real turns to bias."""
        from pi_client.ekf_localization import Ekf2DHeading
        import inspect

        signature = inspect.signature(Ekf2DHeading.predict)
        floor = signature.parameters["turn_rate_floor_rad_s"].default
        bias = signature.parameters["bias_walk_per_s"].default
        self.assertGreater(floor / bias, 50.0)

    def test_a_turn_during_a_visual_dropout_is_still_recovered(self):
        """The behaviour the old parameter was protecting, which the new one
        must not lose: 90 deg turned with vision absent still lands."""
        from pi_client.ekf_localization import Ekf2DHeading

        true_theta, true_bias = math.radians(10.0), math.radians(45.0)
        ekf = Ekf2DHeading.initialize(0.0, 0.0, true_theta, pos_std=0.2, theta_std_deg=10.0)
        R = np.diag([0.2 ** 2, 0.2 ** 2, math.radians(3.0) ** 2])
        for _ in range(40):
            ekf.predict(dt_s=0.5, turned_rad=0.0)
            ekf.update_imu_yaw(true_theta - true_bias, sigma_deg=1.0)
            ekf.update_visual(0.0, 0.0, true_theta, R)

        turned = true_theta + math.radians(90.0)
        for i in range(10):
            step = math.radians(9.0)
            ekf.predict(dt_s=0.2, turned_rad=step)
            ekf.update_imu_yaw(true_theta + math.radians(9.0 * (i + 1)) - true_bias,
                               sigma_deg=1.0)
        self.assertAlmostEqual(math.degrees(ekf.heading_rad), 100.0, delta=3.0)


class VisualCovarianceTests(unittest.TestCase):
    """navindex reports `heading_spread_deg` = 0 when the neighbouring
    keyframes agree perfectly — the best case, and a common one: a single
    surviving neighbour gives a resultant length of exactly 1.0, and any
    tight cluster rounds to 0.0 at one decimal.

    `heading_spread_deg or 30.0` turned every one of those into the
    no-information default, so the most confident fixes got the worst
    covariance."""

    @staticmethod
    def _cov(spread):
        from pi_client.pi_navigation import _visual_covariance

        return _visual_covariance(
            similarity=0.9, heading_spread_deg=spread,
            pos_std_floor=0.15, pos_std_scale=2.0, theta_std_floor_deg=5.0,
        )

    def test_perfect_agreement_is_the_tightest_heading(self):
        perfect = self._cov(0.0)[2, 2]
        scattered = self._cov(45.0)[2, 2]
        self.assertLess(perfect, scattered)

    def test_zero_spread_is_not_silently_promoted_to_thirty_degrees(self):
        """The exact defect: 0.0 must give the floor, not the fallback."""
        self.assertAlmostEqual(
            math.degrees(math.sqrt(self._cov(0.0)[2, 2])), 5.0, places=6
        )

    def test_a_missing_spread_still_falls_back(self):
        """None genuinely means 'no information' and must stay pessimistic."""
        self.assertAlmostEqual(
            math.degrees(math.sqrt(self._cov(None)[2, 2])), 35.0, places=6
        )

    def test_covariance_is_monotonic_in_spread(self):
        values = [self._cov(s)[2, 2] for s in (0.0, 5.0, 20.0, 60.0)]
        self.assertEqual(values, sorted(values))

    def test_position_still_tracks_similarity(self):
        from pi_client.pi_navigation import _visual_covariance

        good = _visual_covariance(1.0, 0.0, 0.15, 2.0, 5.0)[0, 0]
        poor = _visual_covariance(0.2, 0.0, 0.15, 2.0, 5.0)[0, 0]
        self.assertLess(good, poor)


class MotionModelTest(unittest.TestCase):
    """The filter used to have no displacement input at all: driving forward
    left the marker where it was until the next visual fix. These pin the
    motion model that fixed it."""

    def _ekf(self, heading_deg=0.0):
        from pi_client.ekf_localization import Ekf2DHeading

        e = Ekf2DHeading()
        e.x[0], e.x[1], e.x[2] = 0.0, 0.0, math.radians(heading_deg)
        return e

    def test_forward_step_moves_along_the_filters_heading(self) -> None:
        e = self._ekf(90.0)          # facing +Y
        e.predict(dt_s=0.7, forward_m=0.30)
        self.assertAlmostEqual(e.x[0], 0.0, places=6)
        self.assertAlmostEqual(e.x[1], 0.30, places=6)

    def test_no_displacement_keeps_the_old_frozen_behaviour(self) -> None:
        # Every existing caller omits it; the state must not move, or old
        # behaviour would change under them.
        e = self._ekf(37.0)
        e.predict(dt_s=0.7)
        self.assertEqual((e.x[0], e.x[1]), (0.0, 0.0))

    def test_it_uses_theta_not_the_raw_imu_yaw(self) -> None:
        # theta carries the yaw-datum offset `b` that makes IMU yaw mean
        # something on this map. Steering by the raw yaw would rotate every
        # step by that offset.
        e = self._ekf(0.0)
        e.x[3] = math.radians(90.0)   # a large datum offset
        e.predict(dt_s=0.7, forward_m=1.0)
        self.assertAlmostEqual(e.x[0], 1.0, places=6)
        self.assertAlmostEqual(e.x[1], 0.0, places=6)

    def test_a_step_costs_less_certainty_than_not_knowing(self) -> None:
        # The point of measuring: a known 0.2 m step should leave the filter
        # more certain than "it could have gone anywhere at 0.6 m/s".
        known = self._ekf(); known.predict(dt_s=1.0, forward_m=0.20)
        blind = self._ekf(); blind.predict(dt_s=1.0)
        self.assertLess(known.P[0, 0], blind.P[0, 0])

    def test_doubt_grows_with_the_size_of_the_step(self) -> None:
        small = self._ekf(); small.predict(dt_s=0.7, forward_m=0.05)
        large = self._ekf(); large.predict(dt_s=0.7, forward_m=1.50)
        self.assertLess(small.P[0, 0], large.P[0, 0])

    def test_lateral_is_applied_perpendicular(self) -> None:
        e = self._ekf(0.0)           # facing +X
        e.predict(dt_s=0.7, forward_m=0.0, lateral_m=0.25)
        self.assertAlmostEqual(e.x[0], 0.0, places=6)
        self.assertAlmostEqual(e.x[1], 0.25, places=6)

    def test_a_visual_fix_still_corrects_accumulated_drift(self) -> None:
        # The whole reason feeding a noisy step is safe: the fix wins.
        e = self._ekf(0.0)
        for _ in range(5):
            e.predict(dt_s=0.7, forward_m=0.40)   # dead reckons to x=2.0
        self.assertGreater(e.x[0], 1.5)
        import numpy as np
        e.update_visual(0.5, 0.0, 0.0, np.diag([0.0025, 0.0025, 0.05]))
        self.assertLess(abs(e.x[0] - 0.5), 0.2)
