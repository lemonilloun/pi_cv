"""Unit tests for pi_client.focal_check — pinning fx against a tape measure.

Three estimates of this camera's fx at 1536 px disagree by 10% (chessboard
1098, VGGT 991, COLMAP/DA3 1001). Neither estimator can settle it because
both are the same kind of evidence; a tape measure is a different kind.

    python3 -m unittest client.tests.test_focal_check -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "client/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from pi_client.focal_check import implied_fx, summarize  # noqa: E402


class ImpliedFxTests(unittest.TestCase):
    def test_agreement_leaves_fx_unchanged(self):
        self.assertAlmostEqual(implied_fx(1098.0, 1.00, 1.00), 1098.0)

    def test_an_underreported_distance_means_fx_is_too_low(self):
        """d_measured = d_true * fx_assumed / fx_true. If solvePnP places the
        board nearer than the tape does, the real lens is longer than assumed."""
        self.assertAlmostEqual(implied_fx(1000.0, 0.90, 1.00), 1111.11, places=2)

    def test_recovers_the_competing_estimate_from_a_10_percent_error(self):
        """The concrete case this tool exists for: if fx is really 991 but the
        calibration says 1098, a 1.00 m board reports as 1.108 m."""
        self.assertAlmostEqual(implied_fx(1098.0, 1.108, 1.00), 991.0, places=0)

    def test_rejects_a_non_positive_measurement(self):
        with self.assertRaises(ValueError):
            implied_fx(1000.0, 0.0, 1.0)


class SummarizeTests(unittest.TestCase):
    def test_uses_the_median_not_the_mean(self):
        """One frame where the detector snapped to a reflection would drag a
        mean by more than the 10% effect being measured."""
        distances = [1.00, 1.00, 1.00, 1.00, 9.00]
        report = summarize(distances, fx_assumed=1000.0, true_distance_m=1.00)
        self.assertAlmostEqual(report["measured_distance_m"], 1.00)
        self.assertAlmostEqual(report["fx_implied"], 1000.0)

    def test_flags_a_wobbly_capture(self):
        report = summarize([1.00, 1.20, 0.85], fx_assumed=1000.0, true_distance_m=1.00)
        self.assertFalse(report["ok"])
        self.assertTrue(any("varied" in p for p in report["problems"]))

    def test_half_a_metre_is_a_good_working_point(self):
        """1 cm of tape error there is 2% against a 10.8% effect — a 5:1
        margin — and the board is 53 px per square instead of 26 at 1 m."""
        report = summarize([0.50] * 5, fx_assumed=1098.0, true_distance_m=0.50,
                           pitches_px=[53.0] * 5)
        self.assertTrue(report["ok"], report["problems"])

    def test_flags_standing_too_close_to_conclude_anything(self):
        report = summarize([0.20] * 5, fx_assumed=1000.0, true_distance_m=0.20,
                           pitches_px=[120.0] * 5)
        self.assertFalse(report["ok"])
        self.assertTrue(any("too close" in p for p in report["problems"]))

    def test_flags_a_board_too_small_in_frame(self):
        """The gate that actually matters: corner localization, not distance.
        Standing back to please a tape-precision rule hurts solvePnP more."""
        report = summarize([1.50] * 5, fx_assumed=1098.0, true_distance_m=1.50,
                           pitches_px=[12.0] * 5)
        self.assertFalse(report["ok"])
        self.assertTrue(any("move CLOSER" in p for p in report["problems"]))

    def test_clean_agreement_passes(self):
        report = summarize([1.005, 0.998, 1.002], fx_assumed=1098.0, true_distance_m=1.00,
                           pitches_px=[26.0] * 3)
        self.assertTrue(report["ok"], report["problems"])
        self.assertLess(abs(report["distance_error_frac"]), 0.01)

    def test_reports_the_implied_field_of_view(self):
        report = summarize([1.108] * 5, fx_assumed=1098.0, true_distance_m=1.00)
        self.assertAlmostEqual(report["fx_implied"], 991.0, places=0)
        self.assertAlmostEqual(report["hfov_implied_deg"], 75.6, places=1)

    def test_no_detections_is_reported_not_crashed(self):
        report = summarize([], fx_assumed=1000.0, true_distance_m=1.0)
        self.assertFalse(report["ok"])
        self.assertIn("never detected", report["reason"])


class DistanceReadingGuardTests(unittest.TestCase):
    """The board's size in pixels is an independent witness to how far away
    it was. On 2026-08-03 the tool blamed the calibration for a 48% error
    when the real cause was a mistyped distance — it had every number needed
    to notice, and did not."""

    def test_catches_the_real_mistyped_distance(self):
        """Exactly the run that happened: 48.7 px per square claimed at
        1.00 m implies fx 2029 (41 deg) on a sensor named imx708_WIDE."""
        report = summarize([0.5243] * 20, fx_assumed=1028.6, true_distance_m=1.00,
                           pitches_px=[48.7] * 20, square_m=0.024)
        self.assertFalse(report["ok"])
        self.assertTrue(any("CHECK THE TAPE READING" in p for p in report["problems"]))
        self.assertTrue(any("Nothing is wrong with the calibration" in p
                            for p in report["problems"]))

    def test_suggests_the_distance_the_board_size_actually_implies(self):
        report = summarize([0.5243] * 20, fx_assumed=1028.6, true_distance_m=1.00,
                           pitches_px=[48.7] * 20, square_m=0.024)
        problem = next(p for p in report["problems"] if "CHECK THE TAPE" in p)
        self.assertIn("0.49 m", problem)      # ~0.5 m, where the board really was

    def test_the_same_board_at_the_true_distance_passes(self):
        report = summarize([0.5243] * 20, fx_assumed=1028.6, true_distance_m=0.50,
                           pitches_px=[48.7] * 20, square_m=0.024)
        self.assertTrue(report["ok"], report["problems"])

    def test_a_genuine_focal_error_is_still_reported_not_swallowed(self):
        """The guard must not become an excuse that hides real disagreement:
        a plausible fx with a distance mismatch still surfaces normally."""
        report = summarize([0.55] * 20, fx_assumed=1098.0, true_distance_m=0.50,
                           pitches_px=[47.4] * 20, square_m=0.024)
        self.assertTrue(report["ok"])                      # no structural problem
        self.assertGreater(abs(report["distance_error_frac"]), 0.05)   # but flagged
