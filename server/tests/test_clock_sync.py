"""Unit tests for shared.clock_sync.

Observations are stamped on the Pi, actions on the laptop, and there is no NTP
guarantee between them. Joining the two into training pairs needs one shared
timeline good to well under a 10 Hz frame — call it +/-20 ms.

    python3 -m unittest server.tests.test_clock_sync -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.clock_sync import (  # noqa: E402
    ClockOffset,
    SyncSample,
    drift_ns_per_s,
    estimate_offset,
    mac_to_pi_ns,
    pi_to_mac_ns,
)

MS = 1_000_000


def make_sample(true_offset_ns: int, t0: int, out_ns: int, back_ns: int) -> SyncSample:
    """A round trip with explicitly asymmetric legs.

    The server timestamps the moment it sees the request, so its clock reads
    `t0 + out + true_offset`. The reply lands back `back` later.
    """
    return SyncSample(
        t0_pi_ns=t0,
        t1_pi_ns=t0 + out_ns + back_ns,
        mac_ns=t0 + out_ns + true_offset_ns,
    )


class EstimateOffsetTests(unittest.TestCase):
    def test_exact_when_the_legs_are_symmetric(self):
        s = make_sample(true_offset_ns=5_000 * MS, t0=1000, out_ns=2 * MS, back_ns=2 * MS)
        result = estimate_offset([s])
        assert result is not None
        self.assertEqual(result.offset_ns, 5_000 * MS)

    def test_error_is_bounded_by_half_the_asymmetry(self):
        """Not half the RTT — that is the reported worst case, but the actual
        error is driven by how lopsided the two legs were."""
        s = make_sample(true_offset_ns=0, t0=0, out_ns=1 * MS, back_ns=9 * MS)
        result = estimate_offset([s])
        assert result is not None
        self.assertEqual(result.offset_ns, -4 * MS)          # (1 - 9) / 2
        self.assertAlmostEqual(result.quality_ms, 5.0)        # rtt/2 = 10/2

    def test_picks_the_minimum_rtt_not_the_mean(self):
        """The heart of the estimator. Queueing delay is one-sided and
        unbounded above, so averaging RTTs makes the estimate worse the
        noisier the link gets. One clean sample beats twenty muddy ones."""
        true = 7_000 * MS
        clean = make_sample(true, t0=0, out_ns=1 * MS, back_ns=1 * MS)
        muddy = [
            make_sample(true, t0=i * 1000, out_ns=1 * MS, back_ns=(40 + 10 * i) * MS)
            for i in range(1, 20)
        ]
        result = estimate_offset(muddy + [clean])
        assert result is not None
        self.assertEqual(result.offset_ns, true)
        self.assertEqual(result.rtt_ns, 2 * MS)
        self.assertEqual(result.samples, 20)

    def test_a_realistic_burst_lands_well_inside_the_20ms_budget(self):
        true = -3_500 * MS
        samples = [
            make_sample(true, t0=i * 1000, out_ns=(2 + i % 3) * MS, back_ns=(2 + i % 5) * MS)
            for i in range(21)
        ]
        result = estimate_offset(samples)
        assert result is not None
        self.assertLess(abs(result.offset_ns - true), 20 * MS)

    def test_reports_the_rtt_distribution(self):
        samples = [make_sample(0, t0=i, out_ns=(i + 1) * MS, back_ns=(i + 1) * MS)
                   for i in range(5)]
        result = estimate_offset(samples)
        assert result is not None
        self.assertEqual(result.rtt_min_ns, 2 * MS)
        self.assertEqual(result.rtt_max_ns, 10 * MS)
        self.assertEqual(result.rtt_median_ns, 6 * MS)

    def test_drops_non_positive_rtt(self):
        """A monotonic clock cannot run backwards across a round trip; if it
        appears to, the timestamps did not come from the clock we assume."""
        bad = SyncSample(t0_pi_ns=100, t1_pi_ns=50, mac_ns=200)
        good = make_sample(0, t0=0, out_ns=1 * MS, back_ns=1 * MS)
        result = estimate_offset([bad, good])
        assert result is not None
        self.assertEqual(result.samples, 1)

    def test_none_when_nothing_is_usable(self):
        self.assertIsNone(estimate_offset([]))
        self.assertIsNone(estimate_offset([SyncSample(100, 50, 200)]))

    def test_to_dict_is_json_safe(self):
        import json
        result = estimate_offset([make_sample(0, 0, MS, MS)])
        assert result is not None
        json.dumps(result.to_dict())


class ConversionTests(unittest.TestCase):
    def test_round_trips(self):
        offset = ClockOffset(offset_ns=1234 * MS, rtt_ns=2 * MS, samples=21,
                             rtt_min_ns=2 * MS, rtt_median_ns=3 * MS, rtt_max_ns=9 * MS)
        self.assertEqual(mac_to_pi_ns(pi_to_mac_ns(999, offset), offset), 999)


class DriftTests(unittest.TestCase):
    def test_measures_relative_crystal_drift(self):
        """Two consumer crystals drift tens of ppm apart — negligible over a
        30 s episode, but measuring it turns 'negligible' into a fact."""
        start = ClockOffset(0, MS, 21, MS, MS, MS)
        end = ClockOffset(3 * MS, MS, 21, MS, MS, MS)
        self.assertAlmostEqual(drift_ns_per_s(start, end, 30.0), 100_000.0)

    def test_none_for_a_zero_span(self):
        offset = ClockOffset(0, MS, 21, MS, MS, MS)
        self.assertIsNone(drift_ns_per_s(offset, offset, 0.0))
