"""Clock reconciliation between the Pi and the laptop, over the project TCP link.

Observations originate on the Pi and are stamped with its monotonic clock;
actions originate on the laptop and are stamped with its own. Joining the two
into training pairs needs one shared timeline, and there is no NTP guarantee
between these two machines.

Cristian's algorithm over the existing request/response channel:

    Pi:  t0 = monotonic_ns()          --> time_sync request
    Mac: replies mac_monotonic_ns
    Pi:  t1 = monotonic_ns()
         rtt    = t1 - t0
         offset = mac_monotonic_ns + rtt/2 - t1      # pi_mono + offset = mac_mono

The `rtt/2` term assumes the request and reply legs took equal time. That is
false in general, and the error is bounded by half the *asymmetry*, not half
the RTT — so the estimator's quality is dominated by the least-delayed sample,
not the average one.

**Take the minimum-RTT sample, never the mean.** Queueing delay is one-sided
and unbounded above: a sample that sat in a Wi-Fi retransmit queue is pure
noise pulling the mean around, while the fastest sample of a burst is the one
that came closest to the symmetric ideal. Averaging RTTs is a common and
appealing mistake that makes the estimate worse the noisier the link gets.

At 10 Hz control the join needs to be good to well under one frame — call it
±20 ms. On a quiet LAN the minimum RTT of a 21-sample burst is typically a few
milliseconds, which leaves ample margin.

Pure functions, no I/O — the transport lives in pi_client/protocol.py and
mac_server/handlers.py so this stays testable on either machine.
"""

from __future__ import annotations

from dataclasses import dataclass

# Enough samples that a few queue-delayed outliers cannot hide the floor,
# cheap enough to run at both ends of every episode (~21 round trips is well
# under a second on a LAN).
DEFAULT_SAMPLES = 21


@dataclass(frozen=True)
class SyncSample:
    """One completed round trip, all times in nanoseconds."""

    t0_pi_ns: int          # Pi monotonic, just before sending
    t1_pi_ns: int          # Pi monotonic, just after receiving
    mac_ns: int            # Mac monotonic, as the server saw the request

    @property
    def rtt_ns(self) -> int:
        return self.t1_pi_ns - self.t0_pi_ns

    @property
    def offset_ns(self) -> int:
        """Add to a Pi monotonic timestamp to get the Mac's."""
        return int(self.mac_ns + self.rtt_ns / 2 - self.t1_pi_ns)


@dataclass(frozen=True)
class ClockOffset:
    offset_ns: int
    rtt_ns: int            # the RTT of the sample the offset came from
    samples: int
    rtt_min_ns: int
    rtt_median_ns: int
    rtt_max_ns: int

    @property
    def quality_ms(self) -> float:
        """Half the winning RTT: the worst the offset can be wrong by if the
        two legs were maximally asymmetric. The honest error bar to report."""
        return self.rtt_ns / 2e6

    def to_dict(self) -> dict[str, float | int]:
        return {
            "offset_ns": self.offset_ns,
            "rtt_ns": self.rtt_ns,
            "samples": self.samples,
            "rtt_min_ms": round(self.rtt_min_ns / 1e6, 3),
            "rtt_median_ms": round(self.rtt_median_ns / 1e6, 3),
            "rtt_max_ms": round(self.rtt_max_ns / 1e6, 3),
            "quality_ms": round(self.quality_ms, 3),
        }


def estimate_offset(samples: list[SyncSample]) -> ClockOffset | None:
    """Pick the minimum-RTT sample and report its offset. None if empty.

    Samples with a non-positive RTT are dropped rather than trusted: a
    monotonic clock that appears to run backwards across a round trip means
    the timestamps did not come from the clock we think they did.
    """
    usable = [s for s in samples if s.rtt_ns > 0]
    if not usable:
        return None
    rtts = sorted(s.rtt_ns for s in usable)
    best = min(usable, key=lambda s: s.rtt_ns)
    return ClockOffset(
        offset_ns=best.offset_ns,
        rtt_ns=best.rtt_ns,
        samples=len(usable),
        rtt_min_ns=rtts[0],
        rtt_median_ns=rtts[len(rtts) // 2],
        rtt_max_ns=rtts[-1],
    )


def pi_to_mac_ns(pi_ns: int, offset: ClockOffset) -> int:
    return int(pi_ns + offset.offset_ns)


def mac_to_pi_ns(mac_ns: int, offset: ClockOffset) -> int:
    return int(mac_ns - offset.offset_ns)


def drift_ns_per_s(start: ClockOffset, end: ClockOffset, span_s: float) -> float | None:
    """Rate of change between an episode's opening and closing sync.

    Two independent crystals always drift relative to each other; typical
    consumer parts are tens of ppm, i.e. tens of microseconds per second,
    which is negligible over a 30 s episode. Measuring it anyway is what turns
    "negligible" from an assumption into a recorded fact — and a wild value
    here is a much better warning than a quietly skewed dataset.
    """
    if span_s <= 0:
        return None
    return (end.offset_ns - start.offset_ns) / span_s
