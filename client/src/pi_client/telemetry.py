"""System telemetry sampling for the Raspberry Pi session client.

Reads Linux /proc and /sys interfaces directly so no extra dependencies are
needed on the Pi. On platforms without those interfaces (macOS loopback
testing) the corresponding fields are None.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROC_STAT_PATH = Path("/proc/stat")
PROC_MEMINFO_PATH = Path("/proc/meminfo")
THERMAL_ZONE_PATH = Path("/sys/class/thermal/thermal_zone0/temp")


@dataclass(frozen=True)
class _CpuTimes:
    total: int
    busy: int


@dataclass
class TelemetrySample:
    sample_ts: float
    cpu_percent_total: float | None = None
    cpu_percent_per_core: list[float] | None = None
    load_avg_1m: float | None = None
    mem_total_kb: int | None = None
    mem_available_kb: int | None = None
    mem_used_kb: int | None = None
    mem_percent: float | None = None
    temp_c: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_ts": self.sample_ts,
            "cpu_percent_total": self.cpu_percent_total,
            "cpu_percent_per_core": self.cpu_percent_per_core,
            "load_avg_1m": self.load_avg_1m,
            "mem_total_kb": self.mem_total_kb,
            "mem_available_kb": self.mem_available_kb,
            "mem_used_kb": self.mem_used_kb,
            "mem_percent": self.mem_percent,
            "temp_c": self.temp_c,
        }
        payload.update(self.extra)
        return payload


class SystemTelemetrySampler:
    """Samples CPU/memory/temperature. CPU percentages are computed from
    /proc/stat deltas, so the first sample after construction reports None
    for CPU until a second reading exists."""

    def __init__(self) -> None:
        self._previous_cpu: dict[str, _CpuTimes] | None = self._read_cpu_times()

    def sample(self) -> TelemetrySample:
        sample = TelemetrySample(sample_ts=time.time())

        current_cpu = self._read_cpu_times()
        if current_cpu is not None and self._previous_cpu is not None:
            total_pct = _cpu_percent(self._previous_cpu.get("cpu"), current_cpu.get("cpu"))
            per_core: list[float] = []
            core_index = 0
            while True:
                key = f"cpu{core_index}"
                if key not in current_cpu:
                    break
                pct = _cpu_percent(self._previous_cpu.get(key), current_cpu.get(key))
                per_core.append(pct if pct is not None else 0.0)
                core_index += 1
            sample.cpu_percent_total = total_pct
            sample.cpu_percent_per_core = per_core or None
        self._previous_cpu = current_cpu

        meminfo = _read_meminfo()
        if meminfo is not None:
            total_kb, available_kb = meminfo
            sample.mem_total_kb = total_kb
            sample.mem_available_kb = available_kb
            sample.mem_used_kb = total_kb - available_kb
            if total_kb > 0:
                sample.mem_percent = round((total_kb - available_kb) / total_kb * 100.0, 1)

        sample.temp_c = _read_temperature_c()

        try:
            sample.load_avg_1m = round(os.getloadavg()[0], 2)
        except (OSError, AttributeError):
            sample.load_avg_1m = None

        return sample

    @staticmethod
    def _read_cpu_times() -> dict[str, _CpuTimes] | None:
        try:
            lines = PROC_STAT_PATH.read_text(encoding="ascii").splitlines()
        except OSError:
            return None

        times: dict[str, _CpuTimes] = {}
        for line in lines:
            if not line.startswith("cpu"):
                continue
            parts = line.split()
            name = parts[0]
            values = [int(value) for value in parts[1:]]
            if len(values) < 5:
                continue
            total = sum(values)
            idle = values[3]
            iowait = values[4] if len(values) > 4 else 0
            times[name] = _CpuTimes(total=total, busy=total - idle - iowait)
        return times or None


def _cpu_percent(previous: _CpuTimes | None, current: _CpuTimes | None) -> float | None:
    if previous is None or current is None:
        return None
    total_delta = current.total - previous.total
    if total_delta <= 0:
        return 0.0
    busy_delta = current.busy - previous.busy
    return round(max(0.0, min(100.0, busy_delta / total_delta * 100.0)), 1)


def _read_meminfo() -> tuple[int, int] | None:
    try:
        lines = PROC_MEMINFO_PATH.read_text(encoding="ascii").splitlines()
    except OSError:
        return None

    total_kb: int | None = None
    available_kb: int | None = None
    for line in lines:
        if line.startswith("MemTotal:"):
            total_kb = int(line.split()[1])
        elif line.startswith("MemAvailable:"):
            available_kb = int(line.split()[1])
        if total_kb is not None and available_kb is not None:
            return total_kb, available_kb
    return None


def _read_temperature_c() -> float | None:
    try:
        raw = THERMAL_ZONE_PATH.read_text(encoding="ascii").strip()
    except OSError:
        return None
    try:
        return round(int(raw) / 1000.0, 1)
    except ValueError:
        return None


if __name__ == "__main__":
    import json

    sampler = SystemTelemetrySampler()
    print("Sampling every second, Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
            print(json.dumps(sampler.sample().to_payload()))
    except KeyboardInterrupt:
        pass
