"""Guided IMU distance-calibration wizard, driven from the Pi and mirrored
live in the web panel.

The measurement this runs (see imu_distance_analysis.py for the method) is
only valid if the operator actually holds the cart still at both ends of
every run — the whole thing rests on those two boundary conditions. A
protocol that depends on the operator remembering instructions from a chat
log is a protocol that silently produces unusable data, which is exactly
what happened to the first round of logs (`data/imu_tests/forward.csv`:
zero still samples at either end, unanalysable). So the wizard owns the
timeline: it announces each phase, counts it down, and pushes that state to
the server ~5x/second as `imu_cal_state` messages, which the panel renders
as a large instruction plus a progress bar.

Runs entirely on the Pi and needs no camera — the persistent session can
keep the camera for itself while this runs.

Usage (on the Pi):
    ./scripts/run_imu_calibration.sh --truth-m 3.0 --runs 3

The server connection is optional: with no server reachable the wizard
still prints every instruction to the terminal and writes the same CSVs,
so calibration is never blocked on the panel being up.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "client/src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import numpy as np

from pi_client.imu_distance_analysis import RunAnalysis, analyze_run, summarize
from pi_client.network import ClientConnectionError, PiClient
from pi_client.protocol import make_imu_cal_state_message
from shared.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class Phase:
    key: str
    title: str
    detail: str
    seconds: float
    recording: bool


def build_phases(
    run_idx: int, runs: int, truth_m: float, settle_s: float, move_s: float, ready_s: float
) -> list[Phase]:
    """One distance run.

    The still head/tail are what make the bias observable. Two protocol
    decisions come from a real failed session (cal_20260731_143730):

    ALTERNATING DIRECTION. Repeating a push in one direction needs
    runs x truth_m of clear floor — 9 m for three 3 m runs, which no
    ordinary room has. Odd runs go one way, even runs come straight back
    along the same line, so the cart is always already at the start of the
    next run and never has to be repositioned or turned. The analysis
    measures path LENGTH, so direction is irrelevant to it.

    A READY PHASE THAT RECORDS NOTHING. The first version marched from one
    run straight into the next, so the operator had to turn the cart around
    inside the timed window — which put a rotation in the middle of what
    was supposed to be a straight push and destroyed the run. This phase
    exists purely to let the operator walk to the other end and take
    position.
    """
    label = f"Заезд {run_idx}/{runs}"
    heading = "ВПЕРЁД" if run_idx % 2 == 1 else "ОБРАТНО"
    where = (
        "Встаньте у ближнего конца отрезка, тележка носом вперёд."
        if run_idx % 2 == 1
        else "Тележка уже на дальнем конце — перейдите туда, катить будете обратно по той же линии."
    )
    return [
        Phase("ready", "ПРИГОТОВЬТЕСЬ", f"{label} — {where} Ничего не записываю.", ready_s, False),
        Phase(
            "settle_start",
            "СТОЙТЕ НЕПОДВИЖНО",
            f"{label} — замеряю опорный вектор гравитации. Не трогайте тележку.",
            settle_s,
            True,
        ),
        Phase(
            "move",
            f"КАТИТЕ {heading} {truth_m:g} М",
            f"{label} — по прямой, БЕЗ поворотов. Докатили — сразу стойте, не разворачивайте.",
            move_s,
            True,
        ),
        Phase(
            "settle_end",
            "СТОП — СТОЙТЕ НЕПОДВИЖНО",
            f"{label} — замеряю нулевую скорость. Это самая важная фаза, не двигайте тележку.",
            settle_s,
            True,
        ),
    ]


class StatePublisher:
    """Best-effort mirror of wizard state to the server. Never raises into
    the wizard: a dropped panel connection must not abort a physical
    measurement the operator is in the middle of performing."""

    def __init__(self, device_id: str, host: str, port: int) -> None:
        self.device_id = device_id
        self.host = host
        self.port = port
        self._client: PiClient | None = None
        self._warned = False

    def _ensure(self) -> PiClient | None:
        if self._client is not None:
            return self._client
        try:
            client = PiClient(self.host, self.port, timeout_seconds=3.0)
            client.connect()
        except ClientConnectionError:
            if not self._warned:
                logger.warning("Panel mirroring off (no server at %s:%s) — wizard continues", self.host, self.port)
                self._warned = True
            return None
        self._client = client
        return client

    def push(self, payload: dict) -> None:
        client = self._ensure()
        if client is None:
            return
        try:
            client.send_message(make_imu_cal_state_message(self.device_id, payload))
            client.receive_response()
        except (ClientConnectionError, OSError):
            self._client = None

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except OSError:
                pass
            self._client = None


def _write_csv(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_ns", "gx", "gy", "gz", "ax", "ay", "az"])
        writer.writerows(rows)


def run_phase(
    reader,
    phase: Phase,
    publisher: StatePublisher,
    base_state: dict,
    poll_s: float = 0.002,
) -> list[tuple]:
    rows: list[tuple] = []
    last_accel_t: float | None = None
    last_gyro_t: float | None = None
    t_start = time.monotonic()
    t_end = t_start + phase.seconds
    last_push = 0.0

    print(f"\n>>> {phase.title} ({phase.seconds:.0f}s) — {phase.detail}")
    while True:
        now = time.monotonic()
        remaining = t_end - now
        if remaining <= 0:
            break
        sample = reader.latest()
        if sample is not None and phase.recording:
            accel_t = sample.monotonic - sample.accel_age_s
            gyro_t = sample.monotonic - sample.gyro_age_s
            if accel_t != last_accel_t or gyro_t != last_gyro_t:
                last_accel_t, last_gyro_t = accel_t, gyro_t
                gx, gy, gz = sample.gyro_rads
                ax, ay, az = sample.accel_ms2
                rows.append((int(max(accel_t, gyro_t) * 1e9), gx, gy, gz, ax, ay, az))
        if now - last_push >= 0.2:
            last_push = now
            state = dict(base_state)
            state.update(
                {
                    "phase": phase.key,
                    "title": phase.title,
                    "detail": phase.detail,
                    "remaining_s": round(max(0.0, remaining), 1),
                    "phase_seconds": phase.seconds,
                    "progress": round(1.0 - max(0.0, remaining) / max(phase.seconds, 1e-6), 3),
                    "samples": len(rows),
                    "at": time.time(),
                }
            )
            publisher.push(state)
            print(f"    {phase.title}  {remaining:4.1f}s  ({len(rows)} samples)", end="\r", flush=True)
        time.sleep(poll_s)
    print()
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Guided IMU distance calibration")
    p.add_argument("--truth-m", type=float, required=True, help="Tape-measured push distance, metres")
    p.add_argument("--runs", type=int, default=3, help="Repeats — the SPREAD is what shows bias stability")
    p.add_argument("--settle-seconds", type=float, default=4.0)
    p.add_argument("--move-seconds", type=float, default=6.0,
                   help="Integration error grows with the SQUARE of this, the signal only "
                        "linearly — push briskly and keep it short rather than ambling")
    p.add_argument("--ready-seconds", type=float, default=15.0,
                   help="Untimed-feeling gap between runs to walk to the other end; records nothing")
    p.add_argument("--static-seconds", type=float, default=60.0,
                   help="Opening pure-stillness log measuring the gravity leak; 0 to skip")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--calibration", type=Path, default=REPO_ROOT / "config/imu_calibration_shtp.json",
                   help="Stored calibration whose up_imu_reference the live integrator uses — "
                        "the gravity leak is the angle between it and each run's true gravity")
    p.add_argument("--imu-port", default="/dev/ttyUSB0")
    p.add_argument("--imu-baud", type=int, default=3000000)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--device-id", default=None)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(REPO_ROOT / "config/default.json", REPO_ROOT / ".env")

    # `client.server_host` (which .env's PI_CV_SERVER_HOST overrides), NOT
    # `server.host` — the latter is the server's BIND address, 0.0.0.0, which
    # the Pi cannot connect to. Same keys pi_navigation.py uses.
    host = args.host or config.get("client", {}).get("server_host", "127.0.0.1")
    port = args.port or int(config.get("server", {}).get("port", 8765))
    device_id = args.device_id or str(config.get("client", {}).get("device_id", "raspberry_pi_01"))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (REPO_ROOT / "data/imu_tests" / f"cal_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_up = None
    if args.calibration.exists():
        try:
            reference_up = json.loads(args.calibration.read_text(encoding="utf-8")).get(
                "up_imu_reference"
            )
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s (%s) — tilt error will not be measured",
                           args.calibration, exc)
    if reference_up is None:
        logger.warning("No stored gravity reference: runs will still measure distance, but "
                       "the tilt error against the live integrator's reference cannot be "
                       "computed (run scripts/run_imu_calibrate.sh first)")

    from pi_client.imu_shtp_uart import ShtpUartReader

    reader = ShtpUartReader(port=args.imu_port, baudrate=args.imu_baud)
    if not reader.start():
        raise SystemExit(
            f"IMU failed to start on {args.imu_port} — check the PS1/PS0 jumpers are in SHTP "
            f"mode, not RVC (docs/imu_shtp_uart.md; the two are mutually exclusive)"
        )
    time.sleep(1.0)

    publisher = StatePublisher(device_id, host, port)
    base = {
        "session": stamp,
        "truth_m": args.truth_m,
        "runs_total": args.runs,
        "out_dir": str(out_dir),
    }
    results: list[RunAnalysis] = []

    try:
        if args.static_seconds > 0:
            phase = Phase(
                "static",
                "СТОЙТЕ НЕПОДВИЖНО",
                "Базовый замер утечки гравитации. Тележка просто стоит, ничего делать не нужно.",
                args.static_seconds,
                True,
            )
            rows = run_phase(reader, phase, publisher, {**base, "run": 0, "stage": "static"})
            _write_csv(out_dir / "static.csv", rows)
            print(f"    -> {out_dir / 'static.csv'} ({len(rows)} samples)")

        for run_idx in range(1, args.runs + 1):
            rows: list[tuple] = []
            for phase in build_phases(
                run_idx, args.runs, args.truth_m, args.settle_seconds, args.move_seconds,
                args.ready_seconds,
            ):
                rows.extend(
                    run_phase(
                        reader,
                        phase,
                        publisher,
                        {**base, "run": run_idx, "stage": "distance"},
                    )
                )
            csv_path = out_dir / f"run_{run_idx}.csv"
            _write_csv(csv_path, rows)

            arr = np.asarray(rows, dtype=np.float64)
            # The wizard's own phase timings beat re-detecting stillness:
            # it is what told the operator to stop, and the detector has to
            # infer that from a signal where rolling barely differs from
            # parked. Trimmed slightly inwards so a late stop or an early
            # push doesn't contaminate the reference windows.
            margin = min(1.0, args.settle_seconds * 0.25)
            res = analyze_run(
                arr[:, 0] * 1e-9,
                arr[:, 1:4],
                arr[:, 4:7],
                truth_m=args.truth_m,
                head_s=args.settle_seconds - margin,
                tail_s=args.settle_seconds - margin,
                reference_up=reference_up,
            )
            results.append(res)
            if res.ok:
                print(
                    f"    run {run_idx}: distance {res.corrected_distance_m} m "
                    f"(truth {args.truth_m} m, error {res.error_m:+.3f} m), "
                    f"bias |{res.bias_mag_ms2}| m/s²"
                )
            else:
                print(f"    run {run_idx}: UNUSABLE — {res.reason}")
            publisher.push(
                {
                    **base,
                    "run": run_idx,
                    "phase": "run_done",
                    "title": f"Заезд {run_idx} готов",
                    "detail": res.verdict or res.reason,
                    "result": res.as_dict(),
                    "at": time.time(),
                }
            )

        summary = summarize(results)
        (out_dir / "summary.json").write_text(
            json.dumps(
                {"summary": summary, "runs": [r.as_dict() for r in results]},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print("\n=== SUMMARY ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
        publisher.push(
            {
                **base,
                "phase": "done",
                "title": "Калибровка завершена",
                "detail": summary.get("verdict_text", summary.get("reason", "")),
                "summary": summary,
                "results": [r.as_dict() for r in results],
                "at": time.time(),
            }
        )
        print(f"\nArtifacts: {out_dir}")
    finally:
        reader.stop()
        publisher.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
