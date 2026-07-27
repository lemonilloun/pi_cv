"""BNO08x SHTP-over-SPI driver: calibrated gyro+accel at high rate.

Why this exists alongside `imu_rvc.py`: UART-RVC (the old path) is a
stripped-down BNO08x mode that only ever exposes fused yaw/pitch/roll +
accel at 100 Hz — no raw or calibrated gyroscope at all. ORB-SLAM3
mono-inertial needs a real gyroscope (>=200 Hz ideally) for IMU
preintegration, which RVC cannot provide regardless of how it's read.
SHTP is the BNO08x's full protocol; SPI is the recommended transport
(avoids the chip's non-standard I2C clock-stretching, and is faster than
UART-SHTP) per the July 2026 VIO/SLAM research (`real-time vio_slam.md`).

**Hardware prerequisite**: the sensor must be physically wired for SPI —
this is a breakout-board jumper/solder-pad change plus new GPIO wiring
(CS/INT/RESET), not a software setting. See docs/imu_shtp_setup.md for
the exact steps; nothing in this module can be tested until that's done.

**Why the public `adafruit-circuitpython-bno08x` library, not our own SHTP
parser**: SHTP's binary framing (multi-report packets, sensor-specific
payload layouts, timestamp deltas) is substantial to reimplement
correctly, and Adafruit's library already does it, tested, over SPI/I2C/
UART-SHTP. The one real limitation older versions had — `enable_feature()`
silently capping every report at 20 Hz — was fixed upstream: the current
API takes `report_interval` (microseconds) directly. Pin >=1.3.0 in
requirements (client/requirements-imu-shtp.txt) or this cap silently
comes back.

Depends on `adafruit-circuitpython-bno08x` + `adafruit-blinka` (the
CircuitPython compatibility layer that gives `board`/`busio`/`digitalio`
on Linux SBCs) — see client/requirements-imu-shtp.txt. Neither is
installed by default; `start()` fails gracefully (logs, returns False)
if they're missing, same pattern as `imu_rvc.RvcReader` with pyserial.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Defaults only — verify against your actual wiring (docs/imu_shtp_setup.md).
# CS/INT/RESET are plain GPIOs the library drives directly (not the
# hardware-clocked SPI chip-select on GPIO7/8), so almost any free pin
# works EXCEPT the hardware SPI0 bus itself (GPIO7 CE1, GPIO8 CE0, GPIO9
# MISO, GPIO10 MOSI, GPIO11 SCLK — busio.SPI() below already owns
# 9/10/11). Double-check the AI HAT+ doesn't claim these three before
# wiring, since it sits on the 40-pin header even though its data path is
# the separate M.2/PCIe connector.
DEFAULT_CS_PIN = "D5"
DEFAULT_INT_PIN = "D6"
DEFAULT_RESET_PIN = "D13"
DEFAULT_SPI_HZ = 1_000_000  # BNO08x SPI max is ~3 MHz per datasheet; start conservative
DEFAULT_REPORT_HZ = 200.0  # gyro rate the VIO research calls the practical floor


@dataclass(frozen=True)
class ShtpSample:
    """One combined read. Calibrated units (not raw ADC counts) — SHTP
    report IDs 0x01 (accelerometer) / 0x02 (gyroscope), both already bias-
    and scale-corrected on-chip, which is what ORB-SLAM3's preintegration
    expects."""

    monotonic: float
    accel_ms2: tuple[float, float, float]
    gyro_rads: tuple[float, float, float]


class ShtpReader:
    """Background reader: enables calibrated accel+gyro reports at
    `report_hz` and keeps the latest combined sample, timestamped on the
    host's monotonic clock at read time.

    A thread rather than on-demand reads for the same reason
    `imu_rvc.RvcReader` is: the alternative is a caller-driven read every
    time something wants a sample, which either blocks the caller on SPI
    I/O or hands back a stale value with no timestamp discipline.
    """

    def __init__(
        self,
        cs_pin: str = DEFAULT_CS_PIN,
        int_pin: str = DEFAULT_INT_PIN,
        reset_pin: str = DEFAULT_RESET_PIN,
        spi_hz: int = DEFAULT_SPI_HZ,
        report_hz: float = DEFAULT_REPORT_HZ,
    ) -> None:
        self.cs_pin = cs_pin
        self.int_pin = int_pin
        self.reset_pin = reset_pin
        self.spi_hz = spi_hz
        self.report_interval_us = int(1_000_000 / max(report_hz, 1.0))
        self._bno: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: ShtpSample | None = None
        self._reads_ok = 0
        self._reads_failed = 0

    def start(self) -> bool:
        try:
            import board
            import busio
            from adafruit_bno08x import BNO_REPORT_ACCELEROMETER, BNO_REPORT_GYROSCOPE
            from adafruit_bno08x.spi import BNO08X_SPI
            from digitalio import DigitalInOut
        except ImportError as exc:
            logger.error(
                "SHTP driver deps missing (%s) — "
                "pip install -r client/requirements-imu-shtp.txt", exc,
            )
            return False

        try:
            spi = busio.SPI(board.SCK, board.MOSI, board.MISO)
            cs = DigitalInOut(getattr(board, self.cs_pin))
            int_dio = DigitalInOut(getattr(board, self.int_pin))
            reset_dio = DigitalInOut(getattr(board, self.reset_pin))
            self._bno = BNO08X_SPI(spi, cs, int_dio, reset_dio, baudrate=self.spi_hz)
            self._bno.enable_feature(
                BNO_REPORT_ACCELEROMETER, report_interval=self.report_interval_us
            )
            self._bno.enable_feature(
                BNO_REPORT_GYROSCOPE, report_interval=self.report_interval_us
            )
        except Exception as exc:
            logger.error(
                "Cannot open BNO08x over SPI (%s) — check the wiring against "
                "docs/imu_shtp_setup.md (PS1 jumper, CS/INT/RESET pins).", exc,
            )
            self._bno = None
            return False

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="imu-shtp", daemon=True)
        self._thread.start()
        logger.info(
            "SHTP IMU reader started (SPI @ %d Hz, reports requested @ %.0f Hz)",
            self.spi_hz, 1_000_000.0 / self.report_interval_us,
        )
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._bno = None

    def __enter__(self) -> "ShtpReader":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def _run(self) -> None:
        # `.acceleration`/`.gyro` each drain whatever SHTP packets have
        # arrived and return the latest cached reading (Adafruit's
        # documented usage pattern is exactly this: poll in a tight loop,
        # there is no separate blocking "wait for packet" call at this
        # API level) — so the loop's own rate, not report_interval_us, is
        # what actually limits host-side throughput.
        while not self._stop.is_set():
            try:
                accel = self._bno.acceleration
                gyro = self._bno.gyro
            except Exception as exc:
                self._reads_failed += 1
                logger.warning("SHTP read failed: %s", exc)
                time.sleep(0.05)
                continue
            self._reads_ok += 1
            with self._lock:
                self._latest = ShtpSample(
                    monotonic=time.monotonic(),
                    accel_ms2=(float(accel[0]), float(accel[1]), float(accel[2])),
                    gyro_rads=(float(gyro[0]), float(gyro[1]), float(gyro[2])),
                )

    def latest(self) -> ShtpSample | None:
        with self._lock:
            return self._latest

    def stats(self) -> dict[str, Any]:
        return {"reads_ok": self._reads_ok, "reads_failed": self._reads_failed}
