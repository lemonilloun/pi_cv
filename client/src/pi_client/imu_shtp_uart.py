"""BNO08x SHTP-over-UART driver, hand-rolled for one specific noisy link.

Why not `adafruit_bno08x` (see imu_shtp.py's SPI variant for the "normal"
path): that library has three real correctness bugs that matter a lot on a
lossy link and don't matter much on the clean SPI/I2C links it was written
for:

1. `_separate_batch()` aborts the ENTIRE packet the moment it meets a report
   ID it doesn't have a length for (`_AVAIL_SENSOR_REPORTS`/`_REPORT_LENGTHS`
   only cover a subset of what the BNO085 actually emits by default - Tap/
   Sleep/Tilt Detector etc. are not in there) - so a valid accel report
   sitting right next to an unsupported one in the same batch gets thrown
   away too.
2. `_handle_packet()` doesn't clear `self._packet_slices` when that abort
   happens, so a partial batch's already-parsed slices leak into and get
   mixed with the NEXT successful batch - this produced the outright wrong
   readings we saw (e.g. accel z=126) during testing, not just missing data.
3. `.acceleration`/`.gyro` silently return the last successfully parsed
   value with no "how old is this" signal - when fresh reports keep failing
   to parse, you get a stale reading back with no way to tell.

None of these are exotic: they're just untested on a transport (UART) with
real bit errors. This driver reimplements only what we need (accel + gyro),
built to degrade gracefully instead of failing hard:

- Unknown report IDs inside a batch are skipped using a conservative length
  guess (SH2's common "detector" report shape is 6 bytes) instead of
  aborting the whole packet - a wrong guess only mis-walks the REMAINDER of
  THAT one packet, because packet boundaries are the 0x7E delimiters at the
  transport layer, independent of how we interpret a packet's contents. The
  next packet resyncs cleanly regardless.
- Every reading carries `*_age_s`: real elapsed time since that specific
  axis was last freshly parsed, so a caller can decide what "too stale" is
  instead of unknowingly consuming a stuck old value.
- The background thread drains the UART continuously (no caller-imposed
  sleep between reads) - the FTDI adapter's onboard RX FIFO is small at
  3 Mbaud, and gaps in draining were empirically the single biggest source
  of extra corruption during testing.
- Self-healing: no valid frame for `_STALE_AFTER_S` reopens the port and
  redoes the reset+enable sequence, same reconnect-with-backoff spirit as
  `imu_rvc.RvcReader`.

**This will never be as clean as SPI** - the underlying link genuinely drops
and corrupts bytes (see docs/imu_shtp_setup.md for why SPI was the intended
path, and why it's currently blocked on GPIO access being claimed by the AI
HAT+). What this driver optimizes for is: never crash, never silently hand
back a stale value labeled as fresh, and recover data even when the sensor's
default classifier reports (which we do not disable - a burst of Set
Feature Command writes with no interleaved reads was measured to make loss
WORSE by starving the read side, see commit history) are interleaved with
the accel/gyro we actually want.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

_START = 0x7E
_PROTO_SHTP = 0x01
_ESC = 0x7D
_ESC_XOR = 0x20

_CH_COMMAND = 0
_CH_EXECUTABLE = 1
_CH_CONTROL = 2
_CH_SENSOR_REPORTS = 3
_CH_WAKE_SENSOR_REPORTS = 4
_NUM_CHANNELS = 6  # 0..5, per SH2 - indexing outside this range means desync

_SET_FEATURE_COMMAND = 0xFD

_REPORT_ACCELEROMETER = 0x01
_REPORT_GYROSCOPE = 0x02

_ACCEL_SCALAR = 2.0**-8
_GYRO_SCALAR = 2.0**-9

# Total report byte count (4-byte common header + payload), for reports we
# either care about or reliably see often enough to want an exact length
# rather than the generic fallback. Source: adafruit_bno08x's own
# _AVAIL_SENSOR_REPORTS/_REPORT_LENGTHS tables (confirmed against the
# upstream GitHub source), which DO have these right - the gap is only in
# what's *missing* from those tables.
_KNOWN_REPORT_LENGTHS = {
    0x01: 10,  # ACCELEROMETER
    0x02: 10,  # GYROSCOPE
    0x03: 10,  # MAGNETIC_FIELD
    0x04: 10,  # LINEAR_ACCELERATION
    0x06: 10,  # GRAVITY
    0x05: 14,  # ROTATION_VECTOR
    0x09: 14,  # GEOMAGNETIC_ROTATION_VECTOR
    0x08: 12,  # GAME_ROTATION_VECTOR
    0x11: 12,  # STEP_COUNTER
    0xFB: 5,  # BASE_TIMESTAMP
    0xFA: 5,  # TIMESTAMP_REBASE
}
# SH2's common single-enum/flag "detector" report shape (Tap/Sleep/Tilt/
# Pocket/Circle/Shake/Stability/Significant-Motion Detector all match this
# in the reports the sensor actually sends on this board) - used only as a
# fallback so we can skip PAST an unrecognized report instead of losing the
# rest of the packet's data. Getting this wrong for a given report only
# mis-walks the remainder of that one packet (see module docstring).
_FALLBACK_SHORT_LENGTH = 6

_INTER_BYTE_DELAY = 0.001  # datasheet: >=100us between bytes from host
_MAX_PACKET_BYTES = 4096
_STALE_AFTER_S = 3.0

# Physical plausibility floor - belt-and-suspenders on top of the framing
# fix above. BNO08x's gyro tops out around +-2000dps (~34.9 rad/s) at full
# scale; anything past that literally cannot be a real report. Accel bound
# is generous (a robot cart isn't pulling >2.5g) but tight enough to catch
# leftover corruption the framing fix doesn't.
_MAX_PLAUSIBLE_ACCEL_MS2 = 25.0
_MAX_PLAUSIBLE_GYRO_RADS = 35.0


def walk_reports(payload: bytes) -> tuple[list[tuple[int, bytes]], int]:
    """Split a sensor-report-channel payload into individual SH2 reports.

    Pure function (no I/O) so the batching/fallback-length logic can be
    unit tested without a real serial port - see client/tests/
    test_imu_shtp_uart.py. Returns (reports, unknown_report_count).
    """
    reports: list[tuple[int, bytes]] = []
    unknown = 0
    idx = 0
    n = len(payload)
    while idx < n:
        report_id = payload[idx]
        length = _KNOWN_REPORT_LENGTHS.get(report_id)
        if length is None:
            if report_id >= 0xF0:
                break  # control-style ID, not expected here - stop, keep what we have
            length = _FALLBACK_SHORT_LENGTH
            unknown += 1
        if idx + length > n:
            break
        reports.append((report_id, payload[idx : idx + length]))
        idx += length
    return reports, unknown


def decode_vec3(body: bytes, scalar: float) -> tuple[float, float, float]:
    """4-byte common report header, then 3x int16 LE at offset 4.

    ``"<hhh"`` decodes each field as a *signed* 16-bit little-endian short,
    so negative raw readings (e.g. gravity on an axis pointing away from
    Earth's center, or gyro readings for one rotation direction) come back
    with the correct sign and magnitude out of the box - there is no
    separate two's-complement step to get wrong here. Verified directly:
    ``struct.unpack_from("<hhh", ..., 4)`` round-trips ``-32768``/``-1``
    unchanged. See DecodeVec3Tests in test_imu_shtp_uart.py for negative
    coverage on both this function and the plausibility bounds below (which
    use ``abs()`` and are therefore sign-symmetric by construction).

    Physical axis/rotation convention on THIS board's mounting (per the
    silkscreen and user confirmation, 2026-07-28): **X forward, Y right, Z
    up, and rotation about Z is positive CLOCKWISE as viewed from above.**
    That is a LEFT-handed convention for the Z axis - it is the opposite
    sign from the standard right-hand rule (positive Z rotation =
    counter-clockwise from above) that the rest of this repo's math assumes
    (e.g. `imu_rvc.py`'s yaw/heading handling and `mapping/geometry.py`'s
    bearing math, both ordinary right-handed camera/world frames). This
    decoder is a raw byte-to-float step and takes no position on world-frame
    convention - it does not flip anything - but any future consumer that
    treats `gyro_rads[2]` as "positive = turning left" (the usual robotics
    assumption) will get turns backwards unless it explicitly negates the
    value first. Flagging here since this driver has no integration layer
    yet (see docs/imu_shtp_uart.md) where that negation would otherwise
    naturally live.
    """
    x, y, z = struct.unpack_from("<hhh", body, 4)
    return (x * scalar, y * scalar, z * scalar)


def is_plausible_accel(accel: tuple[float, float, float]) -> bool:
    return max(abs(v) for v in accel) <= _MAX_PLAUSIBLE_ACCEL_MS2


def is_plausible_gyro(gyro: tuple[float, float, float]) -> bool:
    return max(abs(v) for v in gyro) <= _MAX_PLAUSIBLE_GYRO_RADS


@dataclass(frozen=True)
class ImuSample:
    monotonic: float
    accel_ms2: tuple[float, float, float]
    gyro_rads: tuple[float, float, float]
    accel_age_s: float
    gyro_age_s: float


class ShtpUartReader:
    """Background reader for BNO08x SHTP over a plain UART (FTDI-class
    adapter), tolerant of a lossy physical link. See module docstring."""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 3_000_000,
        report_hz: float = 100.0,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.report_interval_us = int(1_000_000 / max(report_hz, 1.0))
        self._serial_mod: Any = None
        self._serial: Any = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._seq = [0] * _NUM_CHANNELS
        self._last_accel: Optional[tuple[float, float, float]] = None
        self._last_accel_t = 0.0
        self._last_gyro: Optional[tuple[float, float, float]] = None
        self._last_gyro_t = 0.0
        self._stats = dict(
            packets_ok=0, resyncs=0, unknown_reports=0, timeouts=0, reconnects=0,
            implausible_dropped=0,
        )

    # ---- public API ----

    def start(self) -> bool:
        try:
            import serial
        except ImportError as exc:
            logger.error("pyserial missing (%s) - pip install pyserial", exc)
            return False
        self._serial_mod = serial
        if not self._open():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="imu-shtp-uart", daemon=True)
        self._thread.start()
        logger.info(
            "SHTP-UART IMU reader started (%s @ %d, reports @ %.0f Hz requested)",
            self.port, self.baudrate, 1_000_000.0 / self.report_interval_us,
        )
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    def __enter__(self) -> "ShtpUartReader":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def latest(self) -> Optional[ImuSample]:
        with self._lock:
            if self._last_accel is None or self._last_gyro is None:
                return None
            now = time.monotonic()
            return ImuSample(
                monotonic=now,
                accel_ms2=self._last_accel,
                gyro_rads=self._last_gyro,
                accel_age_s=now - self._last_accel_t,
                gyro_age_s=now - self._last_gyro_t,
            )

    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    # ---- setup ----

    def _open(self) -> bool:
        try:
            self._serial = self._serial_mod.Serial(self.port, self.baudrate, timeout=0.05)
        except Exception as exc:
            logger.error("cannot open %s: %s", self.port, exc)
            return False
        self._seq = [0] * _NUM_CHANNELS
        self._serial.reset_input_buffer()
        self._reset_and_enable()
        return True

    def _reset_and_enable(self) -> None:
        # Announce handshake on the command channel (mirrors the real SH2
        # sequence a working driver uses - skipping straight to the EXE
        # reset below was measured to leave the sensor in a less
        # predictable state).
        self._send_frame(_CH_COMMAND, bytes([0, 1]))
        time.sleep(0.3)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            frame = self._read_frame(timeout=0.2)
            if frame is None:
                break
            if frame != "resync" and frame[0] == _CH_COMMAND:
                break

        # Real reset via the executable channel (value 1), sent twice -
        # matches the vendor protocol, clears any features left enabled
        # from a previous run instead of our old fake sleep-only "reset".
        self._send_frame(_CH_EXECUTABLE, bytes([1]))
        time.sleep(0.5)
        self._send_frame(_CH_EXECUTABLE, bytes([1]))
        time.sleep(0.5)

        self._serial.reset_input_buffer()
        self._set_feature(_REPORT_ACCELEROMETER, self.report_interval_us)
        time.sleep(0.05)
        self._set_feature(_REPORT_GYROSCOPE, self.report_interval_us)
        time.sleep(0.3)

    def _set_feature(self, report_id: int, interval_us: int) -> None:
        payload = bytearray(17)
        payload[0] = _SET_FEATURE_COMMAND
        payload[1] = report_id
        struct.pack_into("<I", payload, 5, interval_us)
        self._send_frame(_CH_CONTROL, payload)

    # ---- low-level framing ----

    def _read_byte(self, deadline: float) -> Optional[int]:
        while time.monotonic() < deadline:
            b = self._serial.read(1)
            if b:
                return b[0]
        return None

    def _read_unstuffed(self, n: int, deadline: float) -> Optional[bytes]:
        out = bytearray(n)
        for i in range(n):
            b = self._read_byte(deadline)
            if b is None:
                return None
            if b == _ESC:
                b2 = self._read_byte(deadline)
                if b2 is None:
                    return None
                b = b2 ^ _ESC_XOR
            elif b == _START:
                # An UNESCAPED 0x7E here means either the declared packet
                # length was wrong or the byte stream is corrupted - 0x7E is
                # the frame delimiter, so blindly treating it as data would
                # read straight through the real frame boundary into the
                # next frame's sync bytes. That produced exactly the
                # "plausible but wrong" values seen in testing (126.xx /
                # 63.xx - literal byte value 126 = 0x7E leaking into a
                # decoded field). Bail out instead of guessing; the caller
                # discards this frame and resyncs cleanly on the next 0x7E.
                self._stats["resyncs"] += 1
                return None
            out[i] = b
        return bytes(out)

    def _read_frame(self, timeout: float = 0.3):
        """Returns (channel, seq, payload), the sentinel "resync", or None
        on timeout."""
        deadline = time.monotonic() + timeout
        while True:
            b = self._read_byte(deadline)
            if b is None:
                return None
            if b != _START:
                continue
            nxt = self._read_byte(deadline)
            if nxt is None:
                return None
            while nxt == _START:
                nxt = self._read_byte(deadline)
                if nxt is None:
                    return None
            if nxt == _PROTO_SHTP:
                break

        header = self._read_unstuffed(4, deadline)
        if header is None:
            return None
        length = (header[0] | (header[1] << 8)) & 0x7FFF
        channel = header[2]
        seq = header[3]
        if not (0 <= channel < _NUM_CHANNELS) or not (4 <= length <= _MAX_PACKET_BYTES):
            self._stats["resyncs"] += 1
            return "resync"

        payload_len = length - 4
        payload = self._read_unstuffed(payload_len, deadline) if payload_len > 0 else b""
        if payload is None:
            return None
        self._read_byte(time.monotonic() + 0.05)  # best-effort trailing 0x7E
        return (channel, seq, payload)

    def _send_frame(self, channel: int, payload: bytes) -> None:
        seq = self._seq[channel]
        total = len(payload) + 4
        header = bytes([total & 0xFF, (total >> 8) & 0xFF, channel, seq])
        body = header + bytes(payload)
        ser = self._serial
        ser.write(bytes([_START]))
        time.sleep(_INTER_BYTE_DELAY)
        ser.write(bytes([_PROTO_SHTP]))
        time.sleep(_INTER_BYTE_DELAY)
        for b in body:
            ser.write(bytes([b]))
            time.sleep(_INTER_BYTE_DELAY)
        ser.write(bytes([_START]))
        self._seq[channel] = (seq + 1) % 256

    # ---- report parsing ----

    def _handle_sensor_payload(self, payload: bytes) -> None:
        reports, unknown = walk_reports(payload)
        self._stats["unknown_reports"] += unknown
        now = time.monotonic()
        for report_id, body in reports:
            if report_id == _REPORT_ACCELEROMETER and len(body) >= 10:
                accel = decode_vec3(body, _ACCEL_SCALAR)
                if is_plausible_accel(accel):
                    with self._lock:
                        self._last_accel = accel
                        self._last_accel_t = now
                else:
                    self._stats["implausible_dropped"] += 1
            elif report_id == _REPORT_GYROSCOPE and len(body) >= 10:
                gyro = decode_vec3(body, _GYRO_SCALAR)
                if is_plausible_gyro(gyro):
                    with self._lock:
                        self._last_gyro = gyro
                        self._last_gyro_t = now
                else:
                    self._stats["implausible_dropped"] += 1
        self._stats["packets_ok"] += 1

    # ---- main loop ----

    def _run(self) -> None:
        last_any = time.monotonic()
        while not self._stop.is_set():
            frame = self._read_frame(timeout=0.3)
            if frame is None:
                self._stats["timeouts"] += 1
                if time.monotonic() - last_any > _STALE_AFTER_S:
                    logger.warning(
                        "no valid IMU frame for %.1fs, reopening %s", _STALE_AFTER_S, self.port,
                    )
                    self._stats["reconnects"] += 1
                    try:
                        self._serial.close()
                    except Exception:
                        pass
                    if not self._open():
                        time.sleep(1.0)
                    last_any = time.monotonic()
                continue
            if frame == "resync":
                continue
            last_any = time.monotonic()
            channel, _seq, payload = frame
            if channel in (_CH_SENSOR_REPORTS, _CH_WAKE_SENSOR_REPORTS):
                self._handle_sensor_payload(payload)
