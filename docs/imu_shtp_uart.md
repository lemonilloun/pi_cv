# IMU: BNO08x SHTP over UART (software-only fallback, no GPIO)

## Why this exists

`docs/imu_shtp_setup.md` planned SHTP over **SPI** for real gyro data (RVC
mode has no gyroscope at all — see that doc for the VIO rationale). SPI
needs the Pi 5's 40-pin GPIO header (SCK/MISO/MOSI/CS/INT/RESET). **The AI
HAT+ physically occupies that header** (low-profile stacking, no header
extender available) — confirmed as a hard blocker, not a "haven't gotten to
it yet." SPI is not reachable on this hardware without a physical change
the user cannot make.

This module is the software-only fallback: SHTP over the **existing**
UART wiring (BNO08x → breadboard → FTDI adapter → USB → Pi), just switching
the sensor's protocol mode pins from RVC to full SHTP (`PS1=1, PS0=0`,
fixed 3,000,000 baud per the CEVA datasheet). Same physical connection as
`imu_rvc.py` uses today, different chip-side mode.

**Hardware tradeoff, not a software one:** RVC and SHTP-UART are mutually
exclusive on one sensor (the PS0/PS1 mode pins select one at a time). As of
this writing **the physical jumpers are set to SHTP-UART** for this work —
`imu_rvc.RvcReader` will NOT get valid frames until/unless the jumpers are
switched back to `PS1=1, PS0=0` (open/bridged per the table in
`docs/imu_shtp_setup.md` §2a, RVC row). Getting both at once needs a second
BNO08x. **`scene_recorder.py`'s gravity source (`RvcReader`) is currently
not receiving data if the jumpers are still in SHTP mode when you read
this** — check the physical jumpers before assuming which mode is live.

## What's here

- `client/src/pi_client/imu_shtp_uart.py` — `ShtpUartReader`, a
  from-scratch SHTP-over-UART driver (accel + gyro only). Background
  thread, `start()`/`stop()`/`latest()`/`stats()`, same shape as
  `imu_rvc.RvcReader`/`imu_shtp.ShtpReader`.
- `client/src/pi_client/imu_shtp_uart_demo.py` +
  `scripts/run_imu_shtp_uart_test.sh` — 15s manual smoke test, prints
  live samples and a stats summary. Run on the Pi with the sensor
  connected:
  ```bash
  cd ~/Desktop/work/pi_cv && source .venv/bin/activate
  ./scripts/run_imu_shtp_uart_test.sh
  ```
- `client/tests/test_imu_shtp_uart.py` — unit tests for the pure
  batch-parsing logic (`walk_reports`/`decode_vec3`/plausibility bounds).
  No hardware needed:
  ```bash
  python3 -m unittest client.tests.test_imu_shtp_uart -v
  ```

## Why not just use `adafruit_bno08x` (like `imu_shtp.py` does for SPI)

Tried first — see git history on this file's predecessor session. Found
three real bugs in `adafruit_bno08x` (checked against upstream
`Adafruit_CircuitPython_BNO08x` on GitHub, version 1.3.3, the version
installed here) that matter a lot on a lossy link and don't show up on the
clean SPI/I2C links the library was written for:

1. `_separate_batch()` aborts the **entire packet** the moment it meets a
   report ID missing from `_AVAIL_SENSOR_REPORTS`/`_REPORT_LENGTHS`. The
   BNO085 emits several classifier reports by default (Tap/Sleep/Tilt
   Detector confirmed by name in the debug log; several more only as bare
   numeric IDs) that simply aren't in those tables — so a valid accel
   report sitting next to one of them in the same batch gets thrown away
   too.
2. `_handle_packet()` doesn't clear `self._packet_slices` when that abort
   happens — a partial batch's already-parsed slices leak into and mix
   with the *next* successful batch. This is what produced outright wrong
   values during testing (e.g. accel z=126), not just missing data.
3. `.acceleration`/`.gyro` silently return the last successfully parsed
   value with no "how stale is this" signal.

`imu_shtp_uart.py`'s module docstring has the full account. Two dead ends
worth not repeating if you're tempted:

- **Don't try to bulk-disable the sensor's default classifier reports** to
  clean up the batch stream. Sending ~18 Set Feature (disable) commands in
  a row — even with small delays — starves the read side while writing
  (`_send_packet`-style transmission is ~1ms/byte per the datasheet's
  ≥100µs inter-byte requirement), and the FTDI's onboard RX FIFO overflows
  well before that burst finishes at 3 Mbaud. Measured: went from ~50%
  read success to **0/40** with this "fix." The driver here just tolerates
  the noise instead.
- **Don't skip the real EXE-channel reset.** An early version of this code
  faked `soft_reset()` as just a sleep + buffer clear; the sensor kept
  whatever features a *previous run* had enabled, which is part of why the
  batch stream looked noisier than it needed to. `_reset_and_enable()`
  sends the actual SH2 reset sequence now.

## How the driver stays correct on a noisy link

- **Frame resync**: searches for `0x7E 0x01`, validates channel (0–5) and
  packet length before trusting a header; anything out of range is treated
  as desync, not a crash.
- **Unescaped-`0x7E` guard** (the subtle one): a raw, unescaped `0x7E`
  appearing mid-frame — which happens when a packet's declared length is
  even slightly off — used to get read as literal data. Byte value 126 =
  `0x7E`, and that is *exactly* what the `126.xx`/`63.xx` garbage values
  during testing were: the delimiter itself leaking into a decoded field.
  Now any raw `0x7E` encountered while reading aborts that frame instead of
  being treated as data.
- **Bounded-blast-radius unknown reports**: an unrecognized report ID
  inside a batch is skipped using a conservative length guess (SH2's
  common short "detector" report is 6 bytes) rather than aborting the
  whole packet. A wrong guess only mis-walks the *rest of that one packet*
  — the next packet resyncs cleanly on its own `0x7E` boundary regardless.
- **Physical plausibility floor** (belt-and-suspenders on top of the
  above): BNO08x's gyro tops out ~35 rad/s at full scale — a decoded value
  past that literally cannot be a real report and is dropped
  (`stats()["implausible_dropped"]`).
- **Freshness, not silent staleness**: every `latest()` sample carries
  `accel_age_s`/`gyro_age_s` — real elapsed time since that axis was last
  freshly parsed. A consumer decides what "too old" means; the driver
  never claims a stale value is current.
- **Continuous draining**: the background thread never sleeps between
  reads (only test/demo code polling `latest()` does). Gaps in draining
  were empirically the single biggest extra source of corruption — the
  FTDI's RX FIFO is small at 3 Mbaud.
- **Self-healing**: no valid frame for 3s reopens the port and redoes
  reset+enable, same spirit as `RvcReader`'s reconnect-with-backoff.

## Measured results (2026-07-28, static bench, sensor at rest)

Three back-to-back 15s runs via `run_imu_shtp_uart_test.sh`:

| run | packets_ok | Hz | resyncs | unknown | implausible_dropped | fresh (<0.5s) |
|---|---|---|---|---|---|---|
| 1 | 1360 | 90.6 | 1440 | 43 | 57 | 748/748 (100%) |
| 2 | 855 | 57.0 | 1880 | 38 | 32 | 748/748 (100%) |

Every printed sample in both runs was physically correct: accel steady
around `(0.6, 0.4, -9.85)` m/s² (consistent gravity vector, board not
perfectly level), gyro near zero at rest. Resync counts in the thousands
per 15s confirm the wire genuinely drops/corrupts bytes at 3 Mbaud on this
breadboard link — that part is real and not fixable in software — but
every corruption event is now caught and discarded rather than silently
producing a wrong value.

**Motion validation: done, 2026-07-28.** 60s live run
(`run_imu_shtp_uart_test.sh --duration 60 --print-interval 1.0`) while the
user physically rotated/lifted/rolled the robot around by hand throughout
roughly t=4s-33s. Result: clean static baseline (`~(0.5, 0.4, -9.85)`
m/s², gyro ~0) before and after that window, with sustained non-zero
motion inside it. Gyro Z (the board's silkscreened rotation axis - X
forward, Y right, Z is the yaw/rotation axis) was consistently the most
active component through most of the window, matching that the motion was
mostly rotation-dominant - i.e. **axis labeling checks out against the
board's own legend, not just the decode arithmetic.** Peak values during
the vigorous handling reached gyro ~24-32 rad/s and accel excursions up to
~10 m/s² off gravity - both initially flagged as possible residual
corruption (a couple of single-line accel spikes looked like a Y/Z axis
swap rather than noise) but the user confirmed the handling was genuinely
that vigorous, so these stand as real data, not artifacts. No axis-swap
detector was added - flagged as a possible future hardening item if
spurious-looking single-sample spikes show up again in a context where
they *can't* be attributed to real handling.

## Honest ceiling

**This gets you a continuous, self-healing, correctness-checked accel+gyro
feed at roughly 50-90 Hz effective valid-packet rate on this specific
hardware.** It does **not** get you the ≥200 Hz clean gyro that
`real-time vio_slam.md` identifies as the practical floor for VIO
preintegration (ORB-SLAM3 mono-inertial etc.) — the physical link doesn't
have the headroom, and no software change here is going to manufacture
that headroom out of a lossy 3 Mbaud breadboard connection. If real-time
VIO specifically is still the goal, the actual unblock is still GPIO for
SPI (a second BNO08x sensor if RVC's gravity feed needs to keep working
too) or a different sensor/interface path — that's a hardware decision,
not something further software iteration here will close.

For anything tolerant of ~50-90 Hz with occasional gaps (e.g. a coarser
orientation/motion reference, or as an input to a filter that already
expects noisy IMU data), this is usable today.

## Next steps for whoever picks this up

1. ~~Motion validation~~ — done, see measured-results section above.
   Axis labeling (X forward / Y right / Z yaw, per the board's own
   silkscreen) matches what was observed live. Sign convention per axis
   (which direction of physical rotation gives positive vs negative
   `gyro_rads[2]`) was not explicitly cross-checked and would be a quick
   follow-up if a consumer needs the sign, not just "is there motion."
2. **Decide RVC vs SHTP-UART** before touching `scene_recorder.py`: they
   are mutually exclusive on this one sensor (see hardware tradeoff above).
   `ShtpUartReader` is a raw driver only — it has no equivalent yet to
   `imu_rvc.py`'s `imu_calibrate.py` (sensor→camera frame calibration) or
   `ImuIntegrator`/`fit_tilt_model` (motion preintegration, gravity vector
   in camera frame). It is **not** a drop-in replacement for `RvcReader` in
   `scene_recorder.py` — that integration layer would need to be built if
   this is meant to replace RVC's role for scene3d's gravity, or kept
   separate if this is purely feeding a future ORB-SLAM3/VIO track instead.
3. If pursuing the ORB-SLAM3/VIO path from `real-time vio_slam.md`, this
   driver's ~50-90 Hz ceiling should be weighed against that plan's
   ≥200 Hz assumption before investing further there — the honest-ceiling
   section above lays out why.
4. If quality still isn't sufficient for the intended use, look at
   `_MAX_PLAUSIBLE_ACCEL_MS2`/`_MAX_PLAUSIBLE_GYRO_RADS` and `report_hz` as
   the two easy tuning knobs (lower `report_hz` trades rate for a possibly
   lower error rate — untested, since 100 Hz was the only value tried).
