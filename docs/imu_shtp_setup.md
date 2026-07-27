# IMU: switching the BNO08x from UART-RVC to SHTP/SPI

Why: UART-RVC (the current wiring) only ever exposes fused yaw/pitch/roll +
accelerometer at 100 Hz — **no gyroscope at all**. ORB-SLAM3 mono-inertial
(and any real VIO/VI-SLAM) needs a real gyroscope, ideally >=200 Hz, for IMU
preintegration. RVC cannot provide that no matter how it's read — this is a
protocol limitation of the BNO08x's stripped-down RVC mode, not a driver bug.
SHTP is the chip's full protocol; SPI is the recommended transport (avoids
the chip's non-standard I2C clock-stretching, faster than UART-SHTP). See
`real-time vio_slam.md` for the full research this is based on.

**This is a one-way change for the current sensor.** Switching its mode
pins from UART to SPI means it stops responding over UART entirely —
`imu_rvc.RvcReader` (and scene3d's gravity/`tilt_model` feature that
depends on it) will have no data source until/unless a SHTP-based
equivalent is written. If you want to keep both working at once, you need
a second BNO08x; if not, that's an accepted tradeoff for this experiment.

## 1. Software (already done, in this repo)

- `client/src/pi_client/imu_shtp.py` — new driver, `ShtpReader`, produces
  calibrated `accel_ms2`/`gyro_rads` samples at a configurable rate.
- `client/requirements-imu-shtp.txt` — `adafruit-circuitpython-bno08x>=1.3.0`
  + `adafruit-blinka`. The version pin matters: older releases silently
  capped every report at 20 Hz regardless of what you asked for; 1.3.0+
  exposes `report_interval` on `enable_feature()` directly.
- `imu_rvc.py` is untouched — nothing here removes UART-RVC support in
  software, only the physical rewiring below makes it stop responding.

## 2. Hardware — what you need to do

**Board assumed: Adafruit BNO08x breakout (STEMMA QT, PID 4754/4998).** If
you have a different breakout, the PS0/PS1 concept is the same (it's the
chip's own reference design) but the exact solder-pad layout may differ —
check your board's silkscreen/schematic before bridging anything.

### 2a. Solder jumpers (protocol select)

The board's `PS1` jumper is currently bridged alone → UART mode (today's
setup). For SPI, **bridge `PS0` as well, in addition to the existing `PS1`
bridge** (both pads shorted, per the Adafruit BNO085 guide — do not
un-bridge PS1, and do not connect either pad to your own 3.3V/5V rail,
only to the pad's own paired trace as the board is designed):

| PS1 | PS0 | Mode |
|---|---|---|
| open | open | I2C (board default) |
| open | bridged | HID I2C |
| **bridged** | open | **UART-RVC (current)** |
| **bridged** | **bridged** | **SPI (target)** |

### 2b. Wiring to the Pi 5's 40-pin header

The breakout's I2C-labeled pins double as SPI pins:

| Breakout pin | Function | Pi 5 physical pin | BCM GPIO |
|---|---|---|---|
| SCL | SPI SCK | 23 | GPIO11 |
| SDA | SPI MISO | 21 | GPIO9 |
| DI | SPI MOSI | 19 | GPIO10 |
| CS | Chip select | 29 | GPIO5 |
| INT | Interrupt (**required** for stable SPI, per Adafruit) | 31 | GPIO6 |
| RST | Reset (**required** for stable SPI) | 33 | GPIO13 |
| 3V | Power | 1 or 17 | 3.3V |
| GND | Ground | any GND pin | — |

CS/INT/RST are plain GPIOs the driver toggles directly, not the Pi's
hardware SPI chip-select — any free GPIO works for them, but the three
above are chosen to avoid the hardware SPI0 bus itself (GPIO7 CE1, GPIO8
CE0, GPIO9 MISO, GPIO10 MOSI, GPIO11 SCLK) and match `imu_shtp.py`'s
defaults. **Double-check none of GPIO5/6/9/10/11/13 are already claimed by
the Hailo AI HAT+** before wiring (it sits on the 40-pin header even
though its actual data path is the separate M.2/PCIe connector) — if one
conflicts, pick a different free GPIO and pass it to `ShtpReader(...)`.

### 2c. Enable SPI on the Pi

```bash
sudo raspi-config nonint do_spi 0   # enables SPI0, no reboot-menu needed
sudo reboot
ls /dev/spidev*                     # should list spidev0.0 (and 0.1)
```

### 2d. Install and smoke-test

```bash
cd ~/Desktop/work/pi_cv
source .venv/bin/activate
pip install -r client/requirements-imu-shtp.txt
python3 -c "
import sys; sys.path.insert(0, 'client/src')
from pi_client.imu_shtp import ShtpReader
import time
r = ShtpReader()
if not r.start():
    sys.exit('failed to start - check wiring/jumpers above')
time.sleep(1.0)
for _ in range(10):
    s = r.latest()
    print(s)
    time.sleep(0.1)
print('stats:', r.stats())
r.stop()
"
```

Expect `accel_ms2` near `(0, 0, ±9.8)` at rest (whichever axis is "up" for
your mounting) and `gyro_rads` near `(0, 0, 0)` — a live sanity check is
gently rotating the board and watching the corresponding gyro axis move.
If `start()` returns `False` or raises, the error message names the
missing dependency or the SPI failure; check jumpers/wiring before
anything else.

## 3. Disk footprint

`adafruit-circuitpython-bno08x` + `adafruit-blinka` are small pure-Python
packages (well under 10 MB installed) — negligible against the Pi's 64 GB
microSD, unlike the ROS2/ORB-SLAM3 step that follows this one.
