"""Measure what the RVC link actually delivers, before changing any code.

Run this FIRST, on the Pi, whenever the IMU is suspected:

    ssh cv-pi.local "cd ~/Desktop/work/pi_cv && python3 scripts/imu_rvc_check.py"

Measured on this rig 2026-08-08 (CH340 adapter, /dev/ttyUSB0):
    100.2 Hz, 0 checksum errors, 0 dropped frames, |a| 9.822 m/s^2 at rest.
    Arrival intervals: median 0.00 ms, p95 20.07 ms, 50.2% under 1 ms.

That last line is the finding that matters. Frames arrive in PAIRS — two
back to back, then a 20 ms wait — because the CH340 buffers, and unlike an
FT232RL it exposes no `latency_timer` to turn that off. So host arrival time
is worth up to 20 ms of error per frame and must not be used to timestamp a
sample. The frame `index` counter runs at exactly 100 Hz and is the correct
time base.


Checks the guide's acceptance criteria directly: frame rate, checksum health,
gaps by the frame index counter, arrival jitter (the CH340 buffering question),
and |a| at rest (which proves the byte parse is right).
"""
import sys, time, struct, statistics
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

ser = serial.Serial(PORT, 115200, timeout=0.02)
ser.reset_input_buffer()
raw = bytearray()
ok = bad = dropped = resync = 0
last_idx = None
arrivals = []      # host time per frame
accel_mag = []
yaws = []
t0 = time.monotonic()
while time.monotonic() - t0 < SECS:
    chunk = ser.read(256)
    t = time.monotonic()
    if not chunk:
        continue
    raw.extend(chunk)
    i = 0
    n = len(raw)
    while i <= n - 19:
        if raw[i] == 0xAA and raw[i+1] == 0xAA:
            fr = raw[i:i+19]
            if (sum(fr[2:18]) & 0xFF) == fr[18]:
                idx = fr[2]
                y, p, r, ax, ay, az = struct.unpack('<hhhhhh', bytes(fr[3:15]))
                if last_idx is not None:
                    gap = (idx - last_idx) & 0xFF
                    if gap == 0: gap = 256
                    if gap > 1: dropped += gap - 1
                last_idx = idx
                ok += 1
                arrivals.append(t)
                accel_mag.append(((ax/1000)**2 + (ay/1000)**2 + (az/1000)**2) ** 0.5 * 9.80665)
                yaws.append(y * 0.01)
                i += 19
                continue
            bad += 1; resync += 1; i += 1; continue
        i += 1
    del raw[:i]
ser.close()

dur = arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 0
gaps = [ (arrivals[k+1]-arrivals[k])*1000 for k in range(len(arrivals)-1) ]
gaps_sorted = sorted(gaps)
print(f"кадров ok        : {ok}   ({ok/max(dur,1e-9):.1f} Гц за {dur:.1f} с)")
print(f"checksum ошибок  : {bad}  ({100*bad/max(ok,1):.2f}% от ok)")
print(f"пропущено по idx : {dropped} ({100*dropped/max(ok,1):.2f}%)")
if gaps:
    print(f"интервал прихода : медиана {gaps_sorted[len(gaps)//2]:.2f} мс, "
          f"p95 {gaps_sorted[int(len(gaps)*.95)]:.2f}, макс {gaps_sorted[-1]:.2f}")
    burst = sum(1 for g in gaps if g < 1.0)
    print(f"  из них <1 мс   : {burst} ({100*burst/len(gaps):.1f}%)  <- признак пачек")
if accel_mag:
    print(f"|a| в покое      : медиана {statistics.median(accel_mag):.3f} м/с²  "
          f"(норма 9.75-9.85), σ {statistics.pstdev(accel_mag):.3f}")
if len(yaws) > 2:
    span = (arrivals[-1]-arrivals[0])/60.0
    unwrapped = [yaws[0]]
    for v in yaws[1:]:
        d = v - unwrapped[-1] % 360
        prev = unwrapped[-1]
        raw_d = v - (prev % 360 if prev >= 0 else prev % 360)
        while raw_d > 180: raw_d -= 360
        while raw_d < -180: raw_d += 360
        unwrapped.append(prev + raw_d)
    print(f"дрейф yaw        : {(unwrapped[-1]-unwrapped[0])/max(span,1e-9):.3f} °/мин "
          f"(за {span*60:.0f} с; норма ≤0.5)")
