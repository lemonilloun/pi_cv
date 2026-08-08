#!/usr/bin/env python3
"""Подбор подстройки колёс по IMU: короткие проезды, без стенки и рулетки.

Запускать НА PI (там IMU).

    python3 scripts/measure_wheel_trim.py

Почему по IMU, а не по боковому сносу. Приёмочный тест B намерил дрейф курса
0.003 град/мин, то есть за двухсекундный проезд датчик врёт на десятитысячные
градуса. Значит ЛЮБОЙ разворот, случившийся при команде «строго вперёд», —
это увод, а не погрешность. Курс ловит его в разы чувствительнее рулетки:
увод в 3 градуса за 2 секунды — это всего 5 см бокового сноса на метре пути,
померить которые по полу почти невозможно, а по курсу видно сразу.

Ни ширина колеи, ни скорость, ни расстояние не нужны: подстройка ищется
итерациями. Скрипт говорит, какое колесо ослабить и насколько, вы вписываете
число, повторяете. Обычно хватает трёх подходов.

Порядок:
  1. Робота на пол, свободного места метра полтора вперёд.
  2. Запустить скрипт, нажать Enter.
  3. Дать РОВНО вперёд (W) на 1.5-2 секунды и отпустить.
  4. Скрипт скажет, куда увело и что поправить.
  5. Вписать предложенные множители в config/default.json -> robot.wheel_trim
     (скрипт делает это сам, если запущен с --apply).
  6. Повторить, пока увод не станет меньше 1 град/с.
"""

from __future__ import annotations

import json
import statistics
import struct
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PORT = "/dev/ttyUSB0"
CONFIG_PATH = REPO_ROOT / "config/default.json"

# Ниже этого увод уже не отличим от неровностей пола и люфта — дальше
# подстраивать значит подгонять под конкретную половицу.
GOOD_ENOUGH_DPS = 1.0
# Доля, на которую ослабляется быстрое колесо за одну итерацию. Меньше 1
# намеренно: перелёт заставляет робота вилять в другую сторону, и итерации
# начинают скакать вокруг решения вместо того чтобы к нему сходиться.
STEP_GAIN = 0.6
# Шасси не поедет прямее, чем позволяет разница самих моторов; ограничение
# страхует от опечатки, которая иначе развернула бы робота на месте.
TRIM_MIN = 0.70


def read_yaw_burst() -> tuple[float, float] | None:
    """Пишет курс, пока идёт проезд. Возвращает (изменение курса, длительность).

    Начало и конец определяются по самому курсу и ускорению: ждём, пока робот
    зашевелится, и останавливаемся, когда он затих. Так не нужно ловить
    нажатие клавиши на другой машине.
    """
    import serial

    ser = serial.Serial(PORT, 115200, timeout=0.02)
    ser.reset_input_buffer()
    raw = bytearray()
    samples: list[tuple[float, float, float]] = []   # t, yaw, |a|
    yaw_unwrapped = None
    yaw_prev = None
    t0 = time.monotonic()
    moving_since = None
    still_since = None

    while time.monotonic() - t0 < 30.0:
        chunk = ser.read(256)
        now = time.monotonic()
        if not chunk:
            continue
        raw.extend(chunk)
        i, n = 0, len(raw)
        while i <= n - 19:
            if raw[i] == 0xAA and raw[i + 1] == 0xAA:
                fr = raw[i:i + 19]
                if (sum(fr[2:18]) & 0xFF) == fr[18]:
                    y, _, _, ax, ay, az = struct.unpack("<hhhhhh", bytes(fr[3:15]))
                    yaw = y * 0.01
                    if yaw_unwrapped is None:
                        yaw_unwrapped = yaw
                    else:
                        yaw_unwrapped += (yaw - yaw_prev + 180.0) % 360.0 - 180.0
                    yaw_prev = yaw
                    mag = (ax * ax + ay * ay + az * az) ** 0.5 / 1000.0 * 9.80665
                    samples.append((now, yaw_unwrapped, mag))
                    i += 19
                    continue
                i += 1
                continue
            i += 1
        del raw[:i]

        if len(samples) < 40:
            continue
        recent = samples[-30:]
        spread = statistics.pstdev([s[2] for s in recent])
        turning = abs(recent[-1][1] - recent[0][1]) / max(
            recent[-1][0] - recent[0][0], 1e-3)
        active = spread > 0.12 or abs(turning) > 3.0

        if active and moving_since is None:
            moving_since = recent[0][0]
            still_since = None
        elif not active and moving_since is not None:
            if still_since is None:
                still_since = now
            elif now - still_since > 0.5:
                break
        elif active:
            still_since = None
    ser.close()

    if moving_since is None:
        return None
    end = still_since or samples[-1][0]
    during = [s for s in samples if moving_since <= s[0] <= end]
    if len(during) < 20:
        return None
    return (during[-1][1] - during[0][1], during[-1][0] - during[0][0])


def load_trim() -> tuple[float, float]:
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return (1.0, 1.0)
    trim = (cfg.get("robot") or {}).get("wheel_trim") or {}
    return (float(trim.get("left", 1.0)), float(trim.get("right", 1.0)))


def save_trim(left: float, right: float) -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg.setdefault("robot", {})["wheel_trim"] = {
        "left": round(left, 4), "right": round(right, 4),
        "measured_by": "scripts/measure_wheel_trim.py",
    }
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")


def main() -> int:
    apply = "--apply" in sys.argv
    left, right = load_trim()
    print(__doc__.split("Порядок:")[1].strip())
    print(f"\nтекущая подстройка: left={left:.3f} right={right:.3f}")
    if not apply:
        print("(запустите с --apply, чтобы скрипт сам вписывал её в конфиг)")

    attempt = 0
    try:
        while True:
            attempt += 1
            input(f"\n[{attempt}] Enter, потом дайте РОВНО вперёд на 1.5-2 с и отпустите...")
            print("    жду проезд…", flush=True)
            burst = read_yaw_burst()
            if burst is None:
                print("    движения не заметил — попробуйте ещё раз")
                continue
            turned_deg, duration = burst
            rate = turned_deg / max(duration, 1e-3)
            side = "ВЛЕВО" if turned_deg > 0 else "ВПРАВО"
            print(f"    проезд {duration:.2f} с, увело на {abs(turned_deg):.1f}° "
                  f"{side}  ({abs(rate):.1f}°/с)")

            if abs(rate) < GOOD_ENOUGH_DPS:
                print(f"    Это уже меньше {GOOD_ENOUGH_DPS}°/с — дальше подстраивать "
                      f"нечего, остальное это неровности пола.")
                break

            # Уводит влево -> левое колесо отстаёт -> ослабляем ПРАВОЕ.
            # Знак курса на этом датчике уже нормализован (YAW_SIGN), поэтому
            # положительный поворот это влево.
            correction = 1.0 - STEP_GAIN * min(0.25, abs(rate) / 45.0)
            if turned_deg > 0:
                right = max(TRIM_MIN, right * correction)
                which = "ПРАВОЕ"
            else:
                left = max(TRIM_MIN, left * correction)
                which = "ЛЕВОЕ"
            # Один множитель всегда 1.0: ослаблять оба значит терять скорость
            # без всякой пользы для прямолинейности.
            top = max(left, right)
            left, right = left / top, right / top
            print(f"    -> ослабляю {which} колесо: left={left:.3f} right={right:.3f}")
            if apply:
                save_trim(left, right)
                print(f"    записано в {CONFIG_PATH}")
                print("    ПЕРЕЗАПУСТИТЕ сервер на маке, иначе новая подстройка не применится")
            else:
                print(f'    впишите вручную: "wheel_trim": {{"left": {left:.3f}, '
                      f'"right": {right:.3f}}}')
    except (KeyboardInterrupt, EOFError):
        print()

    print(f"\nитог: left={left:.3f} right={right:.3f}")
    if not apply:
        print("Скрипт запускался без --apply, конфиг не тронут.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
