#!/usr/bin/env python3
"""Приёмочные тесты IMU по руководству BNO085_RVC_guide.md, §5.

Запускать НА PI. Каждый тест печатает измерение и вердикт — ничего
интерпретировать на глаз не нужно.

    python3 scripts/imu_acceptance.py A     # магнитометр (2 мин)
    python3 scripts/imu_acceptance.py B     # дрейф и шум (по умолчанию 10 мин)
    python3 scripts/imu_acceptance.py E     # повторяемость наклона (5 мин)
    python3 scripts/imu_acceptance.py all   # A, потом B, потом E

Результаты дописываются в data/imu_tests/acceptance_<дата>.json, чтобы их
можно было сравнить после смены крепления или прошивки.

Порядок важен: тест A определяет, можно ли вообще доверять курсу рядом с
моторами, и от этого зависит, сколько сил вкладывать в визуальную коррекцию.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "client/src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

PORT = "/dev/ttyUSB0"


def collect(seconds: float, label: str) -> list[dict]:
    """Пишет поток RVC заданное время. Возвращает список отсчётов."""
    import struct

    import serial

    ser = serial.Serial(PORT, 115200, timeout=0.02)
    ser.reset_input_buffer()
    raw = bytearray()
    out: list[dict] = []
    yaw_unwrapped = None
    yaw_prev = None
    t0 = time.monotonic()
    next_tick = t0 + 5.0
    while time.monotonic() - t0 < seconds:
        chunk = ser.read(256)
        now = time.monotonic()
        if now >= next_tick:
            left = seconds - (now - t0)
            print(f"    {label}: осталось {left:5.0f} с", flush=True)
            next_tick = now + 5.0
        if not chunk:
            continue
        raw.extend(chunk)
        i, n = 0, len(raw)
        while i <= n - 19:
            if raw[i] == 0xAA and raw[i + 1] == 0xAA:
                fr = raw[i:i + 19]
                if (sum(fr[2:18]) & 0xFF) == fr[18]:
                    y, p, r, ax, ay, az = struct.unpack("<hhhhhh", bytes(fr[3:15]))
                    yaw = y * 0.01
                    if yaw_unwrapped is None:
                        yaw_unwrapped = yaw
                    else:
                        yaw_unwrapped += (yaw - yaw_prev + 180.0) % 360.0 - 180.0
                    yaw_prev = yaw
                    out.append({
                        "t": now, "yaw": yaw_unwrapped,
                        "pitch": p * 0.01, "roll": r * 0.01,
                        "ax": ax, "ay": ay, "az": az,
                    })
                    i += 19
                    continue
                i += 1
                continue
            i += 1
        del raw[:i]
    ser.close()
    return out


def _accel_ms2(s: dict) -> float:
    return math.sqrt(s["ax"] ** 2 + s["ay"] ** 2 + s["az"] ** 2) / 1000.0 * 9.80665


def _yaw_span(samples: list[dict]) -> float:
    return max(s["yaw"] for s in samples) - min(s["yaw"] for s in samples)


def test_a() -> dict:
    """Активен ли магнитометр — и портят ли курс сами моторы.

    RVC не сообщает, какой алгоритм фузии внутри. Если магнитометр включён,
    магниты и токи моторов будут уводить курс, и yaw станет пригоден только
    как относительный на коротких окнах. Это решение влияет на всё остальное,
    поэтому тест первый.
    """
    print("\n=== ТЕСТ A: магнитометр и помехи от моторов ===")
    print("Робот НЕПОДВИЖЕН на столе. Моторы выключены.")
    input("Нажмите Enter и не трогайте робота 30 секунд...")
    base = collect(30.0, "фон")

    print("\nТеперь поднесите отвёртку или магнит на ~3 см к модулю IMU.")
    input("Поднесли? Нажмите Enter, держите 20 секунд...")
    magnet = collect(20.0, "магнит")

    print("\nУберите магнит подальше.")
    input("Убрали? Нажмите Enter, ещё 30 секунд покоя...")
    after = collect(30.0, "после")

    print("\nТеперь МОТОРЫ: поставьте робота так, чтобы колёса крутились в воздухе,")
    print("и включите движение вперёд на средней скорости (PWM ~60-80).")
    answer = input("Моторы крутятся? Enter — записываю 30 с, или 's' чтобы пропустить: ")
    motors = None if answer.strip().lower() == "s" else collect(30.0, "моторы")

    base_yaw = statistics.median(s["yaw"] for s in base)
    magnet_shift = max(abs(s["yaw"] - base_yaw) for s in magnet) if magnet else 0.0
    after_yaw = statistics.median(s["yaw"] for s in after)
    returned = abs(after_yaw - base_yaw)
    motor_shift = (max(abs(s["yaw"] - base_yaw) for s in motors) if motors else None)

    mag_active = magnet_shift > 2.0
    print(f"\n  сдвиг от магнита : {magnet_shift:.2f}°  (>2° = магнитометр активен)")
    print(f"  вернулось на     : {returned:.2f}°")
    if motor_shift is not None:
        print(f"  сдвиг от моторов : {motor_shift:.2f}°  (>2° = моторы портят курс)")

    if mag_active:
        verdict = ("МАГНИТОМЕТР АКТИВЕН. Курс использовать ТОЛЬКО как относительный "
                   "на окнах до 10-30 с. Рядом с моторами не доверять. "
                   "Визуальную коррекцию курса делать обязательной.")
    else:
        verdict = ("Магнитометр не влияет — работает как Game Rotation Vector. "
                   "Курс надёжен как относительный на минутах, дрейф предсказуем.")
    if motor_shift is not None and motor_shift > 2.0:
        verdict += (f" ВНИМАНИЕ: моторы уводят курс на {motor_shift:.1f}° — "
                    f"это реальная помеха, а не теория.")
    print(f"\n  ВЕРДИКТ: {verdict}")
    return {"test": "A", "magnet_shift_deg": round(magnet_shift, 3),
            "returned_deg": round(returned, 3),
            "motor_shift_deg": None if motor_shift is None else round(motor_shift, 3),
            "magnetometer_active": mag_active, "verdict": verdict}


def test_b(minutes: float = 10.0) -> dict:
    """Дрейф курса и шум в покое. Даёт σ для R фильтров."""
    print(f"\n=== ТЕСТ B: дрейф и шум в покое ({minutes:.0f} мин) ===")
    print("Робот НЕПОДВИЖЕН, моторы выключены, никто не ходит рядом.")
    input("Нажмите Enter и не трогайте до конца теста...")
    samples = collect(minutes * 60.0, "дрейф")
    if len(samples) < 100:
        return {"test": "B", "error": "слишком мало отсчётов"}

    span_min = (samples[-1]["t"] - samples[0]["t"]) / 60.0
    drift = (samples[-1]["yaw"] - samples[0]["yaw"]) / max(span_min, 1e-9)
    dyaw = [samples[i + 1]["yaw"] - samples[i]["yaw"] for i in range(len(samples) - 1)]
    sigma_yaw = statistics.pstdev(dyaw)
    sigma_pitch = statistics.pstdev([s["pitch"] for s in samples])
    sigma_roll = statistics.pstdev([s["roll"] for s in samples])
    mags = [_accel_ms2(s) for s in samples]
    mean_a, sigma_a = statistics.mean(mags), statistics.pstdev(mags)

    print(f"\n  дрейф yaw        : {drift:+.3f} °/мин   (норма ≤0.5)")
    print(f"  σ приращений yaw : {sigma_yaw:.4f}°       (норма 0.01-0.05)")
    print(f"  σ pitch / roll   : {sigma_pitch:.4f} / {sigma_roll:.4f}°  (норма <0.05)")
    print(f"  |a| среднее      : {mean_a:.3f} м/с²      (норма 9.75-9.85)")
    print(f"  σ |a|            : {sigma_a:.4f} м/с²     (норма <0.05)")

    problems = []
    if abs(drift) > 0.5:
        problems.append("дрейф выше нормы — соблюдать паузу 5 с при старте, проверить вибрацию")
    if not (9.75 <= mean_a <= 9.85):
        problems.append("|a| вне нормы — ошибка масштаба или нужна калибровка акселерометра")
    if sigma_a > 0.05:
        problems.append("шумный акселерометр — вибрация окружения или крепёж")
    verdict = "; ".join(problems) if problems else "Всё в норме."
    print(f"\n  ВЕРДИКТ: {verdict}")
    return {"test": "B", "minutes": span_min,
            "drift_deg_per_min": round(drift, 4),
            "sigma_yaw_deg": round(sigma_yaw, 5),
            "sigma_pitch_deg": round(sigma_pitch, 5),
            "sigma_roll_deg": round(sigma_roll, 5),
            "mean_accel_ms2": round(mean_a, 4),
            "sigma_accel_ms2": round(sigma_a, 5), "verdict": verdict}


def test_e(repeats: int = 10) -> dict:
    """Повторяемость наклона. Прямо задаёт точность выравнивания облака точек."""
    print(f"\n=== ТЕСТ E: повторяемость наклона ({repeats} повторов) ===")
    print("Нужен клин или книга известной толщины. Будете подкладывать под")
    print("одно и то же колесо, снимать и ставить заново.")
    input("Готовы? Enter — сначала запишу РОВНОЕ положение...")
    flat = collect(5.0, "ровно")
    flat_pitch = statistics.median(s["pitch"] for s in flat)
    flat_roll = statistics.median(s["roll"] for s in flat)
    print(f"  ровно: pitch={flat_pitch:+.2f}° roll={flat_roll:+.2f}°")
    print("  ЭТО СМЕЩЕНИЕ УСТАНОВКИ — его надо вычитать всегда (§6.1).")

    tilts = []
    for k in range(repeats):
        input(f"\n  [{k + 1}/{repeats}] Подложите клин, поставьте робота. Enter...")
        block = collect(3.0, f"наклон {k + 1}")
        pitch = statistics.median(s["pitch"] for s in block)
        roll = statistics.median(s["roll"] for s in block)
        tilts.append((pitch, roll))
        print(f"       pitch={pitch:+.2f}° roll={roll:+.2f}°")
        input("       Уберите клин. Enter...")

    spread_pitch = statistics.pstdev([t[0] for t in tilts]) if len(tilts) > 1 else 0.0
    spread_roll = statistics.pstdev([t[1] for t in tilts]) if len(tilts) > 1 else 0.0
    print(f"\n  разброс pitch : {spread_pitch:.3f}°  (норма ≤0.5)")
    print(f"  разброс roll  : {spread_roll:.3f}°  (норма ≤0.5)")
    ok = spread_pitch <= 0.5 and spread_roll <= 0.5
    verdict = ("Наклон повторяем — вертикаль реконструкции будет точной."
               if ok else
               "Разброс велик: люфт крепления или вибрация. Вертикаль будет плавать.")
    print(f"\n  ВЕРДИКТ: {verdict}")
    return {"test": "E", "pitch_offset_deg": round(flat_pitch, 4),
            "roll_offset_deg": round(flat_roll, 4),
            "spread_pitch_deg": round(spread_pitch, 4),
            "spread_roll_deg": round(spread_roll, 4), "verdict": verdict}


CALIB_PATH = REPO_ROOT / "config/imu_calibration.json"


def save_to_calibration(results: list[dict]) -> str | None:
    """Записывает измеренные величины прямо в файл калибровки.

    Никаких «перепишите это число руками»: значение, которое человек должен
    перенести сам, рано или поздно переносится с опечаткой или не переносится
    вовсе, и тогда фильтр работает со старой σ, а никто об этом не знает.

    Существующий файл НЕ перезаписывается целиком — правится только раздел
    `rvc_acceptance`, а рядом кладётся резервная копия. Там же лежит
    гравитационная калибровка, и потерять её из-за приёмочного теста было бы
    несоразмерной ценой.
    """
    if not results:
        return None
    data: dict = {}
    if CALIB_PATH.exists():
        try:
            data = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        backup = CALIB_PATH.with_suffix(".json.bak")
        try:
            backup.write_text(json.dumps(data, indent=1, ensure_ascii=False),
                              encoding="utf-8")
        except OSError:
            pass

    section = data.setdefault("rvc_acceptance", {})
    section["updated_at"] = datetime.now().isoformat(timespec="seconds")
    for result in results:
        which = result.get("test")
        if which == "A":
            section["magnetometer_active"] = result.get("magnetometer_active")
            section["magnet_shift_deg"] = result.get("magnet_shift_deg")
            section["motor_shift_deg"] = result.get("motor_shift_deg")
        elif which == "B" and "sigma_yaw_deg" in result:
            # Эти три идут прямо в R фильтров и в оценку накопленной
            # неопределённости курса.
            section["sigma_yaw_deg"] = result["sigma_yaw_deg"]
            section["sigma_pitch_deg"] = result["sigma_pitch_deg"]
            section["sigma_roll_deg"] = result["sigma_roll_deg"]
            section["sigma_accel_ms2"] = result["sigma_accel_ms2"]
            section["yaw_drift_deg_per_min"] = abs(result["drift_deg_per_min"])
        elif which == "E" and "pitch_offset_deg" in result:
            # Перекос крепления. Вычитается из КАЖДОГО измерения, иначе вся
            # реконструкция систематически наклонена.
            section["pitch_offset_deg"] = result["pitch_offset_deg"]
            section["roll_offset_deg"] = result["roll_offset_deg"]
            section["tilt_repeatability_deg"] = max(
                result["spread_pitch_deg"], result["spread_roll_deg"])

    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIB_PATH.write_text(json.dumps(data, indent=1, ensure_ascii=False),
                          encoding="utf-8")
    return str(CALIB_PATH)


def main() -> int:
    which = (sys.argv[1] if len(sys.argv) > 1 else "all").upper()
    out_dir = REPO_ROOT / "data/imu_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        if which in ("A", "ALL"):
            results.append(test_a())
        if which in ("B", "ALL"):
            minutes = float(sys.argv[2]) if len(sys.argv) > 2 and which == "B" else 10.0
            results.append(test_b(minutes))
        if which in ("E", "ALL"):
            results.append(test_e())
    except KeyboardInterrupt:
        print("\nпрервано")
    if not results:
        print(f"Неизвестный тест {which!r}. Ожидается A, B, E или all.")
        return 2
    path = out_dir / f"acceptance_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    calib = save_to_calibration(results)
    print(f"\nСырые результаты : {path}")
    if calib:
        print(f"Калибровка обновлена: {calib}  (резервная копия рядом, .json.bak)")
    print("Больше ничего от вас не нужно — числа уже на месте.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
