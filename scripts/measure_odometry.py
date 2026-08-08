#!/usr/bin/env python3
"""Замер скорости шасси по вашим же проездам — для одометрии без энкодеров.

Моторы N20 на этом роботе двухпроводные, энкодеров нет, поэтому пройденное
расстояние взять неоткуда. Акселерометр не годится: его надо интегрировать
ДВАЖДЫ, и 0.5 градуса ошибки наклона превращаются в 0.43 м за 10 секунд —
это уже пробовали, маркер улетел с плана.

Но сервер знает КАЖДУЮ команду на моторы: все они проходят через
`RobocarService.drive()`. Если один раз измерить, с какой скоростью шасси
едет при данном PWM, команда становится источником скорости — а скорость,
в отличие от ускорения, интегрируется только один раз, и ошибка растёт
линейно, а не квадратично.

Скрипт сам ловит начало и конец каждого проезда: следит за командой на
моторы и засекает время, пока она не нулевая. От вас нужно только проехать
и назвать расстояние.

    ./scripts/run_measure_odometry.sh

Порядок работы:
  1. Запустите сервер (в вашем tmux) и откройте панель.
  2. Запустите этот скрипт.
  3. Нажмите W, проедьте по прямой сколько удобно, отпустите.
  4. Скрипт напечатает PWM и длительность, спросит расстояние в сантиметрах.
  5. Повторите 3-4 раза на разных скоростях (ползунок Speed в панели).
  6. Нажмите Ctrl+C — скрипт посчитает кривую и запишет её в конфиг.

Чем разнообразнее PWM, тем лучше: одна точка даёт прямую через ноль, а
шасси так себя не ведёт — ниже порога трогания оно вообще стоит.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8080"
POLL_S = 0.05          # 20 Гц: край проезда определяется с точностью ~50 мс
MIN_BURST_S = 0.4      # короче — это промах по клавише, а не проезд
CONFIG_PATH = REPO_ROOT / "config/default.json"


def status() -> dict | None:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/robot/status", timeout=2.0) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def watch_one_burst() -> dict | None:
    """Ждёт проезд и возвращает его параметры, или None если прервали.

    Границы берутся по переходам команды через ноль. Пока клавиша зажата,
    панель шлёт команду каждые 100 мс, а сервер держит её `COMMAND_TTL_S`
    (0.6 с) — поэтому «нулём» считается именно нулевая команда, а не пауза
    между повторами.
    """
    print("\n  ждём проезд… (нажмите W и отпустите)", flush=True)
    while True:
        state = status()
        if state is None:
            print("  сервер не отвечает на 127.0.0.1:8080 — запущен ли он?")
            time.sleep(1.0)
            continue
        drive = state.get("drive") or [0, 0]
        moving = any(abs(int(v)) > 0 for v in drive[:2])
        if not moving:
            time.sleep(POLL_S)
            continue

        # Поехали.
        t_start = time.monotonic()
        samples = [drive[:2]]
        while True:
            time.sleep(POLL_S)
            state = status()
            if state is None:
                break
            drive = state.get("drive") or [0, 0]
            if not any(abs(int(v)) > 0 for v in drive[:2]):
                break
            samples.append(drive[:2])
        duration = time.monotonic() - t_start

        left = [abs(int(s[0])) for s in samples]
        right = [abs(int(s[1])) for s in samples]
        pwm = statistics.median(left + right)
        straight = statistics.median(
            [abs(abs(int(s[0])) - abs(int(s[1]))) for s in samples])

        if duration < MIN_BURST_S:
            print(f"  проезд {duration:.2f} с — слишком короткий, пропускаю")
            continue
        return {"duration_s": round(duration, 3), "pwm": round(pwm, 1),
                "wheel_diff": round(straight, 1), "ticks": len(samples)}


def ask_distance(burst: dict) -> float | None:
    print(f"  проезд: {burst['duration_s']:.2f} с при PWM ~{burst['pwm']:.0f}"
          f"  (разница колёс {burst['wheel_diff']:.0f})")
    if burst["wheel_diff"] > 15:
        print("  ВНИМАНИЕ: колёса крутились по-разному — это была дуга, а не")
        print("  прямая. Для замера скорости нужен проезд ровно вперёд.")
    while True:
        raw = input("  сколько сантиметров проехал? (Enter — выбросить): ").strip()
        if not raw:
            return None
        try:
            value = float(raw.replace(",", "."))
        except ValueError:
            print("  нужно число")
            continue
        if value <= 0:
            return None
        return value / 100.0


def fit(samples: list[dict]) -> dict:
    """Линейная зависимость скорости от PWM с порогом трогания.

    Модель `v = k * (pwm - pwm0)` при `pwm > pwm0`, иначе ноль. Именно такой
    формы, а не пропорциональной: коллекторный мотор с редуктором ниже порога
    не крутится вовсе, и прямая через начало координат предсказывала бы
    движение там, где шасси просто греется.

    По двум и более точкам порог считается, по одной — берётся известный
    MIN_PWM шасси, и об этом честно сообщается.
    """
    points = [(s["pwm"], s["distance_m"] / s["duration_s"]) for s in samples]
    points.sort()
    if len(points) == 1:
        pwm, speed = points[0]
        pwm0 = 40.0        # ESP MIN_PWM
        k = speed / max(1e-6, pwm - pwm0)
        return {"k_ms_per_pwm": round(k, 6), "pwm0": pwm0, "n": 1,
                "note": "одна точка: порог взят как MIN_PWM шасси, не измерен"}

    n = len(points)
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    denom = sum((p[0] - mean_x) ** 2 for p in points)
    if denom < 1e-9:
        return {"error": "все проезды на одном PWM — нужна вариация скорости"}
    k = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points) / denom
    intercept = mean_y - k * mean_x
    pwm0 = -intercept / k if k > 1e-9 else 40.0

    residuals = [abs(p[1] - (k * (p[0] - pwm0))) for p in points]
    return {"k_ms_per_pwm": round(k, 6), "pwm0": round(pwm0, 1), "n": n,
            "residual_ms_median": round(statistics.median(residuals), 4),
            "points": [{"pwm": p[0], "speed_ms": round(p[1], 4)} for p in points]}


def save(model: dict, samples: list[dict]) -> None:
    raw_dir = REPO_ROOT / "data/odometry"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / f"odometry_{datetime.now():%Y%m%d_%H%M%S}.json"
    raw.write_text(json.dumps({"model": model, "samples": samples},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nСырые замеры : {raw}")

    if "error" in model:
        print("Модель не записана:", model["error"])
        return
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    section = config.setdefault("robot", {}).setdefault("odometry", {})
    section.update({
        "k_ms_per_pwm": model["k_ms_per_pwm"],
        "pwm0": model["pwm0"],
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "samples": model["n"],
        # Честный ярлык: пока замеров мало, модель — прикидка, и всё, что от
        # неё зависит, должно это знать.
        "measured": model["n"] >= 3,
    })
    CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"Конфиг обновлён: {CONFIG_PATH} -> robot.odometry")


def main() -> int:
    print(__doc__.split("Порядок работы:")[1].strip() if "Порядок работы:" in __doc__ else "")
    if status() is None:
        print("\nСервер не отвечает на 127.0.0.1:8080. Запустите его в своём tmux и")
        print("повторите — сам я его не запускаю.")
        return 2

    samples: list[dict] = []
    try:
        while True:
            burst = watch_one_burst()
            if burst is None:
                break
            distance = ask_distance(burst)
            if distance is None:
                print("  выброшено")
                continue
            burst["distance_m"] = distance
            burst["speed_ms"] = round(distance / burst["duration_s"], 4)
            samples.append(burst)
            print(f"  -> {distance*100:.0f} см за {burst['duration_s']:.2f} с "
                  f"= {burst['speed_ms']:.3f} м/с при PWM {burst['pwm']:.0f}")
            print(f"  замеров: {len(samples)}  (нужно 3-4 на РАЗНЫХ скоростях)")
    except (KeyboardInterrupt, EOFError):
        print()

    if not samples:
        print("Замеров нет.")
        return 1
    model = fit(samples)
    print("\n=== модель скорости ===")
    for key, value in model.items():
        if key != "points":
            print(f"  {key}: {value}")
    if "k_ms_per_pwm" in model:
        print("\n  проверка на ваших же замерах:")
        for s in samples:
            predicted = max(0.0, model["k_ms_per_pwm"] * (s["pwm"] - model["pwm0"]))
            error = abs(predicted - s["speed_ms"]) / max(s["speed_ms"], 1e-6)
            print(f"    PWM {s['pwm']:5.0f}: измерено {s['speed_ms']:.3f} м/с, "
                  f"модель {predicted:.3f}  ->  ошибка {error*100:.0f}%")
    save(model, samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
