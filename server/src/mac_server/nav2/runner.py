"""Живой цикл: кадры с Pi -> политика -> колёса.

Всё крутится на Mac, потому что там уже и кадры (Pi стримит 30 fps), и
привод (ESP подключается к этому же серверу). Pi ничего нового не делает.

Безопасность здесь важнее функциональности, поэтому она устроена как три
независимых механизма, а не один:

1. **Свежесть кадра.** Нет нового кадра дольше `stale_frame_s` — стоп. Езда
   по устаревшей картинке хуже стояния: политика уверенно ведёт туда, где
   робота уже нет.
2. **Оператор.** Любое ручное управление с панели немедленно снимает
   автономность через хук `RobocarService.add_takeover_hook`. Человек
   побеждает без отдельной команды отмены.
3. **Собственный TTL команды.** Прошивка сама глушит моторы после 400 мс
   тишины, но это защита от обрыва СЕТИ. Зависший здесь поток продолжал бы
   слать последнюю команду, и failsafe бы не сработал — этот случай уже
   ловился на другом потребителе привода.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Такт. Замер офлайн-гейта: 248 мс на кадр на MPS, то есть 4 Гц — потолок,
# а не выбор.
TICK_HZ = 4.0
# Кадр старше этого считается потерянным.
STALE_FRAME_S = 1.0
# Скорость по умолчанию. 120 из 150 MAX_PWM: заметно медленнее ручной езды,
# потому что первые автономные заезды упираются в мебель, а не в секундомер.
DEFAULT_SPEED = 120


class Nav2Runner:
    """Поток автономного движения по топокарте."""

    def __init__(self, policy, topomap: list, robocar, frame_store,
                 speed: int = DEFAULT_SPEED, radius: int = 4,
                 goal_node: int | None = None,
                 stale_frame_s: float = STALE_FRAME_S) -> None:
        from mac_server.nav2.policy import TopologicalLocalizer

        self.policy = policy
        self.robocar = robocar
        self.frame_store = frame_store
        self.speed = speed
        self.goal_node = goal_node
        self.stale_frame_s = stale_frame_s
        self.localizer = TopologicalLocalizer(policy, topomap, radius=radius)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "active": False, "node": 0, "nodes": len(topomap),
            "distance": None, "reason": None, "frames": 0, "pan_note": None,
        }

    # ------------------------------------------------------------ запуск

    def start(self) -> tuple[bool, str]:
        if self._thread is not None and self._thread.is_alive():
            return False, "уже едет"
        if self.robocar is None:
            return False, "привод выключен"
        self._stop.clear()

        # Камера ОБЯЗАНА смотреть вперёд. Политика сравнивает текущий кадр с
        # записанными и молча считает направление взгляда тем же самым;
        # серво, оставшийся повёрнутым после обзора, даёт заезд с косой
        # камерой, где всё выглядит работающим и всё неверно. Отказ
        # центрироваться не останавливает старт — на прошивке без PAN
        # серво просто нет, — но в статус это попадает.
        pan_note = None
        try:
            ok, detail = self.robocar.set_pan(0.0)
            if not ok:
                pan_note = f"камера не отцентрована: {detail}"
        except AttributeError:
            pan_note = None

        self.robocar.add_takeover_hook(self._on_takeover)
        with self._lock:
            self._state.update({"active": True, "reason": None, "frames": 0,
                                "pan_note": pan_note})
        self._thread = threading.Thread(target=self._loop, name="nav2", daemon=True)
        self._thread.start()
        return True, "поехали"

    def stop(self, reason: str = "остановлено") -> None:
        self._stop.set()
        with self._lock:
            if self._state["active"]:
                self._state["reason"] = reason
        # Тормозим здесь же, а не только в потоке: если он завис, именно этот
        # вызов и остановит робота.
        if self.robocar is not None:
            self.robocar.drive(0, 0, _internal=True, source="nav2")

    def _on_takeover(self, reason: str) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.info("nav2: управление перехвачено (%s)", reason)
            self._stop.set()
            with self._lock:
                self._state["reason"] = f"перехват: {reason}"

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    # -------------------------------------------------------------- цикл

    def _loop(self) -> None:
        from PIL import Image
        import io

        from mac_server.nav2.control import drive_command, waypoint_to_drive

        context: list = []
        context_size = self.policy.config.context_size
        period = 1.0 / TICK_HZ
        last_sequence = -1
        last_frame_at = time.monotonic()
        reason = "остановлено"

        try:
            while not self._stop.is_set():
                started = time.monotonic()
                sequence, frame = self.frame_store.latest()
                if frame is None or sequence == last_sequence:
                    # Тот же кадр, что и в прошлый раз. Отсчёт идёт от
                    # ПОСЛЕДНЕГО ЖИВОГО кадра, а не от начала итерации:
                    # иначе таймер сбрасывается каждый такт и не дорастает
                    # до порога никогда — робот едет по замершей картинке
                    # бесконечно.
                    if time.monotonic() - last_frame_at > self.stale_frame_s:
                        reason = "кадры не приходят"
                        break
                    if self._stop.wait(min(period, self.stale_frame_s)):
                        break
                    continue
                last_sequence = sequence
                last_frame_at = time.monotonic()

                context.append(Image.open(io.BytesIO(frame.data)).convert("RGB"))
                context = context[-(context_size + 1):]
                if len(context) < context_size + 1:
                    continue

                out = self.localizer.step(context, goal_node=self.goal_node)
                # Точка на горизонте, а не следующая: ближайшая слишком
                # чувствительна к шуму предсказания и робот дёргается.
                waypoint = out["waypoint"][min(2, len(out["waypoint"]) - 1)]
                left, right = drive_command(waypoint, self.speed)
                throttle, steer = waypoint_to_drive(waypoint)
                self.robocar.drive(left, right, _internal=True, source="nav2",
                                   throttle=throttle, steer=steer, speed=self.speed)

                with self._lock:
                    self._state.update({
                        "node": out["node"],
                        "distance": round(out["distance"], 2),
                        "frames": self._state["frames"] + 1,
                    })
                if out["reached_goal"]:
                    reason = "цель достигнута"
                    break

                elapsed = time.monotonic() - started
                if self._stop.wait(max(0.0, period - elapsed)):
                    break
        except Exception as exc:  # noqa: BLE001 - поток не должен ронять сервер
            logger.exception("nav2: цикл упал")
            reason = f"ошибка: {exc}"
        finally:
            if self.robocar is not None:
                self.robocar.drive(0, 0, _internal=True, source="nav2")
            with self._lock:
                self._state["active"] = False
                self._state["reason"] = self._state["reason"] or reason
            logger.info("nav2: остановлен (%s)", self.status()["reason"])
