"""Приём путевых точек с Pi и передача их на колёса.

Здесь нет ни модели, ни весов, ни torch. Решение принимает робот; ноутбуку
остаётся превратить точку в пару ШИМ — арифметика в десяток строк. Она живёт
именно тут, потому что тут же знание об этом конкретном шасси: зеркальная
распайка моторов в `mix_drive` и пороги страгивания в `drive_profile`.
Продублировать её на Pi значило бы завести вторую версию, которая разойдётся
с первой при первой же перемерке железа.

Две защиты, и обе на этой стороне не случайно:

**TTL.** Прошивка глушит моторы после 400 мс тишины, но это защита от обрыва
СЕТИ. Если Pi замер, продолжая держать соединение, ничего не оборвётся — и
последняя команда осталась бы жить. Поэтому свой сторож.

**Оператор.** Ручное управление немедленно отбирает автономность и глушит
моторы. Робот получает `nav2_stop` при следующем же обмене, но ждать этого
нельзя: между перехватом и доставкой он бы продолжал ехать.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Столько живёт последняя точка. Такт Pi — 4 Гц (250 мс), так что запас
# примерно вдвое: одна пропущенная точка не должна дёргать робота.
WAYPOINT_TTL_S = 0.6
WATCHDOG_PERIOD_S = 0.15


class Nav2Relay:
    """Точка с Pi -> колёса, со сторожем и приоритетом оператора."""

    def __init__(self, robocar) -> None:
        self.robocar = robocar
        self._lock = threading.Lock()
        self._last_at = 0.0
        self._taken_over = False
        self._state: dict[str, Any] = {
            "active": False, "node": None, "nodes": None,
            "distance": None, "reason": None, "waypoints": 0,
        }
        self._stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        if robocar is not None:
            robocar.add_takeover_hook(self._on_takeover)

    # ------------------------------------------------------------- приём

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Обработать сообщение с робота. Возвращает то, что ушло на колёса."""
        from mac_server.nav2.control import waypoint_to_drive
        from mac_server.robocar import mix_drive

        if payload.get("stop"):
            self._halt("робот остановился")
            return {"ok": True, "stopped": True}

        waypoint = payload.get("waypoint")
        if not isinstance(waypoint, (list, tuple)) or len(waypoint) not in (2, 4):
            raise ValueError("nav2_waypoint ждёт waypoint из 2 или 4 чисел")

        with self._lock:
            if self._taken_over:
                # Робот ещё не знает, что им уже рулят. Ответ несёт отказ,
                # чтобы он прекратил, а моторы уже стоят.
                return {"ok": False, "reason": "управление у оператора"}

        throttle, steer = waypoint_to_drive(waypoint)
        speed = int(payload.get("speed", 120))
        left, right = mix_drive(throttle, steer, speed)
        if self.robocar is not None:
            self.robocar.drive(left, right, _internal=True, source="nav2",
                               throttle=throttle, steer=steer, speed=speed)

        with self._lock:
            self._last_at = time.monotonic()
            self._state.update({
                "active": True, "reason": None,
                "node": payload.get("node"), "nodes": payload.get("nodes"),
                "distance": payload.get("distance"),
                "waypoints": self._state["waypoints"] + 1,
            })
        self._ensure_watchdog()
        return {"ok": True, "left": left, "right": right,
                "throttle": round(throttle, 3), "steer": round(steer, 3)}

    def resume(self) -> None:
        """Снова принимать точки после перехвата — по явной команде."""
        with self._lock:
            self._taken_over = False
            self._state["reason"] = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    # ------------------------------------------------------------ защиты

    def _on_takeover(self, reason: str) -> None:
        with self._lock:
            was_active = self._state["active"]
            self._taken_over = True
        if was_active:
            logger.info("nav2: управление перехвачено (%s)", reason)
            self._halt(f"перехват: {reason}")

    def _halt(self, reason: str) -> None:
        with self._lock:
            self._state.update({"active": False, "reason": reason})
        if self.robocar is not None:
            self.robocar.drive(0, 0, _internal=True, source="nav2")

    def _ensure_watchdog(self) -> None:
        if self._watchdog is not None and self._watchdog.is_alive():
            return
        self._stop.clear()
        self._watchdog = threading.Thread(target=self._watch, name="nav2-relay",
                                          daemon=True)
        self._watchdog.start()

    def _watch(self) -> None:
        while not self._stop.wait(WATCHDOG_PERIOD_S):
            with self._lock:
                idle = time.monotonic() - self._last_at
                active = self._state["active"]
            if active and idle > WAYPOINT_TTL_S:
                logger.info("nav2: точки не приходят %.1f с — стоп", idle)
                self._halt("точки перестали приходить")
                return

    def shutdown(self) -> None:
        self._stop.set()
