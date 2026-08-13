"""Автономное движение: решение принимает Pi, ноутбук только передаёт.

Разделение проведено по одной границе — **всё, что думает, живёт здесь**.
Ноутбук получает готовую путевую точку и превращает её в ШИМ колёс: это
чистая арифметика в десяток строк, но она обязана остаться там, потому что
там же знание о зеркальной распайке моторов и о порогах страгивания этого
шасси.

Почему не наоборот. Ноутбук в этом проекте — Intel-машина, где сборки torch
кончились на 2.2.2, а её мост к numpy сломан несовместимой версией; чинить
это нечем. И даже будь он исправен, гонять кадры по Wi-Fi ради предсказания
на пять сантиметров вперёд означает поставить движение в зависимость от сети.
Замерено на этом Pi: такт 239 мс при radius 2 (4.2 Гц), 398 мс при radius 4.

Остановка устроена тремя независимыми способами, и ни один не полагается на
исправность остальных:

1. **Камера молчит** — кадр не пришёл дольше `stale_frame_s`. Ехать по
   устаревшему кадру хуже, чем стоять: политика уверенно ведёт туда, где
   робота уже нет.
2. **Оператор** — ноутбук присылает `nav2_stop`, когда кто-то взялся за
   ручное управление.
3. **Обрыв связи** — если точки перестают доходить, ноутбук сам глушит
   моторы по своему TTL, а прошивка вдобавок по своему failsafe. Здесь же
   разрыв соединения завершает цикл.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Такт. 4 Гц — то же, на чём работает апстрим, и то, что Pi выдаёт с запасом
# при radius 2.
TICK_HZ = 4.0
STALE_FRAME_S = 1.0
# Окно поиска. 2, а не 4: замерено 239 мс против 398, а прыжок узла окном и
# так ограничен. Больше радиус — больше кандидатов, каждый стоит прогона сети.
DEFAULT_RADIUS = 2
DEFAULT_SPEED = 120
# Какую точку предсказанной траектории брать. Ближайшая слишком чувствительна
# к шуму и робота дёргает.
WAYPOINT_INDEX = 2


class Nav2Runner:
    """Цикл автономного движения, живущий на роботе."""

    def __init__(self, policy, topomap: list, source, sender,
                 speed: int = DEFAULT_SPEED, radius: int = DEFAULT_RADIUS,
                 goal_node: int | None = None,
                 stale_frame_s: float = STALE_FRAME_S) -> None:
        from shared.topological import TopologicalLocalizer

        self.policy = policy
        self.source = source
        self.sender = sender
        self.speed = speed
        self.goal_node = goal_node
        self.stale_frame_s = stale_frame_s
        self.localizer = TopologicalLocalizer(policy, topomap, radius=radius)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "active": False, "node": 0, "nodes": len(topomap),
            "distance": None, "reason": None, "frames": 0, "tick_ms": None,
        }

    def start(self) -> tuple[bool, str]:
        if self._thread is not None and self._thread.is_alive():
            return False, "уже едет"
        self._stop.clear()
        with self._lock:
            self._state.update({"active": True, "reason": None, "frames": 0})
        self._thread = threading.Thread(target=self._loop, name="nav2", daemon=True)
        self._thread.start()
        return True, "поехали"

    def stop(self, reason: str = "остановлено") -> None:
        self._stop.set()
        with self._lock:
            if self._state["active"]:
                self._state["reason"] = reason
        self._send_stop()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    # ------------------------------------------------------------- отправка

    def _send_stop(self) -> None:
        """Явный стоп, а не «перестать слать точки».

        Молчание ноутбук тоже поймёт — по TTL, — но между решением
        остановиться и срабатыванием таймаута робот продолжает ехать. Явная
        команда убирает эту задержку.
        """
        try:
            self.sender({"stop": True})
        except Exception:
            logger.exception("nav2: не отправился стоп")

    def _loop(self) -> None:
        # Источник кадров открывается ЗДЕСЬ и закрывается в finally: камера —
        # единственный ресурс этого цикла, и владеть им должен тот, кто
        # владеет циклом, иначе она остаётся занятой после аварийного выхода
        # и следующий запуск падает на «устройство занято».
        opened = False
        try:
            self.source.start()
            opened = True
        except Exception as exc:  # noqa: BLE001
            logger.exception("nav2: камера не открылась")
            with self._lock:
                self._state.update({"active": False,
                                    "reason": f"камера не открылась: {exc}"})
            self._send_stop()
            return

        period = 1.0 / TICK_HZ
        context: list = []
        context_size = self.policy.config.context_size
        last_frame_at = time.monotonic()
        reason = "остановлено"

        try:
            while not self._stop.is_set():
                started = time.monotonic()
                frame = None
                try:
                    frame = self.source.capture_bgr()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("nav2: кадр не снялся: %s", exc)

                if frame is None:
                    # Отсчёт идёт от последнего ЖИВОГО кадра, а не от начала
                    # итерации: иначе таймер сбрасывается каждый такт и до
                    # порога не дорастает никогда.
                    if time.monotonic() - last_frame_at > self.stale_frame_s:
                        reason = "камера молчит"
                        break
                    if self._stop.wait(min(period, self.stale_frame_s)):
                        break
                    continue
                last_frame_at = time.monotonic()

                context.append(frame)
                context = context[-(context_size + 1):]
                if len(context) < context_size + 1:
                    continue

                out = self.localizer.step(context, goal_node=self.goal_node)
                waypoint = out["waypoint"][min(WAYPOINT_INDEX, len(out["waypoint"]) - 1)]
                self.sender({
                    "waypoint": [float(v) for v in waypoint],
                    "node": int(out["node"]),
                    "nodes": len(self.localizer.topomap),
                    "distance": round(float(out["distance"]), 2),
                    "speed": int(self.speed),
                })

                tick_ms = (time.monotonic() - started) * 1000.0
                with self._lock:
                    self._state.update({
                        "node": out["node"],
                        "distance": round(float(out["distance"]), 2),
                        "frames": self._state["frames"] + 1,
                        "tick_ms": round(tick_ms),
                    })
                if out["reached_goal"]:
                    reason = "цель достигнута"
                    break
                if self._stop.wait(max(0.0, period - (time.monotonic() - started))):
                    break
        except Exception as exc:  # noqa: BLE001 - поток не должен ронять клиента
            logger.exception("nav2: цикл упал")
            reason = f"ошибка: {exc}"
        finally:
            if opened:
                try:
                    self.source.stop()
                except Exception:
                    logger.exception("nav2: камера не закрылась")
            self._send_stop()
            with self._lock:
                self._state["active"] = False
                self._state["reason"] = self._state["reason"] or reason
            logger.info("nav2: остановлен (%s)", self.status()["reason"])


def load_topomap(map_dir: Path) -> list:
    """Кадры карты по порядку.

    Сортировка ЧИСЛОМ, а не строкой: лексикографически узел 10 встаёт между
    1 и 2, и граф молча перепутывается — без ошибки, просто карта не той
    комнаты.
    """
    import cv2

    files = sorted(map_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"Пустая топокарта: {map_dir}")
    return [cv2.imread(str(f)) for f in files]


# ------------------------------------------------------------ точка входа


def main(argv: list[str] | None = None) -> int:
    """Запустить автономное движение по топокарте, лежащей на роботе."""
    import argparse
    import signal

    from shared.config import load_config
    from pi_client.camera_session import CaptureSettings, make_capture_source
    from pi_client.nav2_policy import OnnxVintPolicy
    from pi_client.network import PiClient
    from pi_client.protocol import make_nav2_waypoint_message

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    repo = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", help="папка топокарты в data/topomaps на роботе")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--model", default=str(repo / "models/nav2_vint.onnx"))
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    parser.add_argument("--goal-node", type=int, default=None)
    parser.add_argument("--source", default="camera",
                        choices=["camera", "synthetic"])
    parser.add_argument("--synthetic-image", default=str(repo / "data/cat.jpg"))
    args = parser.parse_args(argv)

    config = load_config(None, repo / ".env")
    client_config = config.get("client", {})
    host = args.host or client_config.get("server_host") or config["server"]["host"]
    port = args.port or int(config["server"]["port"])
    device_id = client_config.get("device_id", "raspberry_pi")

    map_dir = Path(args.map)
    if not map_dir.is_absolute():
        map_dir = repo / "data/topomaps" / args.map
    topomap = load_topomap(map_dir)
    policy = OnnxVintPolicy(Path(args.model))
    logger.info("Топокарта %s: %d узлов; сеть %s", map_dir.name, len(topomap),
                policy.config)

    source = make_capture_source(args.source, CaptureSettings(), Path(args.synthetic_image))
    client = PiClient(host, port)
    client.connect()

    def send(payload: dict) -> None:
        client.send_message(make_nav2_waypoint_message(device_id, payload))

    runner = Nav2Runner(policy, topomap, source, send, speed=args.speed,
                        radius=args.radius, goal_node=args.goal_node)

    # Ctrl+C обязан останавливать МОТОРЫ, а не только процесс: иначе робот
    # уезжает, пока оператор смотрит на трассировку.
    def handle_signal(signum, frame):  # noqa: ARG001
        runner.stop("прервано оператором")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    ok, detail = runner.start()
    if not ok:
        logger.error("Не запустилось: %s", detail)
        return 2
    while runner.status()["active"]:
        time.sleep(0.2)
    logger.info("Готово: %s", runner.status()["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
