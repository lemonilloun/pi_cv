"""Телефонная страница управления: маршрут и её API.

Страница бесполезна, если один из вызовов, которые она делает, отвечает не
тем — а по HTML этого не видно. Здесь поднимается настоящий сервер панели и
дёргаются ровно те адреса, что дёргает страница.
"""

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mac_server.preview import FrameStoreHub, MjpegPreviewServer
from mac_server.robocar import RobocarService


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DrivePageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.server = MjpegPreviewServer("127.0.0.1", cls.port, FrameStoreHub(),
                                        robocar=RobocarService())
        # Без этого сокет уже занят конструктором, но обслуживать некому:
        # запросы просто висят, и тест уходит в бесконечное ожидание вместо
        # честного падения.
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=3) as r:
            return r.status, r.read()

    def _post(self, path, payload):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=3) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    def test_the_panel_is_never_cached(self):
        """Панель без Cache-Control застревает в браузере, и добавленные
        кнопки просто не появляются: выглядит как «функция не работает»,
        а на деле пользователь смотрит прошлую версию страницы."""
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=3) as r:
            self.assertEqual(r.headers.get("Cache-Control"), "no-store")

    def test_the_page_is_served(self):
        status, body = self._get("/drive")
        self.assertEqual(status, 200)
        self.assertIn(b"RoboCar", body)

    def test_it_holds_every_endpoint_the_page_calls(self):
        # Страница ссылается на эти адреса из JS; опечатка в любом из них
        # видна только на телефоне и только в момент, когда уже поздно.
        for path in ("/api/robot/status", "/api/nav2/status"):
            with self.subTest(path=path):
                self.assertEqual(self._get(path)[0], 200)

    def test_drive_accepts_the_payload_the_page_sends(self):
        status, body = self._post("/api/robot/drive",
                                  {"throttle": 1, "steer": 0, "speed": 120})
        # Робота в тесте нет — важно, что запрос ПОНЯТ и колёса смикшированы.
        self.assertIn("left", body)
        self.assertEqual(body["left"], body["right"])

    def test_pan_accepts_the_payload_the_page_sends(self):
        status, body = self._post("/api/robot/pan", {"angle_deg": -30})
        self.assertIn("ok", body)


if __name__ == "__main__":
    unittest.main()
