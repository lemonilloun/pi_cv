"""Топокарта: порядок узлов и независимость от исходной сессии."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mac_server.nav2 import topomap


def _session(root: Path, count: int) -> Path:
    session = root / "session_x"
    for i in range(count):
        kf = session / "keyframes" / f"{i:06d}"
        kf.mkdir(parents=True)
        Image.new("RGB", (32, 24), (i, i, i)).save(kf / "rgb.jpg")
    return session


class TopomapTest(unittest.TestCase):
    def test_nodes_are_ordered_numerically_not_lexically(self):
        """Узел 10 при строковой сортировке встаёт между 1 и 2, и граф молча
        перепутывается — без ошибки, просто карта другой комнаты."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topomap.build_from_session(_session(root, 36), root / "map", stride=1)
            frames, _ = topomap.load(root / "map")
            self.assertEqual(len(frames), 36)
            # Кадры были залиты возрастающей яркостью — она и проверяется.
            greys = [f.getpixel((0, 0))[0] for f in frames]
            self.assertEqual(greys, sorted(greys))

    def test_frames_are_copied_so_the_map_outlives_the_session(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = _session(root, 9)
            topomap.build_from_session(session, root / "map", stride=3)
            for path in session.rglob("rgb.jpg"):
                path.unlink()
            frames, _ = topomap.load(root / "map")
            self.assertEqual(len(frames), 3)

    def test_rebuilding_does_not_leave_stale_nodes_behind(self):
        # Иначе карта, пересобранная плотнее, донесёт хвост старой.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = _session(root, 30)
            topomap.build_from_session(session, root / "map", stride=1)
            topomap.build_from_session(session, root / "map", stride=10)
            frames, meta = topomap.load(root / "map")
            self.assertEqual(len(frames), 3)
            self.assertEqual(meta["nodes"], 3)

    def test_the_map_takes_the_session_display_name(self):
        """Пользователь называет запись «bedroom2», а не
        session_20260813_161329. Карта с машинным именем не опознаётся в
        списке, даже если собрана правильно."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = _session(root, 9)
            (session / "session_meta.json").write_text('{"name": "bedroom2"}')
            self.assertEqual(topomap.session_display_name(session), "bedroom2")

    def test_a_session_without_a_name_falls_back_to_its_id(self):
        with TemporaryDirectory() as tmp:
            session = _session(Path(tmp), 9)
            self.assertEqual(topomap.session_display_name(session), session.name)

    def test_a_session_without_frames_is_refused(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "empty").mkdir()
            with self.assertRaises(FileNotFoundError):
                topomap.build_from_session(root / "empty", root / "map")


if __name__ == "__main__":
    unittest.main()
