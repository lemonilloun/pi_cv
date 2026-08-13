"""Шаг navcheck: он обязан быть в конвейере и не обязан его валить."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mac_server.nav2.navcheck_step import run_navcheck_step
from mac_server.scene3d import pipeline


class _Session:
    def __init__(self, root):
        self.root = root


class NavcheckStepTest(unittest.TestCase):
    def test_it_is_the_last_step_and_runs_by_default(self):
        """Проверка «годится ли запись для навигации» должна происходить
        сама. Ручной командой с идентификатором сессии её не делали."""
        self.assertEqual(pipeline.STEP_ORDER[-1], "navcheck")

    def test_the_pipeline_knows_how_to_call_it(self):
        # Шаг в списке, но не в таблице функций — это KeyError в середине
        # получасовой реконструкции.
        functions = pipeline._step_functions()
        for step in pipeline.STEP_ORDER:
            self.assertIn(step, functions)

    def test_missing_weights_skip_instead_of_failing(self):
        """Реконструкция не зависит от навигации: отсутствие весов не
        должно ронять конвейер, который уже отработал полчаса."""
        with TemporaryDirectory() as tmp:
            report = run_navcheck_step(_Session(Path(tmp)), {}, Path(tmp))
        self.assertIn("skipped", report)

    def test_a_broken_session_skips_instead_of_failing(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "models/nav2").mkdir(parents=True)
            (repo / "models/nav2/vint.pth").write_bytes(b"not a checkpoint")
            report = run_navcheck_step(_Session(repo / "empty"), {}, repo)
        self.assertIn("skipped", report)


if __name__ == "__main__":
    unittest.main()
