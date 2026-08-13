"""Шаг конвейера `navcheck`: прогнать гейт сразу после реконструкции.

Стоит последним и запускается сам. Раньше это была ручная команда с
идентификатором сессии в аргументе — то есть проверка, которую надо помнить
и для которой надо знать id. Такую проверку не делают. Здесь она просто
происходит, а результат виден на панели рядом с остальными шагами.

Ничего не ломает при отказе: нет весов или зависимостей — шаг сообщает об
этом и завершается, реконструкция от него не зависит.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Плотность узлов для проверки. Не то же, что при сборке рабочей топокарты:
# здесь узлы разрежены нарочно, чтобы шаг между ними был больше и модели
# было труднее — проверка должна быть строже эксплуатации, а не мягче.
EVAL_STRIDE = 8


def run_navcheck_step(session, config: dict[str, Any], repo_root: Path,
                      progress=lambda msg: None) -> dict[str, Any]:
    from mac_server.nav2 import evaluate

    session_dir = Path(session.root)
    repo = Path(repo_root)
    ckpt = repo / "models/nav2/vint.pth"
    policy_config = repo / "external/visualnav-transformer/train/config/vint.yaml"

    if not ckpt.exists():
        return {"skipped": "нет весов models/nav2/vint.pth",
                "hint": "scripts/setup_nav2.sh"}

    progress("загрузка политики")
    try:
        result = evaluate.evaluate(session_dir, ckpt, policy_config,
                                   stride=EVAL_STRIDE, radius=4, device="mps")
    except Exception as exc:  # noqa: BLE001 - шаг не должен валить конвейер
        logger.exception("navcheck не отработал")
        return {"skipped": f"проверка не выполнилась: {exc}"}

    path = evaluate.save_result(result, repo)
    verdict = "ПРОЙДЕН" if result["passed"] else "НЕ ПРОЙДЕН"
    logger.info("navcheck %s: точность %.1f%%, худший скачок %d узлов -> %s",
                verdict, result["accuracy_within_tolerance"] * 100,
                result["worst_jump_nodes"], path.name)
    # `trace` кладётся в файл, но не в отчёт шага: это сотни строк, которые
    # панель показывать не станет, а отчёт раздуют.
    return {k: v for k, v in result.items() if k != "trace"} | {
        "verdict": verdict,
        "report": str(path.relative_to(repo)),
    }
