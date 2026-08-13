#!/usr/bin/env python3
"""Прогнать гейт навигации вручную.

Обычно этого делать не нужно: гейт — шаг конвейера `navcheck`, он идёт сам
после «Run pipeline» на панели. Скрипт нужен, когда хочется прогнать его
отдельно, на другой сессии или с другой плотностью узлов.

Без аргументов берёт САМУЮ СВЕЖУЮ запись — обычно проверять надо именно
то, что только что сняли, и знать её идентификатор для этого не должно быть
нужно.

Логика самой проверки живёт в mac_server/nav2/evaluate.py, общая с шагом
конвейера: две копии критерия разошлись бы на первой правке порога.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "server/src"))


def main() -> int:
    from mac_server.nav2 import evaluate

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=None,
                    help="идентификатор сессии; по умолчанию самая свежая")
    ap.add_argument("--ckpt", default="models/nav2/vint.pth")
    ap.add_argument("--config",
                    default="external/visualnav-transformer/train/config/vint.yaml")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    sessions = REPO / "data/scene_sessions"
    if args.session:
        session_dir = sessions / args.session
    else:
        session_dir = evaluate.latest_session(sessions)
        if session_dir is None:
            raise SystemExit(f"В {sessions} нет записанных сессий")
        print(f"Сессия не задана — беру свежайшую: {session_dir.name}")
    if not session_dir.exists():
        raise SystemExit(f"Нет сессии {session_dir}")

    try:
        result = evaluate.evaluate(session_dir, REPO / args.ckpt, REPO / args.config,
                                   args.stride, args.radius, args.device)
    except ValueError as exc:
        raise SystemExit(str(exc))
    out_path = evaluate.save_result(result, REPO)

    print(f"\n  узлов                {result['nodes']}")
    print(f"  оценено кадров       {result['evaluated_frames']}")
    print(f"  точность (±{evaluate.NODE_TOLERANCE} узла)   "
          f"{result['accuracy_within_tolerance']:.1%}"
          f"   (порог {evaluate.MIN_ACCURACY:.0%})")
    print(f"  медиана ошибки       {result['median_abs_error_nodes']:.1f} узла")
    print(f"  без окна, глобально  {result['global_search_accuracy']:.1%}"
          f"   (худшая ошибка {result['global_search_worst_error_nodes']} узлов)")
    print(f"  худший скачок        {result['worst_jump_nodes']} узлов "
          f"  (предел {evaluate.MAX_JUMP})")
    print(f"  движение вперёд      {result['monotonic_fraction']:.1%} тактов")
    print(f"  время на кадр        {result['ms_per_frame_median']} мс "
          f"(p95 {result['ms_per_frame_p95']})")
    print(f"\n  ГЕЙТ: {'ПРОЙДЕН' if result['passed'] else 'НЕ ПРОЙДЕН'}")
    print(f"  -> {out_path.relative_to(REPO)}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
