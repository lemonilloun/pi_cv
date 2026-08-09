#!/usr/bin/env python3
"""Гейт N1: годится ли топологическая политика для НАШЕЙ камеры.

Без робота, без батареи, без комнаты — на уже записанных кадрах.

Что проверяется. Модели обучены на широкоугольных камерах (~120°+), у нас
75°. Сузить поле зрения значит убрать из кадра ровно ту периферию, по которой
считается смещение между видами; оценить это рассуждением нельзя, только
замером. Гейт стоит ПЕРЕД инфраструктурой намеренно: если он не проходит,
проект уходит в чистый сбор данных VLA, потратив на выяснение день.

Как. Записанный проезд разрезается на топокарту (каждый `--stride`-й кадр) и
на «живой» поток (все кадры). Робот на самом деле ехал вдоль записи, поэтому
для кадра i истинный узел известен: i / stride. Меряем, попадает ли туда
предсказание.

Критерий (задан заранее, не подгоняется под результат):
  * доля кадров с ошибкой <= NODE_TOLERANCE узлов >= MIN_ACCURACY
  * ни одного скачка узла больше MAX_JUMP за такт

Результат пишется в data/nav2_eval/<метка времени>.json сам.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "server/src"))

NODE_TOLERANCE = 2
MIN_ACCURACY = 0.80
MAX_JUMP = 5


def load_frames(session_dir: Path) -> list[Path]:
    frames = sorted(session_dir.glob("keyframes/*/rgb.jpg"))
    if not frames:
        frames = sorted(session_dir.glob("**/rgb.jpg"))
    return frames


def evaluate(session_dir: Path, ckpt: Path, config: Path, stride: int,
             radius: int, device: str) -> dict:
    from PIL import Image
    from mac_server.nav2.policy import VintPolicy, TopologicalLocalizer

    frames = load_frames(session_dir)
    if len(frames) < stride * 4:
        raise SystemExit(f"Слишком короткая запись: {len(frames)} кадров")

    policy = VintPolicy(ckpt, config, device=device)
    context_size = policy.config.context_size

    node_frames = list(range(0, len(frames), stride))
    topomap = [Image.open(frames[i]).convert("RGB") for i in node_frames]
    print(f"{len(frames)} кадров -> {len(topomap)} узлов (шаг {stride})")

    localizer = TopologicalLocalizer(policy, topomap, radius=radius)
    context: list = []
    errors: list[int] = []
    global_errors: list[int] = []
    jumps: list[int] = []
    trace: list[dict] = []
    previous_node = 0
    times: list[float] = []

    for i, path in enumerate(frames):
        context.append(Image.open(path).convert("RGB"))
        if len(context) < context_size + 1:
            continue
        context = context[-(context_size + 1):]

        t0 = time.perf_counter()
        out = localizer.step(context)
        times.append((time.perf_counter() - t0) * 1000.0)

        # Контрольный замер: тот же кадр против ВСЕХ узлов, без окна. Это
        # ровно то, что делала прошлая навигация (глобальный аргмаксимум по
        # косинусу CLIP) и на чём она ломалась. Если обученная дистанция
        # держит и глобально, значит выигрывает сама постановка задачи, а не
        # только сужение поиска; если нет — окно и есть то, что её спасает.
        # Разница между двумя числами и есть ответ, и он стоит замера.
        global_dists, _ = policy.predict(context, topomap)
        global_node = int(np.argmin(global_dists))

        # Истинный узел: кадр i лежит между узлами i//stride и следующим.
        truth = min(int(round(i / stride)), len(topomap) - 1)
        error = out["node"] - truth
        errors.append(error)
        global_errors.append(global_node - truth)
        jumps.append(abs(out["node"] - previous_node))
        previous_node = out["node"]
        trace.append({"frame": i, "node": out["node"], "truth": truth,
                      "distance": round(out["distance"], 3)})

    errors_arr = np.asarray(errors)
    jumps_arr = np.asarray(jumps)
    global_arr = np.asarray(global_errors)
    accuracy = float(np.mean(np.abs(errors_arr) <= NODE_TOLERANCE))
    global_accuracy = float(np.mean(np.abs(global_arr) <= NODE_TOLERANCE))
    worst_jump = int(jumps_arr.max()) if len(jumps_arr) else 0
    # Монотонность: политика должна ВЕСТИ вдоль записи, а не топтаться.
    nodes = np.asarray([t["node"] for t in trace])
    advanced = float(np.mean(np.diff(nodes) >= 0)) if len(nodes) > 1 else 0.0

    passed = accuracy >= MIN_ACCURACY and worst_jump <= MAX_JUMP
    return {
        "session": session_dir.name,
        "checkpoint": ckpt.name,
        "device": device,
        "frames": len(frames),
        "nodes": len(topomap),
        "stride": stride,
        "radius": radius,
        "evaluated_frames": len(errors),
        "accuracy_within_tolerance": round(accuracy, 4),
        "median_abs_error_nodes": float(np.median(np.abs(errors_arr))),
        "mean_signed_error_nodes": round(float(errors_arr.mean()), 3),
        "worst_jump_nodes": worst_jump,
        "global_search_accuracy": round(global_accuracy, 4),
        "global_search_worst_error_nodes": int(np.abs(global_arr).max()),
        "monotonic_fraction": round(advanced, 4),
        "ms_per_frame_median": round(float(np.median(times)), 1),
        "ms_per_frame_p95": round(float(np.percentile(times, 95)), 1),
        "criteria": {"node_tolerance": NODE_TOLERANCE,
                     "min_accuracy": MIN_ACCURACY, "max_jump": MAX_JUMP},
        "passed": bool(passed),
        "trace": trace,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="session_20260807_171938")
    ap.add_argument("--ckpt", default="models/nav2/vint.pth")
    ap.add_argument("--config",
                    default="external/visualnav-transformer/train/config/vint.yaml")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    session_dir = REPO / "data/scene_sessions" / args.session
    if not session_dir.exists():
        raise SystemExit(f"Нет сессии {session_dir}")

    result = evaluate(session_dir, REPO / args.ckpt, REPO / args.config,
                      args.stride, args.radius, args.device)

    out_dir = REPO / "data/nav2_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stamp}_{args.session}_{Path(args.ckpt).stem}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n  узлов                {result['nodes']}")
    print(f"  оценено кадров       {result['evaluated_frames']}")
    print(f"  точность (±{NODE_TOLERANCE} узла)   {result['accuracy_within_tolerance']:.1%}"
          f"   (порог {MIN_ACCURACY:.0%})")
    print(f"  медиана ошибки       {result['median_abs_error_nodes']:.1f} узла")
    print(f"  без окна, глобально  {result['global_search_accuracy']:.1%}"
          f"   (худшая ошибка {result['global_search_worst_error_nodes']} узлов)")
    print(f"  худший скачок        {result['worst_jump_nodes']} узлов   (предел {MAX_JUMP})")
    print(f"  движение вперёд      {result['monotonic_fraction']:.1%} тактов")
    print(f"  время на кадр        {result['ms_per_frame_median']} мс "
          f"(p95 {result['ms_per_frame_p95']})")
    print(f"\n  ГЕЙТ: {'ПРОЙДЕН' if result['passed'] else 'НЕ ПРОЙДЕН'}")
    print(f"  -> {out_path.relative_to(REPO)}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
