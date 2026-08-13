"""Гейт: справляется ли политика с камерой в её нынешнем положении.

Считается на уже записанных кадрах — без робота, без батареи, без комнаты.
Смысл в том, чтобы узнать это ДО того, как робот поедет: если камеру
подняли, повернули или сменили объектив, узлы перестают узнаваться, и
выясняться это должно на диске, а не в мебели.

Вызывается двумя способами и обоими одинаково: шагом конвейера `navcheck`
(сам, после реконструкции) и вручную `scripts/eval_topomap.py`. Общий
модуль здесь именно поэтому — две копии критерия разошлись бы на первой же
правке порога.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]

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
        # ValueError, а не SystemExit: это библиотека, а SystemExit наследует
        # BaseException и проходит сквозь `except Exception` шага конвейера —
        # короткая запись убила бы конвейер вместо того, чтобы пропустить
        # проверку. Перевод в код возврата — забота скрипта.
        raise ValueError(f"Слишком короткая запись: {len(frames)} кадров "
                         f"(нужно минимум {stride * 4})")

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




def latest_session(sessions_dir: Path) -> Path | None:
    """Самая свежая записанная сессия.

    Нужна затем, чтобы не спрашивать идентификатор: проверять почти всегда
    надо то, что только что сняли.
    """
    candidates = [p for p in sessions_dir.glob("session_*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def save_result(result: dict, repo: Path) -> Path:
    out_dir = repo / "data/nav2_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{stamp}_{result['session']}_{Path(result['checkpoint']).stem}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return path
