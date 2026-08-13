#!/usr/bin/env python3
"""Вывезти ViNT в ONNX, чтобы считать на Pi, а не на ноутбуке.

Torch на Pi не ставится намеренно: два гигабайта на microSD ради модели,
которую onnxruntime исполняет из девяноста мегабайт. Ноутбук же для этого
плохое место вдвойне — он на Intel, где torch остановился на 2.2.2, а его
мост к numpy сломан несовместимой версией.

Ось батча объявлена динамической: число кандидатов задаётся radius'ом окна
поиска и меняется на ходу у краёв карты.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "server/src"))


def main() -> int:
    import torch
    from mac_server.nav2.policy import VintPolicy

    policy = VintPolicy(REPO / "models/nav2/vint.pth",
                        REPO / "external/visualnav-transformer/train/config/vint.yaml",
                        device="cpu")
    width, height = policy.config.image_size
    frames = policy.config.context_size + 1
    obs = torch.zeros(1, 3 * frames, height, width)
    goal = torch.zeros(1, 3, height, width)

    out = REPO / "models/nav2/vint.onnx"
    torch.onnx.export(
        policy.model, (obs, goal), str(out),
        input_names=["obs", "goal"], output_names=["dists", "waypoints"],
        dynamic_axes={"obs": {0: "batch"}, "goal": {0: "batch"},
                      "dists": {0: "batch"}, "waypoints": {0: "batch"}},
        opset_version=17)
    print(f"{out}  {out.stat().st_size / 1e6:.1f} МБ")
    print("на Pi:  scp models/nav2/vint.onnx "
          "cv-pi.local:~/Desktop/work/pi_cv/models/nav2_vint.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
