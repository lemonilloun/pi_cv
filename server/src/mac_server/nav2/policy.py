"""Топологическая политика ViNT: инференс без ROS.

Заменяет метрическую локализацию. Разница не в качестве, а в постановке:
координат нет вообще. «Где я» — это индекс узла в графе кадров, а близость
предсказывает модель как **временну́ю дистанцию** («сколько шагов отсюда
дотуда»), а не как похожесть картинок.

Отсюда свойство, ради которого всё и затевалось: кадр, видящий соседнее
помещение через дверь, находится во МНОГИХ шагах от кадра, снятого внутри
того помещения, — при том что выглядит на него похоже. Косинусное сравнение
эмбеддингов на этом ломалось, обученная дистанция на этом и обучена.

Второе свойство — поиск ведётся в окне `±radius` вокруг текущего узла, а не
по всей карте. Прыжок в другое помещение непредставим: туда просто не
смотрят.

Апстрим (`external/visualnav-transformer`) написан под ROS. Здесь взята
чистая часть — загрузка весов, трансформация кадров, предсказание дистанций
и путевых точек; всё это device-agnostic и переносится дословно. Порядок
трансформации сохранён в точности (resize в (ширина, высота) модели, затем
нормализация ImageNet): любое расхождение здесь тихо смещает вход в
распределение, которого модель не видела.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[4]
UPSTREAM = REPO / "external/visualnav-transformer"

# Нормализация ImageNet — та же, что в обучении апстрима. Не подбирается.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _ensure_upstream_on_path() -> None:
    """Апстрим импортируется как `vint_train.*` из своей папки train/."""
    if not UPSTREAM.exists():
        raise FileNotFoundError(
            f"Нет {UPSTREAM}. Клонируйте: git clone --depth 1 "
            "https://github.com/robodhruv/visualnav-transformer.git "
            "external/visualnav-transformer"
        )
    train_dir = str(UPSTREAM / "train")
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)


# Классы, которые чекпоинт тащит за собой из ОБУЧЕНИЯ и которые к инференсу
# отношения не имеют. Апстрим сохранял весь тренировочный объект целиком, так
# что распаковка требует их присутствия — хотя используются из чекпоинта
# только веса.
#
# Список явный, а не «глушить всё, чего нет», намеренно: отсутствие
# `efficientnet_pytorch` или `vint_train` — это сломанная установка, и она
# должна падать громко, а не подменяться пустышкой, из которой выйдет модель
# со случайными весами.
_TRAINING_ONLY_MODULES = ("warmup_scheduler",)


class _StubFinder:
    """Отдаёт пустые модули под перечисленными корнями и только под ними.

    Распаковка чекпоинта импортирует не сам пакет, а его подмодуль
    (`warmup_scheduler.scheduler`), поэтому одного объекта в `sys.modules`
    мало — нужен участник механизма импорта.
    """

    def __init__(self, roots: tuple[str, ...]) -> None:
        self.roots = roots

    def find_module(self, fullname: str, path=None):  # noqa: D401 - legacy API
        return self if self._owns(fullname) else None

    def find_spec(self, fullname: str, path=None, target=None):
        if not self._owns(fullname):
            return None
        from importlib.machinery import ModuleSpec

        return ModuleSpec(fullname, self, is_package=True)

    def _owns(self, fullname: str) -> bool:
        root = fullname.split(".", 1)[0]
        return root in self.roots

    def create_module(self, spec):
        import types

        module = types.ModuleType(spec.name)
        module.__path__ = []  # пакет, иначе подмодули не ищутся
        module.__file__ = f"<stub {spec.name}>"

        def __getattr__(attr: str, _name: str = spec.name) -> type:
            # Только обычные имена. Синтезировать dunder'ы нельзя: `inspect`
            # и механизм предупреждений обходят sys.modules и читают
            # `__file__`, а класс вместо строки роняет их в стороне от
            # места ошибки — что и произошло.
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            # Класс-пустышка: распаковке нужно куда-то положить состояние,
            # а читаем мы из чекпоинта только веса.
            return type(attr, (), {"__module__": _name})

        module.__getattr__ = __getattr__  # type: ignore[method-assign]
        return module

    def exec_module(self, module) -> None:
        return None


def _stub_training_only_modules() -> None:
    missing = []
    for name in _TRAINING_ONLY_MODULES:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if not missing:
        return
    finder = _StubFinder(tuple(missing))
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, finder)


@dataclass(frozen=True)
class PolicyConfig:
    """Параметры, вшитые в веса. Не настройки — свойства чекпоинта."""

    context_size: int
    len_traj_pred: int
    image_size: tuple[int, int]   # (ширина, высота), как в yaml апстрима
    learn_angle: bool

    @classmethod
    def from_yaml(cls, path: Path) -> "PolicyConfig":
        import yaml

        cfg = yaml.safe_load(path.read_text())
        size = cfg["image_size"]
        return cls(
            context_size=int(cfg["context_size"]),
            len_traj_pred=int(cfg["len_traj_pred"]),
            image_size=(int(size[0]), int(size[1])),
            learn_angle=bool(cfg["learn_angle"]),
        )


class VintPolicy:
    """Загруженные веса ViNT + предсказание дистанций и путевых точек.

    Модель принимает ОКНО из `context_size + 1` кадров, а не один кадр.
    Это не деталь реализации: одиночный кадр не отличает «стою перед
    дверью» от «еду к двери», а вся дистанция измеряется в шагах.
    """

    def __init__(self, ckpt_path: Path, config_path: Path, device: str = "mps") -> None:
        _ensure_upstream_on_path()
        import torch
        from vint_train.models.vint.vint import ViNT

        import yaml

        cfg = yaml.safe_load(Path(config_path).read_text())
        self.config = PolicyConfig.from_yaml(Path(config_path))
        self.device = torch.device(device)
        model = ViNT(
            context_size=cfg["context_size"],
            len_traj_pred=cfg["len_traj_pred"],
            learn_angle=cfg["learn_angle"],
            obs_encoder=cfg["obs_encoder"],
            obs_encoding_size=cfg["obs_encoding_size"],
            late_fusion=cfg["late_fusion"],
            mha_num_attention_heads=cfg["mha_num_attention_heads"],
            mha_num_attention_layers=cfg["mha_num_attention_layers"],
            mha_ff_dim_factor=cfg["mha_ff_dim_factor"],
        )
        # Чекпоинты апстрима сохранены с обучения на нескольких GPU, поэтому
        # ключи несут префикс `module.`; грузим состояние в обёрнутую копию.
        _stub_training_only_modules()
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()

    # ---------------------------------------------------------------- ввод

    def transform(self, images: list) -> Any:
        """PIL-кадры → тензор (1, 3*N, H, W), как ждёт модель.

        Кадры конкатенируются по КАНАЛАМ, а не по батчу: контекст для модели
        — одна многоканальная картинка. Перепутать эти две оси легко, и
        ошибка не падает, а тихо превращает окно в батч одиночных кадров.
        """
        import torch
        from torchvision import transforms

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])
        if not isinstance(images, list):
            images = [images]
        out = []
        for image in images:
            resized = image.resize(self.config.image_size)
            out.append(torch.unsqueeze(to_tensor(resized), 0))
        return torch.cat(out, dim=1)

    # -------------------------------------------------------------- вывод

    def predict(self, context: list, goals: list) -> tuple[np.ndarray, np.ndarray]:
        """Дистанции и путевые точки от окна `context` до каждой цели.

        Возвращает (dists, waypoints): dists — по одной на цель, waypoints —
        (len(goals), len_traj_pred, ...) в системе координат камеры.
        """
        import torch

        obs = self.transform(context)
        batch_obs = torch.cat([obs] * len(goals), dim=0).to(self.device)
        batch_goal = torch.cat([self.transform(g) for g in goals], dim=0).to(self.device)
        with torch.no_grad():
            dists, waypoints = self.model(batch_obs, batch_goal)
        return dists.cpu().numpy().flatten(), waypoints.cpu().numpy()


class TopologicalLocalizer:
    """Текущий узел графа, обновляемый только локальным поиском.

    `radius` — вся защита от «телепорта»: узел может сместиться не более чем
    на radius за такт, поэтому переход между удалёнными частями карты
    физически не выражается в этой структуре. Глобальный аргмаксимум, на
    котором сломалась прошлая навигация, здесь не вычисляется никогда.
    """

    def __init__(self, policy: VintPolicy, topomap: list, radius: int = 4,
                 close_threshold: float = 3.0, start_node: int = 0) -> None:
        self.policy = policy
        self.topomap = topomap
        self.radius = radius
        self.close_threshold = close_threshold
        self.closest_node = start_node

    def step(self, context: list, goal_node: int | None = None) -> dict[str, Any]:
        """Один такт: обновить текущий узел и вернуть путевую точку."""
        goal = len(self.topomap) - 1 if goal_node is None else goal_node
        start = max(self.closest_node - self.radius, 0)
        end = min(self.closest_node + self.radius + 1, max(goal, start))
        goals = self.topomap[start:end + 1]
        dists, waypoints = self.policy.predict(context, goals)
        best = int(np.argmin(dists))

        # Дошли до узла — цель переносится на следующий, иначе робот
        # припаркуется на промежуточной точке и никуда не поедет.
        if dists[best] > self.close_threshold:
            node = start + best
            waypoint = waypoints[best]
        else:
            node = min(start + best + 1, goal)
            waypoint = waypoints[min(best + 1, len(waypoints) - 1)]
        self.closest_node = node
        return {
            "node": node,
            "distance": float(dists[best]),
            "waypoint": waypoint,
            "window": [start, end],
            "reached_goal": node >= goal and dists[best] <= self.close_threshold,
        }
