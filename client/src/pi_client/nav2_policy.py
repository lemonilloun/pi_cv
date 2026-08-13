"""Политика ViNT на Pi через onnxruntime.

**Считает робот, а не ноутбук.** Ноутбук в этом проекте — Intel-машина, где
torch навсегда остановился на 2.2.2 и его мост к numpy сломан несовместимой
версией; вдобавок гонять кадры туда-обратно ради предсказания на 5 см вперёд
значит поставить движение в зависимость от Wi-Fi. Замерено на этом Pi
(4 ядра, onnxruntime CPU, формы боевые):

    кандидатов 1 —  41 мс
    кандидатов 5 — 183 мс
    кандидатов 9 — 334 мс

против 248 мс у ноутбука на девяти. То есть Pi отстаёт на треть, а при
radius 2 даёт 5.5 Гц — быстрее, чем нужно циклу управления.

Torch на Pi не нужен и не ставится: модель вывезена в ONNX
(`scripts/export_vint_onnx.py`), onnxruntime весит десятки мегабайт против
двух гигабайт torch, а microSD здесь дефицит.

Предобработка обязана совпадать с обучением до мелочи: resize в (ширина,
высота) модели и нормализация ImageNet. Расхождение здесь не падает, а тихо
уводит вход в распределение, которого модель не видела.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Та же нормализация, что в обучении апстрима. Не подбирается.
_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True)
class OnnxPolicyConfig:
    """Свойства экспортированной сети, а не настройки."""

    context_size: int
    len_traj_pred: int
    image_size: tuple[int, int]   # (ширина, высота)


class OnnxVintPolicy:
    """Интерфейс тот же, что у серверной VintPolicy: `predict(context, goals)`.

    Совпадение интерфейсов не косметика — `shared.topological` использует
    обе, и именно поэтому «где я» считается на Pi и на ноутбуке одинаково.
    """

    def __init__(self, model_path: Path, threads: int = 4) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), options,
                                            providers=["CPUExecutionProvider"])
        # Формы читаются из самой сети, а не задаются числом: сеть могли
        # переэкспортировать с другим размером входа, и тогда захардкоженный
        # ресайз тихо подаёт мусор вместо ошибки.
        obs_shape = self.session.get_inputs()[0].shape
        channels, height, width = obs_shape[1], obs_shape[2], obs_shape[3]
        waypoints_shape = self.session.get_outputs()[1].shape
        self.config = OnnxPolicyConfig(
            context_size=int(channels) // 3 - 1,
            len_traj_pred=int(waypoints_shape[1]),
            image_size=(int(width), int(height)),
        )

    # ---------------------------------------------------------------- ввод

    def _prepare(self, image) -> np.ndarray:
        """PIL или BGR-массив -> (3, H, W) float32, нормализованный."""
        width, height = self.config.image_size
        # Проверка на ndarray идёт ПЕРВОЙ и по типу, а не по наличию метода.
        # У numpy-массива тоже есть `.resize` — только он меняет буфер на
        # месте и на чужой памяти бросает «cannot resize an array that
        # references...». Утиная проверка `hasattr(image, "resize")` уводила
        # кадр с камеры в ветку для PIL, и падало это не на синтетике, а
        # везде.
        if isinstance(image, np.ndarray):                # BGR с камеры
            import cv2

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            array = cv2.resize(rgb, (width, height)).astype(np.float32)
        else:                                            # PIL
            array = np.asarray(image.resize((width, height)), dtype=np.float32)
        chw = array.transpose(2, 0, 1) / 255.0
        return (chw - _MEAN) / _STD

    def _context_tensor(self, context: list) -> np.ndarray:
        """Окно кадров -> (1, 3*N, H, W).

        Кадры складываются по КАНАЛАМ, а не по батчу: для сети контекст — одна
        многоканальная картинка. Перепутать эти оси легко, и ошибка не падает,
        а превращает окно в набор одиночных кадров.
        """
        return np.concatenate([self._prepare(f) for f in context], axis=0)[None]

    # -------------------------------------------------------------- вывод

    def predict(self, context: list, goals: list) -> tuple[np.ndarray, np.ndarray]:
        obs = self._context_tensor(context)
        batch_obs = np.repeat(obs, len(goals), axis=0)
        batch_goal = np.stack([self._prepare(g) for g in goals], axis=0)
        dists, waypoints = self.session.run(
            None, {"obs": batch_obs.astype(np.float32),
                   "goal": batch_goal.astype(np.float32)})
        return np.asarray(dists, dtype=float).ravel(), np.asarray(waypoints, dtype=float)
