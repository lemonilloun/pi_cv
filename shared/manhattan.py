"""Привязка курса к геометрии здания.

Живёт в `shared/`, потому что нужна ОБЕИМ сторонам: серверный шаг `tsdf`
считает поправку по стенам плана, а навигация на Pi её применяет. Первая
версия лежала в `client/src/pi_client/`, и серверный пайплайн молча падал на
`No module named 'pi_client'` — исключение ловилось, в отчёт уходили честные
нули, и выглядело это как «стен не нашлось», хотя вручную на тех же данных
находилось тринадцать сегментов.
"""

from __future__ import annotations

import math

# Ниже этого стены найдены неуверенно, и поправка — шум. Порог из руководства
# BNO085_RVC_guide.md §13.2.
MANHATTAN_MIN_STRENGTH = 0.7
MANHATTAN_MIN_SEGMENTS = 3


def manhattan_yaw_offset(wall_azimuths_rad: list[float]) -> tuple[float, float]:
    """Поправка курса по стенам комнаты и мера доверия к ней.

    Комнаты почти всегда прямоугольны, поэтому направления стен кучкуются
    кратно 90 градусам. Сворачиваем каждый азимут в [0, 90) и берём круговое
    среднее по УЧЕТВЕРЁННОМУ углу — учетверение сводит четыре направления стен
    в одно, и только после этого среднее вообще имеет смысл.

    Это единственная поправка курса, которая не накапливает ошибку: она
    наблюдает геометрию, а не интегрирует показания датчика.

    Возвращает (поправка, уверенность). Уверенность около единицы означает, что
    стены согласны между собой; около нуля — что стен, достойных внимания, не
    нашлось, и поправку применять нельзя.
    """
    if not wall_azimuths_rad:
        return (0.0, 0.0)
    folded = [a % (math.pi / 2.0) for a in wall_azimuths_rad]
    cos_sum = sum(math.cos(4.0 * a) for a in folded) / len(folded)
    sin_sum = sum(math.sin(4.0 * a) for a in folded) / len(folded)
    return (math.atan2(sin_sum, cos_sum) / 4.0, math.hypot(cos_sum, sin_sum))


def manhattan_is_usable(strength: float, segments: int) -> bool:
    """Стоит ли вообще применять поправку."""
    return strength >= MANHATTAN_MIN_STRENGTH and segments >= MANHATTAN_MIN_SEGMENTS
