"""Отсев невозможных скачков позиции при локализации по месту.

Индекс мест сравнивает эмбеддинг ОДНОГО кадра со всеми записанными и берёт
лучший. У такого сравнения нет памяти: робот, стоящий в комнате и на мгновение
заглянувший в дверной проём, отдаёт кадр, похожий на середину коридора, — и
локализация мгновенно переносит его туда. Наблюдалось ровно это.

Ошибка не в сравнении: кадр действительно похож. Ошибка в том, что физику
никто не спросил. Робот, который 0.7 секунды назад был в комнате, не может
оказаться в восьми метрах — не потому что это маловероятно, а потому что он так
не ездит.

Отсюда два правила, и оба про непрерывность, а не про качество совпадения:

1. **Скорость.** Перемещение, требующее скорости выше физически возможной, —
   выброс, каким бы похожим ни был кадр.
2. **Липкость помещения.** Переход в ДРУГУЮ сессию требует подтверждения
   несколькими кадрами подряд. Один кадр через дверной проём — это взгляд, а не
   перемещение; настоящий переход через дверь виден много кадров подряд.
"""

from __future__ import annotations

import math
from typing import Any

# Шасси едет заметно медленнее, но запас нужен: между фиксами может пройти
# больше времени, чем обычно, а зажимать физику под типичный случай — значит
# отбрасывать верные фиксы после паузы.
MAX_SPEED_MS = 1.5
# Столько кадров подряд должны указывать на новое помещение, прежде чем ему
# поверят. При ~1.5 Гц это около двух секунд — дольше взгляда в дверь и заметно
# короче реального прохода через неё.
SWITCH_CONFIRMATIONS = 3
# Совпадение настолько лучше текущего, что спорить не о чем: робота перенесли
# руками или он въехал в помещение, которого в текущей сессии просто нет.
OVERWHELMING_SIMILARITY = 0.92


class FixGate:
    """Решает, принять ли очередной фикс, помня предыдущий.

    Состояние умышленно крошечное: последний принятый фикс и счётчик
    подтверждений. Фильтр Калмана сглаживает то, что принято, но не может
    отличить взгляд в соседнее помещение от переезда туда — это решение
    принимается ДО него, иначе выброс уже размазан по состоянию.
    """

    def __init__(self, max_speed_ms: float = MAX_SPEED_MS,
                 confirmations: int = SWITCH_CONFIRMATIONS,
                 overwhelming: float = OVERWHELMING_SIMILARITY) -> None:
        self.max_speed_ms = max_speed_ms
        self.confirmations = confirmations
        self.overwhelming = overwhelming
        self.session_id: str | None = None
        self.position: tuple[float, float] | None = None
        self.at: float | None = None
        self._pending_session: str | None = None
        self._pending_count = 0
        self.rejected = 0

    def accept(self, session_id: str, position: tuple[float, float],
               now: float, similarity: float = 0.0) -> dict[str, Any]:
        """Принять или отклонить фикс. Возвращает решение с причиной.

        Причина возвращается всегда: локализация, которая молча игнорирует
        половину фиксов, неотличима от локализации сломанной.
        """
        if self.session_id is None:
            return self._take(session_id, position, now, "первый фикс")

        if session_id != self.session_id:
            # Другое помещение. Одного кадра мало — но подавляющее совпадение
            # принимается сразу: спорить с ним значит залипнуть в комнате,
            # из которой робот давно уехал.
            if similarity >= self.overwhelming:
                return self._take(session_id, position, now,
                                  f"подавляющее совпадение {similarity:.2f}")
            if self._pending_session == session_id:
                self._pending_count += 1
            else:
                self._pending_session = session_id
                self._pending_count = 1
            if self._pending_count >= self.confirmations:
                return self._take(session_id, position, now,
                                  f"подтверждено {self._pending_count} кадрами")
            self.rejected += 1
            return {"accepted": False,
                    "reason": (f"другое помещение, подтверждений "
                               f"{self._pending_count}/{self.confirmations}")}

        # То же помещение: проверяем, что перемещение физически возможно.
        self._pending_session = None
        self._pending_count = 0
        # `or` вместо `is None` здесь ловушка: нулевая метка времени ложна,
        # dt схлопывается и любое движение выглядит бесконечно быстрым.
        dt = max(1e-3, now - (now if self.at is None else self.at))
        moved = math.dist(position, position if self.position is None else self.position)
        speed = moved / dt
        if speed > self.max_speed_ms:
            self.rejected += 1
            return {"accepted": False,
                    "reason": (f"скачок {moved:.2f} м за {dt:.2f} с = "
                               f"{speed:.1f} м/с, предел {self.max_speed_ms}")}
        return self._take(session_id, position, now, "непрерывно")

    def _take(self, session_id: str, position: tuple[float, float],
              now: float, reason: str) -> dict[str, Any]:
        switched = session_id != self.session_id
        self.session_id = session_id
        self.position = position
        self.at = now
        self._pending_session = None
        self._pending_count = 0
        return {"accepted": True, "reason": reason, "switched_session": switched}
