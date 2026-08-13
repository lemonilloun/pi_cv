"""Топокарта: записанный проезд -> граф кадров.

Вся «карта» пространства здесь — папка картинок в порядке проезда. Ни
координат, ни масштаба, ни системы отсчёта: узел это кадр, ребро это «дальше
по записи». Именно поэтому её нечему разъехаться.

Строится из уже записываемых сессий сцены — ничего нового снимать не нужно,
и пользовательское «больше кадров лучше» здесь верно буквально: чем плотнее
узлы, тем меньше шаг, который политике надо преодолеть между ними.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# Плотность узлов. Апстрим кладёт узел раз в секунду при обычной скорости
# ходьбы; сессии сцены пишутся с шагом по параллаксу, около 2-3 кадров в
# секунду, отсюда шаг по умолчанию.
DEFAULT_STRIDE = 3


def session_display_name(session_dir: Path) -> str:
    """Имя, под которым сессия видна на панели.

    Оно, а не машинный идентификатор, — то, как пользователь называет
    запись. Карта с именем `session_20260813_161329` в списке не
    опознаётся, даже если собрана правильно.
    """
    meta = session_dir / "session_meta.json"
    if meta.exists():
        try:
            name = json.loads(meta.read_text()).get("name")
        except (ValueError, OSError):
            name = None
        if name:
            return str(name)
    return session_dir.name


def build_from_session(session_dir: Path, out_dir: Path,
                       stride: int = DEFAULT_STRIDE,
                       name: str | None = None) -> dict:
    """Разложить кадры сессии в топокарту.

    Кадры КОПИРУЮТСЯ, а не берутся ссылкой: сессию можно удалить или
    переснять, а карта, по которой робот ездит, должна пережить это — иначе
    навигация ломается в момент, никак с ней не связанный.
    """
    frames = sorted(session_dir.glob("keyframes/*/rgb.jpg"))
    if not frames:
        raise FileNotFoundError(f"В {session_dir} нет кадров keyframes/*/rgb.jpg")

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    chosen = frames[::stride]
    for index, path in enumerate(chosen):
        shutil.copyfile(path, out_dir / f"{index}.jpg")

    meta = {
        "name": name or out_dir.name,
        "source_session": session_dir.name,
        "nodes": len(chosen),
        "source_frames": len(frames),
        "stride": stride,
    }
    (out_dir / "topomap.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def load(map_dir: Path) -> tuple[list, dict]:
    """Кадры карты по порядку + её описание.

    Имена файлов сортируются ЧИСЛОМ, а не строкой: при лексикографическом
    порядке узел 10 встал бы между 1 и 2, и граф молча перепутался бы
    местами — без единой ошибки, просто карта другой комнаты.
    """
    from PIL import Image

    files = sorted(map_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"Пустая топокарта: {map_dir}")
    meta_path = map_dir / "topomap.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"name": map_dir.name}
    return [Image.open(f).convert("RGB") for f in files], meta


def list_maps(root: Path) -> list[dict]:
    out = []
    for path in sorted(p for p in root.glob("*") if p.is_dir()):
        meta_path = path / "topomap.json"
        if meta_path.exists():
            out.append(json.loads(meta_path.read_text()))
        else:
            out.append({"name": path.name, "nodes": len(list(path.glob("*.jpg")))})
    return out
