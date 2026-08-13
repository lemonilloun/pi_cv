#!/usr/bin/env bash
# Отправить топокарту на робота: считает он, значит и карта нужна ему.
# Кадры копируются, а не монтируются — карта должна пережить пересъёмку и
# удаление исходной сессии на ноутбуке.
set -euo pipefail
cd "$(dirname "$0")/.."
MAP="${1:?укажите имя топокарты, например bedroom2}"
PI="${PI_HOST:-cv-pi.local}"
SRC="data/topomaps/$MAP"
[ -d "$SRC" ] || { echo "Нет топокарты $SRC" >&2; exit 1; }
ssh "$PI" "mkdir -p ~/Desktop/work/pi_cv/data/topomaps"
rsync -a --delete "$SRC/" "$PI:~/Desktop/work/pi_cv/data/topomaps/$MAP/"
echo "Карта $MAP на роботе: $(ls "$SRC"/*.jpg | wc -l | tr -d ' ') узлов"
