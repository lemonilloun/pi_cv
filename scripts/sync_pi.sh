#!/usr/bin/env bash
# Синхронизация кода на Raspberry Pi.
#
# Нужен потому, что git на Pi к GitHub не ходит: remote прописан по HTTPS без
# учётных данных, и по ssh сессия неинтерактивная — `git fetch` падает с
# "could not read Username". Пока это не исправлено ключом, код едет отсюда.
#
# Копировать файлы поштучно нельзя: ровно так config/seg_classes_indoor.json
# один раз остался на Pi без поля `role`, и это НЕ вызвало бы ошибки —
# load_roles подставляет "object" по умолчанию, и стены снова стали бы
# объектами. Молча. Поэтому синхронизируется дерево целиком.
#
#   ./scripts/sync_pi.sh          # синхронизировать
#   ./scripts/sync_pi.sh --check  # только показать расхождения
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PI_HOST="${PI_HOST:-cv-pi.local}"
PI_DIR="${PI_DIR:-Desktop/work/pi_cv}"

# .venv исключён категорически: венв с ноутбука на Pi нерабочий по
# определению — колёса собраны под другую ОС и архитектуру. Один такой
# уже занимал там 6.6 ГБ, включая 2.9 ГБ библиотек CUDA на машине без
# видеокарты. data/ и models/ весят гигабайты и живут на Pi своей жизнью.
EXCLUDES=(
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc'
  --exclude 'data/' --exclude 'models/' --exclude 'external/' --exclude 'research/'
  --exclude '.env' --exclude 'runs/' --exclude '*.hef' --exclude '*.pt'
)

DRY=()
if [ "${1:-}" = "--check" ]; then
  DRY=(--dry-run)
  echo "=== только проверка, ничего не меняю ==="
fi

# --delete намеренно НЕ используется: на Pi лежат файлы, которых нет здесь —
# calibration, снятые сессии, реплей-фикстуры. Удалить их ради чистоты дерева
# было бы обменом рабочих данных на аккуратность.
rsync -az --itemize-changes "${DRY[@]}" "${EXCLUDES[@]}" \
  "$REPO_ROOT/" "$PI_HOST:$PI_DIR/" | grep -vE '^\.d' || true

if [ ${#DRY[@]} -eq 0 ]; then
  echo "=== проверка импортов на Pi ==="
  ssh "$PI_HOST" "cd $PI_DIR && python3 -c \"
import sys; sys.path[:0]=['.','client/src','server/src']
import pi_client.scene_recorder, pi_client.imu_rvc
import json; d=json.load(open('config/seg_classes_indoor.json'))
assert 'role' in d['classes'][0], 'seg_classes_indoor.json без role!'
print('Pi готов: импорты ok, классов', len(d['classes']), 'с ролями')
\""
fi
