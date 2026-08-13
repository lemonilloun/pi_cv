#!/usr/bin/env bash
# Автономное движение по топокарте. Запускается НА РОБОТЕ: политика считается
# здесь, ноутбук только превращает путевую точку в ШИМ колёс.
set -euo pipefail
cd "$(dirname "$0")/.."
exec env PYTHONPATH="client/src:.:${PYTHONPATH:-}" python3 -m pi_client.nav2_runner "$@"
