#!/usr/bin/env bash
# Замер скорости шасси по вашим проездам. Сервер должен быть уже запущен
# (вами, в вашем tmux) — скрипт только слушает его статус.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/.." && pwd)"
exec python3 scripts/measure_odometry.py "$@"
