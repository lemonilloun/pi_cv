#!/usr/bin/env bash
# Live dashboard for the pdfserver fine-tune. Read-only: it cannot disturb
# the run. Open http://127.0.0.1:8090/ once it prints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/watch_training.py" "$@"
