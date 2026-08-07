#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/client/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 scripts/probe_hef.py "$@"
