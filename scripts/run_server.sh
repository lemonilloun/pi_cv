#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/server/src${PYTHONPATH:+:$PYTHONPATH}"
# Safety net: torch 2.1 MPS lacks a few ops (bicubic upsample) used by DA2.
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec python3 -m mac_server.server "$@"
