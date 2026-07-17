#!/usr/bin/env bash
# Clone the yolo26_hailo project (host-side decode for custom yolo26 hefs)
# into external/ on the Raspberry Pi. Run from the repo root on the Pi:
#   ./scripts/setup_yolo26_hailo.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$REPO_ROOT/external/yolo26_hailo"

mkdir -p "$REPO_ROOT/external"
if [ -d "$TARGET/.git" ]; then
  echo "external/yolo26_hailo already cloned — pulling latest"
  git -C "$TARGET" pull --ff-only
else
  git clone https://github.com/DanielDubinsky/yolo26_hailo.git "$TARGET"
fi

echo
echo "Installing python requirements into the current venv..."
python3 -m pip install -r "$TARGET/requirements.txt"

echo
echo "Done. Next steps:"
echo "  1) Compile the hef on an x86 machine: see scripts/compile_yolo26_hef/README.md"
echo "  2) Copy it here as models/yolo26n_hailo8.hef"
echo "  3) Run the session with:"
echo "     ./scripts/run_pi_session.sh --yolo-backend hailo --hailo-arch yolo26 \\"
echo "        --hailo-hef models/yolo26n_hailo8.hef"
