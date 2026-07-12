#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${1:-small}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXTERNAL_DIR="$REPO_ROOT/external"
DA2_DIR="$EXTERNAL_DIR/Depth-Anything-V2"
MODELS_DIR="$REPO_ROOT/models"

case "$MODEL_SIZE" in
  small)
    ENCODER="vits"
    HF_REPO="depth-anything/Depth-Anything-V2-Small"
    CHECKPOINT="depth_anything_v2_vits.pth"
    ;;
  base)
    ENCODER="vitb"
    HF_REPO="depth-anything/Depth-Anything-V2-Base"
    CHECKPOINT="depth_anything_v2_vitb.pth"
    ;;
  large)
    ENCODER="vitl"
    HF_REPO="depth-anything/Depth-Anything-V2-Large"
    CHECKPOINT="depth_anything_v2_vitl.pth"
    ;;
  *)
    echo "Usage: $0 [small|base|large]" >&2
    exit 2
    ;;
esac

mkdir -p "$EXTERNAL_DIR" "$MODELS_DIR"

cd "$REPO_ROOT"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r client/requirements-depth-v2.txt

if ! python3 - <<'PY'
import torch
print(torch.__version__)
PY
then
  echo "Installing PyTorch CPU wheel with the official PyTorch CPU index..."
  python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

if [ ! -d "$DA2_DIR/.git" ]; then
  git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 "$DA2_DIR"
else
  git -C "$DA2_DIR" pull --ff-only
fi

CHECKPOINT_PATH="$MODELS_DIR/$CHECKPOINT"
if [ ! -f "$CHECKPOINT_PATH" ]; then
  URL="https://huggingface.co/$HF_REPO/resolve/main/$CHECKPOINT?download=true"
  echo "Downloading $HF_REPO/$CHECKPOINT ..."
  curl -L --fail --continue-at - "$URL" -o "$CHECKPOINT_PATH"
fi

PYTHONPATH="$REPO_ROOT:$REPO_ROOT/client/src:$DA2_DIR" python3 - <<PY
from pathlib import Path
import cv2
import torch
from depth_anything_v2.dpt import DepthAnythingV2

checkpoint = Path("$CHECKPOINT_PATH")
if checkpoint.stat().st_size < 10_000_000:
    raise SystemExit(f"Checkpoint looks too small: {checkpoint}")

configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
model = DepthAnythingV2(**configs["$ENCODER"])
model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
model.eval()
print("Depth Anything V2 ready")
print(f"encoder=$ENCODER")
print(f"checkpoint={checkpoint}")
print(f"opencv={cv2.__version__}")
print(f"torch={torch.__version__}")
PY

cat <<EOF

Run a real depth test:

  ./scripts/run_cv_client.sh depth \\
    --source camera \\
    --depth-backend depth-anything-v2 \\
    --depth-model-path models/$CHECKPOINT \\
    --depth-encoder $ENCODER \\
    --depth-input-size 392 \\
    --telemetry

EOF
