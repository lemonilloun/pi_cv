#!/usr/bin/env bash
# Mac-side setup for the server CV worker: clones Depth-Anything-V2 (for the
# metric_depth package) and downloads the metric indoor checkpoint, then runs
# one verification inference on data/cat.jpg using the configured device.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXTERNAL_DIR="$REPO_ROOT/external"
DA2_DIR="$EXTERNAL_DIR/Depth-Anything-V2"
MODELS_DIR="$REPO_ROOT/models"

HF_REPO="depth-anything/Depth-Anything-V2-Metric-Hypersim-Small"
CHECKPOINT="depth_anything_v2_metric_hypersim_vits.pth"
CHECKPOINT_PATH="$MODELS_DIR/$CHECKPOINT"

mkdir -p "$EXTERNAL_DIR" "$MODELS_DIR"

echo "== Checking Python dependencies (torch/cv2/numpy must already be installed) =="
python3 - <<'PY'
import cv2, numpy, torch
print(f"torch={torch.__version__} mps_available={torch.backends.mps.is_available()}")
print(f"opencv={cv2.__version__} numpy={numpy.__version__}")
PY

if [ ! -d "$DA2_DIR/.git" ]; then
  echo "== Cloning Depth-Anything-V2 into external/ =="
  git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 "$DA2_DIR"
else
  echo "== Depth-Anything-V2 repo already present =="
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
  URL="https://huggingface.co/$HF_REPO/resolve/main/$CHECKPOINT?download=true"
  echo "== Downloading $HF_REPO/$CHECKPOINT =="
  curl -L --fail --continue-at - "$URL" -o "$CHECKPOINT_PATH"
else
  echo "== Metric checkpoint already present =="
fi

echo "== Verification inference on data/cat.jpg =="
export PYTORCH_ENABLE_MPS_FALLBACK=1
PYTHONPATH="$REPO_ROOT:$REPO_ROOT/server/src" python3 - <<PY
import sys
import time
from pathlib import Path

checkpoint = Path("$CHECKPOINT_PATH")
if checkpoint.stat().st_size < 10_000_000:
    raise SystemExit(f"Checkpoint looks too small: {checkpoint}")

sys.path.insert(0, "$DA2_DIR/metric_depth")
import cv2
import torch
from depth_anything_v2.dpt import DepthAnythingV2

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = DepthAnythingV2(
    encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20.0
)
model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
model = model.to(torch.device(device)).eval()

image = cv2.imread("$REPO_ROOT/data/cat.jpg")
if image is None:
    raise SystemExit("data/cat.jpg not found or unreadable")

# warmup + timed run
with torch.no_grad():
    model.infer_image(image, 392)
    started = time.perf_counter()
    depth = model.infer_image(image, 392)
    elapsed_ms = (time.perf_counter() - started) * 1000

print(f"device={device}")
print(f"depth range: {depth.min():.2f}..{depth.max():.2f} m")
print(f"inference: {elapsed_ms:.0f} ms (~{1000/elapsed_ms:.1f} fps)")
print("Server CV setup OK")
PY
