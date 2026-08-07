#!/usr/bin/env bash
# Deploy/restart gpu_service on pdfserver (GPU1 only, tmux session
# `pi_cv_gpu`). Run this FROM THE LAPTOP after changing anything under
# gpu_service/. Requires the `pdfserver` Host alias in ~/.ssh/config.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REMOTE_DIR="~/cv_research/pi_cv_gpu"

echo "=== rsync gpu_service/ -> pdfserver:$REMOTE_DIR ==="
ssh pdfserver "mkdir -p $REMOTE_DIR"
# --delete plus a remote-only directory is how you lose a dataset: anything
# under $REMOTE_DIR that is not in gpu_service/ gets removed. Training data
# lives in ~/cv_research/ade20k/ (outside this tree) for that reason; these
# excludes are the belt to that braces.
rsync -avP --delete \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' \
    --exclude 'datasets' --exclude 'runs' --exclude 'weights' \
    "$REPO_ROOT/gpu_service/" "pdfserver:$REMOTE_DIR/"

echo "=== VGGT source (facebookresearch/vggt, sibling clone) ==="
ssh pdfserver bash -s <<'REMOTE'
set -euo pipefail
cd ~/cv_research
if [ ! -d vggt ]; then
    git clone --depth 1 https://github.com/facebookresearch/vggt.git
else
    git -C vggt pull --ff-only
fi
REMOTE

echo "=== remote venv + deps ==="
ssh pdfserver bash -s <<'REMOTE'
set -euo pipefail
cd ~/cv_research/pi_cv_gpu
if [ ! -d .venv ]; then
    python3.11 -m venv .venv
fi
source .venv/bin/activate
python3 -m pip install --upgrade pip -q
if ! python3 -c "import torch" 2>/dev/null; then
    echo "installing torch (CUDA 12.4 wheel) - this takes a few minutes the first time"
    pip install torch --index-url https://download.pytorch.org/whl/cu124 -q
fi
if ! python3 -c "import torchvision" 2>/dev/null; then
    pip install torchvision --index-url https://download.pytorch.org/whl/cu124 -q
fi
pip install -r requirements.txt -q
pip install -e ~/cv_research/vggt -q
REMOTE

echo "=== restart tmux session pi_cv_gpu ==="
ssh pdfserver bash -s <<'REMOTE'
set -euo pipefail
tmux kill-session -t pi_cv_gpu 2>/dev/null || true
cd ~/cv_research/pi_cv_gpu
tmux new-session -d -s pi_cv_gpu \
    "CUDA_VISIBLE_DEVICES=1 .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8700"
REMOTE

echo "=== done. Verify from the laptop: curl 127.0.0.1:8700/health ==="
