#!/usr/bin/env bash
# Compile YOLO26 to a Hailo hef inside an x86_64 Docker container.
# Usage: ./run_compile.sh [variant] [target]
#   variant: yolo26n (default) | yolo26s | yolo26m | yolo26l
#   target:  hailo8 (default — the AI HAT+ full chip) | hailo8l
# Prereqs in this directory:
#   - hailo_dataflow_compiler-*.whl  (Hailo Developer Zone, py3.10 x86_64)
#   - calib/                          (~64-200 representative JPEGs)
set -euo pipefail

VARIANT="${1:-yolo26n}"
TARGET="${2:-hailo8}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

DFC_WHL=$(ls hailo_dataflow_compiler-*.whl 2>/dev/null | head -1 || true)
if [ -z "$DFC_WHL" ]; then
  echo "ERROR: put the Hailo Dataflow Compiler wheel here first" >&2
  echo "  https://hailo.ai/developer-zone/software-downloads/ (free account)" >&2
  exit 1
fi
if [ ! -d calib ] || [ -z "$(ls calib 2>/dev/null)" ]; then
  echo "ERROR: put ~64-200 calibration JPEGs into $HERE/calib/" >&2
  exit 1
fi

mkdir -p out
docker build --platform linux/amd64 -t yolo26-hef-builder .

docker run --rm --platform linux/amd64 \
  -v "$HERE:/work" \
  yolo26-hef-builder bash -ec "
    pip install '/work/$DFC_WHL' ultralytics
    if [ ! -d /work/yolo26_hailo ]; then
      git clone https://github.com/DanielDubinsky/yolo26_hailo.git /work/yolo26_hailo
    fi
    cd /work/yolo26_hailo
    pip install -r requirements.txt
    python scripts/download_model.py
    python -m export.cli \
      --variant '$VARIANT' \
      --target '$TARGET' \
      --onnx 'models/$VARIANT.onnx' \
      --calib_dir /work/calib \
      --tag pi_cv
    HEF=\$(ls -t experiments/${VARIANT}_${TARGET}_*/artifacts/3_compiled/model.hef | head -1)
    cp \"\$HEF\" '/work/out/${VARIANT}_${TARGET}.hef'
  "

echo
echo "Done: out/${VARIANT}_${TARGET}.hef"
echo "Copy it to the Pi:  scp out/${VARIANT}_${TARGET}.hef cv-pi.local:~/Desktop/work/pi_cv/models/"
