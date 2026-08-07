#!/usr/bin/env bash
# Compile the fine-tuned yolo11l-seg ONNX into a Hailo-8 .hef, on pdfserver.
#
# BLOCKED until the Hailo Dataflow Compiler wheel is installed. It is not on
# PyPI: it is downloaded from the Hailo Developer Zone behind a login, so it
# needs your account. Once you have it:
#
#   scp hailo_dataflow_compiler-*.whl pdfserver:~/cv_research/pi_cv_train/
#   ssh pdfserver '~/cv_research/pi_cv_train/.venv/bin/pip install ~/cv_research/pi_cv_train/hailo_dataflow_compiler-*.whl'
#
# Then run this. Everything else it needs is already in place: pdfserver is
# x86-64 Linux (the DFC's only supported platform), the ONNX is exported, and
# the end node names below were read out of that exact graph rather than copied
# from a guide.
set -euo pipefail

REMOTE=${REMOTE:-pdfserver}
# Relative to the REMOTE home, deliberately without a tilde: the heredoc below
# is unquoted so the local shell expands these first, and a `~` would resolve
# to the laptop's home before ever reaching the server.
BASE=${BASE:-cv_research/pi_cv_train}
RUN=${RUN:-runs/segment/runs/indoor_yolo11l}

# The ten raw head outputs, in the roles the Pi decoder identifies by channel
# count: 4*(reg_max+1)=64 boxes, 31 classes, 32 mask coefficients per scale,
# plus the 32-channel prototypes. Cutting the graph here leaves the decode on
# the host, which is what `pi_client/seg_postprocess.py` already does.
END_NODES="/model.23/cv2.0/cv2.0.2/Conv,/model.23/cv3.0/cv3.0.2/Conv,/model.23/cv4.0/cv4.0.2/Conv,\
/model.23/cv2.1/cv2.1.2/Conv,/model.23/cv3.1/cv3.1.2/Conv,/model.23/cv4.1/cv4.1.2/Conv,\
/model.23/cv2.2/cv2.2.2/Conv,/model.23/cv3.2/cv3.2.2/Conv,/model.23/cv4.2/cv4.2.2/Conv,\
/model.23/proto/cv3/conv/Conv"

ssh "$REMOTE" bash -s <<REMOTE_SCRIPT
set -euo pipefail
cd "\$HOME/$BASE"
source .venv/bin/activate
# Quantise on the CPU. Two reasons, both found by trying: GPU 0 is another
# user's vLLM holding all 48 GB, so TensorFlow's default device choice fails
# cuDNN init with a misleading "No DNN in stream executor"; and pinning to
# GPU 1 then died mid-quantisation with CUDA_ERROR_INVALID_HANDLE, which is
# the DFC's bundled TensorFlow disagreeing with this box's driver. This is a
# one-off build — slower and finishing beats faster and crashing.
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2

python3 - <<'PY'
import os
from hailo_sdk_client import ClientRunner

# INT8 needs real images to calibrate against, and they must look like what the
# camera will actually see. Using ADE20K validation frames rather than random
# noise or COCO photos is the difference between a quantised model that keeps
# its accuracy and one that quietly loses several points.
import glob, random, numpy as np
from PIL import Image

files = sorted(glob.glob(os.path.expanduser("~/cv_research/ade20k_yolo/images/val/*.jpg")))
random.Random(0).shuffle(files)
calib = np.stack([
    np.array(Image.open(f).convert("RGB").resize((640, 640)))
    for f in files[:256]
]).astype(np.float32)
print("calibration set:", calib.shape, flush=True)
# 256 is inside Hailo's recommended 64-1024 band and keeps a CPU
# quantisation pass to minutes rather than an hour.

runner = ClientRunner(hw_arch="hailo8")
runner.translate_onnx_model(
    "$RUN/weights/best.onnx",
    "indoor_yolo11l_seg",
    start_node_names=["images"],
    end_node_names="$END_NODES".split(","),
    net_input_shapes={"images": [1, 3, 640, 640]},
)
# Normalise ON THE CHIP, not in the calibration data. The Pi's
# `hailo_infer` hands the NPU uint8 0-255 frames, while the network was
# trained on 0-1 floats. Without this layer the graph silently sees inputs
# 255x too large, and the first thing that breaks is quantisation of the
# softmax inside YOLO11's C2PSA attention blocks:
#     NegativeSlopeExponentNonFixable ... desired shift is 8.0, but op has
#     only 8 data bits ... calibration-set is not normalized properly
# which is the DFC telling you exactly this. Keeping the calibration in
# 0-255 and letting the chip divide means the Pi's input format never
# changes.
runner.load_model_script(
    "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n")
runner.optimize(calib)
hef = runner.compile()
open("$RUN/weights/indoor_yolo11l_seg.hef", "wb").write(hef)
print("wrote $RUN/weights/indoor_yolo11l_seg.hef")
PY
REMOTE_SCRIPT

echo "=== pull it to the laptop, then to the Pi ==="
scp "$REMOTE:$BASE/$RUN/weights/indoor_yolo11l_seg.hef" models/ 2>/dev/null \
  || scp "$REMOTE:~/$BASE/$RUN/weights/indoor_yolo11l_seg.hef" models/
echo "then: scp models/indoor_yolo11l_seg.hef cv-pi.local:~/Desktop/work/pi_cv/models/"
echo "and verify BEFORE trusting it:  ./scripts/run_probe_hef.sh models/indoor_yolo11l_seg.hef"
echo "  (it must report num_classes=31, reg_max=15, num_masks=32)"
