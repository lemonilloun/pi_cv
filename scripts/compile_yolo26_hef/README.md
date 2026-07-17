# Compiling YOLO26n to a Hailo-8 .hef

The AI HAT+ carries a full Hailo-8, but the Hailo Model Zoo has no YOLO26 —
the hef must be compiled from ONNX with the **Hailo Dataflow Compiler
(DFC)**, using the MIT-licensed pipeline from
[DanielDubinsky/yolo26_hailo](https://github.com/DanielDubinsky/yolo26_hailo)
(measured on Hailo-8L: yolo26n at **86.5 fps, 0.371 mAP**; the full Hailo-8
is faster still).

## Hard requirement: x86_64 Linux

The DFC only runs on x86_64 Linux — not on the M2 Mac natively and not on
the Pi. Two workable paths:

1. **Docker with x86 emulation on the M2** (this directory): Rosetta-backed
   `--platform linux/amd64` containers work but quantization is slow under
   emulation (expect an hour+ for yolo26n; it is a one-time cost).
   Docker Desktop → Settings → check "Use Rosetta for x86_64/amd64
   emulation" first.
2. **Any real x86 Linux box / cloud VM / Google Colab**: same steps, native
   speed (minutes). If you have access to one, prefer it.

## One-time: download the DFC

The DFC wheel is a free download but needs a (free) account:
<https://hailo.ai/developer-zone/software-downloads/> → *Dataflow Compiler*
→ the Python 3.10 / x86_64 `.whl`. Put it in this directory —
`run_compile.sh` picks up `hailo_dataflow_compiler-*.whl` automatically.

## Calibration images

INT8 quantization needs ~64-200 representative images. Best quality: JPEGs
from your own Pi camera (the actual room!). Drop them into `calib/` here.
Fallback: any COCO val2017 subset works.

## Run

```bash
cd scripts/compile_yolo26_hef
# put the DFC .whl and calib/ images here first
./run_compile.sh yolo26n hailo8
```

Output lands in `out/yolo26n_hailo8.hef`. Copy it to the Pi:

```bash
scp out/yolo26n_hailo8.hef cv-pi.local:~/Desktop/work/pi_cv/models/
```

## On the Pi

```bash
./scripts/setup_yolo26_hailo.sh   # once: clones the host-side decoder
./scripts/run_pi_session.sh --yolo-backend hailo --hailo-arch yolo26 \
    --hailo-hef models/yolo26n_hailo8.hef
```

Or persist it in `.env`:

```env
PI_CV_SESSION_YOLO_BACKEND=hailo
PI_CV_SESSION_HAILO_ARCH=yolo26
PI_CV_SESSION_HAILO_HEF=models/yolo26n_hailo8.hef
```

Until the hef exists, the session default is `ncnn` +
`models/yolo26n_ncnn_model` — same detection quality, just CPU-bound
(~1-3 fps), which monitoring tolerates fine.
