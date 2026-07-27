# ORB-SLAM3 mono-inertial on Pi 5 — install plan (disk-budget aware)

Primary approach per the July 2026 research (`real-time vio_slam.md`):
ORB-SLAM3 mono-inertial under ROS2, running on the Pi. Threshold to fall
back to the hybrid plan (frontend on Pi, backend on the Mac): trekking
sustains <10-12 FPS, or initialization/loop-closure won't converge.

## Why Docker, not a native install

The Pi is on **Debian 13 (trixie)**, confirmed via `cat /etc/os-release`.
Every ROS2 distro (Humble, Jazzy) is **Tier 3 on any Debian derivative** —
no official apt packages, source build only — and the one existing
Raspberry-Pi-5-specific ORB-SLAM3 fork (`eshan-sud/ORB_SLAM3`, likely the
codebase behind the IEEE Access 2025 Pi 5 evaluation the research doc
cites) was built and tested against **Debian 12 (bookworm) + ROS2 Humble**,
not trixie. Building the entire ROS2 stack from source on an OS newer than
anything it was tested against is a real fragility risk (newer glibc/gcc,
newer default library versions) on top of already being the slow, heavy
path.

**Docker sidesteps the OS question entirely**: a `ros:humble-ros-base`
container (Ubuntu 22.04 base — ROS2 Humble's actual Tier 1, full apt
support) gives the exact tested environment regardless of what the host
runs, without installing anything ROS-related on the host OS itself, and
without spending hours building ROS2's ~hundreds of packages from source.
ORB-SLAM3 still needs building from source (it has no apt package
anywhere), but that alone is a much smaller, more contained build than
"ROS2 + ORB-SLAM3" both from source.

## Disk budget (checked before starting anything)

```
df -h /              → 58G total, 20G used, 36G available
du -sh pi_cv/.venv    → 5.0G  (torch/ultralytics/opencv, existing)
du -sh pi_cv/models   → 233M
du -sh pi_cv/external → 540M
```

36 GB free. Budget for this step: Docker engine ~200 MB, `ros:humble-ros-base`
arm64 image (official, confirmed to exist for arm64v8) — a few hundred MB
to ~1 GB, well under a "desktop" variant (which bundles rviz2/Gazebo,
multiple GB, not needed here). ORB-SLAM3's own build dependencies
(OpenCV, Eigen, Pangolin, DBoW2, g2o, Sophus) built inside the container
add more, bounded and cleanable independently of the host — if this step
turns out to need more than a few GB beyond the image itself, that's the
signal to stop and reassess rather than let it grow unbounded.

**Nothing here touches `models/`, `external/`, or anything already on the
host outside the Docker data root** — if the whole thing needs to be
undone, `docker system prune -a` and `apt remove docker.io` return the Pi
to its current state.

## Steps

1. **Install Docker** (~200 MB, one apt install):
   ```bash
   sudo apt update
   sudo apt install -y docker.io
   sudo usermod -aG docker $USER   # then log out/in, or `newgrp docker`
   ```
2. **Pull the base image** (arm64, confirmed to exist as `arm64v8/ros:humble-ros-base` /
   multi-arch `ros:humble-ros-base`):
   ```bash
   docker pull ros:humble-ros-base
   docker run --rm ros:humble-ros-base ros2 --version   # sanity check
   ```
3. **Build ORB-SLAM3 inside a container** built from that base, adding:
   `build-essential cmake libeigen3-dev libopencv-dev libboost-all-dev`
   + Pangolin (built from source, no apt package) + the ORB-SLAM3 source
   itself (`UZ-SLAMLab/ORB_SLAM3`, which already supports mono-inertial
   natively — the `eshan-sud` fork's value is Pi/ARM-specific fixes on
   top, e.g. Pangolin's `-Werror` breaking the ARM build, a stereo
   trajectory-save segfault; worth pulling those patches in rather than
   using vanilla upstream).
4. **ROS2 wrapper node**: a thin `rclpy`/C++ node publishing camera frames
   (from libcamera/picamera2, already working in this repo) as
   `sensor_msgs/Image` and IMU samples (from the new `imu_shtp.ShtpReader`,
   once wired) as `sensor_msgs/Imu`, feeding ORB-SLAM3's mono-inertial
   ROS2 example node. This is new code, not yet written — comes after the
   IMU is physically wired and the SHTP driver is verified against real
   hardware (docs/imu_shtp_setup.md), since there's no point wiring up a
   SLAM node with no real IMU stream to feed it.
5. **Benchmark**: FPS, tracking loss rate, loop-closure success on an
   actual walk — compare against the research doc's cited Pi 5 numbers
   (ORB-SLAM3 ~15 FPS stereo-tracking-only on Cortex-A76; mono should be
   lighter, no cross-camera matching cost).

## What this session has done vs. what's still open

Done: disk-budget check, this plan, `real-time vio_slam.md` research
already in the repo. **Not yet done** (needs your go-ahead or the
IMU-wiring step to land first): installing Docker, pulling the image,
building ORB-SLAM3, writing the ROS2 wrapper node. These are real
time/disk commitments (the ORB-SLAM3 build alone is plausibly 30-90
minutes on a Pi 5) — say the word and the next session picks up at step 1.
