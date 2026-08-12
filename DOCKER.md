# Dockerized Grasping Pipeline

One image, every machine: the laptop (GTX 1650), the RTX 5090 workstation,
and future lab PCs. CUDA 12.8 + torch cu128 wheels cover sm_75 through
sm_120 (Blackwell), so no per-machine torch juggling.

## Prerequisites

| Host | Needs |
|------|-------|
| Linux (lab workstation) | Docker Engine + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| Windows (dev laptop) | Docker Desktop with the WSL2 backend (GPU support is built in; needs a current NVIDIA driver on Windows) |

Verify GPU passthrough first:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

## Build (one command)

From the repo root (the folder containing this file), make sure the
`contact_graspnet_pytorch` submodule is checked out first:

```bash
git submodule update --init --depth 1
docker compose build
```

First build downloads ~4 GB (torch cu128) plus the ~160 MB Contact-GraspNet
checkpoint/test-data (fetched from Hugging Face Hub during the build) —
later builds reuse cached layers.
Code-only changes to `mujoco_grasp_sim/` rebuild in seconds.

## Run (one command)

```bash
docker compose up          # default: --seed 5 --camera lookat --pick-all
```

Results appear on the **host** in `mujoco_grasp_sim/output/<timestamp>/`:
`metrics.json`, `execution.gif`, `observation.png`, `predictions_sim.npz`.

Custom runs — append any `run_sim_grasp_test.py` arguments:

```bash
docker compose run --rm grasp-sim --seed 3 --camera calibrated --execute --top-k 5
docker compose run --rm grasp-sim --pick-all --camera lookat --save-dir output/exp1
```

(`--no-vis` is always enforced inside the container; there is no display.)

## Notes & troubleshooting

- **Calibration**: `calibration_result.yaml` is volume-mounted read-only, so a
  recalibration on the real robot needs no rebuild — just replace the file.
- **Rendering**: the container renders headless via EGL on the GPU. If EGL
  fails on your host (rare WSL2/driver combos; symptoms: `mujoco.FatalError`
  mentioning EGL), fall back to CPU rendering:
  `MUJOCO_GL=osmesa docker compose up`.
- **Interactive viewers** (`--view-sim`, Open3D window) are not available in
  the container; run those from a native environment, or on Linux mount the X
  socket (`-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY` and drop the baked-in
  `--no-vis` by overriding the entrypoint).
- **torch version**: pinned `torch==2.11.0` from the cu128 index (the newest
  published there as of 2026-06-11; the laptop conda env's 2.12.0 exists only
  for cu126, which Blackwell can't use). If the pin ever vanishes from the
  index, bump to the nearest cu128 version in the `Dockerfile`.
- **RAM**: the in-container pipeline is the same code that survives the 8 GB
  laptop (CGN runs as a subprocess per pick-all round), so container memory
  limits are not a concern on the 5090 box.
