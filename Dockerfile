# ContactPilot grasping pipeline: MuJoCo sim + Contact-GraspNet (PyTorch).
#
# CUDA 12.8 so ONE image runs on both lab GPUs:
#   - GTX 1650 laptop (sm_75)
#   - RTX 5090 workstation (sm_120 Blackwell — needs cu128, cu126 will NOT work)
#
# Build (from the repo root):   docker compose build
# Run the default pick-all demo:   docker compose up
# See DOCKER.md for details.

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

# Ubuntu 22.04 ships Python 3.10 — same as the validated cgn_torch env.
# GL/EGL/OSMesa libs cover MuJoCo headless rendering; libgomp/libusb for Open3D.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git \
        libegl1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libosmesa6 libglfw3 libgomp1 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --upgrade pip

WORKDIR /work

# Torch first: biggest layer (~4 GB), changes least often. 2.11.0 is the
# newest on the cu128 index (the laptop env's 2.12.0 only exists for cu126).
RUN pip3 install --no-cache-dir torch==2.11.0 torchvision \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements-docker.txt .
RUN pip3 install --no-cache-dir -r requirements-docker.txt

# pyrender pins PyOpenGL==3.1.0, which lacks EGLDeviceEXT and breaks MuJoCo's
# EGL backend — install it dep-less, then make sure PyOpenGL is current.
RUN pip3 install --no-cache-dir --no-deps pyrender==0.1.45 \
    && pip3 install --no-cache-dir --upgrade "PyOpenGL>=3.1.7"

# Contact-GraspNet PyTorch port (git submodule on the host — make sure
# `git submodule update --init` has run before `docker build`).
COPY contact_graspnet_pytorch/ contact_graspnet_pytorch/
# Equivalent of the conda env's editable install: put the repo root on the
# path (its setup.py is too old for PEP 660 editable installs under pip 26).
# This also exposes the vendored Pointnet_Pointnet2_pytorch helpers.
ENV PYTHONPATH=/work/contact_graspnet_pytorch

# Checkpoint + test scenes aren't committed to git (see the submodule's
# scripts/download_assets.py) — fetch them at build time so the image stays
# self-contained at `docker compose up`.
RUN python3 contact_graspnet_pytorch/scripts/download_assets.py

# Menagerie Panda model + the simulation package
COPY mujoco_menagerie/franka_emika_panda/ mujoco_menagerie/franka_emika_panda/
COPY mujoco_grasp_sim/ mujoco_grasp_sim/

WORKDIR /work/mujoco_grasp_sim

# Reduce CUDA allocator fragmentation: in WSL2/Linux the GPU has a HARD VRAM
# cap (no WDDM spill-to-RAM like native Windows), so 4 GB cards need this.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --no-vis is baked in: no display in the container (GIFs/metrics/npz still
# land in output/, which compose mounts back to the host).
ENTRYPOINT ["python3", "run_sim_grasp_test.py", "--no-vis"]
CMD ["--seed", "5", "--camera", "lookat", "--pick-all"]
