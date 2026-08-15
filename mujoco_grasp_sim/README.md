# mujoco_grasp_sim — Franka Panda tabletop grasping sim for Contact-GraspNet

MuJoCo simulation that mirrors the real-world setup (Franka Panda +
eye-to-hand RGB-D camera + tabletop with unknown objects) to evaluate
Contact-GraspNet-PyTorch end to end before robot deployment.

**Status: validated.** Seed-42 smoke run on a GTX 1650: 4 objects spawned,
52 grasps in ~12 s, table-plane reconstruction within 1 cm of ground truth
(verifies depth conversion + intrinsics + extrinsics simultaneously).

## Architecture

```
        MuJoCo (Menagerie Panda + table + random objects)
                            │  physics settle
                            ▼
            CameraModule (eye-to-hand, 640x480)
            RGB ── Depth [m] ── Segmap ── K ── T_world_cam
                            │
                            ▼
            depth_to_pointcloud  (OpenCV camera frame)
                            │
                            ▼
        ContactGraspNetPredictor (reuses existing repo pipeline)
                            │   {seg_id: (N,4,4) T_cam_grasp}, scores
                            ▼
        GraspFeasibilityChecker (table-collision + underhand filter)
                            │
                            ▼
        metrics.json ── predictions_sim.npz ── Open3D visualization
```

## Directory structure

```
mujoco_grasp_sim/
├── run_sim_grasp_test.py       # main entry point
├── README.md
├── assets/
│   └── objects/                # drop YCB / custom .obj/.stl meshes here
├── sim_grasp/
│   ├── __init__.py
│   ├── frames.py               # FRAME CONVENTIONS — read first
│   ├── scene_generator.py      # SceneGenerator + SceneConfig
│   ├── camera.py               # CameraModule (RGB/depth/seg, K, extrinsics)
│   ├── pointcloud.py           # depth -> point cloud
│   ├── grasp_predictor.py      # GraspPredictor ABC + ContactGraspNetPredictor
│   ├── feasibility.py          # table-collision grasp filter
│   └── visualizer.py           # Open3D + 2D observation dumps
└── output/<run>/               # rgb.png, depth.npy, observation.png,
                                # metrics.json, predictions_sim.npz
```

Sibling dependencies (already in the repo root):
- `../contact_graspnet_pytorch/` — network, checkpoint, visualization (pip-installed editable)
- `../mujoco_menagerie/franka_emika_panda/` — robot model (sparse clone)

## Setup

Everything lives in the existing `cgn_torch` conda env. Only `mujoco` was added:

```powershell
conda activate cgn_torch
pip install mujoco          # 3.9.0 — modern API, NOT mujoco-py
```

(Full env recreation: see `../README.md` — torch cu126/cu128, numpy<2,
open3d, etc., then `pip install mujoco`.)

If the Menagerie assets are missing:

```powershell
git clone --depth 1 --filter=blob:none --sparse --config core.autocrlf=false https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie sparse-checkout set franka_emika_panda
```

**WSL2 note (affects both backends):** running `run_sim_grasp_test.py` /
`benchmark.py` on WSL2 requires `export MUJOCO_GL=osmesa` (install via
`sudo apt-get install libosmesa6 libosmesa6-dev`) — without it, MuJoCo's
segmentation-mask rendering silently returns garbage on WSL2's Mesa/D3D12 GL
stack (there's no native NVIDIA EGL/GLX passthrough in WSL2).

### GraspGen backend setup (optional, `--backend graspgen`)

[NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) is a second grasp
backend (diffusion-based, Franka-Panda only) evaluated alongside
Contact-GraspNet. It needs its own conda env — its dependencies conflict
with `cgn_torch` — and is invoked as a subprocess, never in-process.

**License note:** NVIDIA Research license, not a permissive open-source
license — commercial use requires contacting NVIDIA Research Licensing.

The steps below are the confirmed-working WSL2/Linux setup (this backend has
only ever been validated on this machine under WSL2 — the Franka-Panda
checkpoint uses a PTv3 backbone requiring `torch_scatter`, which fails to
compile on Windows/MSVC with a confirmed open upstream
PyTorch/CUDA/Windows bug; use bash/WSL2 or native Linux, not PowerShell):

```bash
# 1. Clone GraspGen OUTSIDE this repo (not a submodule — nothing here patches it)
cd ~ && git clone https://github.com/NVlabs/GraspGen.git

# 2. Install the CUDA Toolkit (needed to compile GraspGen's pointnet2_ops
#    CUDA extension). On WSL2, use NVIDIA's WSL-Ubuntu apt repo — NOT a
#    normal Linux driver install, since WSL2 already gets GPU passthrough
#    from the Windows host:
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get -y install cuda-toolkit-12-8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
# (On native Linux, not WSL2: install the CUDA Toolkit + matching NVIDIA
# driver the normal way for your distro instead of the WSL-Ubuntu repo above.)

# 3. Create the graspgen_torch env and install torch cu128 (needed for
#    Blackwell/sm_120 GPUs — use cu121 only on older architectures)
conda create -n graspgen_torch python=3.10 -y
/path/to/miniconda3/envs/graspgen_torch/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Install GraspGen.
#    GOTCHA: `pip install -e .` alone silently DOWNGRADES torch to
#    GraspGen's own pinned torch==2.1.0 (breaking Blackwell support), even
#    with --no-build-isolation — that flag only protects the *build* step,
#    not the final dependency-resolution/install step. Work around it:
cd ~/GraspGen
/path/to/graspgen_torch/bin/python -m pip install --no-build-isolation -e .
# The above downgrades torch — restore it:
/path/to/graspgen_torch/bin/python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
# It may also bump numpy past this project's numpy<2 requirement — restore that too:
/path/to/graspgen_torch/bin/python -m pip install --no-deps "numpy==1.26.4"

# 5. Build the pointnet2_ops CUDA extension
cd ~/GraspGen/pointnet2_ops
CUDA_HOME=/usr/local/cuda-12.8 /path/to/graspgen_torch/bin/python -m pip install --no-build-isolation --no-deps .

# 6. Fetch the Franka-Panda checkpoint (from ContactPilot's mujoco_grasp_sim/)
cd ~/ContactPilot/mujoco_grasp_sim
/path/to/cgn_torch/bin/python scripts/download_graspgen_checkpoint.py

# 7. Point ContactPilot at the graspgen_torch interpreter
export GRASPGEN_PYTHON=/path/to/miniconda3/envs/graspgen_torch/bin/python
```

`GRASPGEN_PYTHON` needs to be set (or `--graspgen-python PATH` passed) any
time `--backend graspgen` is used — `run_sim_grasp_test.py` fails fast with
a clear error if it's missing, rather than silently trying to run GraspGen
under `cgn_torch`.

### Promptable selection setup (optional, `--prompt`/`--click`/`--box`)

[Meta SAM 3](https://github.com/facebookresearch/sam3) resolves a text/
click/box prompt to a target object mask. Needs its own conda env (Python
3.12) and, unlike the other dependencies in this repo, its checkpoints are
**gated on Hugging Face** — request access at
https://huggingface.co/facebook/sam3 before anything else here will work.

**License note:** custom Meta "SAM License", not a standard permissive
license — same category of consideration as GraspGen's NVIDIA license.

```bash
# 1. Request access at https://huggingface.co/facebook/sam3, then:
hf auth login

# 2. Clone SAM 3 OUTSIDE this repo
cd ~
git clone https://github.com/facebookresearch/sam3.git

# 3. Create its env (separate from cgn_torch and graspgen_torch)
conda create -n sam3_torch python=3.12 -y
conda activate sam3_torch
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
cd ~/sam3

# 4. Install SAM 3.
#    GOTCHA: `pip install -e .` alone fails two ways — a bare
#    ModuleNotFoundError for pkg_resources (fresh setuptools no longer
#    bundles it), and later ModuleNotFoundError for einops/pycocotools
#    (SAM 3's pyproject.toml only lists those under the optional
#    notebooks/train extras even though core code imports them
#    unconditionally). Work around both:
pip install "setuptools<81"
pip install -e ".[notebooks,train]"
# do NOT install the optional flash-attn-3/cc_torch extras — unnecessary,
# and they're compiled extensions this repo has otherwise avoided needing

# 5. Point ContactPilot at the sam3_torch interpreter
export SAM3_PYTHON=$(which python)   # while sam3_torch is still active
```

`SAM3_PYTHON` needs to be set (or `--sam3-python PATH` passed) any time
`--prompt`/`--click`/`--box` is used — fails fast with a clear error if
missing, same as `GRASPGEN_PYTHON`.

**Accuracy (P5 baseline, seeds 0-4, `benchmark_prompt_selection.py`):
3/5 correct selections (60%), mean IoU 0.733 on the seeds SAM 3 returned a
match for.** When SAM 3 finds the target, localization is excellent
(~0.98 IoU on hits); the failure mode is prompt/color mismatches — it
missed one color entirely and mis-selected on another. This is a working
feature useful for experimentation, not yet reliable enough to trust as a
primary selection path — see `ROADMAP.md` P5 for the full per-seed
breakdown before relying on it.

## Run

```powershell
conda activate cgn_torch
cd mujoco_grasp_sim

python run_sim_grasp_test.py                          # random scene + Open3D vis
python run_sim_grasp_test.py --execute                # ALSO pick the object (IK + control)
python run_sim_grasp_test.py --execute --top-k 5      # retry up to 5 best grasps
python run_sim_grasp_test.py --view-sim               # ALSO open interactive MuJoCo viewer
python run_sim_grasp_test.py --seed 5 --n-objects 4   # reproducible scene (seed 5 = verified pick)
python run_sim_grasp_test.py --no-vis                 # headless
python run_sim_grasp_test.py --forward-passes 3       # denser proposals (needs >=6GB GPU)
python run_sim_grasp_test.py --no-feasibility         # raw CGN output
python run_sim_grasp_test.py --camera lookat          # generic angled camera (pre-calibration setup)
python run_sim_grasp_test.py --camera calibrated      # top-down lab calibration (default)
python run_sim_grasp_test.py --pick-all               # pick EVERY object, place each in the bin
python run_sim_grasp_test.py --save-dir output\myrun  # named output dir
python run_sim_grasp_test.py --camera fused           # P2: fuse lookat + calibrated side cam clouds
python run_sim_grasp_test.py --execute --pick-object 6        # P3: grasp THIS object only
python run_sim_grasp_test.py --execute --grasp-index 2        # P4: run candidate #2 from the printed list
python run_sim_grasp_test.py --recenter --clean-depth         # P1 experimental flags (see ROADMAP)
python run_sim_grasp_test.py --backend graspgen --execute     # NVlabs/GraspGen instead of CGN (needs GRASPGEN_PYTHON — see "GraspGen backend setup")
python run_sim_grasp_test.py --execute --prompt "the red box" # select the target by text description (needs SAM3_PYTHON — see "Promptable selection setup")
python run_sim_grasp_test.py --execute --click 320,240        # select the target by clicking a pixel (observation.png coords)
```

`--camera fused` captures BOTH observation cameras (generic lookat as the
primary + the calibrated side mount from `calibration_result.yaml`), fuses
the depth clouds in the world frame with voxel dedup (`sim_grasp/fusion.py`),
expresses the result in the primary camera frame and runs Contact-GraspNet
on the clouds directly (`predict_clouds` / cloud-mode `cgn_worker`). Grasps
come back in the primary camera frame, so everything downstream is unchanged.

With `--execute`, a ranked candidate table is printed; `--grasp-index I`
executes exactly that candidate and `--pick-object SEG_ID` (also valid with
`--pick-all`) restricts grasping to one chosen object.

By default the eye-to-hand camera is placed from the REAL lab calibration
(`calibration_result.yaml`, interpreted as T_base_cam): top-down at
base-frame (0.49, 0.27, 0.88). `--camera lookat` (or `--calibration none`)
restores the generic angled view. Running both and comparing
`metrics.json` is the intended way to A/B the two setups:

```powershell
python run_sim_grasp_test.py --seed 5 --execute --top-k 5 --camera calibrated --save-dir output\ab_topdown
python run_sim_grasp_test.py --seed 5 --execute --top-k 5 --camera lookat     --save-dir output\ab_lookat
```

With `--execute`, a GIF of the pick attempt is saved to
`output/<run>/execution.gif` and per-attempt results land in `metrics.json`
under `execution` (stage reached, IK errors, object raise height, verdict).

`--pick-all` runs full pick-AND-PLACE for every object: each round the arm
returns to the observe pose, the scene is re-captured, Contact-GraspNet
re-runs (in a per-round subprocess — keeps torch's multi-GB footprint out of
the sim process so 8 GB machines survive), the best-ranked grasp among
still-on-table objects is executed, and the object is carried to the bin
(the grey tray on the table). Objects that get no grasps benefit from
re-observation as the table empties; if none are found at all, one retry
sweep runs with CGN thresholds lowered to 0.08. Per-object retry budget is
3; results land in `metrics.json` under `pick_all` (in_bin / left_on_table /
fell_off_table / per-round log). Validated 2026-06-11: seed 5, `--camera
lookat` → 5/7 objects binned (CGN inference is stochastic, so reruns vary).
The GIF is recorded from a dedicated close-up side camera (`record_cam`,
near the table edge) rather than the observation camera, so finger-object
contact is actually visible; frames are buffered at half resolution with a
hard cap (`GIF_DOWNSAMPLE` / `GIF_MAX_FRAMES` in `sim_grasp/executor.py`)
to bound memory on 8 GB machines.

### Batch evaluation + failure taxonomy (P1)

Reliability changes are judged on success rates over many seeds, never single
runs (CGN inference is stochastic):

```powershell
python benchmark.py --seeds 0-4 --mode pick-all --camera lookat --tag baseline
python benchmark.py --seeds 0-9 --mode execute --top-k 5     # single-pick rate
python analyze_failures.py output\bench_baseline               # taxonomy table
```

`benchmark.py` runs `run_sim_grasp_test.py` headless per seed into
`output/bench_<tag>/seed_N/` and prints aggregate success rates (also saved
to `summary.json`). `analyze_failures.py` classifies every failed attempt in
a benchmark dir (or a single run dir) into the failure taxonomy —
`no_grasp_prediction / ik_unreachable / closed_on_air / object_displaced /
unstable_slip / place_unreachable / missed_bin / knocked_off_table` — and
writes `taxonomy.json`. `sim_grasp/perception.py` holds the countermeasures:
workspace cropping + depth speckle removal (pre-CGN) and grasp re-centering
along the finger-closing axis (post-CGN, targets `closed_on_air`).

Inspect the last generated scene standalone (objects re-drop and settle live —
the XML stores spawn poses, not the settled state):

```powershell
python -m mujoco.viewer --mjcf="..\mujoco_menagerie\franka_emika_panda\_generated_scene.xml"
```

Re-visualize any saved run later (uses the CGN repo's viewer):

```powershell
cd ..\contact_graspnet_pytorch
python contact_graspnet_pytorch\visualize_saved_scene.py --results_path=..\mujoco_grasp_sim\output\test_run\predictions_sim.npz
```

## Coordinate frames (summary — full doc in `sim_grasp/frames.py`)

| Frame | Definition |
|---|---|
| **World** | MuJoCo world, z-up, origin at robot pedestal base on the floor |
| **Robot base** | Franka `link0`; `T_world_base = Trans(0, 0, 0.75)` (tabletop height) |
| **Camera (OpenCV)** | +Z forward, +X right, +Y down. All depth/cloud/grasp data uses this. Converted from MuJoCo's camera (-Z forward) via `R_cv = R_mj @ diag(1,-1,-1)` |
| **Grasp** | CGN/Panda convention: origin at gripper base, +Z approach, +X finger closing line, TCP at +0.1034 m |

Robot execution chain: `T_base_grasp = inv(T_world_base) @ T_world_cam @ T_cam_grasp`.

## Depth conversion — why this code is correct

Modern `mujoco.Renderer` with `enable_depth_rendering()` returns depth
**already linearized to meters** (`near / (1 - z_buf (1 - near/far))`,
done inside `mujoco/renderer.py`). It is perpendicular z-distance — the same
quantity a RealSense reports and what pinhole back-projection expects.
Do NOT apply any extra znear/zfar conversion on top (the classic bug in old
mujoco-py examples). Validation: the reconstructed tabletop plane lands at
z = 0.750 ± 0.01 in the world frame.

## Swapping the grasp backend later

Implement `GraspPredictor` (one method) in a new `sim_grasp/<name>_predictor.py`:

```python
class AnyGraspPredictor(GraspPredictor):
    def predict(self, depth, K, rgb=None, segmap=None) -> GraspPrediction:
        ...  # call AnyGrasp SDK, repackage into GraspPrediction
```

Everything else (scene, camera, feasibility, metrics, visualization) is
backend-agnostic. Same goes for GSNet, GIGA, FoundationPose+planner.

`sim_grasp/graspgen_predictor.py` (`GraspGenPredictor`, backing
`--backend graspgen`) is a realized example of this pattern for a backend
whose dependencies conflict with `cgn_torch` — it always subprocesses out to
a separate conda env rather than running in-process, unlike
`ContactGraspNetPredictor`.

## Adding mesh objects (YCB etc.)

Drop `.obj`/`.stl` files into `assets/objects/` — `_make_mesh_object()` still
auto-scales them to ≤ 12 cm, but the default scene config has
`use_meshes=False` (see "Box-only objects" below). Set `use_meshes=True` and
`mesh_probability>0` to mix them back in.

## Box-only objects, fixed object count (2026-06-14)

The default scene is now **3 box/cuboid objects** (`n_objects_range=(3,3)`,
`use_meshes=False`) at random positions in the reachable spawn region.
Cylinders, spheres, capsules and YCB meshes are no longer generated by
default — curved/non-box shapes dominated pick failures (rolling/slipping
during gripper closing and during the transit-to-bin transfer). 5-seed
fused pick-all benchmark: **box-only/3-objects = 10/15 binned (67%)**,
0 knocked off table, vs the prior box+cylinder+sphere+capsule, 5-10 object
scenes at 21/40 (52%), 7 knocked off table, with the same P1 execution levers
(friction audit, two-phase closing, shape/yaw-aware ranking — see
ROADMAP.md P1).

Override with `--n-objects N` (CLI) or `SceneConfig(n_objects_range=(lo,hi),
use_meshes=True, ...)` to restore the old varied-object behavior.

## Fingertip friction/condim fix (2026-06-14)

The original friction-audit patch (`friction="1.0 0.01 0.004"` on the 5
`fingertip_pad_collision_*` geoms) was a **no-op**: MuJoCo's default
`condim=3` only activates `friction[0]` (sliding) in the contact friction
cone, and 1.0 is also the unpatched default, so nothing changed. This let
grasped boxes slowly twist/slide out of the gripper during lift and during
the transit-to-bin move. Fixed in `_patched_panda_xml()` by patching
`friction="1.5 0.02 0.004" condim="4"` instead — raises sliding friction
above the rigid-body default *and* activates torsional friction, which
resists the off-center-grasp twisting torque. 5-seed fused pick-all
(box-only/3-objects): **14/15 binned (93%)**, up from 10/15 (67%).

## Knobs (`SceneConfig` in `sim_grasp/scene_generator.py`)

- `n_objects_range=(3, 3)` (fixed 3 box objects by default), `spawn_x/spawn_y`, `min_object_spacing=0.09`
- `use_meshes=False`, `mesh_probability=0.4` (only used if `use_meshes=True`)
- `table_height=0.75`, table size/position
- `cam_pos=(1.35, 0, 1.40)`, `cam_target=(0.55, 0, 0.75)`, `cam_fovy_deg=58` (D455-ish)
- `settle_time=3.0` + adaptive extra settling until objects rest

## How the Franka is controlled (joint space vs task space)

The Menagerie Panda uses **joint-space position servos**: `data.ctrl[0..6]`
are joint-angle targets (rad), `data.ctrl[7]` is the gripper tendon
(0 = closed, 255 = 0.08 m open). Cartesian grasp poses are converted by
**differential IK** (damped least squares on the 7 arm joints, multi-seed
restarts, targets the `hand` body frame) in `sim_grasp/executor.py`, then
tracked by linear joint-space interpolation of the ctrl reference:

```
T_cam_grasp --T_world_cam--> T_world_grasp --Rz(±90°)--> T_world_hand
    --DiffIK--> q_target --interpolated ctrl--> position servos --> physics
```

Pick sequence: pre-grasp (10 cm retreat along approach) → approach (+1.2 cm
extra engagement) → close → lift 20 cm → verdict (object raised > 8 cm with
fingers not fully closed).

The ±90° z-rotation maps CGN's grasp frame (+X = closing line) onto the
Menagerie hand frame (fingers slide along ±Y); both signs are physically
identical, IK picks the better-conditioned one.

## Execution findings (validated on GTX 1650, 2026-06-10)

- Verified pick: seed 5, attempt 2 — object raised 0.177 m, IK errors < 4 mm.
- The **top-down calibrated camera is hard mode for Contact-GraspNet**: it
  sees only top faces, so proposals are sparse and low-scored (0.14–0.17 vs
  0.2–0.3 from a ~45° angled view) and several scenes yield no executable
  grasp. The angled-view runs (`--calibration none`) produce 3–10x more and
  better grasps. **Consider this for the real lab mount** — if the camera can
  be angled ~30–60° off vertical, CGN quality improves substantially.
- Execution ranking adds a downward-approach bonus (+0.25·(−approach_z)):
  near-horizontal approaches tend to bulldoze objects during the approach
  move in top-down setups.
- Debug helpers: `debug_execute.py` (side-view GIF of one grasp),
  `debug_geometry.py` (fingertip-vs-object numbers at the close moment).

## Known limitations

- Objects with no feasible CGN grasp (tiny spheres) yield 0 proposals — expected.
- Feasibility check is a corner-sample approximation of the gripper, not exact
  collision geometry; it only filters table hits + underhand approaches.
- Simulated depth is noise-free. For sim-to-real studies add Gaussian +
  pixel-dropout noise to `CameraModule.render_depth()` output.
