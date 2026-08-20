# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

ContactPilot picks up **unseen tabletop objects** (no CAD models, no markers)
with a Franka Panda arm, using [Contact-GraspNet](https://arxiv.org/abs/2103.14127)
(PyTorch port) to predict 6-DoF grasps from RGB-D. There are two independently
runnable pieces:

1. `contact_graspnet_pytorch/` — the vendored/patched CGN network + checkpoint
   + test scenes. Runs standalone against saved `.npy`/`.npz` frames (real
   camera captures or synthetic scenes) — no robot or sim required.
2. `mujoco_grasp_sim/` — a MuJoCo tabletop simulation (Panda + eye-to-hand
   RGB-D camera + random box objects) that drives the CGN pipeline end to end:
   scene → perception → grasp prediction → feasibility filter → ranked
   diff-IK execution → pick-and-place-in-bin.

`mujoco_menagerie/franka_emika_panda/` is a sparse clone providing only the
Panda robot model, consumed by `mujoco_grasp_sim`.

Read `ROADMAP.md` for project history/current status and *why* things are
built the way they are (it documents A/B experiment results, not just a todo
list) — check it before assuming a design choice is arbitrary.

## Environment

Everything runs in one conda env, **`cgn_torch`** (Python 3.10, PyTorch
2.12.0+cu126, **numpy must stay < 2**):

```powershell
conda activate cgn_torch
```

Commands in this file assume that env is active. Full from-scratch recreation
steps (including the CUDA wheel index caveat for RTX 5090/Blackwell, which
needs cu128 instead of cu126) are in `README.md` under "Environment".

## Common commands

### CGN inference on saved/test frames (`contact_graspnet_pytorch/`)

```powershell
cd contact_graspnet_pytorch

# Headless (no blocking GUI windows) — prints grasp counts/scores/timing/VRAM,
# writes results/predictions_<scene>.npz
python test_inference_headless.py --np_path=test_data/7.npy

# Visualize a saved result (Open3D window; close it to exit)
python contact_graspnet_pytorch\visualize_saved_scene.py --results_path=results/predictions_7.npz

# Original upstream interactive inference (GUI, blocking, one window per scene)
python contact_graspnet_pytorch\inference.py --np_path=test_data/7.npy

# Training (lab PC only, needs ACRONYM dataset — see docs/acronym_setup.md, >=24GB VRAM)
python contact_graspnet_pytorch\train.py --data_path acronym/
```

Common flags: `--arg_configs KEY:VALUE` overrides any `config.yaml` entry
(e.g. `--arg_configs DATA.raw_num_points:8192` to fix CUDA OOM on 4GB GPUs,
or `TEST.first_thres:0.19 TEST.second_thres:0.19` to tune grasp count).
`--forward_passes N` trades VRAM/time for more grasp proposals.

### MuJoCo sim pipeline (`mujoco_grasp_sim/`)

```powershell
cd mujoco_grasp_sim

python run_sim_grasp_test.py --seed 5 --execute --top-k 5     # single grasp attempt, reproducible seed
python run_sim_grasp_test.py --pick-all                       # pick+place every object into the bin
python run_sim_grasp_test.py --camera fused                    # fuse two calibrated cameras (best perception)
python run_sim_grasp_test.py --pick-object 6 --grasp-index 0   # target one object / one specific candidate grasp
python run_sim_grasp_test.py --no-vis                          # headless

# Batch evaluation — the ONLY basis for judging a reliability change, since
# CGN inference is stochastic (never trust a single run):
python benchmark.py --seeds 0-4 --mode pick-all --camera lookat --tag baseline
python analyze_failures.py output\bench_baseline    # classifies failures into taxonomy.json
```

Every run writes `output/<run>/`: `metrics.json`, `execution.gif`,
`observation.png`, `predictions_sim.npz`.

There is no automated test suite (`test_inference_headless.py` and
`run_sim_grasp_test.py` runs are the correctness checks — compare
`metrics.json`/console stats against the baselines recorded in `README.md`
and `ROADMAP.md`).

## Architecture — `mujoco_grasp_sim`

```
MuJoCo (Menagerie Panda + table + random box objects)
                    │  physics settle
                    ▼
    CameraModule (eye-to-hand RGB/Depth[m]/Segmap/K/T_world_cam)
                    │
                    ▼
    depth_to_pointcloud (OpenCV camera frame)
                    │
                    ▼
    ContactGraspNetPredictor  →  {seg_id: (N,4,4) T_cam_grasp}, scores
                    │              (runs as a subprocess per pick-all round —
                    │               keeps torch's footprint off the sim
                    │               process so 8GB machines survive)
                    ▼
    GraspFeasibilityChecker (table-collision + underhand filter)
                    │
                    ▼
    ranked execution (diff-IK) → pick → place-in-bin → re-observe loop
                    │
                    ▼
    metrics.json / execution.gif / predictions_sim.npz
```

Key modules in `sim_grasp/`:
- `frames.py` — **read this first**; the canonical doc for every coordinate
  frame in the project (world, robot base, MuJoCo-camera vs OpenCV-camera,
  grasp frame). Chain used everywhere: `T_base_grasp = inv(T_world_base) @ T_world_cam @ T_cam_grasp`.
- `scene_generator.py` — `SceneGenerator` + `SceneConfig` (object count/type,
  spawn region, table/camera placement — see "Knobs" in `mujoco_grasp_sim/README.md`).
- `camera.py` / `pointcloud.py` — rendering and depth→cloud conversion.
- `grasp_predictor.py` — `GraspPredictor` ABC + `ContactGraspNetPredictor`.
  **This is the extension point for swapping grasp backends** (AnyGrasp,
  GSNet, GIGA, …): implement one `predict()` method, everything downstream
  (scene, camera, feasibility, metrics, viz) is backend-agnostic.
- `feasibility.py` — table-collision + underhand-approach filter.
- `fusion.py` — multi-camera point-cloud fusion (world frame, voxel dedup) for `--camera fused`.
- `executor.py` — differential IK (damped least squares, multi-seed restarts)
  + joint-space ctrl interpolation + the pick/place state machine; also
  `GIF_DOWNSAMPLE`/`GIF_MAX_FRAMES` memory caps for recording.
- `visualizer.py` — Open3D + 2D observation dumps.

The Menagerie Panda uses **joint-space position servos** (`ctrl[0..6]` =
joint angles, `ctrl[7]` = gripper tendon), so Cartesian grasp poses always go
through the diff-IK step in `executor.py`, never direct Cartesian control.

Reliability work (see `ROADMAP.md` P1) is evaluated exclusively via
`benchmark.py` success-rate tables across many seeds — a single successful
run proves nothing because CGN inference is stochastic.

## Contact-GraspNet dependency

`contact_graspnet_pytorch/` is a git submodule pointing at
[`VivekSai07/contact_graspnet_pytorch`](https://github.com/VivekSai07/contact_graspnet_pytorch),
pinned to a tag (not a floating branch). All local patches (checkpoint
loading fix for PyTorch >= 2.6, `visualize_saved_scene.py` `--results_path`
flag, a package-relative import fix, the headless inference driver, plus
constant-VRAM `forward_passes` batching and a `torch.cross` compat fix) live
as normal commits on the fork — check its history rather than looking for a
local diff.

The checkpoint (`model.pt`) and the 14 test scenes are not committed to
either repo; they're hosted on Hugging Face Hub and fetched by
`contact_graspnet_pytorch/scripts/download_assets.py`. After
`git submodule update --init`, run that script once to populate
`checkpoints/` and `test_data/` (it's idempotent).

## Known constraints worth knowing before touching things

- **numpy must stay < 2** across the whole stack.
- **RTX 5090 / Blackwell (sm_120) needs cu128 torch wheels** — cu126 has no
  sm_120 kernels; the conda env doc has separate laptop/lab instructions.
- Depth must always be **float32 meters**, never uint16 millimeters (a
  RealSense D455 native output needs `/1000.0` conversion first).
- `mujoco.Renderer` with `enable_depth_rendering()` already returns
  linearized-to-meters perpendicular depth — do not apply any extra
  znear/zfar conversion on top (a classic bug from old mujoco-py examples).
- `--local_regions`/`--filter_grasps` (CGN) require a segmap; without one
  you get ungrouped, scene-wide grasps including the table.
