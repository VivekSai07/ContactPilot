# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read `AGENTS.md` too** — it defines the project's mandatory documentation-sync
workflow (when a bug/fix/new integration requires updating `ROADMAP.md`,
`docs/research/`, or `docs/superpowers/specs/`, and how) plus branch/commit
hygiene. It applies to every task here, for every agent, not just this one.

## What this repo is

ContactPilot picks up **unseen tabletop objects** (no CAD models, no markers)
with a Franka Panda arm, using [Contact-GraspNet](https://arxiv.org/abs/2103.14127)
(PyTorch port) — or, alternatively, [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)
(`--backend graspgen`) — to predict 6-DoF grasps from RGB-D. There are two
independently runnable pieces:

1. `contact_graspnet_pytorch/` — the vendored/patched CGN network + checkpoint
   + test scenes. Runs standalone against saved `.npy`/`.npz` frames (real
   camera captures or synthetic scenes) — no robot or sim required.
2. `mujoco_grasp_sim/` — a MuJoCo tabletop simulation (Panda + eye-to-hand
   RGB-D camera + random box objects) that drives the CGN pipeline end to end:
   scene → perception → grasp prediction → feasibility filter → ranked
   diff-IK execution → pick-and-place-in-bin.

`mujoco_menagerie/franka_emika_panda/` is a sparse-checked-out git submodule
(pinned commit, `franka_emika_panda/` only) providing the Panda robot model,
consumed by `mujoco_grasp_sim`.

Read `ROADMAP.md` for project history/current status and *why* things are
built the way they are (it documents A/B experiment results, not just a todo
list) — check it before assuming a design choice is arbitrary.

**Dexterous hands are shelved, not open ground.** A Shadow Hand E3M5
integration was fully built and live-validated on the unmerged
`shadow-hand-power-grasp` branch, then explicitly not pursued further
(see `ROADMAP.md` P9). Don't re-propose or re-attempt dexterous-hand
integration without the user raising it again — read that branch's
work first if they do.

## Environment

The sim and CGN run in the main conda env, **`cgn_torch`** (Python 3.10,
PyTorch 2.12.0+cu126, **numpy must stay < 2**):

```powershell
conda activate cgn_torch
```

Commands in this file assume that env is active. Full from-scratch recreation
steps (including the CUDA wheel index caveat for RTX 5090/Blackwell, which
needs cu128 instead of cu126) are in `README.md` under "Environment".

Optional features live in **separate conda envs**, invoked as subprocesses
via an interpreter path (see `mujoco_grasp_sim/README.md` for setup):

| Feature | Env | Configure with |
|---|---|---|
| `--backend graspgen` | `graspgen_torch` | `GRASPGEN_PYTHON` or `--graspgen-python` |
| `--prompt` / `--click` / `--box`, `interactive_pick.py` (SAM 3) | `sam3_torch` | `SAM3_PYTHON` or `--sam3-python` |
| `--instruction` (NL parsing via NVIDIA NIM) | — | `NVIDIA_API_KEY` in `.env` (see `.env.example`) |

These fail fast with a clear error when unset — they deliberately never fall
back to `sys.executable`, so don't "fix" that. On WSL2/headless machines,
rendering needs `export MUJOCO_GL=osmesa`.

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
python run_sim_grasp_test.py --backend graspgen --execute      # GraspGen instead of CGN
python run_sim_grasp_test.py --execute --prompt "the red box"  # SAM 3 text-prompted target selection
python run_sim_grasp_test.py --pick-all --instruction "blue cube first, on the left"  # NL pick order/placement
python run_sim_grasp_test.py --pick-all --verbose              # show worker-subprocess output (quiet by default)
python interactive_pick.py --seed 5 --backend graspgen        # click an object in a live window, SAM 3 + pick

# Batch evaluation — the ONLY basis for judging a reliability change, since
# CGN inference is stochastic (never trust a single run):
python benchmark.py --seeds 0-4 --mode pick-all --camera lookat --tag baseline
python analyze_failures.py output\bench_baseline    # classifies failures into taxonomy.json
```

Every run writes `output/<run>/`: `metrics.json`, `execution.gif`,
`observation.png`, `predictions_sim.npz`.

### Tests

There is no pytest suite/runner. `mujoco_grasp_sim/sim_grasp/test_*.py` are
standalone assert-based scripts for pure functions (no MuJoCo/GPU/model
needed); each passes silently or raises. Run one, or all, from
`mujoco_grasp_sim/` — `PYTHONPATH=.` is required because they import
`sim_grasp.*`:

```bash
cd mujoco_grasp_sim
PYTHONPATH=. python sim_grasp/test_reachability.py
for t in sim_grasp/test_*.py; do PYTHONPATH=. python "$t" || echo "FAIL $t"; done
```

New tests follow the same style (top-level asserts, no pytest). End-to-end
correctness is checked by real runs: `test_inference_headless.py` for CGN,
and a `run_sim_grasp_test.py` pick (look for `PICK SUCCESS` /
`execution.success: true` in `metrics.json`), compared against the baselines
in `README.md` and `ROADMAP.md`.

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
    [optional] SAM 3 target selection (--prompt/--click/--box)
                    │
                    ▼
    GraspPredictor (CGN or GraspGen)  →  {seg_id: (N,4,4) T_cam_grasp}, scores
                    │              (runs as a subprocess per pick-all round —
                    │               keeps torch's footprint off the sim
                    │               process so 8GB machines survive)
                    ▼
    GraspFeasibilityChecker (table-collision + underhand filter;
                    │        + neighbor-collision/reachability with --filter-neighbors)
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
- `grasp_predictor.py` — `GraspPredictor` ABC + `ContactGraspNetPredictor`;
  `graspgen_predictor.py` is the second implementation.
  **This is the extension point for swapping grasp backends** (AnyGrasp,
  GSNet, GIGA, …): implement one `predict()` method, everything downstream
  (scene, camera, feasibility, metrics, viz) is backend-agnostic.
- **Worker-subprocess pattern:** heavy models never load in the sim process.
  `cgn_worker.py` / `graspgen_worker.py` / `sam3_worker.py` are the child-side
  scripts; `subprocess_utils.py` is the shared runner (captures output,
  surfaces it only on failure unless `--verbose` / `SIM_GRASP_VERBOSE`).
  A new model-backed feature should follow this pattern.
- `perception.py` — pre-prediction depth cleanup (`--clean-depth`) and
  post-prediction grasp re-centering (`--recenter`).
- `feasibility.py` — table-collision + underhand-approach filter;
  `reachability.py` + `workspace_occupancy.py` add the `--filter-neighbors`
  pre-filter (merged, but needs tuning before being on by default).
- `instruction_parser.py` → `spatial_relation_resolver.py` →
  `placement_planner.py` — the P8 NL-instruction layer: parse an instruction
  into ordered steps, turn each step's spatial relation into a biased
  vision-only bin-placement plan.
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

`GraspGen/` is a submodule pointing at upstream `NVlabs/GraspGen` (unpatched);
it's only used from the `graspgen_torch` env via `graspgen_worker.py`. Its own
`tests/`/`pyproject.toml` belong to upstream, not this project.

`mujoco_menagerie/` is also a git submodule, pointing at
[`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
upstream (pinned commit, sparse-checked-out to `franka_emika_panda/` only —
the full menagerie repo covers dozens of unrelated robots). Nothing in this
repo patches its source on disk: `SceneGenerator` patches `panda.xml`
in-memory and writes the result to `mujoco_grasp_sim/assets/`, never back
into the submodule.

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
