# Design: GraspGen as a second grasp-prediction backend

Date: 2026-08-12

## Problem

`mujoco_grasp_sim` currently has one grasp-prediction backend, Contact-GraspNet
(`ContactGraspNetPredictor`), which sometimes misses grasps (per user
observation and the failure taxonomy already documented in `ROADMAP.md`:
`closed_on_air`, `missed_bin`, etc.). The user wants to evaluate
[NVlabs/GraspGen](https://github.com/NVlabs/GraspGen), a diffusion-based
6-DoF grasp generator that claims SOTA performance on the FetchBench
benchmark (+17% over prior methods) and a discriminator specifically trained
to score/filter generated grasps — a plausible fit for the "generates a
plausible-looking but physically bad grasp" failure mode.

This is deliberately scoped as sub-project 1 of a larger "idea 2"
(promptable/open-vocabulary grasping). Sub-project 2 — promptable object
selection via SAM/SAM-3/Grounded-SAM-2 — is out of scope here and will be
designed separately once it's known which backend(s) it needs to serve.

## Goals

- Add GraspGen as a second concrete `GraspPredictor` implementation,
  selectable via `--backend {cgn,graspgen}`, with zero changes to anything
  downstream of grasp prediction (`feasibility.py`, ranking, `executor.py`,
  `--pick-object`, `--pick-all`).
- Get a real, ContactPilot-specific answer — via `benchmark.py`'s existing
  multi-seed harness — to "does GraspGen actually miss fewer grasps than CGN
  on our box-object scenes," not a README claim.
- Keep the working `cgn_torch` environment completely unmodified/unrisked.

## Non-goals

- Promptable object selection (sub-project 2, separate spec).
- Training or fine-tuning GraspGen — inference only, using NVIDIA's
  released Franka-Panda checkpoint.
- Supporting GraspGen's other grippers (Robotiq-2f-140, suction) — only the
  Franka-Panda config, matching ContactPilot's actual robot.

## Context / research findings (2026-08-12)

- **License**: NVIDIA Research license (commercial use requires contacting
  NVIDIA Research Licensing). Confirmed acceptable — this project's use is
  personal/research, not commercial.
- **Hardware fit**: GraspGen's README explicitly documents a known issue —
  its newer PTv3 backbone doesn't yet run on CUDA 12.8/Blackwell GPUs — and
  recommends the **PointNet++ backbone** as the working path on that
  hardware. This matches the user's RTX PRO 2000 Blackwell GPU. No inference
  VRAM figure is published; the README's "21x less memory" and "realtime
  20Hz (pre-TensorRT)" claims are relative to unspecified prior methods, not
  a hard number — genuinely unconfirmed until tested (see Risks).
- **Grasp-frame convention**: GraspGen's `GRIPPER_DESCRIPTION.md` states its
  convention is "approach direction is the positive Z-axis, gripper finger
  closing direction is along the X-axis" — the **same** convention
  `sim_grasp/frames.py` documents for Contact-GraspNet's Panda grasp frame.
  This means the existing `T_base_grasp = inv(T_world_base) @ T_world_cam @
  T_cam_grasp` chain and the executor's `±90°` Rz mapping to MuJoCo's hand
  frame should apply to GraspGen's output unchanged. The one number that
  still needs verifying is GraspGen's Franka-Panda `depth` value
  (base-link-to-TCP offset in its YAML gripper config) against this
  project's `PANDA_TCP_OFFSET = 0.1034`.
- **Environment conflict**: GraspGen's install docs show
  `torch==2.1.0+cu121` as an optional install step; `cgn_torch` already has
  `torch==2.11.0+cu128`. Whether GraspGen's custom PointNet++ CUDA extension
  (`pointnet2_ops`, built via `install_pointnet.sh`) would compile/run
  correctly against 2.11.0+cu128 is untested and risky to gamble on inside
  the already-validated `cgn_torch` env. Decision: separate `graspgen_torch`
  conda env, invoked as a subprocess — see Architecture.
- **Checkpoints**: already hosted on Hugging Face Hub
  (`adithyamurali/GraspGenModels`), same hosting pattern this project
  already uses (idea 1). The Franka-Panda gripper config
  (`graspgen_franka_panda.yml`) is one of the three released configs.
- GraspGen's own docs note it needs external instance segmentation (they
  suggest SAM2) to run on scene-level clutter — consistent with this
  project's existing segmap-driven `local_regions`/`filter_grasps` pattern,
  and with sub-project 2 (promptable selection) being a natural future
  layer on top of either backend.

## Architecture

```
run_sim_grasp_test.py --backend {cgn|graspgen}
                    │
                    ▼
        GraspPredictor (existing ABC, sim_grasp/grasp_predictor.py, unchanged)
           ├── ContactGraspNetPredictor  (existing; in-process, or cgn_worker.py
           │                              subprocess for --pick-all memory isolation)
           └── GraspGenPredictor         (NEW, sim_grasp/graspgen_predictor.py)
                    │  ALWAYS subprocess — GraspGen needs its own conda env
                    │  (torch version conflict with cgn_torch), not merely
                    │  for the memory-isolation reason CGN's subprocess uses
                    ▼
        sim_grasp/graspgen_worker.py  (NEW; invoked with the graspgen_torch
                    │                  env's python.exe, resolved from the
                    │                  GRASPGEN_PYTHON env var or
                    │                  --graspgen-python CLI override —
                    │                  never sys.executable)
                    ▼
        GraspGen's own inference code (separate env, separate install,
                    │                  Franka-Panda gripper config)
                    ▼
        same {grasps_<sid>, scores_<sid>, contacts_<sid>, openings_<sid>}
        npz format cgn_worker.py already produces — the existing
        _subprocess_predict() result-parsing logic in run_sim_grasp_test.py
        is reused unmodified, just pointed at a different worker script
        and a different interpreter path.
```

`GraspGenPredictor` conforms to the exact same `GraspPredictor` /
`GraspPrediction` interface as `ContactGraspNetPredictor`, so
`feasibility.py`, ranking, `executor.py`, `--pick-object`, `--pick-all`, and
`benchmark.py` all work against it with zero changes — only the predictor
construction and the subprocess dispatch differ per backend.

## Files

- **Create:** `mujoco_grasp_sim/sim_grasp/graspgen_predictor.py` —
  `GraspGenPredictor(GraspPredictor)`. `predict()`/`predict_clouds()` always
  shell out to `graspgen_worker.py` via the resolved `GRASPGEN_PYTHON`
  interpreter (no in-process code path, unlike `ContactGraspNetPredictor`).
- **Create:** `mujoco_grasp_sim/sim_grasp/graspgen_worker.py` — CLI mirrors
  `cgn_worker.py` exactly (`obs.npz out.npz [--forward-passes N] ...`), runs
  under the `graspgen_torch` interpreter, writes the same npz key format.
- **Create:** a checkpoint-fetch script for GraspGen's Franka-Panda
  checkpoint from `adithyamurali/GraspGenModels` on Hugging Face Hub (same
  `huggingface_hub` pattern as `contact_graspnet_pytorch/scripts/download_assets.py`).
- **Modify:** `run_sim_grasp_test.py` — add `--backend {cgn,graspgen}`
  (default `cgn`) and `--graspgen-python PATH` (optional override for
  `GRASPGEN_PYTHON`); thread the choice into predictor construction and into
  the existing `_subprocess_predict`-style dispatch.
- **Modify:** `benchmark.py` — thread `--backend` through to each per-seed
  `run_sim_grasp_test.py` invocation, so `--backend graspgen` runs are
  directly comparable to the recorded CGN baselines (same seeds, same scene
  config).
- **Modify:** `mujoco_grasp_sim/README.md` — document the second backend,
  `graspgen_torch` env setup (mirroring the "Recreate the env from scratch"
  section's style), `GRASPGEN_PYTHON`, and the benchmark-comparison workflow.

## Error handling

- `--backend graspgen` requested but `GRASPGEN_PYTHON` unset and
  `--graspgen-python` not passed: fail fast with a clear message pointing at
  the README setup section — never silently fall back to `sys.executable`
  (which would try to run GraspGen's code under `cgn_torch` and fail
  confusingly deep inside an import error).
- `GRASPGEN_PYTHON` set but the path doesn't exist, or the GraspGen
  checkpoint isn't downloaded yet: fail fast with the specific missing path,
  mirroring `ContactGraspNetPredictor`'s existing
  `FileNotFoundError(f'... checkpoint dir not found: {ckpt_dir}')` pattern.
- `graspgen_worker.py` subprocess failure (non-zero exit / missing output
  file): same handling as `_subprocess_predict()` already does for the CGN
  worker (`raise RuntimeError(f'... worker failed (exit code {r.returncode})')`).

## Testing / verification plan

1. **Standalone GraspGen smoke test**: `graspgen_torch` env installs
   cleanly, GraspGen's own `demo_object_pc.py` sample script runs on this
   GPU. Isolates "does GraspGen even work here" from ContactPilot
   integration — do this before writing any integration code.
2. `graspgen_worker.py` smoke test: feed it a saved scene observation
   (reuse an existing captured `.npz`), confirm it produces a valid
   `{grasps_<sid>, scores_<sid>, ...}` npz output.
3. Single sim run: `run_sim_grasp_test.py --seed 5 --execute --backend
   graspgen` — same seed as CGN's originally validated pick, for a direct
   before/after comparison on one known scene.
4. **The decisive test**: `benchmark.py --seeds 0-4 --mode pick-all
   --backend graspgen --tag graspgen_baseline`, compared against the
   recorded CGN baseline (14/15 binned, 93%, box-only/3-objects/fused
   camera — see `ROADMAP.md` P1). Run `analyze_failures.py` on both to see
   whether the failure taxonomy shifts (fewer `closed_on_air`, etc.) — this
   is the actual answer to "does it miss less than CGN."
5. Verify GraspGen's Franka-Panda `depth` (base-to-TCP offset) against
   `PANDA_TCP_OFFSET = 0.1034` before trusting any executed grasp pose.

## Risks

- `pointnet2_ops` (GraspGen's custom CUDA extension) compiling and running
  correctly on a Blackwell GPU is **unconfirmed** — the README only
  confirms the PTv3-backbone caveat, not full end-to-end Blackwell
  validation with the PointNet++ backbone. This is the first thing Step 1
  of the verification plan checks, before any ContactPilot-side code is
  written.
- Two CUDA/conda environments to maintain and keep working going forward.
- Frame-convention *axes* match CGN's (confirmed above from GraspGen's own
  docs), but the exact TCP-offset number is still unverified — a wrong
  value would silently produce grasps offset along the approach axis.
- No published inference VRAM figure for GraspGen — the "21x less memory"
  README claim is relative to an unspecified baseline, not a number we can
  plan around. Step 1 of the verification plan (standalone smoke test)
  surfaces the real number for this GPU before more work is invested.
