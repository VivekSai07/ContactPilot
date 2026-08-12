# GraspGen Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NVlabs/GraspGen as a second, selectable grasp-prediction backend in `mujoco_grasp_sim`, isolated in its own conda env, and get a real success-rate comparison against the recorded Contact-GraspNet baseline via the existing `benchmark.py` harness.

**Architecture:** `GraspGenPredictor` implements the existing `GraspPredictor` ABC exactly like `ContactGraspNetPredictor` does, but always delegates to a subprocess (`graspgen_worker.py`) run under a separate `graspgen_torch` conda interpreter, because GraspGen's dependencies conflict with `cgn_torch`. Everything downstream of prediction (feasibility, ranking, execution, benchmarking) is backend-agnostic and needs zero changes.

**Tech Stack:** Python 3.10, PyTorch (cu128), NVlabs/GraspGen (external clone, not vendored), Hugging Face Hub (checkpoint hosting), conda.

## Global Constraints

- Gripper: **Franka-Panda only** (`graspgen_franka_panda.yml`) — matches ContactPilot's actual robot. Do not fetch/wire the Robotiq or suction checkpoints.
- Inference only — no training/fine-tuning of GraspGen.
- GraspGen's own checkpoint files must **never be committed to git** (the `_dis.pth` is ~166 MB, the `_gen.pth` is ~907 MB — confirmed via the HF Hub API). Destination directory must be gitignored.
- `graspgen_worker.py` is invoked with the `graspgen_torch` interpreter, resolved from a `GRASPGEN_PYTHON` env var or `--graspgen-python` CLI override — **never** `sys.executable` (that would try to run GraspGen under `cgn_torch` and fail confusingly).
- No frame-conversion code is needed: GraspGen's Franka-Panda grasp convention (origin at gripper base, +Z approach, +X finger closing line) is identical to Contact-GraspNet's (verified from GraspGen's own `GRIPPER_DESCRIPTION.md` and its `config/grippers/franka_panda.yaml`, which has `depth: 0.10527314` vs this project's `PANDA_TCP_OFFSET = 0.1034` — a ~1.9mm difference that does NOT need reconciling in code, because that constant is only ever used in `debug_execute.py`'s diagnostic print, never in `executor.py`'s actual execution path — confirmed by grep). Do not add any pose-transform code for this.
- GraspGen's own collision filtering (`filter_colliding_grasps`, needs a gripper collision mesh) is **out of scope** — `feasibility.py`'s existing table-collision/underhand filter already runs downstream on `GraspGenPredictor`'s output exactly as it does for CGN's. Do not wire GraspGen's own collision filter.
- `benchmark.py`'s recorded CGN baseline to compare against: **14/15 binned (93%), box-only/3-objects, `--camera fused`** (see `ROADMAP.md` P1, 2026-06-14). The decisive benchmark run in Task 6 must use `--camera fused` to be apples-to-apples.

---

### Task 1: Checkpoint download script

**Files:**
- Create: `mujoco_grasp_sim/scripts/download_graspgen_checkpoint.py`
- Modify: `.gitignore` (repo root)

**Interfaces:**
- Produces: `mujoco_grasp_sim/graspgen_checkpoints/graspgen_franka_panda.yml`,
  `graspgen_franka_panda_gen.pth`, `graspgen_franka_panda_dis.pth` on disk.
  Task 2 (standalone smoke test) and Task 4 (`GraspGenPredictor`) both consume
  this directory path.

- [ ] **Step 1: Write the checkpoint download script**

Create `mujoco_grasp_sim/scripts/download_graspgen_checkpoint.py`:

```python
#!/usr/bin/env python3
"""Download the GraspGen Franka-Panda checkpoint from Hugging Face Hub.

Run this once (idempotent) to populate graspgen_checkpoints/, which is not
committed to git (the generator checkpoint alone is ~900 MB).
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "adithyamurali/GraspGenModels"
DEST_DIR = Path(__file__).resolve().parent.parent / "graspgen_checkpoints"

FILES = [
    "checkpoints/graspgen_franka_panda.yml",
    "checkpoints/graspgen_franka_panda_gen.pth",
    "checkpoints/graspgen_franka_panda_dis.pth",
]


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for repo_path in FILES:
        dest = DEST_DIR / Path(repo_path).name
        if dest.exists():
            print(f"Already present: {dest}")
            continue
        downloaded = hf_hub_download(repo_id=REPO_ID, filename=repo_path)
        dest.write_bytes(Path(downloaded).read_bytes())
        print(f"Downloaded {repo_path} -> {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the checkpoint directory to `.gitignore`**

Current `.gitignore` already has a "Generated benchmark outputs" section for
`mujoco_grasp_sim/output/`. Append immediately after it:

```
# GraspGen checkpoint (fetched via mujoco_grasp_sim/scripts/download_graspgen_checkpoint.py — ~1GB, never commit)
mujoco_grasp_sim/graspgen_checkpoints/
```

- [ ] **Step 3: Run it and verify**

`huggingface_hub` is already available in `cgn_torch` from idea 1 — any
Python with it installed works, since this script has no torch/GraspGen
dependency:

```powershell
& "$env:LOCALAPPDATA\miniconda3\envs\cgn_torch\python.exe" mujoco_grasp_sim\scripts\download_graspgen_checkpoint.py
```

Expected output (three lines, one per file):
```
Downloaded checkpoints/graspgen_franka_panda.yml -> ...\graspgen_franka_panda.yml (4868 bytes)
Downloaded checkpoints/graspgen_franka_panda_gen.pth -> ...\graspgen_franka_panda_gen.pth (907408223 bytes)
Downloaded checkpoints/graspgen_franka_panda_dis.pth -> ...\graspgen_franka_panda_dis.pth (165853892 bytes)
```

Verify sizes on disk:
```powershell
Get-ChildItem mujoco_grasp_sim\graspgen_checkpoints | Select-Object Name, Length
```
Expected: exactly the three byte counts above.

Re-run the script once more to confirm idempotency (expected: three
"Already present" lines, no re-download).

- [ ] **Step 4: Confirm the directory is ignored**

```bash
git status --short mujoco_grasp_sim/graspgen_checkpoints/
```
Expected: no output (ignored, not untracked).

- [ ] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/scripts/download_graspgen_checkpoint.py .gitignore
git commit -m "Add GraspGen checkpoint download script"
```

---

### Task 2: `graspgen_torch` environment + standalone smoke test

**This task is the single biggest risk in the whole plan** (per the design
doc: GraspGen's custom PointNet++ CUDA extension on a Blackwell GPU is
unconfirmed). Do this before writing any ContactPilot integration code, and
if it fails, escalate rather than guessing around it — don't attempt any
workaround the design doc didn't already specify (e.g. don't silently switch
to PTv3 or a different torch version without reporting the failure first).

**Files:** none in the ContactPilot repo — this task clones GraspGen and
creates a conda env, both external to this repo.

**Interfaces:**
- Consumes: Task 1's checkpoint at `mujoco_grasp_sim/graspgen_checkpoints/`.
- Produces: a working `graspgen_torch` conda env whose `python.exe` path
  becomes the `GRASPGEN_PYTHON` value used by every later task.

- [ ] **Step 1: Clone GraspGen outside the ContactPilot tree**

GraspGen is NOT a submodule of ContactPilot (unlike `contact_graspnet_pytorch`)
— it's NVIDIA-licensed, not something to vendor inside this repo, and nothing
here patches its source. Clone it as a sibling directory:

```powershell
cd D:\Projects
git clone https://github.com/NVlabs/GraspGen.git
```

- [ ] **Step 2: Create the `graspgen_torch` conda env**

```powershell
$conda = "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe"
& $conda create -n graspgen_torch python=3.10 -y
```

Expected: succeeds the same way `cgn_torch`'s creation did (ToS for the
default channels was already accepted this session).

- [ ] **Step 3: Install PyTorch cu128 (Blackwell-compatible, matching `cgn_torch`)**

GraspGen's own README shows `torch==2.1.0+cu121` as an *optional* install
step, but cu121 wheels have no Blackwell (sm_120) kernels — this machine
needs cu128, exactly like `cgn_torch`. Install the same way:

```powershell
$py = "$env:LOCALAPPDATA\miniconda3\envs\graspgen_torch\python.exe"
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: prints a `+cu128` version and `True` (same pattern verified for
`cgn_torch` earlier this session).

- [ ] **Step 4: Install GraspGen**

```powershell
cd D:\Projects\GraspGen
& $py -m pip install -e .
```

Then the PointNet++ CUDA extension — **this is the step that may fail on
Blackwell; if it does, report BLOCKED with the full error rather than
attempting a workaround**:

```powershell
$env:CC = "cl"; $env:CXX = "cl"   # Windows: use MSVC, not g++ (install_pointnet.sh assumes Linux)
cd D:\Projects\GraspGen\pointnet2_ops
& $py -m pip install --no-build-isolation .
cd D:\Projects\GraspGen
```

If the Windows build fails because `install_pointnet.sh`/`pointnet2_ops`
assumes a Linux toolchain (no MSVC path documented in GraspGen's README),
that's a legitimate BLOCKED outcome — report it with the exact error; do not
attempt to port the build script yourself without asking.

- [ ] **Step 5: Run GraspGen's own smoke test**

Fetch one sample file for the demo (small JSON, not part of Task 1's
checkpoint fetch since it's demo-only, not needed by `graspgen_worker.py`):

```powershell
& $py -c "from huggingface_hub import hf_hub_download; p = hf_hub_download(repo_id='adithyamurali/GraspGenModels', filename='sample_data/real_object_pc/1740787710_319213.json'); import shutil; shutil.copy(p, 'sample_data_1.json')"
```

Then run the demo (headless — no `--visualize` flag, so it doesn't block on
a viser window):

```powershell
mkdir sample_data_dir
Move-Item sample_data_1.json sample_data_dir\1740787710_319213.json
& $py scripts\demo_object_pc.py --sample_data_dir sample_data_dir --gripper_config D:\Projects\ContactPilot\mujoco_grasp_sim\graspgen_checkpoints\graspgen_franka_panda.yml
```

Expected: the script loads the checkpoint, runs inference, and prints a
nonzero grasp count with a score range (matching the style of
`demo_collision_free_grasps.py`'s "Inferred N grasps, with scores ranging
from X - Y" output). No CUDA errors. This confirms GraspGen's PointNet++
path works end-to-end on this GPU — the load-bearing risk this task exists
to retire.

- [ ] **Step 6: Record the interpreter path for later tasks**

```powershell
echo $env:LOCALAPPDATA\miniconda3\envs\graspgen_torch\python.exe
```

This exact path is `GRASPGEN_PYTHON` for every subsequent task and for the
README documentation in Task 7.

No commit — this task touches nothing inside the ContactPilot repo.

---

### Task 3: `graspgen_worker.py` subprocess entry point

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/graspgen_worker.py`

**Interfaces:**
- Consumes: `GraspGenSampler`/`load_grasp_cfg` from GraspGen's `grasp_gen`
  package (importable under the `graspgen_torch` interpreter after Task 2's
  editable install); Task 1's checkpoint directory.
- Produces: same npz key format as `cgn_worker.py`
  (`grasps_<sid>`, `scores_<sid>`, `contacts_<sid>`, `openings_<sid>`) — Task
  4's `GraspGenPredictor` and Task 5's dispatch code both rely on this exact
  format, unchanged from what `_subprocess_predict()` in
  `run_sim_grasp_test.py` already parses.
- CLI: `python graspgen_worker.py obs.npz out.npz --gripper-config PATH
  [--num-grasps N] [--grasp-threshold T]`. `obs.npz` supports both modes
  `cgn_worker.py` supports: depth mode (`depth`, `K`, optional `rgb`,
  `segmap`) and cloud mode (`pc_full`, `pcseg_<sid>` per object) — cloud
  mode is needed because the decisive benchmark comparison (Task 6) uses
  `--camera fused`, which only produces cloud-mode payloads.

- [ ] **Step 1: Write `graspgen_worker.py`**

Create `mujoco_grasp_sim/sim_grasp/graspgen_worker.py`:

```python
"""Subprocess GraspGen worker — runs under the graspgen_torch interpreter.

Mirrors cgn_worker.py's CLI and output format exactly, so
run_sim_grasp_test.py's existing _subprocess_predict() result-parsing code
works unmodified regardless of which backend produced the npz.

Usage:
    python sim_grasp/graspgen_worker.py obs.npz out.npz \
        --gripper-config path/to/graspgen_franka_panda.yml \
        [--num-grasps 200] [--grasp-threshold 0.8]

obs.npz keys, depth mode:  depth (H,W) float32 m, K (3,3), rgb (H,W,3) uint8,
                           segmap (H,W)
            cloud mode:    pc_full (N,3) float32 in a camera frame, plus
                           pcseg_<sid> (Ni,3) per object (P2 fusion path)
out.npz keys: grasps_<sid>, scores_<sid>, contacts_<sid>, openings_<sid>.

Contact points and gripper openings are not produced by GraspGen (unlike
CGN) — contacts_<sid> is written as an empty (0,3) array and openings_<sid>
is omitted, matching GraspPrediction's documented defaults.
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def object_point_clouds(obs) -> dict:
    """Return {seg_id: (Ni,3) float32 object point cloud in camera frame},
    from either depth+segmap or pre-extracted cloud-mode payloads."""
    if 'pc_full' in obs:
        return {float(k.split('_', 1)[1]): np.asarray(obs[k], dtype=np.float32)
                for k in obs.files if k.startswith('pcseg_')}

    from grasp_gen.utils.point_cloud_utils import depth_and_segmentation_to_point_clouds

    depth, K, segmap = obs['depth'], obs['K'], obs['segmap']
    rgb = obs['rgb'] if 'rgb' in obs else None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    out = {}
    for sid in sorted(s for s in np.unique(segmap) if s > 0):
        _scene_pc, object_pc, _scene_c, _obj_c = depth_and_segmentation_to_point_clouds(
            depth_image=depth, segmentation_mask=segmap,
            fx=fx, fy=fy, cx=cx, cy=cy, rgb_image=rgb,
            target_object_id=int(sid), remove_object_from_scene=True)
        out[float(sid)] = np.asarray(object_pc, dtype=np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('obs_npz')
    ap.add_argument('out_npz')
    ap.add_argument('--gripper-config', required=True)
    ap.add_argument('--num-grasps', type=int, default=200)
    ap.add_argument('--grasp-threshold', type=float, default=0.8)
    args = ap.parse_args()

    gripper_config = Path(args.gripper_config)
    if not gripper_config.is_file():
        raise FileNotFoundError(f'GraspGen gripper config not found: {gripper_config}')

    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
    from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal

    obs = np.load(args.obs_npz)
    per_object_pc = object_point_clouds(obs)

    grasp_cfg = load_grasp_cfg(str(gripper_config))
    sampler = GraspGenSampler(grasp_cfg)

    out = {}
    total = 0
    for sid, pc in per_object_pc.items():
        if pc.shape[0] < 30:
            continue
        pc_t = torch.from_numpy(pc)
        pc_filtered, _removed = point_cloud_outlier_removal(pc_t)
        pc_filtered = pc_filtered.numpy()
        grasps_t, conf_t = GraspGenSampler.run_inference(
            pc_filtered, sampler, grasp_threshold=args.grasp_threshold,
            num_grasps=args.num_grasps, topk_num_grasps=-1)
        grasps = grasps_t.cpu().numpy().astype(np.float32) if len(grasps_t) else \
            np.zeros((0, 4, 4), dtype=np.float32)
        scores = conf_t.cpu().numpy().astype(np.float32) if len(conf_t) else \
            np.zeros((0,), dtype=np.float32)
        if len(grasps):
            grasps[:, 3, 3] = 1.0
        key = f'{sid:g}'
        out[f'grasps_{key}'] = grasps
        out[f'scores_{key}'] = scores
        out[f'contacts_{key}'] = np.zeros((0, 3), dtype=np.float32)
        total += len(grasps)

    np.savez(args.out_npz, **out)
    print(f'[graspgen-worker] {total} grasps -> {args.out_npz}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Capture a saved scene observation to test against**

Reuse an existing sim capture rather than fabricating one — run a normal
scene generation and save just the observation arrays (`cgn_torch` env,
depth-mode test):

```powershell
cd D:\Projects\ContactPilot\mujoco_grasp_sim
$py_cgn = "$env:LOCALAPPDATA\miniconda3\envs\cgn_torch\python.exe"
& $py_cgn -c "
import sys; sys.path.insert(0, '.')
from sim_grasp import SceneConfig, SceneGenerator, CameraModule
import numpy as np
cfg = SceneConfig(seed=5)
gen = SceneGenerator(cfg)
model, data = gen.generate()
cam = CameraModule(model, data, cam_name=cfg.cam_name, width=640, height=480)
rgb, depth, segmap, K, T_world_cam = cam.capture(gen.object_body_ids)
np.savez('_test_obs.npz', depth=depth, K=K, rgb=rgb, segmap=segmap)
print('saved _test_obs.npz, segmap ids:', sorted(int(s) for s in np.unique(segmap) if s > 0))
"
```

Expected: prints a list of object segment ids (e.g. `[1, 2, 3]`) and writes
`_test_obs.npz`.

- [ ] **Step 3: Run the worker against it**

```powershell
$py_gg = "<GRASPGEN_PYTHON from Task 2 Step 6>"
& $py_gg sim_grasp\graspgen_worker.py _test_obs.npz _test_out.npz --gripper-config graspgen_checkpoints\graspgen_franka_panda.yml
```

Expected: `[graspgen-worker] N grasps -> _test_out.npz` with N > 0, no
tracebacks.

- [ ] **Step 4: Verify the output format**

```powershell
& $py_cgn -c "
import numpy as np
d = np.load('_test_out.npz')
print(sorted(d.files))
for k in d.files:
    if k.startswith('grasps_'):
        print(k, d[k].shape)
"
```

Expected: keys follow the `grasps_<sid>`, `scores_<sid>`, `contacts_<sid>`
pattern (matching `cgn_worker.py`'s format exactly); each `grasps_<sid>`
array has shape `(Ni, 4, 4)`.

- [ ] **Step 5: Clean up test artifacts and commit**

```bash
rm -f mujoco_grasp_sim/_test_obs.npz mujoco_grasp_sim/_test_out.npz
git add mujoco_grasp_sim/sim_grasp/graspgen_worker.py
git commit -m "Add graspgen_worker.py subprocess entry point"
```

---

### Task 4: `GraspGenPredictor` class

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/graspgen_predictor.py`
- Modify: `mujoco_grasp_sim/sim_grasp/__init__.py`

**Interfaces:**
- Consumes: `GraspPredictor`, `GraspPrediction` from
  `sim_grasp/grasp_predictor.py` (unchanged); Task 3's `graspgen_worker.py`.
- Produces: `GraspGenPredictor` class, importable as
  `from sim_grasp import GraspGenPredictor` (matching how
  `ContactGraspNetPredictor` is already imported in `run_sim_grasp_test.py`).
  Task 5 constructs this class directly.

- [ ] **Step 1: Write `graspgen_predictor.py`**

Create `mujoco_grasp_sim/sim_grasp/graspgen_predictor.py`:

```python
"""GraspGenPredictor: GraspPredictor backed by NVlabs/GraspGen.

Unlike ContactGraspNetPredictor, this ALWAYS runs via subprocess — GraspGen
lives in its own conda env (graspgen_torch) because its dependencies
conflict with cgn_torch's torch version, not merely for the memory-isolation
reason ContactGraspNetPredictor's subprocess path uses. There is no
in-process code path here.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from sim_grasp.grasp_predictor import GraspPredictor, GraspPrediction

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_DIR = _REPO_ROOT / 'graspgen_checkpoints'


def resolve_graspgen_python(override: str | None = None) -> Path:
    """Resolve the graspgen_torch interpreter: --graspgen-python CLI value,
    else GRASPGEN_PYTHON env var. Fails fast with a clear message — never
    falls back to sys.executable (that would run GraspGen under cgn_torch)."""
    candidate = override or os.environ.get('GRASPGEN_PYTHON')
    if not candidate:
        raise RuntimeError(
            'GraspGen backend requested but no interpreter configured. '
            'Set the GRASPGEN_PYTHON environment variable to the '
            'graspgen_torch env\'s python.exe, or pass --graspgen-python. '
            'See mujoco_grasp_sim/README.md "GraspGen backend setup".')
    path = Path(candidate)
    if not path.is_file():
        raise FileNotFoundError(f'GRASPGEN_PYTHON does not exist: {path}')
    return path


class GraspGenPredictor(GraspPredictor):
    def __init__(self, checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
                 graspgen_python: str | None = None,
                 num_grasps: int = 200, grasp_threshold: float = 0.8):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.gripper_config = self.checkpoint_dir / 'graspgen_franka_panda.yml'
        if not self.gripper_config.is_file():
            raise FileNotFoundError(
                f'GraspGen checkpoint not found: {self.gripper_config}. '
                'Run mujoco_grasp_sim/scripts/download_graspgen_checkpoint.py first.')
        self.python = resolve_graspgen_python(graspgen_python)
        self.num_grasps = num_grasps
        self.grasp_threshold = grasp_threshold

    def predict(self, depth, K, rgb=None, segmap=None) -> GraspPrediction:
        payload = dict(depth=depth, K=K)
        if rgb is not None:
            payload['rgb'] = rgb
        if segmap is not None:
            payload['segmap'] = segmap
        return self._run(payload)

    def predict_clouds(self, pc_full: np.ndarray,
                       pc_segments: dict | None = None) -> GraspPrediction:
        payload = {'pc_full': np.asarray(pc_full, dtype=np.float32)}
        for sid, pc in (pc_segments or {}).items():
            payload[f'pcseg_{float(sid):g}'] = np.asarray(pc, dtype=np.float32)
        return self._run(payload)

    def _run(self, payload: dict, work_dir: str | Path = '.') -> GraspPrediction:
        work_dir = Path(work_dir)
        obs_f = work_dir / '_graspgen_obs.npz'
        out_f = work_dir / '_graspgen_out.npz'
        np.savez(obs_f, **payload)
        worker = Path(__file__).parent / 'graspgen_worker.py'
        cmd = [str(self.python), str(worker), str(obs_f), str(out_f),
               '--gripper-config', str(self.gripper_config),
               '--num-grasps', str(self.num_grasps),
               '--grasp-threshold', str(self.grasp_threshold)]
        r = subprocess.run(cmd)
        if r.returncode != 0 or not out_f.exists():
            raise RuntimeError(f'GraspGen worker failed (exit code {r.returncode})')
        parts = {'grasps': {}, 'scores': {}, 'contacts': {}}
        with np.load(out_f) as z:
            for k in z.files:
                kind, sid = k.split('_', 1)
                parts[kind][float(sid)] = z[k]
        obs_f.unlink(missing_ok=True)
        out_f.unlink(missing_ok=True)
        return GraspPrediction(grasps_cam=parts['grasps'], scores=parts['scores'],
                               contact_pts=parts['contacts'], gripper_openings={})
```

- [ ] **Step 2: Export it from `sim_grasp/__init__.py`**

Current:
```python
from sim_grasp.grasp_predictor import GraspPredictor, GraspPrediction, ContactGraspNetPredictor
```

New:
```python
from sim_grasp.grasp_predictor import GraspPredictor, GraspPrediction, ContactGraspNetPredictor
from sim_grasp.graspgen_predictor import GraspGenPredictor
```

Also add one line to the module docstring's list (after the `grasp_predictor`
line):
```
graspgen_predictor  GraspGenPredictor: GraspPredictor backed by NVlabs/GraspGen
                     (separate conda env, always subprocess)
```

- [ ] **Step 3: Smoke-test the class directly**

```powershell
cd D:\Projects\ContactPilot\mujoco_grasp_sim
$env:GRASPGEN_PYTHON = "<GRASPGEN_PYTHON from Task 2 Step 6>"
$py_cgn = "$env:LOCALAPPDATA\miniconda3\envs\cgn_torch\python.exe"
& $py_cgn -c "
import sys; sys.path.insert(0, '.')
from sim_grasp import SceneConfig, SceneGenerator, CameraModule, GraspGenPredictor
cfg = SceneConfig(seed=5)
gen = SceneGenerator(cfg)
model, data = gen.generate()
cam = CameraModule(model, data, cam_name=cfg.cam_name, width=640, height=480)
rgb, depth, segmap, K, T_world_cam = cam.capture(gen.object_body_ids)
predictor = GraspGenPredictor()
pred = predictor.predict(depth, K, rgb=rgb, segmap=segmap)
print('num_grasps:', pred.num_grasps)
print('best:', predictor.predict)
best = pred.best_grasp()
print('best_grasp:', best[0], best[2] if best else None)
"
```

Expected: `num_grasps` > 0, `best_grasp` prints a seg_id and a score, no
exceptions. This is `GraspGenPredictor` exercised through the exact same
`GraspPredictor` interface `ContactGraspNetPredictor` uses — confirms the
class is a correct drop-in.

- [ ] **Step 4: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/graspgen_predictor.py mujoco_grasp_sim/sim_grasp/__init__.py
git commit -m "Add GraspGenPredictor"
```

---

### Task 5: `--backend` flag in `run_sim_grasp_test.py`

**Files:**
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py`

**Interfaces:**
- Consumes: `GraspGenPredictor` from Task 4;
  `predict_in_subprocess`/`predict_clouds_in_subprocess`/`_subprocess_predict`
  (existing, this task extends them with a `backend` parameter).
- Produces: `--backend {cgn,graspgen}` and `--graspgen-python PATH` CLI
  flags. Task 6 (`benchmark.py`) threads `--backend` through to this script.

- [ ] **Step 1: Add the CLI flags**

Current (`run_sim_grasp_test.py:210-235`, right after `--camera`):
```python
    ap.add_argument('--camera', choices=['calibrated', 'lookat', 'fused'],
                    default='calibrated',
                    help='observation camera setup: "calibrated" = real eye-to-hand '
                         'calibration (calibration_result.yaml); "lookat" = '
                         'generic angled look-at camera (the pre-calibration setup); '
                         '"fused" = BOTH (lookat primary + calibrated side cam, '
                         'point clouds fused in world frame — P2). '
                         'Use this to A/B compare Contact-GraspNet performance.')
```

New (add right after it):
```python
    ap.add_argument('--camera', choices=['calibrated', 'lookat', 'fused'],
                    default='calibrated',
                    help='observation camera setup: "calibrated" = real eye-to-hand '
                         'calibration (calibration_result.yaml); "lookat" = '
                         'generic angled look-at camera (the pre-calibration setup); '
                         '"fused" = BOTH (lookat primary + calibrated side cam, '
                         'point clouds fused in world frame — P2). '
                         'Use this to A/B compare Contact-GraspNet performance.')
    ap.add_argument('--backend', choices=['cgn', 'graspgen'], default='cgn',
                    help='grasp-prediction backend: "cgn" = Contact-GraspNet '
                         '(default), "graspgen" = NVlabs/GraspGen (needs the '
                         'graspgen_torch env — see README "GraspGen backend setup")')
    ap.add_argument('--graspgen-python', default=None,
                    help='path to the graspgen_torch env\'s python.exe; overrides '
                         'the GRASPGEN_PYTHON environment variable')
```

- [ ] **Step 2: Thread `backend` through the subprocess dispatch functions**

Current (`run_sim_grasp_test.py:52-97`):
```python
def predict_in_subprocess(depth, K, rgb, segmap, forward_passes, arg_configs,
                          work_dir) -> GraspPrediction:
    """Run one CGN prediction in a child process (sim_grasp/cgn_worker.py).

    PyTorch's multi-GB Windows commit is returned to the OS when the child
    exits, so the sim process keeps enough headroom to render — keeping the
    model resident here OOMs multi-round pick-and-place on 8 GB machines."""
    return _subprocess_predict(dict(depth=depth, K=K, rgb=rgb, segmap=segmap),
                               forward_passes, arg_configs, work_dir)


def predict_clouds_in_subprocess(pc_full_cam, pc_segments_cam, forward_passes,
                                 arg_configs, work_dir) -> GraspPrediction:
    """Cloud-mode subprocess prediction (P2 fusion: fused multi-camera cloud
    expressed in the primary camera frame)."""
    payload = {'pc_full': np.asarray(pc_full_cam, dtype=np.float32)}
    for sid, pc in pc_segments_cam.items():
        payload[f'pcseg_{float(sid):g}'] = np.asarray(pc, dtype=np.float32)
    return _subprocess_predict(payload, forward_passes, arg_configs, work_dir)


def _subprocess_predict(payload, forward_passes, arg_configs,
                        work_dir) -> GraspPrediction:
    import subprocess
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    obs_f, out_f = work_dir / '_cgn_obs.npz', work_dir / '_cgn_out.npz'
    np.savez(obs_f, **payload)
    worker = Path(__file__).parent / 'sim_grasp' / 'cgn_worker.py'
    cmd = [sys.executable, str(worker), str(obs_f), str(out_f),
           '--forward-passes', str(forward_passes)]
    if arg_configs:
        cmd += ['--arg-configs', *arg_configs]
    r = subprocess.run(cmd)
    if r.returncode != 0 or not out_f.exists():
        raise RuntimeError(f'CGN worker failed (exit code {r.returncode})')
    parts = {'grasps': {}, 'scores': {}, 'contacts': {}, 'openings': {}}
    with np.load(out_f) as z:   # context manager: NpzFile keeps the file open
        for k in z.files:       # lazily; Windows can't unlink it otherwise
            kind, sid = k.split('_', 1)
            parts[kind][float(sid)] = z[k]
    obs_f.unlink(missing_ok=True)
    out_f.unlink(missing_ok=True)
    return GraspPrediction(grasps_cam=parts['grasps'], scores=parts['scores'],
                           contact_pts=parts['contacts'],
                           gripper_openings=parts['openings'])
```

New (adds a `backend`/`graspgen_python` parameter, dispatches to
`GraspGenPredictor` for the graspgen case instead of the CGN worker path —
`GraspGenPredictor` already does its own subprocess management, so this just
delegates rather than duplicating `_subprocess_predict`'s npz plumbing):
```python
def predict_in_subprocess(depth, K, rgb, segmap, forward_passes, arg_configs,
                          work_dir, backend='cgn', graspgen_python=None) -> GraspPrediction:
    """Run one grasp prediction in a child process.

    PyTorch's multi-GB Windows commit is returned to the OS when the child
    exits, so the sim process keeps enough headroom to render — keeping the
    model resident here OOMs multi-round pick-and-place on 8 GB machines.
    (For backend='graspgen' this isolation is required regardless, since
    GraspGen needs its own conda env — see GraspGenPredictor.)"""
    if backend == 'graspgen':
        from sim_grasp import GraspGenPredictor
        return GraspGenPredictor(graspgen_python=graspgen_python).predict(
            depth, K, rgb=rgb, segmap=segmap)
    return _subprocess_predict(dict(depth=depth, K=K, rgb=rgb, segmap=segmap),
                               forward_passes, arg_configs, work_dir)


def predict_clouds_in_subprocess(pc_full_cam, pc_segments_cam, forward_passes,
                                 arg_configs, work_dir, backend='cgn',
                                 graspgen_python=None) -> GraspPrediction:
    """Cloud-mode subprocess prediction (P2 fusion: fused multi-camera cloud
    expressed in the primary camera frame)."""
    if backend == 'graspgen':
        from sim_grasp import GraspGenPredictor
        return GraspGenPredictor(graspgen_python=graspgen_python).predict_clouds(
            pc_full_cam, pc_segments_cam)
    payload = {'pc_full': np.asarray(pc_full_cam, dtype=np.float32)}
    for sid, pc in pc_segments_cam.items():
        payload[f'pcseg_{float(sid):g}'] = np.asarray(pc, dtype=np.float32)
    return _subprocess_predict(payload, forward_passes, arg_configs, work_dir)


def _subprocess_predict(payload, forward_passes, arg_configs,
                        work_dir) -> GraspPrediction:
    import subprocess
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    obs_f, out_f = work_dir / '_cgn_obs.npz', work_dir / '_cgn_out.npz'
    np.savez(obs_f, **payload)
    worker = Path(__file__).parent / 'sim_grasp' / 'cgn_worker.py'
    cmd = [sys.executable, str(worker), str(obs_f), str(out_f),
           '--forward-passes', str(forward_passes)]
    if arg_configs:
        cmd += ['--arg-configs', *arg_configs]
    r = subprocess.run(cmd)
    if r.returncode != 0 or not out_f.exists():
        raise RuntimeError(f'CGN worker failed (exit code {r.returncode})')
    parts = {'grasps': {}, 'scores': {}, 'contacts': {}, 'openings': {}}
    with np.load(out_f) as z:   # context manager: NpzFile keeps the file open
        for k in z.files:       # lazily; Windows can't unlink it otherwise
            kind, sid = k.split('_', 1)
            parts[kind][float(sid)] = z[k]
    obs_f.unlink(missing_ok=True)
    out_f.unlink(missing_ok=True)
    return GraspPrediction(grasps_cam=parts['grasps'], scores=parts['scores'],
                           contact_pts=parts['contacts'],
                           gripper_openings=parts['openings'])
```

- [ ] **Step 3: Wire `--backend` into the three call sites**

There are three places `predict_in_subprocess`/`predict_clouds_in_subprocess`
are called (the initial single prediction, and the two branches inside the
`--pick-all` round loop) plus one non-subprocess `ContactGraspNetPredictor`
construction (the non-`--pick-all` path). Update each:

Current (`run_sim_grasp_test.py:347-366`, initial prediction):
```python
    t0 = time.time()
    if args.pick_all:
        # pick-all re-runs CGN every round: keep torch OUT of this process
        # (see predict_in_subprocess) or MuJoCo rendering OOMs on 8 GB RAM
        print('[cgn] pick-all: running Contact-GraspNet in a subprocess per round...')
        predictor = None
        if fused:
            pred = predict_clouds_in_subprocess(pc_fused_cam, seg_fused_cam,
                                                args.forward_passes, arg_configs,
                                                save_dir)
        else:
            pred = predict_in_subprocess(depth, K, rgb, segmap,
                                         args.forward_passes, arg_configs, save_dir)
    else:
        print('[cgn] loading Contact-GraspNet...')
        predictor = ContactGraspNetPredictor(forward_passes=args.forward_passes,
                                             arg_configs=arg_configs)
        pred = predictor.predict_clouds(pc_fused_cam, seg_fused_cam) if fused \
            else predictor.predict(depth, K, rgb=rgb, segmap=segmap)
    print(f'[cgn] {pred.num_grasps} grasps in {time.time() - t0:.1f}s')
```

New:
```python
    t0 = time.time()
    if args.pick_all:
        # pick-all re-runs the backend every round: keep torch OUT of this
        # process (see predict_in_subprocess) or MuJoCo rendering OOMs on
        # 8 GB RAM. For backend='graspgen' this isolation is required
        # regardless of --pick-all, since it needs its own conda env.
        print(f'[{args.backend}] pick-all: running in a subprocess per round...')
        predictor = None
        if fused:
            pred = predict_clouds_in_subprocess(pc_fused_cam, seg_fused_cam,
                                                args.forward_passes, arg_configs,
                                                save_dir, backend=args.backend,
                                                graspgen_python=args.graspgen_python)
        else:
            pred = predict_in_subprocess(depth, K, rgb, segmap,
                                         args.forward_passes, arg_configs, save_dir,
                                         backend=args.backend,
                                         graspgen_python=args.graspgen_python)
    elif args.backend == 'graspgen':
        print('[graspgen] loading GraspGen...')
        from sim_grasp import GraspGenPredictor
        predictor = GraspGenPredictor(graspgen_python=args.graspgen_python)
        pred = predictor.predict_clouds(pc_fused_cam, seg_fused_cam) if fused \
            else predictor.predict(depth, K, rgb=rgb, segmap=segmap)
    else:
        print('[cgn] loading Contact-GraspNet...')
        predictor = ContactGraspNetPredictor(forward_passes=args.forward_passes,
                                             arg_configs=arg_configs)
        pred = predictor.predict_clouds(pc_fused_cam, seg_fused_cam) if fused \
            else predictor.predict(depth, K, rgb=rgb, segmap=segmap)
    print(f'[{args.backend}] {pred.num_grasps} grasps in {time.time() - t0:.1f}s')
```

Current (`run_sim_grasp_test.py:463-471`, inside the `--pick-all` round loop):
```python
                if fused:
                    pc_f, seg_f = capture_fused(
                        (rgb_r, depth_r, segmap_r, K_r, T_wc))
                    pred_r = predict_clouds_in_subprocess(
                        pc_f, seg_f, args.forward_passes, cfgs_r, save_dir)
                else:
                    pred_r = predict_in_subprocess(depth_r, K_r, rgb_r, segmap_r,
                                                   args.forward_passes, cfgs_r,
                                                   save_dir)
```

New:
```python
                if fused:
                    pc_f, seg_f = capture_fused(
                        (rgb_r, depth_r, segmap_r, K_r, T_wc))
                    pred_r = predict_clouds_in_subprocess(
                        pc_f, seg_f, args.forward_passes, cfgs_r, save_dir,
                        backend=args.backend, graspgen_python=args.graspgen_python)
                else:
                    pred_r = predict_in_subprocess(depth_r, K_r, rgb_r, segmap_r,
                                                   args.forward_passes, cfgs_r,
                                                   save_dir, backend=args.backend,
                                                   graspgen_python=args.graspgen_python)
```

- [ ] **Step 4: Single-run smoke test**

```powershell
cd D:\Projects\ContactPilot\mujoco_grasp_sim
$env:GRASPGEN_PYTHON = "<GRASPGEN_PYTHON from Task 2 Step 6>"
$py_cgn = "$env:LOCALAPPDATA\miniconda3\envs\cgn_torch\python.exe"
& $py_cgn run_sim_grasp_test.py --seed 5 --execute --backend graspgen --no-vis
```

Expected: runs to completion, prints `[graspgen] N grasps in ...s`, writes
`metrics.json` and `execution.gif` under `output/<timestamp>/`, same as a
normal CGN run. Same seed CGN was originally validated on (seed 5), so this
is a direct before/after comparison point.

- [ ] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/run_sim_grasp_test.py
git commit -m "Add --backend flag to run_sim_grasp_test.py"
```

---

### Task 6: `--backend` in `benchmark.py` + the decisive comparison

**Files:**
- Modify: `mujoco_grasp_sim/benchmark.py`

**Interfaces:**
- Consumes: Task 5's `--backend`/`--graspgen-python` flags on
  `run_sim_grasp_test.py`.
- Produces: `output/bench_<tag>/summary.json` for both backends, directly
  comparable (same seeds, same `--camera fused`, same scene config).

- [ ] **Step 1: Add `--backend` to `benchmark.py`'s CLI and pass-through**

Current (`benchmark.py:39-52`, `run_one`):
```python
def run_one(seed: int, args, run_dir: Path) -> dict:
    cmd = [sys.executable, str(HERE / 'run_sim_grasp_test.py'),
           '--seed', str(seed), '--no-vis', '--camera', args.camera,
           '--save-dir', str(run_dir)]
    if args.n_objects:
        cmd += ['--n-objects', str(args.n_objects)]
    if args.recenter:
        cmd += ['--recenter']
    if args.clean_depth:
        cmd += ['--clean-depth']
    if args.mode == 'execute':
        cmd += ['--execute', '--top-k', str(args.top_k)]
    elif args.mode == 'pick-all':
        cmd += ['--pick-all']
```

New:
```python
def run_one(seed: int, args, run_dir: Path) -> dict:
    cmd = [sys.executable, str(HERE / 'run_sim_grasp_test.py'),
           '--seed', str(seed), '--no-vis', '--camera', args.camera,
           '--save-dir', str(run_dir), '--backend', args.backend]
    if args.backend == 'graspgen' and args.graspgen_python:
        cmd += ['--graspgen-python', args.graspgen_python]
    if args.n_objects:
        cmd += ['--n-objects', str(args.n_objects)]
    if args.recenter:
        cmd += ['--recenter']
    if args.clean_depth:
        cmd += ['--clean-depth']
    if args.mode == 'execute':
        cmd += ['--execute', '--top-k', str(args.top_k)]
    elif args.mode == 'pick-all':
        cmd += ['--pick-all']
```

Current (`benchmark.py:90-104`, `main`'s argparse setup):
```python
    ap.add_argument('--recenter', action='store_true',
                    help='forward --recenter to the run script')
    ap.add_argument('--clean-depth', action='store_true',
                    help='forward --clean-depth to the run script')
    ap.add_argument('--tag', default=None, help='output/bench_<tag>/')
```

New:
```python
    ap.add_argument('--recenter', action='store_true',
                    help='forward --recenter to the run script')
    ap.add_argument('--clean-depth', action='store_true',
                    help='forward --clean-depth to the run script')
    ap.add_argument('--backend', choices=['cgn', 'graspgen'], default='cgn',
                    help='forward --backend to the run script')
    ap.add_argument('--graspgen-python', default=None,
                    help='forward --graspgen-python to the run script (or rely '
                         'on the GRASPGEN_PYTHON env var, same as the run script)')
    ap.add_argument('--tag', default=None, help='output/bench_<tag>/')
```

- [ ] **Step 2: Verify the CLI wiring with `--help`**

```powershell
cd D:\Projects\ContactPilot\mujoco_grasp_sim
$py_cgn = "$env:LOCALAPPDATA\miniconda3\envs\cgn_torch\python.exe"
& $py_cgn benchmark.py --help
```

Expected: `--backend {cgn,graspgen}` and `--graspgen-python` appear in the
printed help text.

- [ ] **Step 3: Commit the wiring**

```bash
git add mujoco_grasp_sim/benchmark.py
git commit -m "Add --backend flag to benchmark.py"
```

- [ ] **Step 4: Run the decisive benchmark comparison**

This is the actual answer to "does GraspGen miss fewer grasps than CGN" —
same seeds and camera config as the recorded CGN baseline (`ROADMAP.md` P1,
2026-06-14: 14/15 binned, 93%, box-only/3-objects, `--camera fused`):

```powershell
$env:GRASPGEN_PYTHON = "<GRASPGEN_PYTHON from Task 2 Step 6>"
& $py_cgn benchmark.py --seeds 0-4 --mode pick-all --camera fused --backend graspgen --tag graspgen_baseline
```

Expected: prints per-seed results as it goes, then a final summary line
`[bench] objects binned: X/Y (Z%), knocked off table: N` — this is the
number to compare against the recorded 14/15 (93%).

- [ ] **Step 5: Run the failure-taxonomy comparison**

```powershell
& $py_cgn analyze_failures.py output\bench_graspgen_baseline
```

Expected: a `taxonomy.json` classifying failures into the same categories
`ROADMAP.md` already uses (`closed_on_air`, `missed_bin`,
`knocked_off_table`, etc.) — compare this distribution against the recorded
CGN taxonomy (`closed_on_air` was 78% of CGN's original failures) to see
whether GraspGen shifts the failure *mode*, not just the raw rate.

- [ ] **Step 6: Record the result**

This step has no fixed expected output — it depends on what the benchmark
shows. Write the actual binned rate, knocked-off count, and failure
taxonomy comparison into `ROADMAP.md` (repo root, `D:\Projects\ContactPilot\ROADMAP.md`
— not inside `mujoco_grasp_sim/`) under a new dated entry, matching the
existing P1 entry style, so this becomes the recorded baseline for any
future GraspGen work, the same way the CGN numbers are recorded today.

```bash
git add ROADMAP.md
git commit -m "Record GraspGen benchmark results (seeds 0-4, pick-all, fused camera)"
```

---

### Task 7: Documentation

**Files:**
- Modify: `mujoco_grasp_sim/README.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Add a "GraspGen backend setup" section**

Insert a new section right after the existing `## Setup` section (before
`## Run`):

Current end of `## Setup` (`mujoco_grasp_sim/README.md:70-76`):
```markdown
If the Menagerie assets are missing:

```powershell
git clone --depth 1 --filter=blob:none --sparse --config core.autocrlf=false https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie sparse-checkout set franka_emika_panda
```

## Run
```

New:
```markdown
If the Menagerie assets are missing:

```powershell
git clone --depth 1 --filter=blob:none --sparse --config core.autocrlf=false https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie sparse-checkout set franka_emika_panda
```

### GraspGen backend setup (optional, `--backend graspgen`)

[NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) is a second grasp
backend (diffusion-based, Franka-Panda only) evaluated alongside
Contact-GraspNet. It needs its own conda env — its dependencies conflict
with `cgn_torch` — and is invoked as a subprocess, never in-process.

**License note:** NVIDIA Research license, not a permissive open-source
license — commercial use requires contacting NVIDIA Research Licensing.

```powershell
# 1. Clone GraspGen OUTSIDE this repo (not a submodule — nothing here patches it)
cd D:\Projects
git clone https://github.com/NVlabs/GraspGen.git

# 2. Create its env (separate from cgn_torch)
conda create -n graspgen_torch python=3.10 -y
conda activate graspgen_torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cd GraspGen
pip install -e .
cd pointnet2_ops && pip install --no-build-isolation . && cd ..

# 3. Fetch the Franka-Panda checkpoint (from ContactPilot's mujoco_grasp_sim/)
cd D:\Projects\ContactPilot\mujoco_grasp_sim
conda activate cgn_torch   # huggingface_hub already installed there (idea 1)
python scripts\download_graspgen_checkpoint.py

# 4. Point ContactPilot at the graspgen_torch interpreter
$env:GRASPGEN_PYTHON = "C:\path\to\miniconda3\envs\graspgen_torch\python.exe"
```

`GRASPGEN_PYTHON` needs to be set (or `--graspgen-python PATH` passed) any
time `--backend graspgen` is used — `run_sim_grasp_test.py` fails fast with
a clear error if it's missing, rather than silently trying to run GraspGen
under `cgn_torch`.

## Run
```

- [ ] **Step 2: Add `--backend` examples to the `## Run` command list**

Current (`mujoco_grasp_sim/README.md:93-98`):
```markdown
python run_sim_grasp_test.py --pick-all               # pick EVERY object, place each in the bin
python run_sim_grasp_test.py --save-dir output\myrun  # named output dir
python run_sim_grasp_test.py --camera fused           # P2: fuse lookat + calibrated side cam clouds
python run_sim_grasp_test.py --execute --pick-object 6        # P3: grasp THIS object only
python run_sim_grasp_test.py --execute --grasp-index 2        # P4: run candidate #2 from the printed list
python run_sim_grasp_test.py --recenter --clean-depth         # P1 experimental flags (see ROADMAP)
```
```

New:
```markdown
python run_sim_grasp_test.py --pick-all               # pick EVERY object, place each in the bin
python run_sim_grasp_test.py --save-dir output\myrun  # named output dir
python run_sim_grasp_test.py --camera fused           # P2: fuse lookat + calibrated side cam clouds
python run_sim_grasp_test.py --execute --pick-object 6        # P3: grasp THIS object only
python run_sim_grasp_test.py --execute --grasp-index 2        # P4: run candidate #2 from the printed list
python run_sim_grasp_test.py --recenter --clean-depth         # P1 experimental flags (see ROADMAP)
python run_sim_grasp_test.py --backend graspgen --execute     # NVlabs/GraspGen instead of CGN (needs GRASPGEN_PYTHON — see "GraspGen backend setup")
```
```

- [ ] **Step 3: Update "Swapping the grasp backend later" to reference the realized example**

Current (`mujoco_grasp_sim/README.md:200-211`):
```markdown
## Swapping the grasp backend later

Implement `GraspPredictor` (one method) in `sim_grasp/grasp_predictor.py`:

```python
class AnyGraspPredictor(GraspPredictor):
    def predict(self, depth, K, rgb=None, segmap=None) -> GraspPrediction:
        ...  # call AnyGrasp SDK, repackage into GraspPrediction
```

Everything else (scene, camera, feasibility, metrics, visualization) is
backend-agnostic. Same goes for GSNet, GIGA, FoundationPose+planner.
```

New:
```markdown
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
```

- [ ] **Step 4: Verify**

```bash
grep -n "graspgen" mujoco_grasp_sim/README.md
```

Expected: matches in the new setup section, the Run examples, and the
"Swapping the grasp backend later" section — at least 5 lines.

- [ ] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/README.md
git commit -m "Document the GraspGen backend"
```
