# Promptable Object Selection (SAM 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add text/click/box-prompted object selection to `mujoco_grasp_sim`, via NVlabs... via Meta's SAM 3, resolving to a segmap the existing grasp pipeline (CGN or GraspGen, unchanged) already knows how to consume.

**Architecture:** `PromptSelector` runs SAM 3 in its own conda env (`sam3_torch`) via subprocess (`sam3_worker.py`), exactly mirroring `GraspGenPredictor`'s isolation pattern. The resolved mask becomes a single-object segmap that replaces MuJoCo's ground-truth segmap for that run — real segmentation on rendered RGB, transferable to a live camera later. A new benchmark script grades selection accuracy against ground truth without ever feeding ground truth into the runtime pipeline.

**Tech Stack:** Python 3.12, PyTorch 2.10+cu128, Meta SAM 3 (`facebookresearch/sam3`, gated HF checkpoint), conda.

## Global Constraints

- SAM 3 checkpoint access is **gated on Hugging Face** (`facebook/sam3`) — requires requesting access and waiting for approval, then `hf auth login`. Start this immediately; it's not scriptable.
- SAM 3 needs its own conda env (`sam3_torch`, Python 3.12+) — do not attempt to install into `cgn_torch` or `graspgen_torch`.
- No custom CUDA extension compilation is needed for SAM 3's core functionality — do NOT install the optional `flash-attn-3`/`cc_torch` extras; they're unnecessary and reintroduce exactly the compile risk this plan is designed to avoid.
- `SAM3_PYTHON` env var / `--sam3-python` CLI override resolves the interpreter — **never** `sys.executable`, same rule as `GRASPGEN_PYTHON`.
- `add_geometric_prompt`'s box format is **`[center_x, center_y, width, height]`, normalized to [0,1]** — not corner coordinates, not pixel coordinates. Verified against `sam3/model/sam3_image_processor.py` source directly.
- `--prompt`/`--click`/`--box` are mutually exclusive with each other AND with `--pick-object` (same "pick this one object" purpose via different mechanisms — combining them is ambiguous, not additive).
- Zero changes to `GraspPredictor`, `GraspPrediction`, `feasibility.py`, `executor.py`, or either grasp backend — this is a perception-layer addition upstream of grasp prediction only.
- Ground truth (`model.geom_rgba`, MuJoCo's real segmap) is used ONLY by the benchmark script for grading — never by the runtime selection pipeline.

---

### Task 1: `sam3_torch` environment + standalone smoke test

**This is the task with the biggest genuine unknown** (untested on this
GPU/WSL2 combination), though much lower-risk than GraspGen's Task 2 —
SAM 3's core install has no custom CUDA extension to compile.

**Files:** none in the ContactPilot repo — this task requests HF access,
clones SAM 3, and creates a conda env, all external to this repo.

**Interfaces:**
- Produces: a working `sam3_torch` conda env whose `python`'s full path
  becomes `SAM3_PYTHON` for every later task.

- [ ] **Step 1: Request Hugging Face access to SAM 3's checkpoints (do this first — it's the only step here that isn't scriptable and may take time to be approved)**

Visit https://huggingface.co/facebook/sam3 and request access. Once
approved, authenticate:

```bash
pip install -U huggingface_hub
hf auth login
```

If access isn't approved yet, everything else in this task (through Step 4)
can still proceed — only Step 5 (loading the actual checkpoint) needs it.

- [ ] **Step 2: Clone SAM 3 outside the ContactPilot tree**

Same reasoning as GraspGen: not a submodule, nothing here patches it, keep
it a sibling clone.

```bash
cd ~
git clone https://github.com/facebookresearch/sam3.git
```

- [ ] **Step 3: Create the `sam3_torch` conda env (Python 3.12)**

```bash
conda create -n sam3_torch python=3.12 -y
conda activate sam3_torch
```

- [ ] **Step 4: Install PyTorch and SAM 3**

```bash
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
cd ~/sam3
pip install -e .
```

Do NOT run the "optional dependencies for faster inference" step
(`flash-attn-3`, `cc_torch`) — skip it entirely; it's unnecessary and
reintroduces compiled-extension risk.

Verify the install:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from sam3.model_builder import build_sam3_image_model; print('import OK')"
```
Expected: a `+cu128` torch version, `True`, and `import OK` with no errors.

- [ ] **Step 5: Run SAM 3's own smoke test (requires Step 1's HF access to be approved)**

```bash
cd ~/sam3
python -c "
from PIL import Image
import numpy as np
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)

# any real image works for this smoke test — using a synthetic one to avoid
# needing a sample asset
img = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype('uint8'))
state = processor.set_image(img)
output = processor.set_text_prompt(prompt='an object', state=state)
print('masks:', output['masks'].shape if hasattr(output['masks'], 'shape') else len(output['masks']))
print('scores:', output['scores'])
"
```

Expected: prints a masks shape/count and scores array, no exceptions, no
CUDA errors. (A random-noise image won't match anything meaningful — the
point is confirming inference runs end-to-end on this GPU, not that it
finds a real object. Task 2 onward test against real captured scenes.)

- [ ] **Step 6: Record the interpreter path**

```bash
which python
```

This full path (e.g. `/home/vivek/miniconda3/envs/sam3_torch/bin/python`)
is `SAM3_PYTHON` for every later task.

No commit — this task touches nothing inside the ContactPilot repo.

---

### Task 2: Color-naming utility

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/color_utils.py`

**Interfaces:**
- Produces: `rgb_to_color_name(rgb) -> str`. Task 6's benchmark script
  consumes this directly.

No dependency on Task 1 (`sam3_torch`) — pure numpy, runs under `cgn_torch`
or any Python with numpy.

- [ ] **Step 1: Write the failing test**

Create `mujoco_grasp_sim/sim_grasp/test_color_utils.py` (a plain script,
not pytest — this codebase has no test suite; running it directly with
assertions is the convention used elsewhere in this plan):

```python
"""Standalone check for color_utils.rgb_to_color_name — run directly, no
pytest (this codebase has no automated test suite)."""
from sim_grasp.color_utils import rgb_to_color_name

# Clear, unambiguous cases
assert rgb_to_color_name((0.9, 0.15, 0.15)) == 'red', 'pure red misclassified'
assert rgb_to_color_name((0.15, 0.8, 0.15)) == 'green', 'pure green misclassified'
assert rgb_to_color_name((0.15, 0.15, 0.9)) == 'blue', 'pure blue misclassified'
assert rgb_to_color_name((0.9, 0.9, 0.15)) == 'yellow', 'pure yellow misclassified'

# Returns a string for any input in the valid [0,1]^3 range, never crashes
import numpy as np
rng = np.random.default_rng(0)
for _ in range(20):
    rgb = rng.uniform(0.15, 0.95, size=3)
    name = rgb_to_color_name(rgb)
    assert isinstance(name, str) and len(name) > 0

print('All color_utils checks passed.')
```

- [ ] **Step 2: Run it to verify it fails (module doesn't exist yet)**

```bash
cd mujoco_grasp_sim
python sim_grasp/test_color_utils.py
```
Expected: `ModuleNotFoundError: No module named 'sim_grasp.color_utils'` (or
similar import error).

- [ ] **Step 3: Write `color_utils.py`**

Create `mujoco_grasp_sim/sim_grasp/color_utils.py`:

```python
"""Nearest-named-color matching for scene objects.

Scene objects get uniform-random RGB in [0.15, 0.95]^3 (see
scene_generator.py's _rand_rgba) — not perceptually distributed. This maps
an arbitrary RGB to the closest name in a small curated table, used only to
build ground-truth prompts for the promptable-selection benchmark. It is
never part of the runtime selection pipeline.
"""
import numpy as np

_NAMED_COLORS = {
    'red': (0.85, 0.2, 0.2),
    'orange': (0.9, 0.5, 0.15),
    'yellow': (0.9, 0.9, 0.2),
    'green': (0.25, 0.7, 0.3),
    'cyan': (0.2, 0.8, 0.8),
    'blue': (0.2, 0.3, 0.85),
    'purple': (0.55, 0.25, 0.75),
    'pink': (0.9, 0.5, 0.75),
    'brown': (0.5, 0.35, 0.2),
    'gray': (0.55, 0.55, 0.55),
}


def rgb_to_color_name(rgb) -> str:
    """Nearest named color to `rgb` (any 3+-length sequence in [0,1]), by
    Euclidean distance in RGB space."""
    rgb = np.asarray(rgb, dtype=float)[:3]
    best_name, best_dist = None, float('inf')
    for name, ref in _NAMED_COLORS.items():
        d = float(np.linalg.norm(rgb - np.asarray(ref, dtype=float)))
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd mujoco_grasp_sim
python sim_grasp/test_color_utils.py
```
Expected: `All color_utils checks passed.`

- [ ] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/color_utils.py mujoco_grasp_sim/sim_grasp/test_color_utils.py
git commit -m "Add RGB-to-color-name utility for promptable-selection benchmarking"
```

---

### Task 3: `sam3_worker.py` subprocess entry point

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/sam3_worker.py`

**Interfaces:**
- Consumes: `Sam3Processor`/`build_sam3_image_model` from SAM 3 (importable
  under the `sam3_torch` interpreter after Task 1).
- Produces: an npz with `masks` `(K,H,W)` bool, `scores` `(K,)` float,
  `boxes` `(K,4)` float (pixel `[x1,y1,x2,y2]`, computed from each mask's
  own bounding box — NOT trusted from SAM 3's raw output, since that
  format wasn't independently confirmed). Task 4's `PromptSelector`
  consumes this exact format.
- CLI: `python sam3_worker.py rgb.npy out.npz [--prompt TEXT] [--click X,Y] [--box X1,Y1,X2,Y2]`
  — exactly one of `--prompt`/`--click`/`--box` required.

- [ ] **Step 1: Write `sam3_worker.py`**

Create `mujoco_grasp_sim/sim_grasp/sam3_worker.py`:

```python
"""Subprocess SAM 3 worker — runs under the sam3_torch interpreter.

Usage:
    python sim_grasp/sam3_worker.py rgb.npy out.npz --prompt "the red box"
    python sim_grasp/sam3_worker.py rgb.npy out.npz --click 320,240
    python sim_grasp/sam3_worker.py rgb.npy out.npz --box 100,100,300,300

rgb.npy: (H,W,3) uint8 RGB image.
out.npz keys: masks (K,H,W) bool, scores (K,) float32, boxes (K,4) float32
    (pixel [x1,y1,x2,y2], computed from each mask's own bounding box).
"""
import argparse

import numpy as np
from PIL import Image


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    """Pixel-space [x1,y1,x2,y2] bounding box of the True region of a 2D
    boolean mask. Computed directly from the mask rather than trusted from
    the model's raw box output, whose exact coordinate convention wasn't
    independently verified against the installed version."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('rgb_npy')
    ap.add_argument('out_npz')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompt', type=str)
    group.add_argument('--click', type=str, help='X,Y pixel coordinates')
    group.add_argument('--box', type=str, help='X1,Y1,X2,Y2 pixel coordinates')
    ap.add_argument('--click-radius-px', type=int, default=15,
                    help='half-width of the box synthesized around a --click point')
    args = ap.parse_args()

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    rgb = np.load(args.rgb_npy)
    H, W = rgb.shape[:2]
    image = Image.fromarray(rgb)

    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    state = processor.set_image(image)

    if args.prompt:
        output = processor.set_text_prompt(prompt=args.prompt, state=state)
    else:
        if args.click:
            x, y = (float(v) for v in args.click.split(','))
            r = args.click_radius_px
            x1, y1, x2, y2 = x - r, y - r, x + r, y + r
        else:
            x1, y1, x2, y2 = (float(v) for v in args.box.split(','))
        cx, cy = (x1 + x2) / 2.0 / W, (y1 + y2) / 2.0 / H
        w, h = (x2 - x1) / W, (y2 - y1) / H
        output = processor.add_geometric_prompt(
            box=[cx, cy, w, h], label=True, state=state)

    masks_t, scores_t = output['masks'], output['scores']
    masks = masks_t.cpu().numpy().astype(bool) if hasattr(masks_t, 'cpu') \
        else np.asarray(masks_t, dtype=bool)
    scores = scores_t.cpu().numpy().astype(np.float32) if hasattr(scores_t, 'cpu') \
        else np.asarray(scores_t, dtype=np.float32)
    if masks.ndim == 2:          # single instance: normalize to (1,H,W)
        masks = masks[None]
        scores = scores.reshape(1)

    boxes = np.array([mask_bbox(m) for m in masks], dtype=np.float32)

    np.savez(args.out_npz, masks=masks, scores=scores, boxes=boxes)
    print(f'[sam3-worker] {len(masks)} match(es), scores={scores.tolist()} -> {args.out_npz}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Capture a real scene RGB to test against (run in `cgn_torch`, mirrors GraspGen Task 3's approach)**

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
np.save('_test_rgb.npy', rgb)
print('saved _test_rgb.npy', rgb.shape)
"
```

(On WSL2, adjust the interpreter path to wherever `cgn_torch`'s equivalent
env lives there, or use system Python with the same sim dependencies
installed — whichever this session's own earlier setup already provides.)

- [ ] **Step 3: Run the worker with a text prompt**

```bash
python sim_grasp/sam3_worker.py _test_rgb.npy _test_out.npz --prompt "a box"
```
Expected: `[sam3-worker] N match(es), scores=[...] -> _test_out.npz` with
N >= 1, no tracebacks (seed 5 spawns 3 box objects — see
`scene_generator.py`'s box-only default).

- [ ] **Step 4: Run it with a click**

```bash
python sim_grasp/sam3_worker.py _test_rgb.npy _test_out2.npz --click 320,240
```
Expected: `[sam3-worker] 1 match(es), ...` (a click always yields exactly
one target — it's a positive point prompt, not an open-vocabulary query).

- [ ] **Step 5: Verify the output format**

```bash
python -c "
import numpy as np
d = np.load('_test_out.npz')
print('masks', d['masks'].shape, d['masks'].dtype)
print('scores', d['scores'])
print('boxes', d['boxes'])
"
```
Expected: `masks` shape `(N, 480, 640)` dtype `bool`; `scores` length `N`;
`boxes` shape `(N, 4)`.

- [ ] **Step 6: Clean up test artifacts and commit**

```bash
rm -f mujoco_grasp_sim/_test_rgb.npy mujoco_grasp_sim/_test_out.npz mujoco_grasp_sim/_test_out2.npz
git add mujoco_grasp_sim/sim_grasp/sam3_worker.py
git commit -m "Add sam3_worker.py subprocess entry point"
```

---

### Task 4: `PromptSelector` class

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/prompt_selector.py`

**Interfaces:**
- Consumes: Task 3's `sam3_worker.py`.
- Produces: `PromptSelector` class and `SelectionResult` dataclass,
  importable as `from sim_grasp.prompt_selector import PromptSelector, SelectionResult`.
  Task 5 constructs and calls `PromptSelector.select(...)` directly.

- [ ] **Step 1: Write `prompt_selector.py`**

Create `mujoco_grasp_sim/sim_grasp/prompt_selector.py`:

```python
"""PromptSelector: resolves a text/click/box prompt to a target mask via
Meta SAM 3.

Always runs via subprocess — SAM 3 lives in its own conda env (sam3_torch,
Python 3.12) because it needs a newer Python than cgn_torch's 3.10. There is
no in-process code path here, matching GraspGenPredictor's isolation
pattern in sim_grasp/graspgen_predictor.py.
"""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SelectionResult:
    masks: np.ndarray    # (K,H,W) bool
    scores: np.ndarray   # (K,) float32
    boxes: np.ndarray    # (K,4) float32, pixel [x1,y1,x2,y2]

    @property
    def is_ambiguous(self) -> bool:
        return len(self.scores) > 1

    @property
    def is_empty(self) -> bool:
        return len(self.scores) == 0


def resolve_sam3_python(override: str | None = None) -> Path:
    """Resolve the sam3_torch interpreter: --sam3-python CLI value, else
    SAM3_PYTHON env var. Fails fast — never falls back to sys.executable
    (that would run SAM 3 under cgn_torch, which lacks Python 3.12)."""
    candidate = override or os.environ.get('SAM3_PYTHON')
    if not candidate:
        raise RuntimeError(
            'Promptable selection requested but no SAM 3 interpreter '
            'configured. Set the SAM3_PYTHON environment variable to the '
            'sam3_torch env\'s python, or pass --sam3-python. See '
            'mujoco_grasp_sim/README.md "Promptable selection setup".')
    path = Path(candidate)
    if not path.is_file():
        raise FileNotFoundError(f'SAM3_PYTHON does not exist: {path}')
    return path


class PromptSelector:
    def __init__(self, sam3_python: str | None = None, click_radius_px: int = 15):
        self.python = resolve_sam3_python(sam3_python)
        self.click_radius_px = click_radius_px

    def select(self, rgb: np.ndarray, prompt: str | None = None,
              click: tuple[float, float] | None = None,
              box: tuple[float, float, float, float] | None = None,
              work_dir: str | Path = '.') -> SelectionResult:
        modes = [m for m in (prompt, click, box) if m is not None]
        if len(modes) != 1:
            raise ValueError('Exactly one of prompt, click, box must be given')

        work_dir = Path(work_dir)
        rgb_f = work_dir / '_sam3_rgb.npy'
        out_f = work_dir / '_sam3_out.npz'
        np.save(rgb_f, np.asarray(rgb, dtype=np.uint8))

        worker = Path(__file__).parent / 'sam3_worker.py'
        cmd = [str(self.python), str(worker), str(rgb_f), str(out_f)]
        if prompt is not None:
            cmd += ['--prompt', prompt]
        elif click is not None:
            cmd += ['--click', f'{click[0]},{click[1]}']
        else:
            cmd += ['--box', f'{box[0]},{box[1]},{box[2]},{box[3]}']
        cmd += ['--click-radius-px', str(self.click_radius_px)]

        r = subprocess.run(cmd)
        if r.returncode != 0 or not out_f.exists():
            raise RuntimeError(f'SAM 3 worker failed (exit code {r.returncode})')

        with np.load(out_f) as z:
            result = SelectionResult(masks=z['masks'], scores=z['scores'], boxes=z['boxes'])
        rgb_f.unlink(missing_ok=True)
        out_f.unlink(missing_ok=True)
        return result
```

- [ ] **Step 2: Smoke-test the class directly (mirrors GraspGenPredictor's Task 4 Step 3)**

```powershell
cd D:\Projects\ContactPilot\mujoco_grasp_sim
$env:SAM3_PYTHON = "<SAM3_PYTHON from Task 1 Step 6>"
$py_cgn = "$env:LOCALAPPDATA\miniconda3\envs\cgn_torch\python.exe"
& $py_cgn -c "
import sys; sys.path.insert(0, '.')
from sim_grasp import SceneConfig, SceneGenerator, CameraModule
from sim_grasp.prompt_selector import PromptSelector
cfg = SceneConfig(seed=5)
gen = SceneGenerator(cfg)
model, data = gen.generate()
cam = CameraModule(model, data, cam_name=cfg.cam_name, width=640, height=480)
rgb, depth, segmap, K, T_world_cam = cam.capture(gen.object_body_ids)
selector = PromptSelector()
result = selector.select(rgb, prompt='a box')
print('matches:', len(result.scores), 'ambiguous:', result.is_ambiguous)
print('scores:', result.scores)
"
```

Expected: matches >= 1, no exceptions. This exercises `PromptSelector`
through the exact interface Task 5 will use.

- [ ] **Step 3: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/prompt_selector.py
git commit -m "Add PromptSelector"
```

---

### Task 5: CLI wiring in `run_sim_grasp_test.py`

**Files:**
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py`

**Interfaces:**
- Consumes: `PromptSelector`, `SelectionResult` from Task 4.
- Produces: `--prompt`, `--click`, `--box`, `--prompt-index`, `--sam3-python`
  CLI flags.

- [ ] **Step 1: Add the CLI flags**

Current (`run_sim_grasp_test.py`, right after the `--pick-object` argument):
```python
    ap.add_argument('--pick-object', type=int, default=None, metavar='SEG_ID',
                    help='only grasp THIS object (segmentation instance id, '
                         'see the printed per-object table / observation.png). '
                         'Works with --execute and --pick-all.')
```

New (add right after it):
```python
    ap.add_argument('--pick-object', type=int, default=None, metavar='SEG_ID',
                    help='only grasp THIS object (segmentation instance id, '
                         'see the printed per-object table / observation.png). '
                         'Works with --execute and --pick-all.')
    prompt_group = ap.add_mutually_exclusive_group()
    prompt_group.add_argument('--prompt', type=str, default=None,
                    help='select the target object by text description '
                         '(e.g. "the red box"), via SAM 3 on the rendered '
                         'RGB. Mutually exclusive with --click/--box/--pick-object.')
    prompt_group.add_argument('--click', type=str, default=None, metavar='X,Y',
                    help='select the target object by clicking a pixel '
                         '(observation.png coordinates), via SAM 3.')
    prompt_group.add_argument('--box', type=str, default=None, metavar='X1,Y1,X2,Y2',
                    help='select the target object by a pixel bounding box, via SAM 3.')
    ap.add_argument('--prompt-index', type=int, default=None, metavar='I',
                    help='with an ambiguous --prompt (multiple matches): pick '
                         'match #I from the printed ranked list')
    ap.add_argument('--sam3-python', default=None,
                    help='path to the sam3_torch env\'s python; overrides '
                         'the SAM3_PYTHON environment variable')
```

- [ ] **Step 2: Reject `--pick-object` combined with a prompt flag**

Current (`run_sim_grasp_test.py`, right after `args = ap.parse_args()`):
```python
    args = ap.parse_args()
    if args.pick_all:
        args.execute = True   # pick-all implies execution (thresholds, GPU prep)
```

New:
```python
    args = ap.parse_args()
    if args.pick_all:
        args.execute = True   # pick-all implies execution (thresholds, GPU prep)
    if args.pick_object is not None and (args.prompt or args.click or args.box):
        sys.exit('[prompt] --pick-object and --prompt/--click/--box are mutually '
                 'exclusive — both select a single target object, pick one mechanism.')
```

- [ ] **Step 3: Resolve the prompt to a segmap right after camera capture**

Current (`run_sim_grasp_test.py`, right after the depth-cleaning block and
before the `visible_ids`/`vis.save_observation` lines):
```python
    if args.clean_depth:
        n0 = int((depth > 0).sum())
        depth = clean_depth(depth, K, T_world_cam)
        print(f'[perception] workspace crop + speckle removal: '
              f'{n0 - int((depth > 0).sum())} px dropped')
    visible_ids = sorted(int(s) for s in np.unique(segmap) if s > 0)
```

New:
```python
    if args.clean_depth:
        n0 = int((depth > 0).sum())
        depth = clean_depth(depth, K, T_world_cam)
        print(f'[perception] workspace crop + speckle removal: '
              f'{n0 - int((depth > 0).sum())} px dropped')

    if args.prompt or args.click or args.box:
        from sim_grasp.prompt_selector import PromptSelector
        selector = PromptSelector(sam3_python=args.sam3_python)
        click_xy = tuple(float(v) for v in args.click.split(',')) if args.click else None
        box_xyxy = tuple(float(v) for v in args.box.split(',')) if args.box else None
        result = selector.select(rgb, prompt=args.prompt, click=click_xy, box=box_xyxy)
        if result.is_empty:
            sys.exit(f'[prompt] no object matched: '
                     f'{args.prompt or args.click or args.box!r}')
        if result.is_ambiguous and args.prompt_index is None:
            print(f'[prompt] {len(result.scores)} matches for '
                  f'{args.prompt!r} — pass --prompt-index to disambiguate:')
            for i, (s, b) in enumerate(zip(result.scores, result.boxes)):
                print(f'  [{i}] score {float(s):.3f}  box {[round(float(v), 1) for v in b]}')
            sys.exit(1)
        idx = args.prompt_index if result.is_ambiguous else 0
        if not 0 <= idx < len(result.scores):
            sys.exit(f'[prompt] --prompt-index {idx} out of range (0..{len(result.scores) - 1})')
        segmap = np.zeros(rgb.shape[:2], dtype=segmap.dtype)
        segmap[result.masks[idx]] = 1
        print(f'[prompt] resolved to 1 object, score {float(result.scores[idx]):.3f}')

    visible_ids = sorted(int(s) for s in np.unique(segmap) if s > 0)
```

- [ ] **Step 4: Single-run smoke test**

```powershell
cd D:\Projects\ContactPilot\mujoco_grasp_sim
$env:SAM3_PYTHON = "<SAM3_PYTHON from Task 1 Step 6>"
$py_cgn = "$env:LOCALAPPDATA\miniconda3\envs\cgn_torch\python.exe"
& $py_cgn run_sim_grasp_test.py --seed 5 --execute --prompt "a box" --no-vis
```

Expected: prints `[prompt] resolved to 1 object, score ...` (or the
ambiguous-candidates list if seed 5's 3 boxes all match "a box" — in that
case, re-run with a more specific prompt like a real color from
`observation.png`, or with `--prompt-index 0`), then proceeds through
grasp prediction/execution exactly as a normal `--pick-object` run would,
writing `metrics.json`/`execution.gif`.

- [ ] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/run_sim_grasp_test.py
git commit -m "Add --prompt/--click/--box flags to run_sim_grasp_test.py"
```

---

### Task 6: Accuracy benchmark + decisive results

**Files:**
- Create: `mujoco_grasp_sim/benchmark_prompt_selection.py`

**Interfaces:**
- Consumes: `PromptSelector` (Task 4), `rgb_to_color_name` (Task 2),
  `SceneGenerator`/`CameraModule` (existing).
- Produces: `output/bench_prompt_<tag>/summary.json` + a printed accuracy
  table.

- [ ] **Step 1: Write the benchmark script**

Create `mujoco_grasp_sim/benchmark_prompt_selection.py`:

```python
"""Promptable-selection accuracy benchmark.

For each seed: generate a scene, capture RGB, read each object's REAL
spawned color from the compiled MuJoCo model (model.geom_rgba — not scene
metadata), build a genuine ground-truth prompt ("the {color} box") for one
target object, run PromptSelector, and check whether the resolved mask
actually overlaps the intended object's ground-truth segmap region (IoU).
Ground truth is used only for this grading step, never fed into the
selection pipeline itself.

Usage:
    python benchmark_prompt_selection.py --seeds 0-4 --tag baseline
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim_grasp import SceneConfig, SceneGenerator, CameraModule
from sim_grasp.color_utils import rgb_to_color_name
from sim_grasp.prompt_selector import PromptSelector

HERE = Path(__file__).resolve().parent


def parse_seeds(spec: str) -> list[int]:
    seeds = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def object_geom(model, body_name):
    """Mirrors run_sim_grasp_test.py's _object_geom() — (bid, gid, gtype)
    for a named object body."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    gid = model.body_geomadr[bid]
    return bid, gid, int(model.geom_type[gid])


def run_one(seed: int, selector: PromptSelector) -> dict:
    cfg = SceneConfig(seed=seed)
    gen = SceneGenerator(cfg)
    model, data = gen.generate()
    on_table = gen.objects_on_table()
    if not on_table:
        return {'seed': seed, 'skipped': 'no objects on table'}

    cam = CameraModule(model, data, cam_name=cfg.cam_name, width=640, height=480)
    rgb, depth, segmap, K, T_world_cam = cam.capture(gen.object_body_ids)
    cam.close()

    label_of = {name: i + 1 for i, name in enumerate(gen.object_names)}
    target_body = on_table[0]
    target_sid = label_of[target_body]
    _bid, gid, _gtype = object_geom(model, target_body)
    color_name = rgb_to_color_name(model.geom_rgba[gid][:3])
    prompt = f'the {color_name} box'

    result = selector.select(rgb, prompt=prompt)
    if result.is_empty:
        return {'seed': seed, 'prompt': prompt, 'matched': False, 'reason': 'no matches'}

    # best-scoring match (benchmark grades accuracy; it doesn't exercise
    # the CLI's disambiguation-required behavior)
    idx = int(np.argmax(result.scores))
    mask = result.masks[idx]
    gt_mask = segmap == target_sid
    intersection = np.logical_and(mask, gt_mask).sum()
    union = np.logical_or(mask, gt_mask).sum()
    iou = float(intersection) / float(union) if union > 0 else 0.0

    return {
        'seed': seed, 'prompt': prompt, 'target_object': target_sid,
        'num_matches': len(result.scores), 'matched': iou > 0.5, 'iou': round(iou, 3),
        'score': round(float(result.scores[idx]), 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', default='0-4', help='e.g. "0-4" or "1,3,7"')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--sam3-python', default=None)
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    tag = args.tag or time.strftime('%m%d_%H%M')
    bench_dir = HERE / 'output' / f'bench_prompt_{tag}'
    bench_dir.mkdir(parents=True, exist_ok=True)

    selector = PromptSelector(sam3_python=args.sam3_python)
    results = []
    for seed in seeds:
        print(f'[bench-prompt] seed {seed}...', flush=True)
        r = run_one(seed, selector)
        results.append(r)
        print(f'[bench-prompt]   {r}', flush=True)

    (bench_dir / 'summary.json').write_text(json.dumps(results, indent=2))
    graded = [r for r in results if 'matched' in r]
    n_correct = sum(1 for r in graded if r['matched'])
    print(f'\n[bench-prompt] ===== {n_correct}/{len(graded)} correct selections '
          f'({100 * n_correct / max(len(graded), 1):.0f}%) =====')
    if graded:
        mean_iou = sum(r['iou'] for r in graded) / len(graded)
        print(f'[bench-prompt] mean IoU: {mean_iou:.3f}')
    print(f'[bench-prompt] full details: {bench_dir / "summary.json"}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the decisive benchmark**

```powershell
cd D:\Projects\ContactPilot\mujoco_grasp_sim
$env:SAM3_PYTHON = "<SAM3_PYTHON from Task 1 Step 6>"
$py_cgn = "$env:LOCALAPPDATA\miniconda3\envs\cgn_torch\python.exe"
& $py_cgn benchmark_prompt_selection.py --seeds 0-4 --tag baseline
```

Expected: prints per-seed results, then a final
`[bench-prompt] X/Y correct selections (Z%)` line and a mean IoU. This has
no fixed pass value ahead of time — it depends on SAM 3's actual accuracy
on this scene distribution.

- [ ] **Step 3: Record the result in `ROADMAP.md`**

Add a new dated entry to `ROADMAP.md` (repo root,
`D:\Projects\ContactPilot\ROADMAP.md`) documenting the actual accuracy
number and mean IoU from Step 2, in the same style as the GraspGen entry
already there — this becomes the recorded baseline for future promptable-
selection work. There's no fixed text to copy here since it depends on the
real result; write it factually, citing the exact numbers `summary.json`
produced.

```bash
git add mujoco_grasp_sim/benchmark_prompt_selection.py ROADMAP.md
git commit -m "Add promptable-selection benchmark; record baseline accuracy (seeds 0-4)"
```

---

### Task 7: Documentation

**Files:**
- Modify: `mujoco_grasp_sim/README.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Add a "Promptable selection setup" section**

Insert right after the existing "GraspGen backend setup" section (added by
the previous plan) and before `## Run`:

```markdown
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
pip install -e .
# do NOT install the optional flash-attn-3/cc_torch extras — unnecessary,
# and they're compiled extensions this repo has otherwise avoided needing

# 4. Point ContactPilot at the sam3_torch interpreter
export SAM3_PYTHON=$(which python)   # while sam3_torch is still active
```

`SAM3_PYTHON` needs to be set (or `--sam3-python PATH` passed) any time
`--prompt`/`--click`/`--box` is used — fails fast with a clear error if
missing, same as `GRASPGEN_PYTHON`.
```

- [ ] **Step 2: Add examples to the `## Run` command list**

Current (end of the flag examples list in `mujoco_grasp_sim/README.md`):
```markdown
python run_sim_grasp_test.py --backend graspgen --execute     # NVlabs/GraspGen instead of CGN (needs GRASPGEN_PYTHON — see "GraspGen backend setup")
```
```

New:
```markdown
python run_sim_grasp_test.py --backend graspgen --execute     # NVlabs/GraspGen instead of CGN (needs GRASPGEN_PYTHON — see "GraspGen backend setup")
python run_sim_grasp_test.py --execute --prompt "the red box" # select the target by text description (needs SAM3_PYTHON — see "Promptable selection setup")
python run_sim_grasp_test.py --execute --click 320,240        # select the target by clicking a pixel (observation.png coords)
```
```

- [ ] **Step 3: Verify**

```bash
grep -n "sam3\|SAM3\|Promptable selection" mujoco_grasp_sim/README.md
```
Expected: matches in the new setup section and the Run examples — at
least 5 lines.

- [ ] **Step 4: Commit**

```bash
git add mujoco_grasp_sim/README.md
git commit -m "Document promptable object selection"
```
