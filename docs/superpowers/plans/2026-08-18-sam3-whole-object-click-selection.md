# SAM3 Whole-Object Click Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix click-based object selection so it returns a full-object mask instead of just the locally-clicked face, by using a generic text-prompt instance detection pass (real whole-object segmentation) and using the click only to pick which detected instance the user meant.

**Architecture:** `PromptSelector.select(rgb, click=...)` currently synthesizes a tiny box around the click and feeds it to SAM3 as a *visual exemplar* — SAM3 treats this as "find things that look like this patch," which on a flat, uniformly-lit cuboid face returns only that face (measured IoU ~0.29 against the true full-object mask; got *worse*, not better, when the box was grown). A generic category text prompt (e.g. `"a block"`) instead makes SAM3 do genuine per-instance object detection, returning one full-object mask per instance in the scene (measured IoU 0.89-0.99 across all 3 objects in the test scene, reproducible across repeated calls). The fix adds `PromptSelector.click_to_select()`: run the text-prompt detection, then keep only the candidate(s) whose mask contains the clicked pixel — click becomes pure disambiguation, not a segmentation input. `run_sim_grasp_test.py --click` and `interactive_pick.py`'s click loop switch to this method; `--prompt`/`--box` modes are untouched (not reported broken, out of scope).

**Tech Stack:** Python, NumPy, Meta SAM 3 (subprocess, `sam3_torch` conda env) — no new dependencies, no changes to `sam3_worker.py` (the fix lives entirely in `sim_grasp/prompt_selector.py`'s orchestration layer).

## Global Constraints

- Repo has no automated test suite. Standalone assertion scripts live next to the module they test (e.g. `sim_grasp/test_resolve_real_label.py`) and are run directly: `PYTHONPATH=. python sim_grasp/test_x.py`, always from the `mujoco_grasp_sim/` directory, always with `/home/vivek/miniconda3/envs/cgn_torch/bin/python` (or `python` after `conda activate cgn_torch`) — plain `python`/`python3` on PATH lacks numpy.
- Commit messages are plain text — **never** add a "Co-Authored-By: Claude" trailer or similar.
- Any step that needs a live SAM3 call requires `export SAM3_PYTHON=/home/vivek/miniconda3/envs/sam3_torch/bin/python` (and `MUJOCO_GL=osmesa`, `GRASPGEN_PYTHON=...` for full pipeline runs) — see `mujoco_grasp_sim/README.md` "Promptable selection setup" / "GraspGen backend setup".
- Ground truth (`segmap`) must never influence which mask/pixels get used for prediction — only for after-the-fact bookkeeping (`resolve_real_label`'s existing docstring states this explicitly; do not violate it).
- All new/modified Python files must pass `get_errors` with no reported problems before moving to the next task.

---

### Task 1: `filter_selection_by_click` pure function + test

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/prompt_selector.py` (add function after `resolve_real_label`, before `resolve_sam3_python`, around line 47)
- Create: `mujoco_grasp_sim/sim_grasp/test_filter_selection_by_click.py`

**Interfaces:**
- Consumes: `SelectionResult` dataclass (already defined in `prompt_selector.py`: `masks: np.ndarray (K,H,W) bool`, `scores: np.ndarray (K,) float32`, `boxes: np.ndarray (K,4) float32`, properties `is_empty`, `is_ambiguous`).
- Produces: `filter_selection_by_click(result: SelectionResult, click: tuple[float, float]) -> SelectionResult` — used by Task 2's `click_to_select`.

- [ ] **Step 1: Write the failing test**

Create `mujoco_grasp_sim/sim_grasp/test_filter_selection_by_click.py`:

```python
"""Standalone check for prompt_selector.filter_selection_by_click — run
directly, no pytest (this codebase has no automated test suite)."""
import numpy as np
from sim_grasp.prompt_selector import SelectionResult, filter_selection_by_click

# Two candidates, only the second contains the click pixel (5,5) — keep only it
m0 = np.zeros((10, 10), dtype=bool)
m0[0:2, 0:2] = True
m1 = np.zeros((10, 10), dtype=bool)
m1[4:7, 4:7] = True  # contains (row=5, col=5)
result = SelectionResult(
    masks=np.stack([m0, m1]),
    scores=np.array([0.9, 0.5], dtype=np.float32),
    boxes=np.zeros((2, 4), dtype=np.float32),
)
kept = filter_selection_by_click(result, click=(5.0, 5.0))
assert len(kept.scores) == 1, 'expected exactly one candidate kept'
assert kept.scores[0] == 0.5, 'kept the wrong candidate'
assert bool(kept.masks[0][5, 5]), 'kept mask must contain the click pixel'

# Both candidates overlap the click pixel — keep both, order preserved
m2 = np.zeros((10, 10), dtype=bool)
m2[3:8, 3:8] = True  # also contains (5,5)
result2 = SelectionResult(
    masks=np.stack([m1, m2]),
    scores=np.array([0.7, 0.6], dtype=np.float32),
    boxes=np.zeros((2, 4), dtype=np.float32),
)
kept2 = filter_selection_by_click(result2, click=(5.0, 5.0))
assert len(kept2.scores) == 2, 'expected both overlapping candidates kept'
assert list(kept2.scores) == [0.7, 0.6], 'order must be preserved'

# No candidate contains the click pixel — result is empty
result3 = SelectionResult(
    masks=np.stack([m0]),
    scores=np.array([0.9], dtype=np.float32),
    boxes=np.zeros((1, 4), dtype=np.float32),
)
kept3 = filter_selection_by_click(result3, click=(5.0, 5.0))
assert kept3.is_empty, 'expected empty result when no mask contains the click'

# Already-empty input is returned as empty (no crash on zero candidates)
empty_in = SelectionResult(
    masks=np.zeros((0, 10, 10), dtype=bool),
    scores=np.zeros((0,), dtype=np.float32),
    boxes=np.zeros((0, 4), dtype=np.float32),
)
kept4 = filter_selection_by_click(empty_in, click=(5.0, 5.0))
assert kept4.is_empty, 'expected empty-in to stay empty'

print('All filter_selection_by_click checks passed.')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mujoco_grasp_sim && PYTHONPATH=. /home/vivek/miniconda3/envs/cgn_torch/bin/python sim_grasp/test_filter_selection_by_click.py`
Expected: FAIL with `ImportError: cannot import name 'filter_selection_by_click'`

- [ ] **Step 3: Write minimal implementation**

In `mujoco_grasp_sim/sim_grasp/prompt_selector.py`, add this function directly after `resolve_real_label` (before `def resolve_sam3_python`):

```python
def filter_selection_by_click(result: SelectionResult,
                              click: tuple[float, float]) -> SelectionResult:
    """Pure filter: keep only candidates in `result` whose mask contains
    the `click` pixel (x, y). No subprocess/model call here -- safe to
    unit-test directly with synthetic SelectionResult objects.

    Used to turn a category-wide detection pass (`select(rgb, prompt=...)`,
    which returns one mask per object instance in the scene) into a
    click-disambiguated selection: the click no longer needs to localize
    the object geometrically (SAM 3's click-as-box-exemplar mode only
    matches the locally clicked face's appearance -- measured IoU ~0.29
    against the true full-object mask, see prompt_selector click_to_select
    docstring), it only needs to land on the intended instance's mask."""
    if result.is_empty:
        return result
    x, y = int(click[0]), int(click[1])
    keep = [i for i in range(len(result.scores)) if result.masks[i][y, x]]
    if not keep:
        return SelectionResult(
            masks=np.zeros((0,) + result.masks.shape[1:], dtype=bool),
            scores=np.zeros((0,), dtype=np.float32),
            boxes=np.zeros((0, 4), dtype=np.float32))
    return SelectionResult(masks=result.masks[keep], scores=result.scores[keep],
                           boxes=result.boxes[keep])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mujoco_grasp_sim && PYTHONPATH=. /home/vivek/miniconda3/envs/cgn_torch/bin/python sim_grasp/test_filter_selection_by_click.py`
Expected: PASS — prints `All filter_selection_by_click checks passed.`

- [ ] **Step 5: Check for errors and commit**

Run `get_errors` on both files, confirm no problems, then:

```bash
git add mujoco_grasp_sim/sim_grasp/prompt_selector.py mujoco_grasp_sim/sim_grasp/test_filter_selection_by_click.py
git commit -m "Add filter_selection_by_click, a pure click-disambiguation filter"
```

---

### Task 2: `PromptSelector.click_to_select` method

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/prompt_selector.py` (add method to `PromptSelector` class, after `select`, around line 101)

**Interfaces:**
- Consumes: `PromptSelector.select(rgb, prompt=..., work_dir=...) -> SelectionResult` (existing); `filter_selection_by_click` (Task 1).
- Produces: `PromptSelector.click_to_select(rgb: np.ndarray, click: tuple[float, float], category: str = 'a block', work_dir: str | Path = '.') -> SelectionResult` — used by Task 3 and Task 4.

- [ ] **Step 1: Add the method**

In `mujoco_grasp_sim/sim_grasp/prompt_selector.py`, add this method to the `PromptSelector` class, directly after `select` (which currently ends the file):

```python
    def click_to_select(self, rgb: np.ndarray, click: tuple[float, float],
                        category: str = 'a block',
                        work_dir: str | Path = '.') -> SelectionResult:
        """Click-based selection that returns a full-object mask, not just
        the locally-clicked face/color: a click-as-box-exemplar prompt
        (the old `select(rgb, click=...)` path) makes SAM 3 match the
        clicked region's *appearance*, which on a uniformly-lit cuboid
        face returns only that face (measured IoU ~0.29 against the true
        full-object mask on a real repro case, 2026-08-18). Instead, this
        runs a category-wide text-prompt detection pass -- genuine
        per-instance object detection, measured IoU 0.89-0.99 against the
        true full-object masks for every instance in the same scene --
        then keeps only the instance(s) whose mask contains the click
        pixel. `category` currently defaults to box/cuboid wording since
        this project's scenes only spawn box-shaped objects (see
        ROADMAP.md); pass a different category if that changes."""
        result = self.select(rgb, prompt=category, work_dir=work_dir)
        return filter_selection_by_click(result, click)
```

- [ ] **Step 2: Manual smoke-test against the real repro case**

This method makes a live SAM3 call and can't be unit-tested without a model, matching how `select()` itself has no direct unit test in this codebase (exercised via real runs instead). Verify manually:

```bash
cd mujoco_grasp_sim
export MUJOCO_GL=osmesa SAM3_PYTHON=/home/vivek/miniconda3/envs/sam3_torch/bin/python
/home/vivek/miniconda3/envs/cgn_torch/bin/python -c "
from sim_grasp import SceneConfig, SceneGenerator, CameraModule
from sim_grasp.prompt_selector import PromptSelector
cfg = SceneConfig(seed=5)
cfg.calibration_file = 'calibration_result.yaml'
gen = SceneGenerator(cfg)
model, data = gen.generate()
cam = CameraModule(model, data, cam_name=cfg.cam_name, width=640, height=480)
rgb, depth, segmap, K, T = cam.capture(gen.object_body_ids)
sel = PromptSelector()
r = sel.click_to_select(rgb, (389.0, 273.0))
print('n_matches:', len(r.scores), 'mask_pixels:', int(r.masks[0].sum()) if len(r.scores) else None)
"
```

Expected: `n_matches: 1`, `mask_pixels` around 1500 (the object's true full extent is 1537px in this scene) — NOT ~450 (the old top-face-only size).

- [ ] **Step 3: Check for errors and commit**

Run `get_errors` on `prompt_selector.py`, confirm no problems, then:

```bash
git add mujoco_grasp_sim/sim_grasp/prompt_selector.py
git commit -m "Add PromptSelector.click_to_select for whole-object click selection"
```

---

### Task 3: Wire `run_sim_grasp_test.py --click` to `click_to_select`

**Files:**
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py` (CLI args around line 253; click-handling block around lines 334-341)

**Interfaces:**
- Consumes: `PromptSelector.click_to_select` (Task 2).
- Produces: `--category` CLI flag, used identically in Task 4.

- [ ] **Step 1: Add the `--category` CLI flag**

In `mujoco_grasp_sim/run_sim_grasp_test.py`, find this existing block (around line 253):

```python
    ap.add_argument('--sam3-python', default=None,
```

Add a new argument directly before it:

```python
    ap.add_argument('--category', type=str, default='a block',
                    help='text category for click-based whole-object detection '
                         "(e.g. 'a block', 'a cube'); only applies to --click, "
                         'not --prompt/--box (default: a block)')
    ap.add_argument('--sam3-python', default=None,
```

- [ ] **Step 2: Switch the click path to `click_to_select`**

Find this block (around line 334):

```python
    if args.prompt or args.click or args.box:
        from sim_grasp.prompt_selector import PromptSelector, resolve_real_label
        selector = PromptSelector(sam3_python=args.sam3_python)
        click_xy = tuple(float(v) for v in args.click.split(',')) if args.click else None
        box_xyxy = tuple(float(v) for v in args.box.split(',')) if args.box else None
        result = selector.select(rgb, prompt=args.prompt, click=click_xy, box=box_xyxy)
```

Replace the last line with:

```python
    if args.prompt or args.click or args.box:
        from sim_grasp.prompt_selector import PromptSelector, resolve_real_label
        selector = PromptSelector(sam3_python=args.sam3_python)
        click_xy = tuple(float(v) for v in args.click.split(',')) if args.click else None
        box_xyxy = tuple(float(v) for v in args.box.split(',')) if args.box else None
        if click_xy is not None:
            result = selector.click_to_select(rgb, click_xy, category=args.category)
        else:
            result = selector.select(rgb, prompt=args.prompt, box=box_xyxy)
```

(Everything below this — `is_empty`, `is_ambiguous`/`prompt_index` disambiguation, `resolve_real_label` — is unchanged; `click_to_select` returns the same `SelectionResult` type.)

- [ ] **Step 3: Regression-test the exact reported failure case**

```bash
cd mujoco_grasp_sim
export MUJOCO_GL=osmesa SAM3_PYTHON=/home/vivek/miniconda3/envs/sam3_torch/bin/python GRASPGEN_PYTHON=/home/vivek/miniconda3/envs/graspgen_torch/bin/python
/home/vivek/miniconda3/envs/cgn_torch/bin/python run_sim_grasp_test.py --seed 5 --execute --backend graspgen --click 389,273 --no-vis --top-k 3
```

Expected: `[prompt] resolved to object 2, score ...` (same object as before), and — check `output/<latest>/observation.png` or the printed feasibility/grasp stats — grasps should no longer be uniformly "pushing" poses. This is a manual/visual check (no automated assertion for full pipeline behavior in this repo).

- [ ] **Step 4: Check for errors and commit**

Run `get_errors` on `run_sim_grasp_test.py`, confirm no problems, then:

```bash
git add mujoco_grasp_sim/run_sim_grasp_test.py
git commit -m "Use click_to_select for --click in run_sim_grasp_test.py"
```

---

### Task 4: Wire `interactive_pick.py`'s click loop to `click_to_select`

**Files:**
- Modify: `mujoco_grasp_sim/interactive_pick.py` (CLI args around line 44; click loop around line 96)

**Interfaces:**
- Consumes: `PromptSelector.click_to_select` (Task 2).

- [ ] **Step 1: Add the `--category` CLI flag**

In `mujoco_grasp_sim/interactive_pick.py`, find:

```python
    ap.add_argument('--click-radius-px', type=int, default=15,
                    help='half-width of the box synthesized around your click for SAM 3')
```

Add directly after it:

```python
    ap.add_argument('--category', type=str, default='a block',
                    help='text category for whole-object detection '
                         "(e.g. 'a block', 'a cube'); default: a block")
```

- [ ] **Step 2: Switch the click loop to `click_to_select`**

Find this block (around line 96):

```python
        print(f'[interactive] click at {xy} — running SAM 3...')
        result = viewer.run_blocking(
            rgb, lambda: selector.select(rgb, click=(float(xy[0]), float(xy[1]))),
            message='Running SAM 3 segmentation...')
```

Replace with:

```python
        print(f'[interactive] click at {xy} — running SAM 3...')
        result = viewer.run_blocking(
            rgb, lambda: selector.click_to_select(
                rgb, (float(xy[0]), float(xy[1])), category=args.category),
            message='Running SAM 3 segmentation...')
```

(`--click-radius-px` stays defined and passed to `PromptSelector.__init__` — it's now unused by the click path but harmless to leave; other callers of `select(rgb, click=...)` may still rely on it.)

- [ ] **Step 3: Manual end-to-end verification**

```bash
cd mujoco_grasp_sim
export MUJOCO_GL=osmesa SAM3_PYTHON=/home/vivek/miniconda3/envs/sam3_torch/bin/python GRASPGEN_PYTHON=/home/vivek/miniconda3/envs/graspgen_torch/bin/python DISPLAY=:0
/home/vivek/miniconda3/envs/cgn_torch/bin/python interactive_pick.py --seed 5 --backend graspgen
```

Click directly on the flat top face of a cuboid (the exact scenario that previously produced a top-face-only mask). Expected: the mask overlay now highlights the *entire* visible cuboid (all faces), not just the clicked face. Ask the human operator to confirm this visually (GUI verification — this repo's established pattern since there's no automated way to inspect a cv2 window's contents).

- [ ] **Step 4: Check for errors and commit**

Run `get_errors` on `interactive_pick.py`, confirm no problems, then:

```bash
git add mujoco_grasp_sim/interactive_pick.py
git commit -m "Use click_to_select for whole-object masks in interactive_pick.py"
```

---

### Task 5: Update README documentation

**Files:**
- Modify: `mujoco_grasp_sim/README.md` (the "Promptable selection" and "Interactive live pick" sections)

**Interfaces:**
- Consumes: nothing new; documents Tasks 1-4's behavior change.
- Produces: nothing consumed by other tasks (last task in this plan).

- [ ] **Step 1: Document the fix and the `--category` flag**

Find the "Promptable selection" section's known-limitation paragraph (the one citing "3/5 correct selections (60%), mean IoU 0.733") and add a short note directly after it:

```markdown
**2026-08-18 update:** click-based selection (`--click`) no longer uses the
click as a SAM 3 box exemplar (that mode only matches the locally-clicked
face's appearance -- e.g. clicking a cuboid's top face returned just that
face, IoU ~0.29 against the true object). It now runs a category-wide text
detection pass (`--category`, default `'a block'`) and uses the click only
to pick which detected instance you meant, giving full-object masks
(IoU 0.89-0.99 measured across a real scene). `--category` assumes
box/cuboid-shaped objects, matching this project's current scene
generation (see `ROADMAP.md`) -- pass a different category if that changes.
`--prompt`/`--box` selection modes are unchanged.
```

Then find the "Interactive live pick" section's paragraph about the pre-warmed CGN predictor / retry logic, and add one line after it:

```markdown
Click-based selection uses the same whole-object category detection as
`run_sim_grasp_test.py --click` (see "Promptable selection" above,
2026-08-18 update) -- pass `--category` to override the default `'a block'`
if your scene uses different-shaped objects.
```

- [ ] **Step 2: Commit**

```bash
git add mujoco_grasp_sim/README.md
git commit -m "Document whole-object click selection and --category flag"
```
