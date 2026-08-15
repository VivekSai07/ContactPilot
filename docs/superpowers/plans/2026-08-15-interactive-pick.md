# Interactive Live Pick Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a new interactive entry point (`interactive_pick.py`) that opens a live camera-feed window, lets the user click an object to select it via SAM 3, shows the segmented mask for confirmation, then runs grasp prediction (CGN or GraspGen) and executes the pick while the window keeps updating live — closing the loop idea 2 was originally motivated by.

**Architecture:** A new `LiveViewer` (cv2-based window + mouse-click capture + mask-overlay drawing) sits on top of the existing, unmodified pipeline (`SceneGenerator`, `CameraModule`, `PromptSelector`, `GraspGenPredictor`/`ContactGraspNetPredictor`, `GraspFeasibilityChecker`, `GraspExecutor`). `GraspExecutor` gets one small additive change (an optional per-frame callback) so execution can be watched live, not just saved to a GIF afterward. The real-object-label lookup already implemented inline in `run_sim_grasp_test.py` (from the promptable-selection sub-project's final-review fix) is extracted into a small shared function so this new script doesn't duplicate it.

**Tech Stack:** Python, OpenCV (`cv2`, already an existing `cgn_torch` dependency — nothing new to install), MuJoCo, existing `sim_grasp` modules.

**Spec:** `docs/superpowers/specs/2026-08-15-interactive-pick-design.md`

## Global Constraints

- `MUJOCO_GL=osmesa` must be set in the environment before running anything in this plan that touches `CameraModule`/`mujoco.Renderer` on this WSL2 machine — same requirement as every other script in this project. Do NOT attempt to switch GL backends mid-process; `osmesa`'s rendered frames are plain numpy arrays and display in `cv2` exactly like any other backend's output, so there is no reason to use a different backend anywhere in this plan.
- `SAM3_PYTHON`/`GRASPGEN_PYTHON` env vars (or `--sam3-python`/`--graspgen-python` CLI overrides) resolve those interpreters — **never** `sys.executable`. This plan does not introduce any new interpreter-resolution code; it reuses `resolve_sam3_python()`/`resolve_graspgen_python()`/`GraspGenPredictor`/`PromptSelector` exactly as they exist today.
- Zero changes to `GraspPredictor`, `feasibility.py`'s `GraspFeasibilityChecker`, `graspgen_predictor.py`, `prompt_selector.py`'s `PromptSelector`/`SelectionResult`/`resolve_sam3_python`, or `sam3_worker.py`/`graspgen_worker.py`/`cgn_worker.py` — every task in this plan either adds new files or makes small, additive (non-breaking) modifications to `executor.py` and `run_sim_grasp_test.py`.
- No automated test suite exists in this repo (documented convention) — verification is via standalone assertion scripts (matching `test_color_utils.py`'s convention) for pure functions, and manual smoke-testing for anything involving a window or live hardware-in-the-loop behavior.
- Commit messages are plain — **do not add a `Co-Authored-By: Claude` trailer to any commit** (corrected twice already in this project's history).

---

### Task 1: `resolve_real_label` — extract the real-object-label lookup into a shared function

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/test_resolve_real_label.py`
- Modify: `mujoco_grasp_sim/sim_grasp/prompt_selector.py`

**Interfaces:**
- Produces: `resolve_real_label(gt_segmap: np.ndarray, mask: np.ndarray) -> int | None` in `sim_grasp/prompt_selector.py`. Task 2 (refactoring `run_sim_grasp_test.py`) and Task 5 (`interactive_pick.py`) both consume this directly.

This logic already exists, inlined, in `run_sim_grasp_test.py` (added by the promptable-selection sub-project's final-review fix, commit `dbb50eb`) — this task only extracts it into a reusable, independently-testable function. No behavior change.

- [x] **Step 1: Write the failing test**

Create `mujoco_grasp_sim/sim_grasp/test_resolve_real_label.py`:

```python
"""Standalone check for prompt_selector.resolve_real_label — run directly,
no pytest (this codebase has no automated test suite)."""
import numpy as np
from sim_grasp.prompt_selector import resolve_real_label

# Mask entirely inside a single real object's region
gt = np.zeros((10, 10), dtype=np.int32)
gt[2:5, 2:5] = 3
mask = np.zeros((10, 10), dtype=bool)
mask[3:4, 3:4] = True
assert resolve_real_label(gt, mask) == 3, 'single-object overlap failed'

# Mask straddling two objects — majority (by pixel count) wins
gt2 = np.zeros((10, 10), dtype=np.int32)
gt2[0:5, :] = 1
gt2[5:10, :] = 2
mask2 = np.zeros((10, 10), dtype=bool)
mask2[3:5, :] = True   # rows 3,4 -> all label 1 (20 px)
mask2[5, :] = True     # row 5 -> label 2 (10 px, minority)
assert resolve_real_label(gt2, mask2) == 1, 'majority-overlap failed'

# Mask entirely over background (no real object underneath)
gt3 = np.zeros((10, 10), dtype=np.int32)
mask3 = np.zeros((10, 10), dtype=bool)
mask3[0:2, 0:2] = True
assert resolve_real_label(gt3, mask3) is None, 'background-only case failed'

print('All resolve_real_label checks passed.')
```

- [x] **Step 2: Run it to verify it fails (function doesn't exist yet)**

```bash
cd mujoco_grasp_sim
python sim_grasp/test_resolve_real_label.py
```
Expected: `ImportError: cannot import name 'resolve_real_label' from 'sim_grasp.prompt_selector'` (or similar).

- [x] **Step 3: Add `resolve_real_label` to `prompt_selector.py`**

Add this function to `mujoco_grasp_sim/sim_grasp/prompt_selector.py`, right after the `SelectionResult` dataclass and before `resolve_sam3_python`:

```python
def resolve_real_label(gt_segmap: np.ndarray, mask: np.ndarray) -> int | None:
    """Real object label the resolved `mask` (H,W bool) actually overlaps
    most, per `gt_segmap` (H,W int, 0 = background). Returns None if the
    mask overlaps no real object (entirely over background).

    Sim-only bookkeeping: maps a SAM 3 mask (real perception, no ground
    truth involved in producing it) onto the ground-truth body-name/
    success-detection machinery the rest of the pipeline uses. A real
    camera deployment has no ground-truth segmap to compare against;
    success there would be graded some other way. Ground truth here is
    used ONLY to label an already-SAM3-selected mask, never to influence
    which mask/object gets selected in the first place."""
    overlap_labels = gt_segmap[mask]
    overlap_labels = overlap_labels[overlap_labels > 0]
    if len(overlap_labels) == 0:
        return None
    return int(np.bincount(overlap_labels.astype(int)).argmax())
```

- [x] **Step 4: Run test to verify it passes**

```bash
cd mujoco_grasp_sim
python sim_grasp/test_resolve_real_label.py
```
Expected: `All resolve_real_label checks passed.`

- [x] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/prompt_selector.py mujoco_grasp_sim/sim_grasp/test_resolve_real_label.py
git commit -m "Extract resolve_real_label as a shared, testable function"
```

---

### Task 2: Refactor `run_sim_grasp_test.py` to use `resolve_real_label`

**Files:**
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py`

**Interfaces:**
- Consumes: `resolve_real_label` from Task 1.

Pure refactor — replaces the inline label-lookup block with a call to the now-shared function. Behavior must be identical; this task is a regression check, not new functionality.

- [x] **Step 1: Update the import**

Current (`run_sim_grasp_test.py`, inside the `if args.prompt or args.click or args.box:` block):
```python
    if args.prompt or args.click or args.box:
        from sim_grasp.prompt_selector import PromptSelector
```

New:
```python
    if args.prompt or args.click or args.box:
        from sim_grasp.prompt_selector import PromptSelector, resolve_real_label
```

- [x] **Step 2: Replace the inline label-lookup block**

Current:
```python
        mask = result.masks[idx]
        # Determine which real object this mask actually overlaps — sim-only
        # bookkeeping that maps a SAM 3 mask (real perception) onto the
        # ground-truth body-name/success-detection machinery below. A real
        # camera deployment has no ground-truth segmap to compare against;
        # success there would be graded some other way. SAM 3's own mask
        # still determines WHICH pixels are selected — no ground truth is
        # used to pick the target, only to label it correctly downstream.
        overlap_labels = segmap[mask]
        overlap_labels = overlap_labels[overlap_labels > 0]
        if len(overlap_labels) == 0:
            sys.exit('[prompt] resolved mask does not overlap any known object')
        real_label = int(np.bincount(overlap_labels.astype(int)).argmax())
        new_segmap = np.zeros(rgb.shape[:2], dtype=segmap.dtype)
        new_segmap[mask] = real_label
        segmap = new_segmap
        print(f'[prompt] resolved to object {real_label}, score {float(result.scores[idx]):.3f}')
```

New:
```python
        mask = result.masks[idx]
        real_label = resolve_real_label(segmap, mask)
        if real_label is None:
            sys.exit('[prompt] resolved mask does not overlap any known object')
        new_segmap = np.zeros(rgb.shape[:2], dtype=segmap.dtype)
        new_segmap[mask] = real_label
        segmap = new_segmap
        print(f'[prompt] resolved to object {real_label}, score {float(result.scores[idx]):.3f}')
```

- [x] **Step 3: Regression smoke test**

```bash
cd mujoco_grasp_sim
export MUJOCO_GL=osmesa
export GRASPGEN_PYTHON=/home/vivek/miniconda3/envs/graspgen_torch/bin/python
export SAM3_PYTHON=/home/vivek/miniconda3/envs/sam3_torch/bin/python
/home/vivek/miniconda3/envs/cgn_torch/bin/python run_sim_grasp_test.py --seed 5 --execute --backend graspgen --click 320,240 --no-vis
```
Expected: identical behavior to before this refactor — `[prompt] resolved to object N, score X.XXX` printed with a real object label (not always `1`), pipeline proceeds exactly as it did prior to this task.

- [x] **Step 4: Commit**

```bash
git add mujoco_grasp_sim/run_sim_grasp_test.py
git commit -m "Use shared resolve_real_label in run_sim_grasp_test.py"
```

---

### Task 3: `LiveViewer` — the interactive window

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/live_viewer.py`
- Create: `mujoco_grasp_sim/sim_grasp/test_live_viewer_overlay.py`

**Interfaces:**
- Produces: `compose_mask_overlay(rgb, mask, color, alpha) -> np.ndarray` (pure function, no window) and `LiveViewer` class, importable as `from sim_grasp.live_viewer import LiveViewer, compose_mask_overlay`. Task 5 (`interactive_pick.py`) consumes `LiveViewer` directly: `show_frame(rgb)`, `wait_for_click(rgb) -> (x, y) | None`, `show_mask_overlay(rgb, mask)`, `wait_for_confirm() -> bool`, `close()`.

The mask-overlay compositing math is factored into a standalone pure
function (per the spec's testing plan: "`LiveViewer`'s non-interactive
pieces ... testable directly with synthetic frames/masks, no window
needed") so it's testable without opening a `cv2` window — `LiveViewer`'s
constructor always opens one, so anything bundled inside a `LiveViewer`
method can't be exercised standalone.

**This task also includes the spec's flagged risk check**: confirm `cv2.imshow` actually opens a visible window on this WSL2 machine before writing the rest of the class — if it doesn't, escalate rather than guessing around it (same "retire the biggest unknown first" principle as the GraspGen/SAM 3 sub-projects' own Task 1s).

- [x] **Step 1: Write the failing test for the pure overlay function**

Create `mujoco_grasp_sim/sim_grasp/test_live_viewer_overlay.py`:

```python
"""Standalone check for live_viewer.compose_mask_overlay — run directly, no
pytest (this codebase has no automated test suite), no window needed."""
import numpy as np
from sim_grasp.live_viewer import compose_mask_overlay

rgb = np.zeros((10, 10, 3), dtype=np.uint8)
mask = np.zeros((10, 10), dtype=bool)
mask[2:5, 2:5] = True

# Outside the mask: pixel unchanged
out = compose_mask_overlay(rgb, mask, color=(255, 0, 0), alpha=0.5)
assert tuple(out[0, 0]) == (0, 0, 0), 'unmasked pixel should be unchanged'

# Inside the mask: 50/50 blend of black background and red overlay color
assert tuple(out[3, 3]) == (127, 0, 0) or tuple(out[3, 3]) == (128, 0, 0), \
    f'masked pixel should be ~50% red, got {tuple(out[3, 3])}'

# alpha=0 leaves the image untouched everywhere
out0 = compose_mask_overlay(rgb, mask, color=(255, 0, 0), alpha=0.0)
assert np.array_equal(out0, rgb), 'alpha=0 should not change the image'

# alpha=1 fully replaces masked pixels with the overlay color
out1 = compose_mask_overlay(rgb, mask, color=(255, 0, 0), alpha=1.0)
assert tuple(out1[3, 3]) == (255, 0, 0), 'alpha=1 should fully replace masked pixels'
assert tuple(out1[0, 0]) == (0, 0, 0), 'alpha=1 should not touch unmasked pixels'

print('All compose_mask_overlay checks passed.')
```

- [x] **Step 2: Run it to verify it fails (module doesn't exist yet)**

```bash
cd mujoco_grasp_sim
python sim_grasp/test_live_viewer_overlay.py
```
Expected: `ModuleNotFoundError: No module named 'sim_grasp.live_viewer'` (or similar import error).

- [x] **Step 3: Confirm `cv2.imshow` works on this machine**

```bash
cd mujoco_grasp_sim
/home/vivek/miniconda3/envs/cgn_torch/bin/python -c "
import cv2
import numpy as np
img = (np.random.rand(240, 320, 3) * 255).astype('uint8')
cv2.imshow('smoke test', img)
print('window created — press any key in the window (or wait 3s) to close')
cv2.waitKey(3000)
cv2.destroyAllWindows()
print('OK: cv2 window smoke test passed')
"
```
Expected: a window titled "smoke test" actually appears on screen (via WSLg, same display path `mujoco.viewer.launch()` already uses successfully in this project) showing random-noise colors, and the script prints `OK: cv2 window smoke test passed` after it closes. If no window appears, STOP and report BLOCKED with the exact error/behavior — do not proceed by guessing at a workaround.

- [x] **Step 4: Write `live_viewer.py`**

Create `mujoco_grasp_sim/sim_grasp/live_viewer.py`:

```python
"""LiveViewer — a cv2-based window for interactive pick: shows the live
camera feed, captures mouse clicks, overlays SAM 3 mask results, and waits
for a keypress to confirm or retry a selection.

BGR vs RGB: OpenCV's imshow expects BGR pixel order; this project's cameras
(sim_grasp/camera.py) produce RGB. All display methods here convert
RGB -> BGR internally, so callers always pass RGB, matching every other
module in sim_grasp — no OpenCV-specific color convention leaks out.
"""
import cv2
import numpy as np


def compose_mask_overlay(rgb: np.ndarray, mask: np.ndarray,
                         color: tuple[int, int, int] = (255, 80, 40),
                         alpha: float = 0.45) -> np.ndarray:
    """Pure compositing: `rgb` (H,W,3 uint8) with `mask` (H,W bool) blended
    toward `color` (RGB tuple) by `alpha` (0 = unchanged, 1 = fully
    replaced). No window/display side effects — safe to unit-test directly
    with synthetic arrays."""
    out = np.ascontiguousarray(rgb).astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


class LiveViewer:
    def __init__(self, window_name: str = 'ContactPilot -- interactive pick'):
        self.window_name = window_name
        self._click_xy: tuple[int, int] | None = None
        self._closed = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_xy = (x, y)

    def _pump(self, delay_ms: int = 1) -> int:
        """Process window events once; returns the pressed key code (-1 if
        none). Also detects the window being closed via its X button."""
        key = cv2.waitKey(delay_ms) & 0xFF
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 0
        if visible < 1:
            self._closed = True
        return key

    @property
    def closed(self) -> bool:
        return self._closed

    def show_frame(self, rgb: np.ndarray) -> None:
        """Display an RGB frame; pumps the event loop once (non-blocking)."""
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, bgr)
        self._pump()

    def wait_for_click(self, rgb: np.ndarray) -> tuple[int, int] | None:
        """Display rgb and block (repeatedly re-showing it, pumping the
        window) until the user clicks a pixel or closes the window. Returns
        (x, y) pixel coordinates, or None if the window was closed."""
        self._click_xy = None
        while not self._closed:
            self.show_frame(rgb)
            if self._click_xy is not None:
                xy, self._click_xy = self._click_xy, None
                return xy
        return None

    def show_mask_overlay(self, rgb: np.ndarray, mask: np.ndarray,
                          color: tuple[int, int, int] = (255, 80, 40),
                          alpha: float = 0.45) -> None:
        """Display rgb with `mask` (H,W bool) highlighted in a translucent
        color (given as an RGB tuple)."""
        self.show_frame(compose_mask_overlay(rgb, mask, color=color, alpha=alpha))

    def wait_for_confirm(self) -> bool:
        """Block until Enter/Space confirms or Esc/'c' cancels-and-retries,
        or the window closes (treated as cancel). Returns True if
        confirmed."""
        while not self._closed:
            key = self._pump(delay_ms=30)
            if key in (13, 32):        # Enter, Space
                return True
            if key in (27, ord('c')):  # Esc, 'c'
                return False
        return False

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
```

- [x] **Step 5: Run the pure-function test to verify it now passes**

```bash
cd mujoco_grasp_sim
python sim_grasp/test_live_viewer_overlay.py
```
Expected: `All compose_mask_overlay checks passed.`

- [x] **Step 6: Manual smoke test of the class itself**

```bash
cd mujoco_grasp_sim
/home/vivek/miniconda3/envs/cgn_torch/bin/python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from sim_grasp.live_viewer import LiveViewer

img = (np.random.rand(240, 320, 3) * 255).astype('uint8')
v = LiveViewer('manual test')
print('Click anywhere in the window...')
xy = v.wait_for_click(img)
print('clicked at:', xy)
mask = np.zeros((240, 320), dtype=bool)
if xy:
    x, y = xy
    mask[max(0,y-20):y+20, max(0,x-20):x+20] = True
v.show_mask_overlay(img, mask)
print('Press Enter to confirm, Esc to cancel...')
confirmed = v.wait_for_confirm()
print('confirmed:', confirmed)
v.close()
"
```
Expected: a window opens, your click is captured and printed, a highlighted box appears around the click point, and pressing Enter/Esc prints the correct `confirmed` value.

- [x] **Step 7: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/live_viewer.py mujoco_grasp_sim/sim_grasp/test_live_viewer_overlay.py
git commit -m "Add LiveViewer for interactive click/confirm/live-frame display"
```

---

### Task 4: `GraspExecutor` optional live-frame callback

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/executor.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `GraspExecutor.__init__`'s new `on_frame` parameter. Task 5 (`interactive_pick.py`) passes `LiveViewer.show_frame` as this callback.

Purely additive — defaults to `None`, so every existing caller (`run_sim_grasp_test.py`'s two `GraspExecutor(...)` construction sites) is completely unaffected.

- [x] **Step 1: Add the `on_frame` parameter**

Current (`mujoco_grasp_sim/sim_grasp/executor.py`, `__init__`):
```python
    def __init__(self, model, data, camera_module=None, record_gif=False,
                 record_dir=None, gif_frame_interval=0.08):
        self.model, self.data = model, data
        self.ik = DiffIK(model)
        self.cam = camera_module       # reused for GIF recording (optional)
        self.record = record_gif
        # With record_dir, frames are streamed to disk as JPEGs and the GIF is
        # assembled at the end — peak RAM stays flat (essential when CGN stays
        # loaded during multi-round pick-and-place on 8 GB machines).
        self.record_dir = Path(record_dir) if (record_gif and record_dir) else None
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
        self.frames: list[np.ndarray] = []
        self._n_frames = 0
        self._frame_interval = gif_frame_interval  # sim seconds between frames
        self._last_frame_t = -1.0
```

New:
```python
    def __init__(self, model, data, camera_module=None, record_gif=False,
                 record_dir=None, gif_frame_interval=0.08,
                 on_frame: 'Callable[[np.ndarray], None] | None' = None):
        self.model, self.data = model, data
        self.ik = DiffIK(model)
        self.cam = camera_module       # reused for GIF recording (optional)
        self.record = record_gif
        # With record_dir, frames are streamed to disk as JPEGs and the GIF is
        # assembled at the end — peak RAM stays flat (essential when CGN stays
        # loaded during multi-round pick-and-place on 8 GB machines).
        self.record_dir = Path(record_dir) if (record_gif and record_dir) else None
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
        self.frames: list[np.ndarray] = []
        self._n_frames = 0
        self._frame_interval = gif_frame_interval  # sim seconds between frames
        self._last_frame_t = -1.0
        # Optional live-display hook (e.g. LiveViewer.show_frame) — called
        # with the same frame captured for the GIF, at the same cadence.
        # Requires record_gif=True (the cadence/frame-capture logic lives in
        # _maybe_record below); this does not change GIF-saving behavior.
        self.on_frame = on_frame
```

- [x] **Step 2: Add the `Callable` import**

Current (top of `executor.py`):
```python
import shutil
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
```

New:
```python
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
```

- [x] **Step 3: Call `on_frame` from `_maybe_record`**

Current:
```python
    def _maybe_record(self):
        if self.record and self.cam is not None and \
                self.data.time - self._last_frame_t >= self._frame_interval:
            if self._n_frames >= GIF_MAX_FRAMES:
                print(f'[executor] GIF frame cap ({GIF_MAX_FRAMES}) reached '
                      '— recording stopped')
                self.record = False
                return
            frame = np.ascontiguousarray(
                self.cam.render_rgb()[::GIF_DOWNSAMPLE, ::GIF_DOWNSAMPLE])
            if self.record_dir is not None:
                imageio.imwrite(self.record_dir / f'{self._n_frames:05d}.jpg', frame)
            else:
                self.frames.append(frame)
            self._n_frames += 1
            self._last_frame_t = self.data.time
```

New:
```python
    def _maybe_record(self):
        if self.record and self.cam is not None and \
                self.data.time - self._last_frame_t >= self._frame_interval:
            if self._n_frames >= GIF_MAX_FRAMES:
                print(f'[executor] GIF frame cap ({GIF_MAX_FRAMES}) reached '
                      '— recording stopped')
                self.record = False
                return
            frame = np.ascontiguousarray(
                self.cam.render_rgb()[::GIF_DOWNSAMPLE, ::GIF_DOWNSAMPLE])
            if self.record_dir is not None:
                imageio.imwrite(self.record_dir / f'{self._n_frames:05d}.jpg', frame)
            else:
                self.frames.append(frame)
            self._n_frames += 1
            self._last_frame_t = self.data.time
            if self.on_frame is not None:
                self.on_frame(frame)
```

- [x] **Step 4: Regression smoke test — confirm existing (non-interactive) execution is unaffected**

```bash
cd mujoco_grasp_sim
export MUJOCO_GL=osmesa
export GRASPGEN_PYTHON=/home/vivek/miniconda3/envs/graspgen_torch/bin/python
/home/vivek/miniconda3/envs/cgn_torch/bin/python run_sim_grasp_test.py --seed 5 --execute --backend graspgen --no-vis
```
Expected: identical behavior to before this change (this run doesn't pass `on_frame`, so nothing new should happen) — `PICK SUCCESS`/`FAIL` printed, `execution.gif` written, same as always.

- [x] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/executor.py
git commit -m "Add optional live-frame callback to GraspExecutor"
```

---

### Task 5: `interactive_pick.py` — the orchestration script

**Files:**
- Create: `mujoco_grasp_sim/interactive_pick.py`

**Interfaces:**
- Consumes: `LiveViewer` (Task 3), `GraspExecutor`'s `on_frame` (Task 4), `resolve_real_label` (Task 1), and everything else unchanged (`SceneGenerator`, `CameraModule`, `PromptSelector`, `GraspGenPredictor`, `ContactGraspNetPredictor`, `GraspFeasibilityChecker`, `filter_feasible` — imported directly from `run_sim_grasp_test.py`, since it's a small self-contained function defined there and this avoids duplicating it).

This script deliberately does LESS than `run_sim_grasp_test.py`: single scene, single camera (default `calibrated`, no `--camera fused`), single grasp attempt on the resolved object (no top-k retry loop, no `--pick-all`) — matching the design spec's "no new flag semantics, just fewer of them, since this script is inherently single-object/single-run."

- [x] **Step 1: Write `interactive_pick.py`**

Create `mujoco_grasp_sim/interactive_pick.py`:

```python
"""Interactive live pick: click an object in a live camera window, SAM 3
segments it, confirm the mask, then watch GraspGen/CGN pick it — live, in
the same window.

Usage:
    export MUJOCO_GL=osmesa
    export GRASPGEN_PYTHON=/path/to/graspgen_torch/bin/python
    export SAM3_PYTHON=/path/to/sam3_torch/bin/python
    python interactive_pick.py --seed 5 --backend graspgen

Controls: click an object in the window to select it. Once SAM 3 resolves
a mask, it's highlighted — press Enter/Space to confirm and execute the
pick, or Esc/'c' to try a different click. Close the window at any point
to cancel.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_sim_grasp_test import filter_feasible
from sim_grasp import (SceneConfig, SceneGenerator, CameraModule,
                       ContactGraspNetPredictor)
from sim_grasp.live_viewer import LiveViewer
from sim_grasp.prompt_selector import PromptSelector, resolve_real_label


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=None, help='randomization seed')
    ap.add_argument('--backend', choices=['cgn', 'graspgen'], default='cgn',
                    help='grasp-prediction backend')
    ap.add_argument('--graspgen-python', default=None,
                    help='path to graspgen_torch\'s python; overrides GRASPGEN_PYTHON')
    ap.add_argument('--sam3-python', default=None,
                    help='path to sam3_torch\'s python; overrides SAM3_PYTHON')
    ap.add_argument('--click-radius-px', type=int, default=15,
                    help='half-width of the box synthesized around your click for SAM 3')
    ap.add_argument('--save-dir', default=None, help='output dir (default: output/<timestamp>)')
    args = ap.parse_args()

    save_dir = Path(args.save_dir) if args.save_dir else \
        Path(__file__).parent / 'output' / time.strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f'[run] outputs -> {save_dir}')

    cfg = SceneConfig(seed=args.seed)
    default_cal = Path(__file__).parent / 'calibration_result.yaml'
    if default_cal.exists():
        cfg.calibration_file = str(default_cal)
        print(f'[camera] using real eye-to-hand calibration: {default_cal}')
    else:
        print(f'[camera] using generic look-at camera: pos={cfg.cam_pos}, target={cfg.cam_target}')
    gen = SceneGenerator(cfg)
    model, data = gen.generate()
    on_table = gen.objects_on_table()
    print(f'[scene] {len(gen.object_names)} objects spawned, {len(on_table)} on table')

    cam = CameraModule(model, data, cam_name=cfg.cam_name, width=640, height=480)
    rgb, depth, segmap, K, T_world_cam = cam.capture(gen.object_body_ids)

    viewer = LiveViewer()
    selector = PromptSelector(sam3_python=args.sam3_python, click_radius_px=args.click_radius_px)

    real_label = None
    mask = None
    print('[interactive] click an object in the window to select it '
          '(close the window to cancel)...')
    while real_label is None:
        xy = viewer.wait_for_click(rgb)
        if xy is None:
            print('[interactive] window closed — cancelled, nothing executed')
            viewer.close()
            cam.close()
            return
        print(f'[interactive] click at {xy} — running SAM 3...')
        result = selector.select(rgb, click=(float(xy[0]), float(xy[1])))
        if result.is_empty:
            print('[interactive] no object found at that point — click again')
            continue
        mask = result.masks[0]
        viewer.show_mask_overlay(rgb, mask)
        print(f'[interactive] SAM 3 match, score {float(result.scores[0]):.3f} — '
              'Enter/Space to confirm, Esc/c to retry')
        if not viewer.wait_for_confirm():
            if viewer.closed:
                print('[interactive] window closed — cancelled, nothing executed')
                viewer.close()
                cam.close()
                return
            print('[interactive] retry — click an object in the window again...')
            continue
        real_label = resolve_real_label(segmap, mask)
        if real_label is None:
            print('[interactive] resolved mask does not overlap any known '
                  'object — click again')
            continue

    new_segmap = np.zeros(rgb.shape[:2], dtype=segmap.dtype)
    new_segmap[mask] = real_label
    print(f'[interactive] confirmed: object {real_label}')

    if args.backend == 'graspgen':
        print('[graspgen] loading GraspGen...')
        from sim_grasp import GraspGenPredictor
        predictor = GraspGenPredictor(graspgen_python=args.graspgen_python)
    else:
        print('[cgn] loading Contact-GraspNet...')
        predictor = ContactGraspNetPredictor(forward_passes=3)
    t0 = time.time()
    pred = predictor.predict(depth, K, rgb=rgb, segmap=new_segmap)
    print(f'[{args.backend}] {pred.num_grasps} grasps in {time.time() - t0:.1f}s')

    grasps_cam, scores = pred.grasps_cam, pred.scores
    if pred.num_grasps > 0:
        grasps_cam, scores, feas_stats = filter_feasible(
            grasps_cam, scores, pred.gripper_openings, T_world_cam, cfg.table_height)
        print(f"[feasibility] kept {feas_stats['n_after']}/{feas_stats['n_before']} "
              f"({feas_stats['n_rejected']} table-colliding/underhand rejected)")

    s = np.asarray(scores.get(real_label, []))
    if len(s) == 0:
        print('[interactive] no feasible grasp found for the selected object — '
              'nothing to execute')
        viewer.close()
        cam.close()
        return

    i = int(np.argmax(s))
    T_cam_grasp = grasps_cam[real_label][i]
    score = float(s[i])
    T_world_grasp = T_world_cam @ np.asarray(T_cam_grasp)
    label_to_body = {lbl: gen.object_names[lbl - 1]
                     for lbl in gen.object_body_ids.values()}
    body = label_to_body[real_label]

    from sim_grasp.executor import GraspExecutor
    rec_cam = CameraModule(model, data, cam_name=cfg.record_cam_name, width=640, height=480)
    executor = GraspExecutor(model, data, camera_module=rec_cam, record_gif=True,
                             record_dir=save_dir / '_gif_frames',
                             on_frame=viewer.show_frame)
    print(f'[execute] object {real_label} ({body}), score {score:.3f} — watch the window...')
    res = executor.execute(T_world_grasp, target_body=body)
    res.update(object=real_label, score=score)
    print(f'[execute]   -> {res}')
    if res['success']:
        print(f"[execute] PICK SUCCESS (object raised {res['object_raised_m']} m)")
    else:
        print('[execute] pick failed')
    executor.save_gif(save_dir / 'execution.gif')
    (save_dir / 'metrics.json').write_text(json.dumps(
        {'seed': args.seed, 'backend': args.backend, 'object': real_label,
         'score': score, 'execution': res}, indent=2))
    print(f'[execute] video saved: {save_dir / "execution.gif"}')

    print('[interactive] done — close the window to exit')
    while not viewer.closed:
        viewer.show_frame(rec_cam.render_rgb())
    viewer.close()
    cam.close()
    rec_cam.close()


if __name__ == '__main__':
    main()
```

- [x] **Step 2: Manual end-to-end smoke test**

```bash
cd mujoco_grasp_sim
export MUJOCO_GL=osmesa
export GRASPGEN_PYTHON=/home/vivek/miniconda3/envs/graspgen_torch/bin/python
export SAM3_PYTHON=/home/vivek/miniconda3/envs/sam3_torch/bin/python
/home/vivek/miniconda3/envs/cgn_torch/bin/python interactive_pick.py --seed 5 --backend graspgen
```
Expected: a live window opens showing the scene; click an object; after a
couple seconds a highlighted mask overlay appears; press Enter; the window
keeps updating as the arm moves; final message is either
`[execute] PICK SUCCESS ...` or `[execute] pick failed`, and
`output/<timestamp>/execution.gif`/`metrics.json` are written either way.
Try this at least twice — once confirming a real object, once pressing
Esc/'c' after a click to confirm the retry path works, and once (if you
can trigger it) clicking empty space away from any object to confirm the
"no object found" retry message.

- [x] **Step 3: Commit**

```bash
git add mujoco_grasp_sim/interactive_pick.py
git commit -m "Add interactive_pick.py: live click-to-select and watch the pick execute"
```

---

### Task 6: Documentation

**Files:**
- Modify: `mujoco_grasp_sim/README.md`

**Interfaces:** none (documentation only).

- [x] **Step 1: Add an "Interactive live pick" section**

Insert right after the existing "Promptable selection setup" section (added by the prior sub-project) and before `## Run`:

```markdown
### Interactive live pick (`interactive_pick.py`)

A live, click-to-select alternative to `run_sim_grasp_test.py --click X,Y`
(sub-project 2 above still works exactly as documented — this is an
additional, separate entry point, not a replacement). Opens a window
showing the robot's live camera feed; click the object you want, SAM 3
segments it and shows the mask for confirmation (Enter/Space to confirm,
Esc/`c` to retry), then GraspGen or CGN predicts a grasp and you watch the
pick execute live in the same window.

Needs the same `SAM3_PYTHON`/`GRASPGEN_PYTHON` env vars as the setups
above — see "Promptable selection setup" and "GraspGen backend setup" if
you haven't configured those yet. No new dependency: uses `opencv-python`,
already installed in `cgn_torch`.

```bash
export MUJOCO_GL=osmesa
export GRASPGEN_PYTHON=/path/to/graspgen_torch/bin/python
export SAM3_PYTHON=/path/to/sam3_torch/bin/python
python interactive_pick.py --seed 5 --backend graspgen
```

Single scene, single click-selected object, single grasp attempt per
run — unlike `run_sim_grasp_test.py`, this script has no `--pick-all`,
`--top-k`, or `--camera fused` support; it's built for watching one pick
happen interactively, not batch runs or benchmarking.
```

- [x] **Step 2: Add a run example to the `## Run` command list**

Current (end of the flag examples list in `mujoco_grasp_sim/README.md`, after the promptable-selection examples added by the prior sub-project):
```markdown
python run_sim_grasp_test.py --execute --click 320,240        # select the target by clicking a pixel (observation.png coords)
```
```

New:
```markdown
python run_sim_grasp_test.py --execute --click 320,240        # select the target by clicking a pixel (observation.png coords)
python interactive_pick.py --seed 5 --backend graspgen        # click live in a window instead of typing coordinates — see "Interactive live pick"
```
```

- [x] **Step 3: Verify**

```bash
grep -n "interactive_pick\|Interactive live pick" mujoco_grasp_sim/README.md
```
Expected: at least 3 matches (the new section header, the code block usage line, and the Run example).

- [x] **Step 4: Commit**

```bash
git add mujoco_grasp_sim/README.md
git commit -m "Document interactive_pick.py"
```
