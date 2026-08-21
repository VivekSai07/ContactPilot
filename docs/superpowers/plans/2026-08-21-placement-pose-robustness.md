# Placement Pose Robustness Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two real, code-verified bugs observed in live testing of the
(not-yet-merged) `intelligent-bin-placement` branch: objects always landing
in the same bin corner/touching the bin wall, and objects slipping out of
the gripper during the carry-to-bin transit.

**Architecture:** `OccupancyPlacementPlanner.plan()`'s candidate-scoring
function gets a wall-clearance term so it no longer degenerates to the
first-scanned corner when the bin is empty (its only signal today is
distance to *other objects*, which is meaningless — infinite for every
candidate — before any object has been placed). `GraspExecutor._step_to()`
gets an optional smoothstep ease-in-ease-out interpolation profile, applied
only to `place()`'s two object-carrying motions, replacing linear
interpolation's instantaneous start-of-motion velocity jump — a classic
cause of a held object slipping right as a move begins.

**Tech Stack:** Python, NumPy (existing dependencies only, nothing new).

## Global Constraints

- Branch: `intelligent-bin-placement` (already has an open PR — these
  fixes land as additional commits on the same branch, not a new one).
- This codebase has **no pytest suite** — tests are standalone `test_*.py`
  scripts run directly (`python sim_grasp/test_name.py`, from
  `mujoco_grasp_sim/` with `PYTHONPATH=.`), plain `assert` statements,
  ending with a `print('All ... checks passed.')` line.
- `conda activate cgn_torch` before running anything in this repo.
- Do not touch `GraspFeasibilityChecker`/`EXTRA_APPROACH`/the pick sequence
  in `GraspExecutor.execute()` — the short-object finger-table-collision
  issue found during this investigation is explicitly **out of scope**,
  deferred to its own future plan (Task 3 below just records it).
- Do not change `_step_to()`'s behavior for any call site other than the
  two named in Task 2 — the existing pick/close/lift sequence in
  `execute()` is already empirically tuned; don't destabilize it.

---

## Task 1: Wall-aware clearance scoring in `OccupancyPlacementPlanner`

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/placement_planner.py` (the `plan()`
  method's main search loop and its fallback loop)
- Modify: `mujoco_grasp_sim/sim_grasp/test_placement_planner.py` (extend
  the existing empty-bin assertions)

**Interfaces:**
- Consumes: nothing new — same `ObjectFootprint`/`BinHeightmap` inputs,
  same `OccupancyPlacementPlanner.__init__` signature.
- Produces: no interface change — `plan()` still returns
  `PlacementPose | None`, same fields. Callers (already wired in
  `run_sim_grasp_test.py`/`interactive_pick.py`) need no changes.

**The bug, precisely:** in the main search loop, the score is:
```python
clearance = (float(np.min(np.hypot(occ_x - cx, occ_y - cy)))
             if len(occ_x) else float('inf'))
if best is None or clearance > best[0]:
    best = (clearance, cx, cy, yaw)
```
When `occ_x` is empty (an empty bin — true for the first object placed),
`clearance` is `inf` for *every* candidate. Since the update requires
strict `>`, the very first candidate scanned (`yaw_offset=0.0`, `cx=x_min`,
`cy=y_min` — the near-corner of the search region) can never be beaten by
any later one. This is exactly "every time it goes to the same corner and
touches the edge." The design spec always said the objective should be
"clearance to the nearest occupied cell **or wall**" — only the
occupied-cell half was ever actually implemented.

- [ ] **Step 1: Write the failing test**

Open `mujoco_grasp_sim/sim_grasp/test_placement_planner.py` and find this
existing block (search for `pose = planner.plan(small_footprint, empty_map)`):

```python
planner = OccupancyPlacementPlanner(bin_center=(0.45, -0.30), bin_inner_half=0.12)
pose = planner.plan(small_footprint, empty_map)
assert pose is not None, 'a 3cm object must fit in an empty 24cm bin'
assert abs(pose.release_z - 0.75) < 1e-6, f'release_z={pose.release_z}'
assert abs(pose.x - 0.45) < 0.12 and abs(pose.y - (-0.30)) < 0.12
```

Immediately after that last `assert` line, add:

```python
# Regression test for a real bug: on an EMPTY bin every candidate had
# infinite "clearance to nearest occupied cell" (there are no occupied
# cells yet), so the tie-break silently picked the first-scanned candidate
# -- the near-corner of the search region -- every single time. Fix adds a
# wall-clearance term so an empty bin centers the placement instead. (bin
# half-extent is 0.12m, so a corner would be off by ~0.10m from center --
# 0.03m is a tight, meaningful "not a corner" bound.)
assert abs(pose.x - 0.45) < 0.03 and abs(pose.y - (-0.30)) < 0.03, \
    f'expected a near-center placement on an empty bin, got ({pose.x}, {pose.y})'
```

- [ ] **Step 2: Run the test to verify it fails**

Run (from `mujoco_grasp_sim/`): `PYTHONPATH=. python sim_grasp/test_placement_planner.py`
Expected: `AssertionError: expected a near-center placement on an empty bin, got (0.348, -0.402)` (or similar corner-ish coordinates — the exact values don't matter, only that it's clearly not near `(0.45, -0.30)`).

- [ ] **Step 3: Fix the main search loop**

In `mujoco_grasp_sim/sim_grasp/placement_planner.py`, find this block inside
`OccupancyPlacementPlanner.plan()`:

```python
            for cx in np.arange(x_min, x_max + 1e-9, self.search_stride):
                for cy in np.arange(y_min, y_max + 1e-9, self.search_stride):
                    in_box = ((np.abs(grid_x - cx) <= half_ex) &
                              (np.abs(grid_y - cy) <= half_ey))
                    if np.any(heightmap.heights[in_box] > occupied_z):
                        continue
                    clearance = (float(np.min(np.hypot(occ_x - cx, occ_y - cy)))
                                 if len(occ_x) else float('inf'))
                    if best is None or clearance > best[0]:
                        best = (clearance, cx, cy, yaw)
```

Replace with:

```python
            for cx in np.arange(x_min, x_max + 1e-9, self.search_stride):
                for cy in np.arange(y_min, y_max + 1e-9, self.search_stride):
                    in_box = ((np.abs(grid_x - cx) <= half_ex) &
                              (np.abs(grid_y - cy) <= half_ey))
                    if np.any(heightmap.heights[in_box] > occupied_z):
                        continue
                    # Clearance to the nearest occupied cell OR the bin
                    # wall, whichever is smaller -- without the wall term,
                    # an empty bin gives every candidate infinite
                    # occupied-clearance, so the tie-break always
                    # degenerates to the first-scanned candidate (a corner
                    # of the search region).
                    wall_clearance = min(cx - x_min, x_max - cx,
                                         cy - y_min, y_max - cy)
                    occ_clearance = (float(np.min(np.hypot(occ_x - cx, occ_y - cy)))
                                     if len(occ_x) else float('inf'))
                    clearance = min(wall_clearance, occ_clearance)
                    if best is None or clearance > best[0]:
                        best = (clearance, cx, cy, yaw)
```

- [ ] **Step 4: Fix the fallback loop's tie-break the same way**

Find this block (the fallback path, after `if not any_orientation_fits:`):

```python
        best_fallback = None
        for cx in np.arange(x_min, x_max + 1e-9, self.search_stride):
            for cy in np.arange(y_min, y_max + 1e-9, self.search_stride):
                in_box = ((np.abs(grid_x - cx) <= half_ex) &
                          (np.abs(grid_y - cy) <= half_ey))
                cell_heights = heightmap.heights[in_box]
                if len(cell_heights) == 0:
                    continue
                max_h = float(np.max(cell_heights))
                if best_fallback is None or max_h < best_fallback[0]:
                    best_fallback = (max_h, cx, cy)
        if best_fallback is None:
            return None
        max_h, x, y = best_fallback
```

Replace with (same fix applied to the fallback's tie-break: prefer lowest
stack height first as before, but break ties on wall clearance instead of
scan order):

```python
        best_fallback = None
        for cx in np.arange(x_min, x_max + 1e-9, self.search_stride):
            for cy in np.arange(y_min, y_max + 1e-9, self.search_stride):
                in_box = ((np.abs(grid_x - cx) <= half_ex) &
                          (np.abs(grid_y - cy) <= half_ey))
                cell_heights = heightmap.heights[in_box]
                if len(cell_heights) == 0:
                    continue
                max_h = float(np.max(cell_heights))
                wall_clearance = min(cx - x_min, x_max - cx,
                                     cy - y_min, y_max - cy)
                if (best_fallback is None or max_h < best_fallback[0] or
                        (max_h == best_fallback[0] and
                         wall_clearance > best_fallback[1])):
                    best_fallback = (max_h, wall_clearance, cx, cy)
        if best_fallback is None:
            return None
        max_h, _, x, y = best_fallback
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `PYTHONPATH=. python sim_grasp/test_placement_planner.py`
Expected: `All placement_planner footprint checks passed.` /
`All placement_planner heightmap checks passed.` /
`All placement_planner search checks passed.` (all three, no assertion
errors) — the third line's underlying test now includes the new
near-center check.

- [ ] **Step 6: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/placement_planner.py mujoco_grasp_sim/sim_grasp/test_placement_planner.py
git commit -m "Fix placement always landing in the same bin corner

OccupancyPlacementPlanner's clearance scoring only measured distance to
other objects, which is infinite for every candidate on an empty bin --
the tie-break silently picked the first-scanned candidate (a corner) every
time. Adds a wall-clearance term so an empty bin centers the placement
instead, matching what the design spec always said the objective should
be. Same fix applied to the fallback path's tie-break for consistency."
```

---

## Task 2: Smooth (ease-in-ease-out) motion for the two object-carrying transit steps

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/executor.py` (`_step_to()` and
  `place()`)
- Test: `mujoco_grasp_sim/sim_grasp/test_executor_ease.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_ease(t: float, smooth: bool) -> float` (new module-level
  function, importable for tests); `GraspExecutor._step_to(..., smooth:
  bool = False)` (new optional parameter, default preserves existing
  behavior at every other call site).

**The bug, precisely:** `_step_to()` linearly interpolates joint targets
over a fixed duration — `a = (i + 1) / n` — which has an *instantaneous*
velocity jump from 0 to full speed at the very start of the motion (and an
equally abrupt stop at the end). `place()`'s transit-to-hover and
lower-to-release steps run this while the gripper is closed holding an
object — that abrupt start-of-motion jerk, right after lift-off, is a
well-known cause of a marginally-gripped object slipping. A smoothstep
profile (zero velocity at both ends) removes the discontinuity without
changing the overall motion duration.

- [ ] **Step 1: Write the failing test**

Create `mujoco_grasp_sim/sim_grasp/test_executor_ease.py`:

```python
"""Standalone check for executor._ease -- run directly, no pytest (this
codebase has no automated test suite)."""
import numpy as np

from sim_grasp.executor import _ease

# Linear (smooth=False) is unchanged: identity function.
for t in (0.0, 0.25, 0.5, 0.75, 1.0):
    assert _ease(t, smooth=False) == t, f'_ease({t}, False) should equal {t}'

# Smoothstep (smooth=True): passes through the same endpoints...
assert _ease(0.0, smooth=True) == 0.0
assert _ease(1.0, smooth=True) == 1.0
# ...is symmetric around the midpoint...
assert abs(_ease(0.5, smooth=True) - 0.5) < 1e-9
# ...and has ZERO slope at both ends (the actual fix for the slip bug --
# estimate the derivative numerically at t=0 and t=1, it must be ~0, unlike
# linear interpolation's constant slope of 1.0 everywhere).
eps = 1e-4
slope_at_start = (_ease(eps, smooth=True) - _ease(0.0, smooth=True)) / eps
slope_at_end = (_ease(1.0, smooth=True) - _ease(1.0 - eps, smooth=True)) / eps
assert slope_at_start < 0.01, f'slope at t=0 should be ~0, got {slope_at_start}'
assert slope_at_end < 0.01, f'slope at t=1 should be ~0, got {slope_at_end}'

# Monotonically increasing (no overshoot/oscillation) across the whole range.
ts = np.linspace(0, 1, 50)
values = [_ease(float(t), smooth=True) for t in ts]
assert all(b >= a for a, b in zip(values, values[1:])), \
    'smoothstep must be monotonically increasing'

print('All executor _ease checks passed.')
```

- [ ] **Step 2: Run the test to verify it fails**

Run (from `mujoco_grasp_sim/`): `PYTHONPATH=. python sim_grasp/test_executor_ease.py`
Expected: `ImportError: cannot import name '_ease' from 'sim_grasp.executor'`

- [ ] **Step 3: Add `_ease()` and thread it through `_step_to()`**

In `mujoco_grasp_sim/sim_grasp/executor.py`, find the existing `_step_to`
method:

```python
    def _step_to(self, q_target: np.ndarray, duration: float,
                 gripper_ctrl: float | None = None):
        """Linearly interpolate joint position references over `duration`
        seconds of sim time; the position servos do the tracking."""
        model, data = self.model, self.data
        n = max(1, int(duration / model.opt.timestep))
        q_start = data.ctrl[:7].copy()
        for i in range(n):
            a = (i + 1) / n
            data.ctrl[:7] = (1 - a) * q_start + a * q_target
            if gripper_ctrl is not None:
                data.ctrl[7] = gripper_ctrl
            mujoco.mj_step(model, data)
            self._maybe_record()
```

Replace with:

```python
    def _step_to(self, q_target: np.ndarray, duration: float,
                 gripper_ctrl: float | None = None, smooth: bool = False):
        """Interpolate joint position references over `duration` seconds of
        sim time; the position servos do the tracking. `smooth=True` uses
        a smoothstep ease-in-ease-out profile (zero velocity at both ends)
        instead of linear interpolation -- linear interpolation has an
        instantaneous velocity jump at t=0, a real cause of a held object
        slipping right as a transit move begins."""
        model, data = self.model, self.data
        n = max(1, int(duration / model.opt.timestep))
        q_start = data.ctrl[:7].copy()
        for i in range(n):
            t = (i + 1) / n
            a = _ease(t, smooth)
            data.ctrl[:7] = (1 - a) * q_start + a * q_target
            if gripper_ctrl is not None:
                data.ctrl[7] = gripper_ctrl
            mujoco.mj_step(model, data)
            self._maybe_record()
```

Add the `_ease` function at module level, right above the `_candidate_hand_orientations`
function (search for `def _candidate_hand_orientations`):

```python
def _ease(t: float, smooth: bool) -> float:
    """Interpolation parameter for _step_to: linear (smooth=False) or a
    smoothstep ease-in-ease-out curve (smooth=True, zero velocity at t=0
    and t=1) -- pulled out as its own function so the easing math is
    unit-testable without a live MuJoCo model."""
    return (3 * t ** 2 - 2 * t ** 3) if smooth else t
```

- [ ] **Step 4: Use `smooth=True` for the two object-carrying motions in `place()`**

Find this block in `place()`:

```python
        ik_pre, ik_rel = plan
        self._step_to(ik_pre.qpos, 2.2, gripper_ctrl=GRIPPER_CLOSED)   # transit
        self._step_to(ik_rel.qpos, 1.0, gripper_ctrl=GRIPPER_CLOSED)   # lower
        self._hold(0.2)
        self._step_to(ik_rel.qpos, 0.6, gripper_ctrl=GRIPPER_OPEN)     # release
        self._hold(0.4)
        self._step_to(ik_pre.qpos, 0.8, gripper_ctrl=GRIPPER_OPEN)     # retract
        return {'placed': True, 'stage': 'place_done'}
```

Replace with (only the first two calls change — release/retract happen
open-handed, no object to slip, left as linear on purpose):

```python
        ik_pre, ik_rel = plan
        # smooth=True only for the two motions performed while still
        # holding the object -- linear interpolation's instant velocity
        # jump at the start of a move is a real cause of slip; release and
        # retract happen open-handed, so they're left as linear.
        self._step_to(ik_pre.qpos, 2.2, gripper_ctrl=GRIPPER_CLOSED, smooth=True)   # transit
        self._step_to(ik_rel.qpos, 1.0, gripper_ctrl=GRIPPER_CLOSED, smooth=True)   # lower
        self._hold(0.2)
        self._step_to(ik_rel.qpos, 0.6, gripper_ctrl=GRIPPER_OPEN)     # release
        self._hold(0.4)
        self._step_to(ik_pre.qpos, 0.8, gripper_ctrl=GRIPPER_OPEN)     # retract
        return {'placed': True, 'stage': 'place_done'}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `PYTHONPATH=. python sim_grasp/test_executor_ease.py`
Expected: `All executor _ease checks passed.`

Also re-run the existing executor test to confirm the new optional
parameter didn't disturb anything:
`PYTHONPATH=. python sim_grasp/test_executor_place_orientation.py`
Expected: `All executor orientation checks passed.`

- [ ] **Step 6: Smoke-test a real pick-and-place run**

```bash
cd mujoco_grasp_sim
MUJOCO_GL=osmesa GRASPGEN_PYTHON=/home/vivek/miniconda3/envs/graspgen_torch/bin/python \
  python run_sim_grasp_test.py --pick-all --camera fused --backend graspgen --seed 0 --no-vis
```

Expected: completes with no traceback, same as every prior smoke test in
this project's history for this script.

- [ ] **Step 7: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/executor.py mujoco_grasp_sim/sim_grasp/test_executor_ease.py
git commit -m "Smooth the two object-carrying transit motions in place()

_step_to() linearly interpolated joint targets, which has an instantaneous
velocity jump at t=0 -- a real cause of a held object slipping right as a
transit move begins. Adds an optional smoothstep ease-in-ease-out profile
(_ease()), applied only to place()'s transit-to-hover and lower-to-release
steps (the two motions performed while still holding the object); release/
retract and the entire pick sequence in execute() are untouched."
```

---

## Task 3: Validate, record results, and log the deferred short-object issue

**Files:**
- Modify: `ROADMAP.md` (append to the P7 section)
- Modify: `README.md` (update the progress table/chart if the numbers
  materially change)
- Create: `docs/research/2026-08-21-short-object-finger-table-collision.md`
- Modify: `docs/research/README.md` (index entry for the new file)

**Interfaces:** none (validation + documentation task).

- [ ] **Step 1: Re-run the same 10-seed benchmark P7 already used**

```bash
cd mujoco_grasp_sim
conda activate cgn_torch
MUJOCO_GL=osmesa GRASPGEN_PYTHON=/home/vivek/miniconda3/envs/graspgen_torch/bin/python \
  python benchmark.py --seeds 0-9 --mode pick-all --camera fused --backend graspgen --tag placement_robustness
python analyze_failures.py output/bench_placement_robustness
```

Record the resulting objects-binned success rate and the failure taxonomy
breakdown — specifically whether `missed_bin` and `unstable_slip` (if
present) failures decreased versus P7's recorded 29/30 (96.7%), 3
`missed_bin` failures.

- [ ] **Step 2: Record the result in `ROADMAP.md`**

Find the `## P7 — Intelligent bin placement (vision-only)` section (added
by the prior benchmark run) and append a new dated sub-entry directly
under it (match the existing entry's format/style) summarizing: the two
bugs found (corner-lock tie-break, linear-interpolation slip), the fixes,
and the new benchmark numbers from Step 1 compared against the prior
29/30 (96.7%) recording.

- [ ] **Step 3: Update `README.md` if the headline number changed materially**

If Step 1's new success rate differs meaningfully from the existing
"Progress at a glance" table/chart entry for intelligent bin placement,
update that row/bar with the new number and a one-line note; if it's
within noise of the existing number, leave `README.md` unchanged and say
so in the commit message.

- [ ] **Step 4: Log the deferred short-object finger-table-collision issue**

Create `docs/research/2026-08-21-short-object-finger-table-collision.md`:

```markdown
# Short-object finger-table collision during grasp closing

**Status:** deferred — root-caused, not yet fixed. Separate future plan,
to be started only after the placement-pose robustness fixes (this same
investigation's other two findings) are implemented, verified, and merged.

## Symptom

For short objects, the gripper fingers hit the table and close against it
while attempting to grasp the object, rather than cleanly grasping the
object itself.

## Root-cause hypothesis (code-verified, not yet empirically confirmed)

`GraspFeasibilityChecker.is_feasible()` (`sim_grasp/feasibility.py`)
validates the grasp pose `T_world_grasp` as originally predicted by
CGN/GraspGen. But `GraspExecutor.execute()`
(`sim_grasp/executor.py`) advances the *actually executed* pose further
along the approach axis before closing:

    T_hand_grasp[:3, 3] = T_world_grasp[:3, 3] + EXTRA_APPROACH * T_world_grasp[:3, 2]

`EXTRA_APPROACH = 0.012` (12mm) is never re-validated against the table
plane. For a short object, the grasp's TCP is already close to the table,
leaving little margin -- this unchecked 12mm deepening is a plausible
direct cause of fingers contacting the table specifically for short
objects, since taller objects have more margin to spare.

## Proposed fix direction (not yet designed in detail)

Re-run (or extend) the feasibility check against the *post-EXTRA_APPROACH*
pose, not just the originally predicted one -- either by checking both
poses, or by folding `EXTRA_APPROACH` into what
`GraspFeasibilityChecker.is_feasible()` validates in the first place.

## Scope note

This is a **pick-phase** issue (`GraspFeasibilityChecker`/
`GraspExecutor.execute()`), not a placement-phase issue -- unrelated to
the `intelligent-bin-placement` branch's `placement_planner.py`/`place()`
work, found incidentally during the same round of live-testing feedback.
```

- [ ] **Step 5: Update the research notes index**

In `docs/research/README.md`, add an index entry for the new file
alongside the existing ones (match the existing bullet format).

- [ ] **Step 6: Commit**

```bash
git add ROADMAP.md README.md docs/research/2026-08-21-short-object-finger-table-collision.md docs/research/README.md
git commit -m "Record placement-robustness benchmark results; log deferred short-object grasp issue"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers the corner/wall-touching symptom, Task 2
  covers the transit-slip symptom, Task 3 covers validation + the explicit
  user instruction to scope out (not fix) the short-object issue as a
  separate future plan. All three user-reported symptoms are accounted
  for — two fixed, one deliberately deferred and documented, matching the
  user's own stated scope decision.
- **No placeholders:** every code step above is the literal, complete diff
  to apply — nothing marked TBD/TODO.
- **Type/interface consistency:** `_ease(t: float, smooth: bool) -> float`
  is defined once in Task 2 and used consistently in the same task's
  `_step_to()` edit; no other task references it. `OccupancyPlacementPlanner.plan()`'s
  return type (`PlacementPose | None`) is unchanged by Task 1 — only the
  internal scoring changed, so no caller-facing signature drift to check
  against other tasks.
