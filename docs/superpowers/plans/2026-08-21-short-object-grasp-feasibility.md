# Short-Object Finger-Table Collision Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the root cause documented in
`docs/research/2026-08-21-short-object-finger-table-collision.md`:
`GraspFeasibilityChecker.is_feasible()` validates the grasp pose as
originally predicted by CGN/GraspGen, but `GraspExecutor.execute()`
advances the actually-executed pose `EXTRA_APPROACH` (12mm) further along
the approach axis before closing, unvalidated. For short objects (whose
predicted grasp already has little margin above the table), that
unchecked 12mm deepening is a plausible direct cause of fingers hitting
and closing against the table instead of the object.

**Architecture:** `GraspFeasibilityChecker` gains an `extra_approach`
constructor parameter (meters, default `0.0` — fully backward compatible).
When set, `is_feasible()`'s table-collision check is run against the pose
advanced by `extra_approach` along the grasp +Z axis — i.e. the pose that
will actually be executed — instead of only the originally predicted pose.
The upward-approach-direction check is unaffected (it depends only on
orientation, not position). `run_sim_grasp_test.py`'s `filter_feasible()`
(the single call site used by both `run_sim_grasp_test.py` and
`interactive_pick.py`) is updated to construct the checker with
`extra_approach=EXTRA_APPROACH`, imported from `sim_grasp.executor` — the
same constant `execute()` already uses — so the two stay in sync by
construction rather than by duplicated magic numbers.

**Tech Stack:** Python, NumPy (existing dependencies only, nothing new).

## Global Constraints

- Branch: `fix-short-object-grasp-feasibility`, based on current `main`
  (post `intelligent-bin-placement` merge).
- This codebase has **no pytest suite** — tests are standalone `test_*.py`
  scripts run directly (`python sim_grasp/test_name.py`, from
  `mujoco_grasp_sim/` with `PYTHONPATH=.`), plain `assert` statements,
  ending with a `print('All ... checks passed.')` line.
- `conda activate cgn_torch` before running anything in this repo.
- Do not change `EXTRA_APPROACH`'s value (0.012) or any other executor
  pick-sequence constant/timing — this plan only makes the existing
  feasibility check aware of the existing deepening, it does not change
  how grasps are executed.
- Do not touch `placement_planner.py`/`executor.py`'s `_step_to()`/`place()`
  — that work is already merged (P7); this plan is scoped entirely to the
  pick-phase feasibility filter (`sim_grasp/feasibility.py` and its one
  call site in `run_sim_grasp_test.py`).
- `GraspFeasibilityChecker.is_feasible()`'s existing public signature
  (`T_world_grasp`, `opening`) must not change — `extra_approach` is
  configured once at construction time (`__init__`), not per-call, since
  it is a property of the executor's pick sequence, not of an individual
  grasp.

---

## Task 1: Fold `EXTRA_APPROACH` into the feasibility table-collision check

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/feasibility.py`
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py` (`filter_feasible()`)
- Create: `mujoco_grasp_sim/sim_grasp/test_feasibility.py`

**Interfaces:**
- `GraspFeasibilityChecker.__init__(self, table_height: float, margin:
  float = 0.005, reject_upward_approach: bool = True, max_up_z: float =
  0.15, extra_approach: float = 0.0)` — new keyword-only-by-convention
  param, stored as `self.extra_approach`.
- `is_feasible()` and `filter()` keep their exact existing signatures —
  no caller other than `__init__` needs to know about `extra_approach`.

**Step 1: Write the failing test**

Create `mujoco_grasp_sim/sim_grasp/test_feasibility.py`. It must construct
a `T_world_grasp` (4x4 numpy array, grasp frame: +Z approach, origin at
gripper base — see `sim_grasp/frames.py` for the convention) for a grasp
that is:
- **Feasible without the fix**: all `_gripper_sample_points()` corners (in
  world frame) sit comfortably above `table_height + margin` at the
  originally predicted pose.
- **Infeasible after the executor's real `EXTRA_APPROACH` deepening**:
  advancing the same pose by `0.012` m along its own +Z axis (grasp frame,
  which for a straight-down top-down grasp is world `-Z`) pushes at least
  one finger-box corner below `table_height + margin`.

Concretely: use a top-down grasp orientation (grasp +Z pointing straight
down in world, i.e. `T_world_grasp[:3, :3] = R` such that
`R @ [0,0,1] = [0,0,-1]`) with the grasp origin positioned so the lowest
finger-box corner (`z0=0.066` in the grasp frame, per
`_gripper_sample_points()`'s finger box) sits at `table_height + margin +
0.003` (3mm of margin at the predicted pose — enough to pass today's
check, too little to survive a 12mm further descent).

Assertions:
1. `GraspFeasibilityChecker(table_height, extra_approach=0.0).is_feasible(T)`
   is `True` (today's behavior, unchanged when the fix is not engaged —
   confirms backward compatibility).
2. `GraspFeasibilityChecker(table_height, extra_approach=0.012).is_feasible(T)`
   is `False` (the fix catches the case the bug report describes — this
   is the regression test for the actual bug).
3. A grasp with ample table clearance (e.g. 5cm above `table_height +
   margin` at the predicted pose) stays feasible with
   `extra_approach=0.012` too (the fix must not reject well-clear grasps).
4. The upward-approach rejection (`reject_upward_approach`) still works
   unchanged regardless of `extra_approach` (construct a grasp whose +Z
   points upward in world and confirm it's still rejected).

End the file with `print('All feasibility checks passed.')`. Run it
(`PYTHONPATH=. python sim_grasp/test_feasibility.py` from
`mujoco_grasp_sim/`) and confirm assertion 2 fails against the current
`feasibility.py` (proving the test actually exercises the bug) before
writing the fix.

- [ ] **Step 1: Write the failing test** (as above; confirm it fails first)

**Step 2: Implement the fix**

In `mujoco_grasp_sim/sim_grasp/feasibility.py`:
- Add `extra_approach: float = 0.0` to `GraspFeasibilityChecker.__init__`,
  store as `self.extra_approach`.
- In `is_feasible()`, before the table-collision check, compute the
  pose to actually validate:
  ```python
  T_check = T_world_grasp.copy()
  T_check[:3, 3] = T_world_grasp[:3, 3] + self.extra_approach * T_world_grasp[:3, 2]
  ```
  Use `T_check` (not `T_world_grasp`) when transforming
  `_gripper_sample_points()` for the table-collision test. The
  upward-approach check (`T_world_grasp[2, 2] > self.max_up_z`) keeps
  using the original `T_world_grasp` — orientation is unaffected by the
  translation-only advance.
- Update the module/class docstring's one-line description of what's
  validated to mention the optional post-approach-advance check (one or
  two lines, not a rewrite).

In `mujoco_grasp_sim/run_sim_grasp_test.py`:
- Add `EXTRA_APPROACH` to the existing `from sim_grasp.executor import
  PLACE_RELEASE` line (`from sim_grasp.executor import EXTRA_APPROACH,
  PLACE_RELEASE`).
- In `filter_feasible()`, change
  `GraspFeasibilityChecker(table_height=table_height)` to
  `GraspFeasibilityChecker(table_height=table_height,
  extra_approach=EXTRA_APPROACH)`.

- [ ] **Step 2: Implement the fix and confirm all 4 assertions in
      `test_feasibility.py` pass**
- [ ] **Step 3: Run the full existing test suite** (every
      `sim_grasp/test_*.py` file) to confirm no regression, and run one
      `--pick-all --no-vis` smoke test (any backend/seed) to confirm the
      pipeline still runs end-to-end with real grasps now passing through
      the updated filter.

**Verification:**
- [ ] `PYTHONPATH=. python sim_grasp/test_feasibility.py` → `All
      feasibility checks passed.`
- [ ] All pre-existing `sim_grasp/test_*.py` files still pass unchanged.
- [ ] One live `run_sim_grasp_test.py --pick-all --no-vis` run completes
      without error and reports a `[feasibility] kept X/Y` line (proving
      the new `extra_approach` param doesn't crash the real pipeline).

---

## Task 2: Empirical validation + docs

**Files:**
- Modify: `ROADMAP.md` (P1 section)
- Modify: `docs/research/2026-08-21-short-object-finger-table-collision.md`
  (update `Status` from "deferred" to fixed, with a dated finding)
- Modify: `docs/research/README.md` (add the missing index entry for the
  short-object research file — this was never added in the prior
  placement-robustness work; a genuine gap, not new scope creep)

**Step 1: Quantify the fix's effect**

Run a benchmark comparison to measure the fix's real-world effect on
grasp feasibility filtering, using the existing
`mujoco_grasp_sim/benchmark.py` harness (see `ROADMAP.md` P1 for the
established seeds/config convention: box-only objects, fused camera,
pick-all). Since this codebase's objects have randomized heights
(`sz` half-height uniform in `[0.02, 0.055]` — see
`sim_grasp/scene_generator.py`'s `_make_primitive()`), a plain multi-seed
run will naturally include some short objects; there is no dedicated
"short objects only" scene config, so do not add one — this plan fixes
the feasibility filter, not the object distribution.

Compare, over the same seeds, with and without the fix (toggle by
constructing `GraspFeasibilityChecker` with `extra_approach=0.0` vs
`extra_approach=EXTRA_APPROACH` — e.g. a small standalone script or a
temporary local edit, not a permanent CLI flag):
1. The raw `[feasibility] kept X/Y` counts across several seeds — the fix
   should show a small additional rejection count (grasps that pass the
   old check but fail the new one), concentrated on short objects.
2. A `benchmark.py` run (5-10 seeds, pick-all, fused camera, whichever
   backend is available in this environment) with the fix active, to
   confirm no regression in overall success rate versus the last recorded
   P1 baseline in `ROADMAP.md`.

Be honest in the writeup if the sample size is too small for a solid
before/after delta (precedent: the placement-pose-robustness fix's
writeup in `ROADMAP.md` P7 made the same honest caveat) — the qualitative
argument (a real, previously-unvalidated 12mm blind spot is now checked)
stands on its own even if the quantitative delta is noisy.

- [ ] **Step 1: Run the before/after comparison and the regression
      benchmark; record the raw numbers**

**Step 2: Update docs**

- `ROADMAP.md` P1: add a dated bullet under the existing checklist
  documenting the fix (mirroring the style of the existing dated bullets:
  what was broken, the fix, the numbers from Step 1).
- `docs/research/2026-08-21-short-object-finger-table-collision.md`:
  change the `Status:` line from "deferred — root-caused, not yet fixed"
  to a fixed/resolved status with today's date and a one-line pointer to
  this plan file and the `ROADMAP.md` bullet.
- `docs/research/README.md`: add an index entry for
  `2026-08-21-short-object-finger-table-collision.md`, following the
  existing entry's format (topic + one-line status).

- [ ] **Step 2: Update all three docs**

**Verification:**
- [ ] `ROADMAP.md` P1 has a new dated bullet with real numbers.
- [ ] The research doc's `Status:` line reflects the fix.
- [ ] `docs/research/README.md`'s index lists the short-object file.
