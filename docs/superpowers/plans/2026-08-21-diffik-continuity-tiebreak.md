# DiffIK Continuity-First Seed Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `DiffIK.solve()` from silently swapping in a differently-postured (elbow-flipped) IK solution when the current-pose-seeded solve already converged well enough — fix the visible "elbow flips between waypoints" symptom in pick-and-place sequences.

**Architecture:** `DiffIK.solve()` (`mujoco_grasp_sim/sim_grasp/executor.py`) tries 4 seeds (the caller's current joint config `q_init`, a canonical elbow-down pose, and two random perturbations) and today picks whichever *converged* result has the lowest position error — with no penalty for jumping away from `q_init`. Fix: if the `q_init`-seeded result converges at all (existing `converged` flag, no new threshold), use it unconditionally; only fall through to comparing the other seeds if `q_init` itself fails to converge. Extract this decision into a standalone, pure function (`_pick_best_seed_result`) so it's unit-testable with synthetic `IKResult` objects, mirroring how `_ease()` was already pulled out of `_step_to` for the same reason (see `test_executor_ease.py`).

**Tech Stack:** Python, NumPy — no new dependencies.

**Spec:** No separate spec file — scoped as a bounded task in chat during brainstorming (design approved 2026-08-21).

## Global Constraints

- No behavior change to `_solve_single()`'s DLS iteration math itself — this plan only changes which of the 4 already-computed results `solve()` returns, not how any individual seed is solved. (A nullspace-redundancy-resolution follow-up, if wanted, is an explicit separate future decision, not part of this plan.)
- `converged` continues to mean exactly what it means today (`pos_err < pos_tol * 2 and ori_err < ori_tol * 2`, `executor.py:182`) — no new tolerance constant.
- This repo has no automated test suite by design — tests are standalone `test_*.py` scripts run directly with `python`, plain `assert` statements, ending with `print('All ... checks passed.')`. No pytest.
- Commit messages are plain text — never add a `Co-Authored-By: Claude` trailer.
- Every change goes through its own branch → PR → merge (already on branch `diffik-continuity-tiebreak`, forked from `main`).
- Validation requires both: (a) the new unit test, and (b) a live before/after comparison — `benchmark.py` for success-rate regression, plus a visual/log comparison of a multi-round `--pick-all` run for the elbow-flip symptom itself.

---

### Task 1: Continuity-first seed selection + unit test

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/executor.py` (the `DiffIK.solve()` method, `executor.py:128-144`)
- Create: `mujoco_grasp_sim/sim_grasp/test_executor_seed_selection.py`

**Interfaces:**
- Consumes: nothing new (uses the existing `IKResult` dataclass, `executor.py:105-110`).
- Produces: `_pick_best_seed_result(results: list[IKResult]) -> IKResult` (module-level, importable for tests). `results[0]` is always assumed to be the `q_init`-seeded result — callers (i.e. `solve()`) must preserve that ordering.

- [ ] **Step 1: Write the failing test**

Create `mujoco_grasp_sim/sim_grasp/test_executor_seed_selection.py`:

```python
"""Standalone check for executor._pick_best_seed_result -- run directly, no
pytest (this codebase has no automated test suite). Uses synthetic
IKResult objects (no live MuJoCo model needed), mirroring
test_executor_ease.py's pattern for testing pulled-out pure logic."""
import numpy as np

from sim_grasp.executor import _pick_best_seed_result, IKResult

def _r(pos_err, ori_err, converged, tag):
    # qpos payload is just a tag so we can identify which result won
    return IKResult(qpos=np.array([tag]), pos_err=pos_err, ori_err=ori_err,
                    converged=converged)

# 1. q_init converges tightly -> picked immediately, even if a later seed
#    (canonical elbow-down, tag=1) has a lower position error.
q_init_tight = _r(0.001, 0.005, converged=True, tag=0)
competitor_lower_err = _r(0.0005, 0.003, converged=True, tag=1)
picked = _pick_best_seed_result([q_init_tight, competitor_lower_err])
assert picked.qpos[0] == 0, 'q_init must win when it converges, even if a later seed is more accurate'

# 2. q_init converges loosely (still `converged=True` under the existing
#    2x-tolerance definition) but a later seed is numerically more
#    accurate -- q_init must STILL win (this is the bug this plan fixes:
#    "converged" already means "good enough", no further comparison).
q_init_loose = _r(0.0075, 0.018, converged=True, tag=0)  # within 2x tol, not 1x
competitor_more_accurate = _r(0.001, 0.002, converged=True, tag=1)
picked = _pick_best_seed_result([q_init_loose, competitor_more_accurate])
assert picked.qpos[0] == 0, \
    'q_init must win whenever it converged, regardless of a competitor being more accurate'

# 3. q_init fails to converge entirely -> fall through to the best
#    CONVERGED competitor (lowest pos_err among converged results).
q_init_failed = _r(0.05, 0.1, converged=False, tag=0)
worse_converged = _r(0.006, 0.015, converged=True, tag=1)
better_converged = _r(0.002, 0.006, converged=True, tag=2)
picked = _pick_best_seed_result([q_init_failed, worse_converged, better_converged])
assert picked.qpos[0] == 2, 'when q_init fails, the best CONVERGED competitor must win'

# 4. Nothing converges at all -> fall back to lowest pos_err overall
#    (existing "best effort" behavior, unchanged).
none_converged = [_r(0.05, 0.1, converged=False, tag=0),
                  _r(0.03, 0.08, converged=False, tag=1),
                  _r(0.09, 0.2, converged=False, tag=2)]
picked = _pick_best_seed_result(none_converged)
assert picked.qpos[0] == 1, 'with nothing converged, lowest pos_err wins as a best-effort result'

# 5. Single-result list (only q_init tried, e.g. early-exit callers) ->
#    returns that result unconditionally.
only = [_r(0.002, 0.005, converged=True, tag=0)]
picked = _pick_best_seed_result(only)
assert picked.qpos[0] == 0

print('All executor seed-selection checks passed.')
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `mujoco_grasp_sim/`): `PYTHONPATH=. python sim_grasp/test_executor_seed_selection.py`
Expected: `ImportError: cannot import name '_pick_best_seed_result'`

- [ ] **Step 3: Implement `_pick_best_seed_result` and rewrite `solve()`**

In `mujoco_grasp_sim/sim_grasp/executor.py`, add this function directly above the `DiffIK` class (i.e. just before line 113, `class DiffIK:`):

```python
def _pick_best_seed_result(results: list) -> 'IKResult':
    """Continuity-first seed selection: `results[0]` is always the
    q_init-seeded (current-pose) attempt. If it converged at all, it wins
    unconditionally -- 'converged' already means 'good enough' (see
    IKResult.converged's definition), so a numerically lower error from a
    differently-postured seed is not a reason to abandon joint-space
    continuity with the arm's current pose. Only when q_init itself fails
    to converge do we fall through to the best of the remaining attempts
    (preferring convergence, then lowest position error) -- unchanged from
    the prior behavior for that case."""
    q_init_result = results[0]
    if q_init_result.converged:
        return q_init_result
    best = q_init_result
    for res in results[1:]:
        if (res.converged and not best.converged) or \
                (res.converged == best.converged and res.pos_err < best.pos_err):
            best = res
    return best
```

Then replace the body of `DiffIK.solve()` (`executor.py:128-144`):

Current:
```python
    def solve(self, T_world_hand: np.ndarray, q_init: np.ndarray) -> IKResult:
        """DLS with restarts: try the given seed, a canonical elbow-down pose,
        and two perturbed seeds; return the best converged solution."""
        seeds = [q_init,
                 np.array([0.0, 0.35, 0.0, -1.8, 0.0, 2.2, -0.785]),
                 q_init + np.random.default_rng(0).uniform(-0.4, 0.4, 7),
                 q_init + np.random.default_rng(1).uniform(-0.7, 0.7, 7)]
        best = None
        for s in seeds:
            res = self._solve_single(T_world_hand, np.clip(
                s, self.jnt_range[:, 0], self.jnt_range[:, 1]))
            if best is None or (res.converged and not best.converged) or \
                    (res.converged == best.converged and res.pos_err < best.pos_err):
                best = res
            if best.converged and best.pos_err < self.pos_tol:
                break
        return best
```

New:
```python
    def solve(self, T_world_hand: np.ndarray, q_init: np.ndarray) -> IKResult:
        """DLS with restarts: try the given seed (the caller's current joint
        configuration) first. If it converges, use it -- continuity with the
        arm's current pose beats a numerically lower error from a
        differently-postured seed (see _pick_best_seed_result). Only try the
        canonical elbow-down pose and two perturbed seeds if q_init itself
        fails to converge."""
        seeds = [q_init,
                 np.array([0.0, 0.35, 0.0, -1.8, 0.0, 2.2, -0.785]),
                 q_init + np.random.default_rng(0).uniform(-0.4, 0.4, 7),
                 q_init + np.random.default_rng(1).uniform(-0.7, 0.7, 7)]
        results = []
        for s in seeds:
            res = self._solve_single(T_world_hand, np.clip(
                s, self.jnt_range[:, 0], self.jnt_range[:, 1]))
            results.append(res)
            if res.converged:
                break   # q_init converged (or, having fallen through, this
                        # later seed did) -- _pick_best_seed_result will
                        # still apply the continuity-first rule below
        return _pick_best_seed_result(results)
```

Note the early-exit change: the loop now breaks as soon as ANY seed converges (not just when it's tight), since `_pick_best_seed_result` already knows to keep `q_init`'s result if it was the one that converged, and to only consider later seeds when `q_init` didn't converge. This is both simpler and cheaper (fewer wasted IK solves in the common case).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python sim_grasp/test_executor_seed_selection.py`
Expected: `All executor seed-selection checks passed.`

Also run the existing executor-adjacent tests to confirm nothing else broke:
`PYTHONPATH=. python sim_grasp/test_executor_ease.py && PYTHONPATH=. python sim_grasp/test_executor_place_orientation.py`
Expected: both print their existing "All ... checks passed." lines.

- [ ] **Step 5: Live validation**

From `mujoco_grasp_sim/` (with `conda activate cgn_torch`, `MUJOCO_GL=osmesa`, and `GRASPGEN_PYTHON` set):

```bash
python benchmark.py --seeds 0-9 --mode pick-all --camera fused --backend graspgen --tag diffik_continuity
python analyze_failures.py output/bench_diffik_continuity
```

Expected: success rate/taxonomy comparable to (not worse than) the last recorded GraspGen baseline in `ROADMAP.md` — this change should not affect *whether* picks/places succeed, only *which* converged joint configuration is chosen. Report the actual before/after numbers; if there's a regression, that's a real finding to investigate, not something to wave away.

Also run one multi-round `--pick-all` with a fixed seed and inspect the resulting `execution.gif` (or the console's per-round joint-space behavior if the GIF isn't conclusive) to confirm the elbow no longer visibly flips between waypoints within the same pick sequence — this is the actual symptom being fixed, so a benchmark success-rate match alone doesn't confirm the fix.

- [ ] **Step 6: Record the result and commit**

If the live validation confirms no regression, add a dated bullet to `ROADMAP.md`'s P1 section (or wherever `--n-objects`/executor-reliability fixes are recorded — check the existing convention) describing the bug, the fix, and the before/after numbers from Step 5 — per `AGENTS.md`'s mandatory documentation-sync workflow (a bug found and fixed must get a `ROADMAP.md` bullet as part of finishing the task, not a separate ask).

```bash
git add mujoco_grasp_sim/sim_grasp/executor.py mujoco_grasp_sim/sim_grasp/test_executor_seed_selection.py ROADMAP.md
git commit -m "Fix DiffIK elbow-flip: prefer continuity with current pose over lowest IK error"
```

---
