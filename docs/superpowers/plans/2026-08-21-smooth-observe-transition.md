# Smooth Observe-Pose Transition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the abrupt "snap" feel of the arm's transition from a finished place back into the fixed observation pose between `--pick-all` rounds, by applying the already-proven `smooth=True` easing (used for `place()`'s object-carrying motions) to the two motions that bracket this transition.

**Architecture:** `GraspExecutor._step_to()` already supports linear (default) or smoothstep (`smooth=True`, zero velocity at both ends) interpolation — `place()`'s two carrying motions already use `smooth=True`; its final open-handed retract move and `go_observe()`'s move to `ARM_OBSERVE_QPOS` currently don't. This plan adds `smooth=True` to both, so two consecutive motions (retract-after-place, then go-to-observe) each ramp smoothly to zero velocity at their shared boundary instead of truncating abruptly — turning one visible "snap-stop-snap-start" into a fluid deceleration/re-acceleration. No new interpolation logic — `_ease()` itself is unchanged and already fully unit-tested (`test_executor_ease.py`).

**Tech Stack:** Python, MuJoCo — no new dependencies.

**Spec:** No separate spec file — scoped as a bounded task in chat during brainstorming (design approved 2026-08-21).

## Global Constraints

- No change to `_ease()`'s math, `DiffIK`, or any pose/target computation — this plan only changes which existing `_step_to()` calls pass `smooth=True`, and updates the one stale comment that claims smoothing is only used for object-carrying motions.
- Do NOT touch hold durations (`_hold(0.2)`, `_hold(0.3)`, `_hold(0.4)` anywhere in `executor.py`) or remove any hold call — those serve either object-settling physics (place's pre/post-release holds) or camera-capture-accuracy margin (go_observe's post-move hold), and are explicitly out of scope for this plan. (A more invasive follow-up — skipping the fixed observe pose entirely when not needed for camera-frustum clearance — is recorded as a deferred idea in `ROADMAP.md`, not part of this plan.)
- This repo has no automated test suite by design — tests are standalone `test_*.py` scripts, plain `assert`, no pytest.
- Commit messages are plain text — never add a `Co-Authored-By: Claude` trailer.
- Already on branch `smooth-observe-transition`, forked from `main`.
- Validation requires both: (a) `benchmark.py` for a success-rate regression check (this change only affects HOW the arm moves between waypoints, never which waypoints are targeted, so no regression is expected — confirm it), and (b) a live/visual check of the actual "snap" symptom being gone (extract frames around a round transition from a real `--pick-all` GIF, same technique used to validate the diff-IK elbow-flip fix).

---

### Task 1: Smooth the retract-then-observe transition

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/executor.py` (`GraspExecutor.place()`'s final retract line, and `GraspExecutor.go_observe()`)

**Interfaces:**
- Consumes: existing `_step_to(..., smooth: bool = False)` parameter (no signature change).
- Produces: nothing new for later tasks (this is the only task in this plan).

- [ ] **Step 1: Update `place()`'s final retract call**

In `mujoco_grasp_sim/sim_grasp/executor.py`, find this line (near the end of `place()`):

```python
        self._step_to(ik_pre.qpos, 0.8, gripper_ctrl=GRIPPER_OPEN)     # retract
```

Replace with:

```python
        self._step_to(ik_pre.qpos, 0.8, gripper_ctrl=GRIPPER_OPEN, smooth=True)     # retract
```

Then find the comment just above `place()`'s two already-smoothed lines (currently reads, in full):

```python
        # smooth=True only for the two motions performed while still
        # holding the object -- linear interpolation's instant velocity
        # jump at the start of a move is a real cause of slip; release and
        # retract happen open-handed, so they're left as linear.
```

Replace with:

```python
        # smooth=True on the transit/lower motions guards against slip
        # (linear interpolation's instant velocity jump at the start of a
        # move is a real cause of it) while still holding the object.
        # The retract below is also smoothed -- not for slip (nothing is
        # held), but so it decelerates to zero velocity right where the
        # next round's go_observe() move picks up, instead of stopping
        # abruptly and immediately snapping into a new motion.
```

- [ ] **Step 2: Update `go_observe()`**

Find:

```python
    def go_observe(self, q_observe, duration=2.5):
        """Joint move back to the observation pose (gripper open) so the next
        capture sees the table without the arm in the frustum."""
        self._step_to(np.asarray(q_observe, dtype=float), duration,
                      gripper_ctrl=GRIPPER_OPEN)
        self._hold(0.3)
```

Replace with:

```python
    def go_observe(self, q_observe, duration=2.5):
        """Joint move back to the observation pose (gripper open) so the next
        capture sees the table without the arm in the frustum. Smoothed so
        this move (and the retract that typically precedes it in place())
        together read as one continuous decelerate-then-reaccelerate,
        instead of two separately-abrupt stop/start segments."""
        self._step_to(np.asarray(q_observe, dtype=float), duration,
                      gripper_ctrl=GRIPPER_OPEN, smooth=True)
        self._hold(0.3)
```

- [ ] **Step 3: Syntax/import sanity check**

```bash
cd mujoco_grasp_sim
python -c "import ast; ast.parse(open('sim_grasp/executor.py').read()); print('SYNTAX OK')"
```

- [ ] **Step 4: Regression benchmark**

From `mujoco_grasp_sim/` (`conda activate cgn_torch`, `MUJOCO_GL=osmesa`, `GRASPGEN_PYTHON=/home/vivek/miniconda3/envs/graspgen_torch/bin/python`):

```bash
python benchmark.py --seeds 0-9 --mode pick-all --camera fused --backend graspgen --tag smooth_observe
python analyze_failures.py output/bench_smooth_observe
```

Expected: success rate/taxonomy matching (not worse than) the last recorded GraspGen baseline in `ROADMAP.md` (30/30 binned, 100%, from the diff-IK elbow-flip fix's own validation) — this change only affects the SHAPE of the joint trajectory between waypoints, never which waypoints are targeted or whether a pick/place succeeds, so no regression is expected. Report the actual numbers.

- [ ] **Step 5: Visual confirmation of the fix**

Using a saved `execution.gif` from Step 4's benchmark (pick any seed with 2+ rounds), extract several frames spanning a round-transition (end of one object's place → go_observe → start of the next object's pick) and inspect them for smoother, less "snappy" motion compared to before. This is a qualitative check (there's no numeric "smoothness" metric in this codebase) — describe what you actually observe rather than asserting success you can't back up; report `DONE_WITH_CONCERNS` if the frames aren't conclusive one way or the other.

- [ ] **Step 6: Record the result and commit**

Add a dated bullet to `ROADMAP.md`'s P1 section (matching the format of the neighboring DiffIK elbow-flip and short-object-collision entries) describing the change and the before/after benchmark numbers, per `AGENTS.md`'s mandatory documentation-sync workflow. Also add a `[ ]` deferred-idea bullet (if not already adequately covered by the diff-IK fix's own deferred nullspace note) capturing the Tier 2 idea from brainstorming: skipping the fixed `ARM_OBSERVE_QPOS` detour entirely when it isn't actually needed for camera-frustum clearance for the next target.

```bash
git add mujoco_grasp_sim/sim_grasp/executor.py ROADMAP.md
git commit -m "Smooth the retract-then-observe transition between pick-all rounds"
```

---
