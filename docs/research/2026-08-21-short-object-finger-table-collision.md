# Short-object finger-table collision during grasp closing

**Status:** fixed (2026-08-21) — see
`docs/superpowers/plans/2026-08-21-short-object-grasp-feasibility.md` and
the dated `ROADMAP.md` P1 entry. `GraspFeasibilityChecker` now validates
the post-`EXTRA_APPROACH` pose (the one `GraspExecutor.execute()` actually
closes on), not just the originally predicted one.

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

## Fix (2026-08-21)

`GraspFeasibilityChecker` gained an `extra_approach` constructor parameter
(default `0.0`, backward compatible). `run_sim_grasp_test.py`'s
`filter_feasible()` now constructs the checker with
`extra_approach=EXTRA_APPROACH` (the same constant `GraspExecutor.execute()`
uses), so the table-collision check validates the pose actually closed on.

Validation (5 seeds, box-only/3-objects, fused camera, GraspGen backend):
- **Before/after on identical raw predicted grasps** (same prediction, two
  feasibility passes): 27/1687 grasps (1.6%) across the 5 seeds that were
  previously accepted are now correctly rejected once the 12mm
  post-approach descent is accounted for — 14, 3, 3, 7, 0 grasps caught
  on seeds 0-4 respectively (concentrated on 4 of 5 seeds; one seed had
  zero, consistent with the effect being object-height-dependent rather
  than universal).
- **Regression check** (`benchmark.py --seeds 0-4 --mode pick-all --camera
  fused --backend graspgen`, fix active): **15/15 objects binned (100%)**,
  0 knocked off table, 0 crashes — matches the last recorded P1 GraspGen
  baseline, confirming the extra rejections don't cost overall throughput.

As with the placement-pose-robustness validation, the sample size is small
and this run's object heights weren't deliberately biased toward the
short-object case the bug report describes, so the 1.6% figure is a lower
bound on the effect for a mixed-height object distribution, not a precise
measurement isolated to short objects specifically. The qualitative result
— a real, previously-unvalidated 12mm blind spot in the feasibility check
is now closed — is the more meaningful takeaway.

## Scope note

This is a **pick-phase** issue (`GraspFeasibilityChecker`/
`GraspExecutor.execute()`), not a placement-phase issue -- unrelated to
the `intelligent-bin-placement` branch's `placement_planner.py`/`place()`
work, found incidentally during the same round of live-testing feedback
that led to the placement-pose robustness fixes (corner-lock,
transit-slip — see `ROADMAP.md` P7 and
`docs/superpowers/plans/2026-08-21-placement-pose-robustness.md`).
