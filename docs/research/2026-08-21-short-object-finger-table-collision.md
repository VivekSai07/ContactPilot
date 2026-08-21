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
work, found incidentally during the same round of live-testing feedback
that led to the placement-pose robustness fixes (corner-lock,
transit-slip — see `ROADMAP.md` P7 and
`docs/superpowers/plans/2026-08-21-placement-pose-robustness.md`).
