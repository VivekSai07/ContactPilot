# Handoff: Intelligent Bin Placement — implementation ready to start

**For:** whichever Claude Code session picks this up next (this was
brainstormed/planned in a different session, on a different machine, than
whichever one implements it).

## TL;DR

A full design + implementation plan for "make the pick-and-place bin
placement as intelligent as the pick phase" is written, self-reviewed, and
pushed. **Nothing has been implemented yet.** Your job: check out the
branch below and execute the plan task-by-task.

```bash
git fetch origin
git checkout intelligent-bin-placement
git pull --ff-only
```

Branch is based on `origin/main` at commit `529d673` (no conflicts expected
if your local main is up to date — `git log --oneline -1 origin/main` to
confirm). It currently contains exactly 2 new files and nothing else:
- `docs/superpowers/specs/2026-08-20-intelligent-bin-placement-design.md` — the approved design
- `docs/superpowers/plans/2026-08-20-intelligent-bin-placement.md` — the 7-task implementation plan (has its own required-header pointing at the `subagent-driven-development` skill)

## What this feature is

`mujoco_grasp_sim`'s **pick** phase is intelligent (GraspGen/Contact-GraspNet
+ SAM3 object selection), but the **place** phase is currently hardcoded:
every object is dropped at the exact same fixed world point
(`SceneGenerator.bin_drop_point()`) at a fixed release height
(`executor.PLACE_RELEASE`). Two concrete bugs result:

1. The 2nd/3rd object in a `--pick-all` run lands on top of the 1st
   (no XY offset between placements) — loses the grasp or knocks the first
   object out of the bin.
2. A tall cuboid gets picked correctly but the fixed release height doesn't
   account for its actual height, jamming/crushing it against the bin floor.

The fix (full design in the spec doc) replaces this with a **vision-only**
placement planner — no MuJoCo internal-state queries (`model.geom_size`,
`data.body().xpos`, etc.), only depth/segmap/intrinsics/extrinsics through
the existing `CameraModule`/`depth_to_pointcloud()` pipeline — specifically
so the same logic is portable to a real RealSense camera later (this was an
explicit user requirement, not my own addition).

## How to execute

1. Read `docs/superpowers/plans/2026-08-20-intelligent-bin-placement.md` in full.
2. It requires the `subagent-driven-development` skill (recommended) or
   `executing-plans` — both are named explicitly in the plan's required
   header. Follow whichever this session's `superpowers` skill set provides.
3. The plan is 7 tasks, in dependency order:
   - Tasks 1-3: a new pure-Python module `sim_grasp/placement_planner.py`
     (object footprint from a segmap mask, bin heightmap from depth, a
     free-space + orientation search). Each has full TDD steps with
     complete test code and complete implementation code already written
     out in the plan — no placeholders, should be close to copy-paste.
   - Task 4: `executor.py`'s `GraspExecutor.place()` changes signature from
     `place(drop_pos)` to `place(x, y, release_z, yaw=0.0)`.
   - Tasks 5-6: wire the new planner into `run_sim_grasp_test.py` (both the
     `--pick-all` loop and the single `--execute` path) and
     `interactive_pick.py`.
   - Task 7: validation — visual sanity runs, a `benchmark.py` before/after
     comparison (multi-seed, matching this project's established "never
     trust a single run" convention), and recording the real resulting
     numbers in `ROADMAP.md`/`README.md`.
4. This codebase has **no pytest** — the plan's unit tests follow the
   existing convention exactly: standalone `test_*.py` scripts in
   `sim_grasp/`, run directly (`python sim_grasp/test_foo.py`), plain
   `assert` statements, ending with a `print('All ... checks passed.')`
   line. See `sim_grasp/test_resolve_real_label.py` for the reference style
   if you want another example beyond what's already in the plan.
5. During my own self-review of the plan I found and fixed one real bug in
   Task 2's synthetic test (an out-of-bounds pixel coordinate from reusing
   the wrong camera intrinsics) — the version in the plan now is the
   corrected one, already verified by hand. Still worth double-checking
   the numbers if anything looks off when you actually run it.

## Things NOT to touch / be aware of

- **Do not modify `graspgen-submodule-plan` or anything related to it** —
  it was already merged (PR #15, GraspGen is now a git submodule pinned to
  `2dd8852`). If your session is the one that did that, disregard; just
  flagging in case this note is stale by the time you read it — check
  `git log --oneline --all | grep -i graspgen` if unsure.
- There's also a `docker-support` branch preserved on origin (Docker
  support was added then deliberately removed from `main`, per PR #16 —
  the branch is kept around in case it's wanted later, not because it's
  in-progress work). Leave it alone unless separately asked.
- If another session is concurrently active on this repo, **check
  `git branch -a` and `git log --oneline -20` before assuming the state
  described here is still current** — this repo has had multiple sessions
  (mine and at least one other, on a different OS) both actively merging to
  `main` recently.

## Environment notes (cross-checked against `/memories/repo/wsl2-gpu-display.md`)

Some of these are WSL2/Linux-specific and won't apply verbatim on Windows,
but the underlying lessons do:
- Never `conda install` anything (especially `ffmpeg`) into the `cgn_torch`
  env — it can pull in shared libraries that break `pyrender`. Use a
  separate throwaway env for any non-Python tooling.
- `ContactGraspNetPredictor` must never be constructed in-process in a
  script that's already rendered with MuJoCo in the same process — route
  through the existing subprocess helper
  (`predict_in_subprocess`/`cgn_worker.py`) instead. Not directly relevant
  to this plan (the new placement code has nothing to do with CGN
  in-process rendering), but worth knowing if you touch `run_sim_grasp_test.py`
  near that code.
- CGN's grasp proposals are stochastic even for a fixed seed — if a
  demo/validation run needs a specific outcome, expect to retry.
- This project's rule for any reliability-affecting change: never trust a
  single sim run — validate via `benchmark.py` across multiple seeds
  (`--seeds 0-9` or similar), which Task 7 of the plan already accounts for.

## Current git state at handoff time

```
Branch: intelligent-bin-placement (pushed to origin, based on main @ 529d673)
Local main is up to date with origin/main.
Working tree otherwise clean.
```
