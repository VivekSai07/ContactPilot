# ContactPilot Follow-up Workstreams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the five suggestions in a safe sequence, with SP3 first and independently reviewable follow-ups.

**Architecture:** SP3 has its own design and task plan. The environment doctor, test-runner decision, post-SP3 metric audit, and viewer lifecycle each use separate branches/PRs so a failure in one cannot obscure another.

**Tech Stack:** Python 3.10, MuJoCo, NumPy, existing standalone assert scripts, GraspGen, NVIDIA NIM, GitHub PRs.

**Spec:** `docs/superpowers/specs/2026-10-06-semantic-sorting-sp3-perceived-routing-design.md` for SP3; the other items are bounded follow-ups described below.

## Global Constraints

- Preserve all existing user edits until their exact targets are reviewed; never commit directly to `main`.
- Real benchmark evidence, not one smoke run, is required for grasp/sorting reliability claims.
- A broken viewer path must not become a public CLI option.
- Keep the default boxes path and oracle sorting path stable.

## Review Focus

- Missing optional model envs should be reported without printing credentials (workstream 2).
- Test-runner changes must expose, not hide, missing prop assets and the settled-qpos failure (workstream 3).
- A vision error that lands in the chosen but wrong category bin must score wrong (workstream 4).
- Closing the passive viewer mid-motion must not crash execution (workstream 5).
- Real RGB-D bin overlap in XY must not be assumed equivalent to physical bin containment (real-footage validation below).

---

This is a sequencing plan for five independent suggestions. P10 SP3 has its
own [design](../specs/2026-10-06-semantic-sorting-sp3-perceived-routing-design.md)
and [implementation plan](2026-10-06-semantic-sorting-sp3-perceived-routing.md).
Each remaining change gets its own feature branch and PR under `AGENTS.md`.
Do not hold SP3 behind the environment/test-runner work.

## 1. P10 SP3 — do first

Implement the linked SP3 plan. Route from graph category, preserve oracle
routing as a baseline, compare paired 10-seed GraspGen/fused batches, and
keep simulator truth confined to offline evaluation. This work also addresses
suggestion 4's immediate leakage risk in the sorting pipeline.

**2026-10-06 status:** implemented and validated on the SP3 branch; paired
correct-bin scores are oracle 36/40, graph/perceived 29/40, and
graph/oracle-identity 33/40, with 10/10 completed in each cohort. The
29/40 result leaves a real performance gap; the post-SP3 metrics audit and
identification improvements remain follow-up work.

## 2. Reproducible environment check — after SP3

**Goal:** A single fast command reports whether the main `cgn_torch` runner
can import and identify the required packages, and whether optional model
interpreters and assets are configured. It does not initialize GPU models
or make network calls.

**Files:** `mujoco_grasp_sim/scripts/check_environment.py` (new), a standalone
assert script for its pure checks, `CLAUDE.md`, and
`mujoco_grasp_sim/README.md`.

**Steps:**

- [ ] Define required checks for Python, NumPy `<2`, `mujoco`, `imageio`, `cv2`,
   `open3d`, the Panda submodule, and default render mode. Report interpreter
   path, import/version result, and exact setup guidance on failure. Report
   `GRASPGEN_PYTHON`, `SAM3_PYTHON`, and `NVIDIA_API_KEY` as optional feature
   readiness without printing the key or importing their heavy packages.
- [ ] Test the pure check renderer with synthetic pass/fail results, including
   a missing `imageio` and NumPy 2.x; confirm nonzero exit on required failure.
- [ ] Run the command in `cgn_torch` and a deliberately incomplete interpreter.
   Document both outputs and the command. Do not claim that an import check
   proves CUDA kernels, rendering, NIM service, or grasping work.
- [ ] Commit on a new branch and open a PR with the exact passing and failing
   check outputs. Acceptance: one command identifies each required missing
   dependency with a nonzero exit, never reveals the NVIDIA key, and does
   not start a model worker or network request.

The earlier `imageio` import failure came from invoking system Python, not
proof of a missing dependency in `cgn_torch`.

## 3. Standalone tests versus pytest — evaluate, then migrate only if useful

**Goal:** Better collection and failure reporting without losing direct
script execution or masking the settled-`qpos` issue.

**Files if migration is justified:** `pyproject.toml` or test config,
`sim_grasp/test_*.py` as needed, and `CLAUDE.md`.

**Steps:**

- [ ] Inventory all test scripts and their prerequisites (pure, MuJoCo/Panda,
   downloaded GSO assets, GPU/network). Run the current loop in `cgn_torch`
   and record the baseline failures.
- [ ] Prototype pytest collection on a temporary branch without rewriting
   assertions. Compare collection time, failure diagnostics, and ability to
   run the same scripts directly. If there is no practical gain, record the
   finding and keep the existing runner.
- [ ] If there is a gain, add minimal collection/configuration and dependency
   documentation. Keep assertions and direct execution working, and mark
   missing optional assets explicitly instead of silently hiding failures.
- [ ] Investigate the settled-`qpos` hash against a same-MuJoCo-version pre-P10
   commit before changing that guard. An updated hash from one host is not a
   valid fix. Run both runners and show matching results before the PR.
- [ ] If migration proceeds, commit its config/docs/tests on a separate
   branch and open a PR. Acceptance: `python -m pytest sim_grasp` and a loop
   invoking each `test_*.py` directly cover the same cases;
   the known seed-0 hash failure remains visible until independently fixed.
   If migration is rejected, record the comparison and reason in the PR
   instead of adding a pytest dependency for its own sake.

## 4. Decision versus evaluation metrics — audit after SP3

SP3's spec establishes `decision` and `evaluation` fields for sorting.

- [ ] After its PR, audit every `gen.object_categories`, `gen.object_props`,
  `objects_on_table`, and `objects_in_bins` read in the graph route. Trace
  whether each read affects ranking, retry, placement, termination, or only
  offline scoring.
- [ ] Add an adversarial case where simulator truth disagrees with the graph;
  verify the route and retry sequence are unchanged.
- [ ] Document metric provenance in `mujoco_grasp_sim/README.md` and the
  benchmark output. Create a broader metric schema spec only if another
  consumer demonstrably needs one.
- [ ] Commit the audit/tests on a new branch and open a PR. Acceptance:
  changing oracle truth while holding the graph observation fixed changes
  only truth-labelled evaluation, not chosen bin, retry budget, or next
  candidate; summarize any remaining simulator ID association explicitly.

## 5. Finish the pending live-viewer edit — separate branch/PR

**Original starting state:** the main checkout had an uncommitted
`executor.py` edit that accepts a passive viewer and calls `viewer.sync()`
each physics step. No caller currently passes a viewer to `GraspExecutor`,
so that edit alone cannot provide live execution viewing.

**2026-10-06 update:** the edit and a real-physics/fake-viewer regression are
committed in draft PR #33; the main checkout is clean. A direct passive-viewer
smoke opened and synced, then hung after close and exited 139 under both
OSMesa and GLFW on WSL. The attempted CLI wiring was removed. This remains
an investigation, not a shipped live-viewer feature.

**Files:** `mujoco_grasp_sim/sim_grasp/executor.py`,
`mujoco_grasp_sim/run_sim_grasp_test.py`, and, only if its UI contract needs
the feature, `interactive_pick.py`; update ROADMAP with observed behavior.

**Steps:**

- [x] Preserve the existing diff on a feature branch and inspect `--view-sim`'s
   viewer lifecycle and decide which run paths own/pass the handle. Add a
   small regression only if it checks an actual lifecycle or sync failure,
   rather than mirroring a one-line `sync()` call.
- [ ] Wire the handle to the executor only after its WSL shutdown behavior is
   understood; ensure it closes after execution, and
   keep headless runs free of viewer calls. Check that syncing every physics
   step does not cause unacceptable slowdown; cap cadence if measurement
   shows a problem.
- [ ] Run headless executor tests and a real live-viewer `--execute` smoke with
   visual confirmation. Record observed output and any limitation in a dated
   ROADMAP P1/P6 entry, then open its own PR.
- [ ] Measure the live sync overhead against headless execution and keep
   PR #33 draft until the WSL close/exit path returns cleanly (exit 0, no
   hang/segfault). Acceptance: fake-viewer regression, headless run, and
   live pick-and-place all pass; do not merge on fake-viewer evidence alone.

## Clarifications from the earlier review

- `PromptSelector.select(click=...)` passes `click_radius_px` to the SAM 3
  worker, where it sets the geometric prompt box. Public click picking calls
  `click_to_select()` and uses text detection plus click disambiguation, so
  `interactive_pick.py --click-radius-px` does not change that current path.
  Clarify the Claude memory note; do not delete the low-level option as part
  of SP3.
- The README progress table already reached P7 on 2026-08-21; PR #32 adds
  P8-P10 rows. Resolve PR #32 before writing another conflicting progress
  summary.
- A perceived route can be evaluated with simulator truth, but its
  *correct-bin* score is an offline truth-labelled score, not perception-only.
- SP2's XY-only bin rule agreed 100% with simulator locations in its ten
  props seeds. Before hardware use, collect real RGB-D plus object masks,
  annotate table/bin/unknown locations, and measure false bin positives,
  especially objects overlapping a bin in XY but above/outside it.
- The default-scene hash failure was reproduced under MuJoCo 3.11.0:
  generated XML matches, settled `qpos` differs. PR #32 records it. Do not
  rewrite the expected hash without a same-version baseline investigation.
