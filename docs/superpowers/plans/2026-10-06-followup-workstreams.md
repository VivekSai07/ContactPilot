# ContactPilot Follow-up Workstreams (2026-10-06)

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

## 2. Reproducible environment check — after SP3

**Goal:** A single fast command reports whether the main `cgn_torch` runner
can import and identify the required packages, and whether optional model
interpreters and assets are configured. It does not initialize GPU models
or make network calls.

**Files:** `mujoco_grasp_sim/scripts/check_environment.py` (new), a standalone
assert script for its pure checks, `CLAUDE.md`, and
`mujoco_grasp_sim/README.md`.

**Steps:**

1. Define required checks for Python, NumPy `<2`, `mujoco`, `imageio`, `cv2`,
   `open3d`, the Panda submodule, and default render mode. Report interpreter
   path, import/version result, and exact setup guidance on failure. Report
   `GRASPGEN_PYTHON`, `SAM3_PYTHON`, and `NVIDIA_API_KEY` as optional feature
   readiness without printing the key or importing their heavy packages.
2. Test the pure check renderer with synthetic pass/fail results, including
   a missing `imageio` and NumPy 2.x; confirm nonzero exit on required failure.
3. Run the command in `cgn_torch` and a deliberately incomplete interpreter.
   Document both outputs and the command. Do not claim that an import check
   proves CUDA kernels, rendering, NIM service, or grasping work.

The earlier `imageio` import failure came from invoking system Python, not
proof of a missing dependency in `cgn_torch`.

## 3. Standalone tests versus pytest — evaluate, then migrate only if useful

**Goal:** Better collection and failure reporting without losing direct
script execution or masking the settled-`qpos` issue.

**Files if migration is justified:** `pyproject.toml` or test config,
`sim_grasp/test_*.py` as needed, and `CLAUDE.md`.

**Steps:**

1. Inventory all test scripts and their prerequisites (pure, MuJoCo/Panda,
   downloaded GSO assets, GPU/network). Run the current loop in `cgn_torch`
   and record the baseline failures.
2. Prototype pytest collection on a temporary branch without rewriting
   assertions. Compare collection time, failure diagnostics, and ability to
   run the same scripts directly. If there is no practical gain, record the
   finding and keep the existing runner.
3. If there is a gain, add minimal collection/configuration and dependency
   documentation. Keep assertions and direct execution working, and mark
   missing optional assets explicitly instead of silently hiding failures.
4. Investigate the settled-`qpos` hash against a same-MuJoCo-version pre-P10
   commit before changing that guard. An updated hash from one host is not a
   valid fix. Run both runners and show matching results before the PR.

## 4. Decision versus evaluation metrics — audit after SP3

SP3's spec establishes `decision` and `evaluation` fields for sorting.
After its PR, audit every `gen.object_categories`, `gen.object_props`,
`objects_on_table`, and `objects_in_bins` read in the graph route. Trace
whether each read affects ranking, retry, placement, termination, or only
offline scoring. Add a synthetic adversarial case in which simulator truth
disagrees with the graph and verify the route and retry sequence are
unchanged. Document metric provenance in `mujoco_grasp_sim/README.md` and
in the benchmark output. Put a broader metric schema migration in its own
spec only if the audit finds another consumer needs it.

## 5. Finish the pending live-viewer edit — separate branch/PR

**Starting state:** the main checkout has an uncommitted `executor.py` edit
that accepts a passive viewer and calls `viewer.sync()` each physics step.
No caller currently passes a viewer to `GraspExecutor`, so that edit alone
cannot provide live execution viewing.

**Files:** `mujoco_grasp_sim/sim_grasp/executor.py`,
`mujoco_grasp_sim/run_sim_grasp_test.py`, and, only if its UI contract needs
the feature, `interactive_pick.py`; update ROADMAP with observed behavior.

**Steps:**

1. Preserve the existing diff on a feature branch. Inspect `--view-sim`'s
   viewer lifecycle and decide which run paths own/pass the handle. Add a
   small regression only if it checks an actual lifecycle or sync failure,
   rather than mirroring a one-line `sync()` call.
2. Wire the handle to the executor, ensure it closes after execution, and
   keep headless runs free of viewer calls. Check that syncing every physics
   step does not cause unacceptable slowdown; cap cadence if measurement
   shows a problem.
3. Run headless executor tests and a real `--view-sim --execute` smoke with
   visual confirmation. Record observed output and any limitation in a dated
   ROADMAP P1/P6 entry, then open its own PR.

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
