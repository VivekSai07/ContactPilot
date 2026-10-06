# P10 SP3 Perceived-Category Sorting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route props to bins using the selected scene-graph node's category and measure the result against a paired oracle-routing benchmark.

**Architecture:** An explicit routing mode preserves the existing oracle default. A small pure policy resolves a category to a configured bin; the pick-all consumer passes either graph or oracle category to it. Simulator truth is used only for offline outcome metrics in graph mode.

**Tech Stack:** Python 3.10, MuJoCo, NumPy, existing standalone assert scripts, GraspGen subprocess, NVIDIA NIM.

**Spec:** `docs/superpowers/specs/2026-10-06-semantic-sorting-sp3-perceived-routing-design.md`

## Global Constraints

- `--routing oracle` remains the default; boxes and single-pick behavior stay unchanged.
- `--routing scene-graph` requires props, pick-all, and `--scene-graph`.
- `--identity oracle` supplies a true name to NIM, not a true category.
- Missing/invalid perceived category raises; never fall back to simulator truth.
- GraspGen/fused, seeds 0-9, props pick-all is the paired benchmark configuration.
- Keep simulator truth out of route and retry decisions; retain it for offline scoring.
- Follow `AGENTS.md`: feature branch + PR; dated ROADMAP entry with actual evidence for new capability; explain every fix site in a one-line comment.

## Review Focus

- A graph node that vanishes after ranking must fail clearly, never route from truth (Task 1).
- A category outside the two-bin mapping must fail clearly (Task 1).
- A successfully picked object that remains on the table must still get a bounded retry without using the qpos oracle (Task 2).
- A wrong-bin placement must count as binned but incorrectly sorted (Task 3).
- Partial benchmark completion must show crashed seeds rather than inflating success (Task 3).

---

### Task 1: Routing policy and CLI contract

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/sorting_policy.py`
- Create: `mujoco_grasp_sim/sim_grasp/test_sorting_policy.py`
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py` (argument parsing, validation, bin decision)
- Modify: `mujoco_grasp_sim/benchmark.py` (argument parsing and passthrough)

**Interfaces:**
- `target_bin(category: str, bins: list[BinSpec]) -> BinSpec` raises `ValueError` for unknown/missing category or ambiguous mapping.
- `run_sim_grasp_test.py --routing {oracle,scene-graph}` and `benchmark.py --routing {oracle,scene-graph}`.

- [x] Write synthetic policy assertions: food→A, non_food→B, missing/unknown category raises, duplicate category mapping raises; use a graph node category with a deliberately conflicting truth-category sentinel.
- [x] Run `PYTHONPATH=. python sim_grasp/test_sorting_policy.py` from `mujoco_grasp_sim/`; verify red before implementation.
- [x] Implement `target_bin` and the CLI guards. In pick-all, after candidate ranking use the current graph node category for scene-graph routing and the existing truth category only in oracle routing. Keep single `--execute` on its existing oracle path.
- [x] Re-run the policy test and CLI `--help`/invalid-combination checks; verify green and early rejection.
- [x] Commit policy, tests, and CLI wiring.

### Task 2: Graph-only retry control and decision provenance

**Files:**
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py` (pick-all retry branch and per-round log)
- Create: `mujoco_grasp_sim/sim_grasp/test_sorting_round.py` if the retry/decision helper is extracted; otherwise extend `test_sorting_policy.py`

**Interfaces:**
- Graph-route per-round `decision = {category_source, category, target_bin}`.
- Retry helper or inline branch takes `routing`, pick success, and prior failure count; it does not take simulator bin membership for graph routing.

- [x] Write a failing test for graph-routed attempt accounting that is unchanged whether the simulator reports a landed bin or none.
- [x] Run the targeted standalone script; confirm red.
- [x] Implement graph-route attempt accounting and decision logging. Leave oracle branch behavior unchanged; keep truth-based `landed_bin` only as post-action evaluation.
- [x] Re-run targeted tests and inspect `run_sim_grasp_test.py` for reads of `gen.object_categories`/`gen.objects_in_bins()` inside graph-route decisions.
- [x] Commit the retry/decision change.

### Task 3: Offline evaluation and benchmark reporting

**Files:**
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py` (per-round `evaluation`, final metrics)
- Modify: `mujoco_grasp_sim/benchmark.py` (mode label, aggregate counts)
- Modify: `mujoco_grasp_sim/analyze_failures.py` (wrong-category landed-in-target case)
- Create or extend: `mujoco_grasp_sim/sim_grasp/test_sorting_metrics.py`
- Extend: `mujoco_grasp_sim/sim_grasp/test_analyze_failures_wrong_bin.py`

**Interfaces:**
- Keep `pick_all.in_bin`, `in_correct_bin`, `in_wrong_bin` as existing truth-scored lists.
- Add per-round `evaluation = {gt_category, landed_bin, correct_bin}`; use `None` when no bin result is available.
- Benchmark output identifies `routing` and `identity` and retains `crashed` per seed.

- [x] Write failing synthetic assertions for correct bin, wrong bin, unbinned object, a wrong graph category that lands in its selected bin, and a crashed seed excluded from numerator but reported in completed/crashed counts.
- [x] Run the targeted scripts; confirm red.
- [x] Implement metric grouping, wrong-bin failure taxonomy, and benchmark labels/counts without changing existing JSON list meanings.
- [x] Re-run targeted tests and `git diff --check`.
- [x] Commit metrics/reporting.

### Task 4: End-to-end gate and documentation

**Files:**
- Modify: `ROADMAP.md` P10
- Modify: `mujoco_grasp_sim/README.md` P10 usage
- Modify: `docs/superpowers/specs/2026-10-06-semantic-sorting-sp3-perceived-routing-design.md` Implementation notes only if the implementation diverges

**Interfaces:** benchmark command line and `metrics.json` from Tasks 1-3.

- [x] Run all `sim_grasp/test_*.py` in `cgn_torch` with `MUJOCO_GL=osmesa`; 26 passed, one known settled-qpos hash failure; prop tests ran with local downloaded assets.
- [x] Run one `--scene props --pick-all --scene-graph --routing scene-graph --identity perceived --camera fused --backend graspgen` smoke; seed 0 sorted 4/4 with matching `decision` and `evaluation` fields.
- [x] Run three paired `benchmark.py --seeds 0-9 --mode pick-all --scene props --camera fused --backend graspgen` batches: oracle; graph routing with perceived identity; graph routing with oracle identity. Use distinct tags and preserve summary files. All three cohorts completed 10/10.
- [x] Record exact aggregate output, per-seed comparison, crashes, wrong-bin and failure stages in ROADMAP. The graph decision/retry path has no simulator-truth input; the perceived 29/40 vs fresh oracle 36/40 gap is documented, not relabelled as a performance win.
- [x] Run a boxes-mode regression smoke and `git diff --check`; commit the evidence-backed docs. The live boxes seed-0 smoke completed 3/3, and the three raw summaries are tracked under `docs/benchmarks/2026-10-06-sp3/`.
- [x] Push branch and open PR #34 to `main`; include commands, outputs, limitations, and no AI attribution. Review found and prompted the unbinned-evaluation and default-identity reporting corrections, each covered by a red/green standalone test.
