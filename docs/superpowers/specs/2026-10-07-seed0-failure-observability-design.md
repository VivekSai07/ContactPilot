# Seed-0 sorting failure observability — design

**Status:** proposed for review. **Scope:** the first, diagnostic stage of the
post-SP3 reliability sequence. Transport, recovery, placement, and perception
changes require later designs and separate evidence gates.

## Purpose and observed case

The 2026-10-06 SP3 perceived-routing cohort finished with 29/40 props in the
correct bin, 5/40 in the wrong bin, and 6/40 left on the table. Its seed-0
`execution.gif` shows the white Fondant box tip onto the tabletop between
frames 40 and 42 while the blue Crunch box is carried toward a bin. Seed-0
perceived-routing metrics end with Fondant (`obj_1`) on the table, not off it;
later scene-graph rounds still call it a table object, but no further pick is
logged. The oracle-route seed 0 finishes 4/4. This is an observed failure
sequence, **not** proof of which bodies contacted or why no later grasp was
available. The raw GIF and per-seed metrics are local, ignored artifacts under
`mujoco_grasp_sim/output/bench_sp3_graph_perceived_20261006/seed_0/`; the
tracked cohort summary is in `docs/benchmarks/2026-10-06-sp3/`.

The outcome of this stage is an attributable answer to two questions:

1. Which contact or motion first displaced Fondant, in which execution phase?
2. At each later observation, did GraspGen return no Fondant grasps, did a
   filter remove them, or did eligibility/retry control reject them?

No reliability improvement is claimed merely because tracing is added.

## Boundary and approach

Use a compact, opt-in diagnostic trace rather than full per-step state dumps.
Add `--trace-pick-all` to the run script and pass it through `benchmark.py`.
Reject it outside `--pick-all`. Default runs retain current motion, routing,
retry, and output behavior. Tracing writes `execution_trace.json` beside
`metrics.json` and a small `pick_all.diagnostics` pointer/summary in metrics.
The trace must not include API keys, NIM requests, full images, or model
checkpoints.

All MuJoCo contacts and object poses in the trace are **offline simulator
diagnostics**. They must not enter `sorting_policy`, candidate ranking,
placement planning, retry decisions, or scene-graph node construction. The
trace must label simulator-truth fields separately from observation/pipeline
fields. Existing SP3 `decision` and `evaluation` provenance remains intact.

## Trace contracts

The trace is versioned (`schema_version: 1`) and identifies scene, seed,
backend, camera, identity/routing modes, code revision, and whether optional
filters were enabled. Stable object names/segmentation IDs and the current
target body are recorded for analysis only. Timestamps are simulation time;
wall time is a separate optional field.

### Execution phases and contacts

`GraspExecutor` emits named phase boundaries for pre-grasp, approach, close,
lift, carry-to-hover, lower, release, retract, and return-to-observe. A
collector observes `mjData` after each step when tracing is enabled, without
altering controls or advancing physics itself. For each unordered body pair
and phase, retain first/last contact time, step count, maximum penetration,
and maximum normal force when available. Preserve the underlying geom IDs so
a later analysis can distinguish the held prop, fingers, arm, bin, and another
prop. Record all prop-involving pairs; classify an interaction as expected or
unexpected only during analysis, not by silently discarding contacts at
capture time. Deduplicate in memory so the JSON is bounded by phase and pair,
not simulation steps.

At every phase boundary, snapshot world pose and linear velocity for each
prop under a `sim_truth` key. Save selected grasp and placement targets,
commanded joint waypoints/durations, the starting `qpos`/`qvel`/controls,
and a hash plus local copy of the generated scene XML needed for a later
inference-free replay. The replay still uses MuJoCo and the prop assets; it
must not call NIM or GraspGen.
These fields distinguish contact preceding a fall from contact after a fall.
If normal force is unavailable, record `null` and retain contact/time/pose
evidence; never fabricate a zero force.

### Candidate funnel and termination

Record one diagnostic entry for **every observation round**, including rounds
with no pick (today `pick_all.rounds` contains only attempted actions). For
each observed `seg_id`, count predicted grasps, grasps remaining after table/
width feasibility, after the optional neighbor/reachability filter, and
after graph/instruction eligibility. Record table-node status, retry count,
selected candidate if any, and an explicit stop/continue reason. Do not
conflate "predictor returned none" with "filter removed all" or "visible
but ineligible". In GraspGen mode, label the existing lowered-CGN-threshold
retry as ineffective for that backend; this stage reports that fact but does
not change its retry policy.

The final trace links each attempted pick to its phase/contact summary, then
reports immediate post-release bin assessment and the end-of-episode outcome
as distinct observations. A later changed result must remain visible, not
be overwritten by the final score. This stage adds no extra settling steps;
stable-placement verification belongs to the later placement stage.

## Failure handling and cost

Trace collection is opt-in because per-step contact inspection has overhead.
Normal benchmark runs must not create a trace file. A missing or unreadable
trace must not be interpreted as "zero contacts"; mark diagnostics incomplete
and preserve the run's existing result. If the trace writer fails after an
episode, report its error explicitly without silently changing the sorting
score. Bound output by aggregating contacts and phase snapshots, not by
dropping the last phases when a frame cap is reached. Do not infer a physical
cause from GIF proximity alone.

## Validation and diagnostic gate

- Unit checks with a small fake model/data fixture establish geom-to-body
  attribution, phase grouping, first-contact timing, force-unavailable
  handling, and no mutation of control or simulation state by the collector.
- Funnel checks cover no predictions, all candidates filtered, graph
  ineligibility, retry cap, and global round limit. They assert that a round
  with no action still appears in diagnostics and has exactly one reason.
- A deterministic small MuJoCo/scripted-waypoint replay exercises a prop
  contact and verifies that tracing reports the pair and phase. It does not
  need hosted NIM or GraspGen; a replay fixture is captured only after a
  traced collision, not invented from the old GIF.
- Run the tracked seed-0 perceived-routing command with tracing, retaining
  the exact invocation and raw trace locally. If stochastic inference means
  that run does not repeat the collision, record that outcome and repeat
  with a stated limit; do not claim root cause until a collision is traced.
  Compare the oracle and oracle-identity seed-0 runs as working/differing
  examples, without treating paired seeds as identical trajectories.
- The gate is met only when a trace identifies the first Fondant-displacing
  event and phase (a contact/body pair, or a clearly traced non-contact
  mechanism), explains the subsequent no-pick candidate funnel, and supports
  a repeatable scripted regression. Existing sorting and placement checks
  must pass, and tracing on/off must produce the same deterministic scripted
  controls/outcome. Report any unresolved causal gap.

After the gate, write a dated evidence bullet in `ROADMAP.md`, with before/
after diagnostic behavior and exact commands/results. If investigation notes
are added under `docs/research/`, update its README index in the same commit.
The next transport/recovery spec may use this evidence; it must not assume a
particular collision mechanism before the diagnostic gate is met.

## Implementation notes

None yet. Record any approved-design divergence here when implementation
lands, without rewriting the design rationale above.
