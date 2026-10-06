# P10 SP3 — Perceived-Category Sorting — Design

**Parent:** `2026-09-29-semantic-sorting-scene-graph-design.md`.
**Starting point:** SP1 routes 37/40 props correctly using simulator categories;
SP2 builds a vision-only graph, names and categorizes 33/40 correctly with
perceived identity, but still routes by simulator truth. This spec completes
the planned consumer. Results quoted here are the recorded 2026-10-03 gates,
not a prediction of SP3 performance.

## Goal and boundary

In a props `--pick-all --scene-graph` run, let the category stored on the
selected graph node choose bin A (`food`) or B (`non_food`). Neither the
simulator's prop identity/category nor its object pose/bin occupancy may be
consulted to choose a destination or decide whether to retry. Simulator
truth remains available after decisions for offline scoring and diagnostics.
Boxes mode, single `--execute`, and existing oracle-routing runs retain
their behavior. Real-camera footage and hardware operation are outside SP3.

## Interface and modes

- Add `--routing {oracle,scene-graph}` to `run_sim_grasp_test.py` and
  `benchmark.py`, default `oracle`. `scene-graph` requires `--scene props`,
  `--pick-all`, and `--scene-graph`; reject invalid combinations before
  simulation/model startup. The explicit switch preserves the meaning of
  prior SP1/SP2 commands and metrics.
- Keep `--identity {perceived,oracle}` independent of routing. The primary
  SP3 mode is `--routing scene-graph --identity perceived` (or the default
  identity when `--scene-graph` is supplied). The oracle-identity mode still
  feeds the true *name* into the NIM categorizer, giving a knowledge-only
  comparison, not true-category routing.
- Keep `--routing oracle --scene-graph` as the SP2 measurement mode.
  `--routing oracle` without a graph remains the SP1 mode.

## Decision path

1. The existing per-round graph and `KnowledgeCache` fill each node's
   `identity` and `category`; cached categories are reused across rounds.
2. After ranking a candidate, look up the selected `seg_id` in the graph.
   For scene-graph routing, require that node to exist, be a current table
   node, and have a category exactly equal to one of the configured bin
   categories. Resolve the destination from category to `BinSpec`. There is
   no fallback to `gen.object_categories` on missing/invalid data; the
   run raises an actionable error. In oracle mode, retain the existing
   truth-category route.
3. Feed the chosen `BinSpec` to the existing footprint, heightmap, free-space
   planner, and same-bin fallback drop point. No placement geometry changes.
4. In graph-routing mode, graph table nodes remain the source of remaining
   work. A successful physical pick consumes an attempt in the per-object
   retry budget; the next graph observation decides whether that object is
   still pickable. This avoids the current `gen.objects_in_bins()` result
   changing retry state. Failed picks continue to consume an attempt.
   The existing global round limit remains in force.

## Data separation and metrics

`metrics.json` records `routing: "oracle" | "scene-graph"` and keeps the
existing `pick_all.in_bin`, `in_correct_bin`, `in_wrong_bin` lists for
compatibility. These final outcome lists are explicitly **offline
simulator-truth evaluation**; a real robot cannot compute them unaided.

Each props round records a `decision` object containing `category_source`,
`category`, and `target_bin`. The optional `evaluation` object records
`gt_category`, `landed_bin`, and `correct_bin` from simulator truth after
the action. Keep legacy top-level per-round fields if existing consumers
need them, but never read `evaluation` to make decisions. The benchmark
summary names both routing and identity modes, aggregates correct-bin and
wrong-bin counts over all spawned props, and reports category accuracy and
location agreement separately. A run with a crashed seed does not silently
report success on only completed seeds: show completed/crashed counts and
preserve every seed's error.

`scene_graph` node category and location remain perception-derived. SP3 does
not claim a perception-only *correctness* metric: correctness needs an
independent category label, supplied by the sim only for offline evaluation.

## Validation

Use standalone assert scripts with synthetic graph nodes and bins to prove
category-to-bin routing, invalid/missing-category rejection, oracle mode,
and no truth-category read in the graph decision path. Test CLI validation,
retry accounting, metric provenance, and benchmark aggregation without NIM.
Run the existing `test_*.py` scripts, recording the known settled-qpos hash
failure separately if it reproduces; downloaded GSO assets are required for
the prop tests. Run a live one-seed smoke for graph routing, then paired
GraspGen/fused props pick-all batches on seeds 0-9:

1. `--routing oracle` (current-branch baseline);
2. `--scene-graph --routing scene-graph --identity perceived` (SP3);
3. `--scene-graph --routing scene-graph --identity oracle` (knowledge-only).

Report numerator/denominator, wrong-bin, knocked-off, crashes, category
accuracy, and the paired per-seed gap from the fresh oracle baseline.
The historical 37/40 SP1 and 36/40 SP2 scores are context, not substitutes
for the paired comparison. A release candidate requires 10/10 completed
runs, no truth-derived routing/retry decision in code review, and no boxes
regression. Do not invent a correct-bin result or lower a gate after seeing
the batch; if the graph route underperforms, report it and diagnose it.

## Known limits and out of scope

SP2's bin membership uses XY votes without a height cutoff. Its measured
100% agreement in the props sim is evidence for that scene only; validate
with real RGB-D/segmentation footage before hardware use. Model availability
is hosted NIM dependent. A local VLM fallback, pytest migration, general
environment doctor, and the pending viewer integration are separate tasks.

## Implementation notes

Record any divergence from this design here with the reason and observed
behavior; keep the original design statements intact.
