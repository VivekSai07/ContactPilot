# Neighbor-Object Collision + Workspace-Reachability Pre-Filter — Design

## Problem

`GraspFeasibilityChecker` (`sim_grasp/feasibility.py`) only rejects two
things: table collision and upward/underhand approach direction. It has
no way to reject a grasp whose gripper geometry would clip a *neighboring*
object on the table, and no cheap way to reject a grasp whose target is
kinematically implausible for the Panda before ever running IK — today
that only shows up as an `ik_unreachable` failure discovered live during
execution (`analyze_failures.py`'s taxonomy shows this happening on 3-4
out of ~20-40 attempts per benchmark run — real, but modest).

## Decisions (confirmed with the user)

- **Scope of "neighbor collision"**: reject a candidate grasp if the
  gripper's approximated geometry (same box-corner approach already used
  for the table check) intersects any *other* on-table object's
  geometry at that pose — extending the existing pre-filter philosophy,
  not building a new motion-planning subsystem.
- **Occupancy representation**: full 3D voxel occupancy built from the
  round's own depth/segmap capture (not a 2.5D top-down heightmap like
  `placement_planner.py`'s `BinHeightmap` — that can't represent a
  gripper reaching under part of a tall neighbor).
- **Gripper sampling**: densify `_gripper_sample_points()` beyond its
  current corners-only set (add edge-midpoints and face-centers) for
  better coverage against thin/small neighbors.
- **Reachability approach**: **A** — a cheap analytical position + azimuth
  check, no IK. **C** — ship this narrower, honestly-scoped version first
  and require a real benchmark comparison (not just "should work") before
  considering any deeper kinematic modeling. Full wrist/orientation
  reachability has no cheap closed form for a 7-DOF arm and is explicitly
  out of scope — the existing `max_up_z` approach-angle check remains the
  orientation safety net it already is.
- **Rejected**: a reduced-iteration `DiffIK` probe as the reachability
  check (Approach B) — `DiffIK.solve()` costs up to 4 seeds × 200 DLS
  iterations; run against the 300-400+ raw candidates seen per round in
  practice, it would likely cost more than today's status quo of letting
  the rare unreachable candidate fail naturally during the top-k
  execution attempts.

## Architecture

Two new, small, composable modules — neither modifies
`GraspFeasibilityChecker` itself, matching this codebase's existing
pattern of composable filtering steps (`spatial_relation_resolver.py`
composing with `placement_planner.py` without touching it):

```
sim_grasp/workspace_occupancy.py
    build_workspace_occupancy(depth, segmap, K, T_world_cam, table_height)
        -> WorkspaceOccupancy   # voxel -> which seg_id(s) occupy it
    collides_with_neighbors(T_world_grasp, opening, exclude_seg_id,
                            occupancy) -> bool

sim_grasp/reachability.py
    is_reachable(T_world_grasp, base_xy=(0.0, 0.0)) -> bool
```

Both are called from `run_sim_grasp_test.py`'s existing `filter_feasible()`
as an *additional* pass after `GraspFeasibilityChecker.filter()`'s
existing table/approach-angle check — same two call sites as today (the
initial single-shot capture and each `--pick-all` round), both of which
already have `depth`/`segmap`/`K` in scope, just not currently threaded
into `filter_feasible()`.

**Opt-in, matching `--recenter`/`--clean-depth`'s existing pattern**: a
new `--filter-neighbors` flag (default off). With it off, `filter_feasible()`
behaves byte-for-byte as it does today — the whole point of Approach C is
a clean, direct benchmark A/B (`--filter-neighbors` vs. without), not a
silent behavior change to every existing benchmark baseline.

## `workspace_occupancy.py` — neighbor collision

`build_workspace_occupancy()`: back-projects every `segmap > 0` pixel to
world points (reusing `depth_to_pointcloud`/`transform_points`, the same
path `compute_object_footprint` already uses), keeps points above
`table_height + 0.005` (matching `GraspFeasibilityChecker`'s existing
`margin=0.005` convention, excluding table-level noise), and voxelizes
each point into a grid-index tuple `(floor(x/voxel_size), floor(y/voxel_size),
floor(z/voxel_size))` at `voxel_size=0.005` (matching `BinHeightmap`'s
existing cell size convention). Result: a plain Python `dict[(int,int,int),
frozenset[int]]` mapping each occupied voxel to the seg_id(s) that put a
point there — a `dict`/`set`-based lookup, not a KD-tree or new dependency,
giving O(1) "is this voxel occupied by someone other than me" queries.

`collides_with_neighbors(T_world_grasp, opening, exclude_seg_id,
occupancy)`: transforms the densified gripper sample points into world
frame, voxelizes each, and returns `True` if any sample point's voxel is
occupied by a seg_id set whose difference from `{exclude_seg_id}` is
non-empty (i.e., some *other* object occupies that voxel). `exclude_seg_id`
is the object currently being picked — its own points must never count as
a "collision" with itself.

Gripper sample density: `_gripper_sample_points()` in `feasibility.py`
already returns each box's 8 corners. A new function in
`workspace_occupancy.py`, `_densified_box_points()`, extends
`_box_corners()`'s pattern to also include the 6 face-centers and 12
edge-midpoints per box (26 points/box instead of 8) — same
palm/finger-left/finger-right box definitions, just denser sampling.

## `reachability.py` — analytical position + azimuth check

```python
PANDA_JOINT1_RANGE = (-2.8973, 2.8973)   # radians; real Panda spec, matches
                                          # mujoco_menagerie's panda.xml default
                                          # class (no override on joint1) --
                                          # a physical robot fact, not a
                                          # MuJoCo-model query, so this module
                                          # stays model-independent like the
                                          # rest of the vision-only pipeline
MAX_REACH = 0.54    # meters; already established (scene_generator.py's
                     # spawn-region comment: "within arm reach (0.54 m from
                     # the base)")
MIN_REACH = 0.15    # meters; conservative estimate for the "too close to
                     # fold the elbow" dead zone -- NOT a hard physical
                     # constant like the two above, flagged as
                     # benchmark-tunable if it turns out too conservative
                     # or not conservative enough
AZIMUTH_MARGIN = 0.1   # radians; safety margin inside the raw joint1 range
```

`is_reachable(T_world_grasp, base_xy=(0.0, 0.0))`: computes the grasp
origin's XY distance from `base_xy` (the robot base sits at world XY
`(0, 0)` per `frames.py`'s documented convention — `T_world_base =
Trans(0, 0, table_height)`) and rejects if outside `[MIN_REACH, MAX_REACH]`;
computes the azimuth `atan2(dy, dx)` and rejects if outside
`PANDA_JOINT1_RANGE` narrowed by `AZIMUTH_MARGIN` on each side.

## Error handling

- `build_workspace_occupancy()` on a mask that yields zero points for
  every object (degenerate capture): returns an occupancy with an empty
  `voxel_to_segids` dict — `collides_with_neighbors()` then always
  returns `False` for it, degrading to "no neighbor check" rather than
  raising, matching `compute_object_footprint`'s existing "return `None`,
  don't raise" philosophy for degenerate masks.
- `--filter-neighbors` without `--pick-all`/`--execute`: no error needed —
  it's simply inert if nothing calls `filter_feasible()`.

## Testing

Standalone `test_*.py` scripts (no pytest, matching this codebase's
convention), synthetic depth/segmap/K arrays (same fixture style as
`test_placement_planner.py`), no live MuJoCo model needed for either new
module (both are pure geometry, consuming only numpy arrays and plain
Python data structures) — mirrors how `spatial_relation_resolver.py`'s
tests use a fake `PromptSelector` rather than a real SAM 3 subprocess.

## Validation

Per Approach C: `benchmark.py --seeds 0-4 --mode pick-all` run with and
without `--filter-neighbors`, comparing binned/knocked-off-table/failure-
taxonomy numbers — not merged/declared done on "should work" alone.
