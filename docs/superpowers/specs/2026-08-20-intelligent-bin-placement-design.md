# Intelligent Bin Placement — Design

## Problem

`mujoco_grasp_sim`'s pick phase is intelligent (GraspGen/Contact-GraspNet +
SAM3), but the place phase is not: every object is dropped at the same
hardcoded world point (`SceneGenerator.bin_drop_point()`) at a fixed release
height (`executor.PLACE_RELEASE`). This causes two concrete failures:

1. **Stacking**: the 2nd/3rd object in a `--pick-all` run lands on top of the
   1st (no XY offset between placements), often losing the grasp or
   knocking the first object out of the bin.
2. **Crushing**: a tall cuboid is picked correctly but the fixed release
   height doesn't account for its actual height, jamming it against the bin
   floor and losing the grasp.

## Constraint

The fix must be computed **entirely from vision** (RGB-D + segmap via the
existing `CameraModule`/`depth_to_pointcloud()` pipeline) — no MuJoCo
internal-state queries (`model.geom_size`, `data.body().xpos`, etc.) — so the
same code is a straightforward port to a real RealSense camera later.

## Decisions (confirmed with the user)

- **Slot search**: general free-space search over a heightmap of the bin
  (not a fixed N-slot grid), so it generalizes if the object count changes.
- **Scope**: both `run_sim_grasp_test.py --pick-all` (and its single
  `--execute` path) and `interactive_pick.py`.
- **Orientation**: placement also picks a release yaw to best fit the
  object's footprint into its chosen slot (in addition to position).

## Architecture

New module `mujoco_grasp_sim/sim_grasp/placement_planner.py`, mirroring the
existing `GraspPredictor` ABC pattern used for CGN/GraspGen:

```
[existing full-scene depth+segmap capture, already taken every round]
        │                                  │
        ▼                                  ▼
compute_object_footprint()          build_bin_heightmap()
 (XY size + yaw + height of the      (top-down height grid of the
  object about to be picked, from    bin interior, from whatever
  its own segmap mask)               is already in the bin)
        │                                  │
        └────────────────┬─────────────────┘
                          ▼
             OccupancyPlacementPlanner.plan()
          (free-space + yaw search over the heightmap
           for the object's footprint)
                          │
                          ▼
              PlacementPose(x, y, yaw, release_z)
                          │
                          ▼
       executor.place(x, y, release_z, yaw) — dynamic
       release, replacing the fixed drop_pos/PLACE_RELEASE
```

No new camera captures are needed: both entry points already capture a
full-scene depth/segmap before every pick, which already contains both the
target object and the bin's current contents.

The **grasp offset** (how far below the gripper the object's bottom sits
once grasped) is computed from data already on hand mid-pipeline — the
object's own bottom-Z (measured from its footprint before the grasp) versus
the executed grasp's world Z — no new sensing required for that either.

## Algorithm

**`compute_object_footprint(depth, segmap, seg_id, K, T_world_cam)`**:
masks the object's own depth pixels, back-projects + transforms to world
frame (existing `depth_to_pointcloud()` + `transform_points()`), then
`cv2.minAreaRect()` on the world XY points gives natural width/height and
yaw; world Z min/max give height. Returns `None` if the mask yields too few
points (graceful degradation, see Error handling).

**`build_bin_heightmap(depth, segmap, K, T_world_cam, bin_center,
bin_inner_half, exclude_seg_id, cell_size=5mm)`**: crops the same capture to
the bin's world XY region (excluding the currently-picked object's own
mask), discretizes into a small grid, stores max observed world-Z per cell.
The grid's own low-percentile height self-calibrates the "empty floor"
reference (no hardcoded floor height assumed).

**`OccupancyPlacementPlanner.plan(footprint, heightmap)`**: tries 4 candidate
yaws (footprint's natural yaw + 0°/45°/90°/135°); for each, computes the
axis-aligned bounding box of the rotated footprint (a deliberate, documented
simplification over full oriented-rectangle collision, acceptable given the
objects here are only mildly rectangular); slides it across the bin at a 1cm
stride, keeping candidates where every covered cell is at/near the floor
reference; picks the candidate maximizing clearance to the nearest occupied
cell/wall. Fallback (should not trigger at 3 objects in a 24cm² bin): the
region with the lowest max-height anywhere in the bin.

**Release orientation nuance**: the computed yaw is only applied to the
canonical top-down fallback hand orientation, not the "keep current hand
orientation" first try — a top-down grasp preserves the object's on-table
yaw through the pick, so rotating the wrist's yaw there reliably rotates the
placed object; for non-top-down grasps there's no such guarantee, so yaw is
skipped there rather than faked.

## Error handling

- Footprint/heightmap computation fails (degenerate mask) → log a warning,
  fall back to the legacy fixed `bin_drop_point()`/`PLACE_RELEASE` behavior
  for that one placement.
- No free slot found → place at the least-occupied region found, log a
  warning.
- IK infeasible at the computed pose → existing retry pattern extended with
  one more canonical-orientation-no-yaw attempt (never worse than today).

## Testing

1. Unit tests for the new pure-geometry functions in `placement_planner.py`
   and the small `executor.py` orientation helper (synthetic arrays, no
   MuJoCo — matches this codebase's existing `test_*.py` convention: run
   directly with `assert`, no pytest).
2. Visual sanity: a few `run_sim_grasp_test.py --pick-all --camera fused`
   runs — confirm 3 objects land in distinct, non-overlapping slots and a
   deliberately tall cuboid isn't crushed.
3. `benchmark.py --seeds 0-4 --mode pick-all --camera fused` before/after
   comparison (matches the project's existing reliability-change convention
   from `ROADMAP.md`).
