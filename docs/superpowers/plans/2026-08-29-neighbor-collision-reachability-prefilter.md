# Neighbor-Object Collision + Workspace-Reachability Pre-Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the grasp feasibility pipeline with two new, opt-in
pre-filters — neighbor-object collision (3D voxel occupancy) and
workspace-reachability (analytical position + azimuth) — per
`docs/superpowers/specs/2026-08-29-neighbor-collision-reachability-prefilter-design.md`.

**Architecture:** Two new, small, composable modules
(`sim_grasp/workspace_occupancy.py`, `sim_grasp/reachability.py`), neither
modifying `GraspFeasibilityChecker` itself. Both are wired into
`run_sim_grasp_test.py`'s existing `filter_feasible()` as an additional
pass, gated behind a new `--filter-neighbors` flag (default off, so every
existing benchmark baseline is unaffected unless explicitly opted in).

**Tech Stack:** Python, NumPy (existing dependencies only, nothing new).

## Global Constraints

- This codebase has **no pytest suite** — tests are standalone `test_*.py`
  scripts run directly (`python sim_grasp/test_name.py`, from
  `mujoco_grasp_sim/` with `PYTHONPATH=.`), plain `assert` statements,
  ending with a `print('All ... checks passed.')` line.
- `conda activate cgn_torch` before running anything in this repo.
- Branch: `neighbor-collision-reachability-prefilter` (already created off
  `main`, holds the design spec commit).
- Do not modify `GraspFeasibilityChecker`'s public behavior
  (`is_feasible()`/`filter()`) — the new checks are a separate, composable
  pass, not a change to the existing table-collision/approach-angle logic.
- `--filter-neighbors` is opt-in (default off). With it off,
  `filter_feasible()` must be byte-for-byte unchanged from before this
  plan — every existing benchmark baseline stays valid.
- `interactive_pick.py`'s existing call to `filter_feasible()` (5
  positional args, no new flag) must keep working unchanged — new
  parameters are optional with defaults, not required. Wiring
  `--filter-neighbors` into `interactive_pick.py` itself is explicitly
  out of scope for this plan.
- No new third-party dependency — plain Python `dict`/`set` for voxel
  occupancy, no KD-tree library.
- `PANDA_JOINT1_RANGE = (-2.8973, 2.8973)` (radians) is a real Panda
  robot spec constant (confirmed against
  `mujoco_menagerie/franka_emika_panda/panda.xml`'s `<default class="panda">`
  block — joint1 has no per-joint override, so it uses this class default
  verbatim) — not a MuJoCo-model query, keeping `reachability.py`
  model-independent like the rest of the vision-only pipeline.
- `MAX_REACH = 0.54` meters is already an established constant
  (`scene_generator.py`'s spawn-region comment: "within arm reach (0.54 m
  from the base)"). `MIN_REACH = 0.15` meters is an approximate,
  benchmark-tunable estimate, not a hard physical fact — say so in the
  code comment, don't present it as equally certain as the other two
  constants.
- The robot base sits at world XY `(0.0, 0.0)` (`frames.py`'s documented
  convention: `T_world_base = Trans(0, 0, table_height)`).

---

## Task 1: `sim_grasp/workspace_occupancy.py` — 3D occupancy + neighbor collision

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/feasibility.py` (extract `_hand_boxes()`)
- Create: `mujoco_grasp_sim/sim_grasp/workspace_occupancy.py`
- Create: `mujoco_grasp_sim/sim_grasp/test_workspace_occupancy.py`

**Interfaces:**
- Produces: `WorkspaceOccupancy` (dataclass: `voxel_size: float`,
  `voxel_to_segids: dict[tuple, frozenset]`),
  `build_workspace_occupancy(depth, segmap, K, T_world_cam, table_height,
  voxel_size=0.005) -> WorkspaceOccupancy`,
  `collides_with_neighbors(T_world_grasp, opening, exclude_seg_id,
  occupancy) -> bool`. Task 3 imports both functions and the dataclass.
- Consumes: `sim_grasp.pointcloud.depth_to_pointcloud`,
  `sim_grasp.frames.transform_points` (both unmodified), and
  `sim_grasp.feasibility._hand_boxes` (new, extracted in this task).

**Step 1: Extract `_hand_boxes()` in `feasibility.py` (DRY — same physical
gripper geometry needed by both the existing corner-only check and the
new densified check)**

Find:

```python
def _box_corners(x0, x1, y0, y1, z0, z1):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])


def _gripper_sample_points(opening: float = 0.08) -> np.ndarray:
    """Corner samples of the simplified Panda hand in the grasp frame."""
    half_open = opening / 2.0
    palm = _box_corners(-0.102, 0.102, -0.0315, 0.0315, -0.012, 0.066)
    finger_l = _box_corners(-half_open - 0.012, -half_open + 0.012, -0.012, 0.012, 0.066, 0.112)
    finger_r = _box_corners(half_open - 0.012, half_open + 0.012, -0.012, 0.012, 0.066, 0.112)
    return np.vstack([palm, finger_l, finger_r])
```

Replace with:

```python
def _box_corners(x0, x1, y0, y1, z0, z1):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])


def _hand_boxes(opening: float = 0.08) -> list:
    """The 3 boxes (palm, left finger, right finger) approximating the Panda
    hand in the grasp frame, as (x0, x1, y0, y1, z0, z1) tuples -- the
    single source of truth for hand geometry, shared by this module's
    corner-only sampling and workspace_occupancy.py's denser sampling."""
    half_open = opening / 2.0
    return [
        (-0.102, 0.102, -0.0315, 0.0315, -0.012, 0.066),                        # palm
        (-half_open - 0.012, -half_open + 0.012, -0.012, 0.012, 0.066, 0.112),  # finger_l
        (half_open - 0.012, half_open + 0.012, -0.012, 0.012, 0.066, 0.112),    # finger_r
    ]


def _gripper_sample_points(opening: float = 0.08) -> np.ndarray:
    """Corner samples of the simplified Panda hand in the grasp frame."""
    return np.vstack([_box_corners(*box) for box in _hand_boxes(opening)])
```

- [ ] **Step 1: Extract `_hand_boxes()`, confirm `_gripper_sample_points()`
      still returns the same 24 points as before** (run
      `PYTHONPATH=. python sim_grasp/test_feasibility.py` from
      `mujoco_grasp_sim/` — must still pass unchanged, since this is a
      pure refactor with no behavior change)

**Step 2: Write the failing tests**

Create `mujoco_grasp_sim/sim_grasp/test_workspace_occupancy.py`. This test
is split into two independent parts on purpose: `collides_with_neighbors()`
is tested against a **manually-constructed** `WorkspaceOccupancy` (no
camera math involved) so the check is exact and deterministic by
construction, and `build_workspace_occupancy()` is tested separately
against a **real synthetic depth/segmap capture** but only for its own
actual contract (non-empty, correct per-object seg_id separation, empty
input degrades cleanly) rather than trying to predict which exact 5mm
voxel a reverse-projected camera pixel lands in — a synthetic top-down
camera's per-pixel world-step can easily be coarser than the voxel size,
making single-voxel-precision prediction through the camera path
fragile/flaky, not a property of the real implementation:

```python
"""Standalone checks for workspace_occupancy.py -- run directly, no
pytest (this codebase has no automated test suite)."""
import numpy as np

from sim_grasp.workspace_occupancy import (
    build_workspace_occupancy, collides_with_neighbors, WorkspaceOccupancy,
    _densified_gripper_sample_points)

TABLE_HEIGHT = 0.75
VOXEL_SIZE = 0.005

# ============================================================================
# Part A: collides_with_neighbors() against a manually-constructed
# occupancy -- exact and deterministic, no camera reverse-projection needed.
# ============================================================================
grip_pts = _densified_gripper_sample_points(0.08)
origin = np.array([0.30, 0.10, 0.85])
T = np.eye(4)
T[:3, 3] = origin
world_pts = grip_pts + origin   # identity rotation: world = local + origin
occ_voxel = tuple(np.floor(world_pts[0] / VOXEL_SIZE).astype(int))  # a REAL
                                                                    # gripper-
                                                                    # sample voxel

manual_occupancy = WorkspaceOccupancy(voxel_size=VOXEL_SIZE,
                                      voxel_to_segids={occ_voxel: frozenset({2})})
assert collides_with_neighbors(T, 0.08, exclude_seg_id=1, occupancy=manual_occupancy), \
    'a gripper-sample voxel occupied by a DIFFERENT seg_id must collide'
assert not collides_with_neighbors(T, 0.08, exclude_seg_id=2, occupancy=manual_occupancy), \
    "excluding the occupying object's own seg_id must not collide"

empty_occupancy = WorkspaceOccupancy(voxel_size=VOXEL_SIZE, voxel_to_segids={})
assert not collides_with_neighbors(T, 0.08, exclude_seg_id=1, occupancy=empty_occupancy), \
    'an empty occupancy must never register a collision'

print('All workspace_occupancy collision-logic checks passed.')

# ============================================================================
# Part B: build_workspace_occupancy() against a real synthetic depth/segmap
# capture -- checks its own contract (non-empty, correct seg_id separation,
# degenerate-input handling), not exact voxel prediction.
# ============================================================================
H, W = 200, 200
fx = fy = 300.0
cx = cy = 100.0
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
cam_height = 1.5
T_world_cam = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, cam_height],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

depth = np.zeros((H, W), dtype=np.float32)
segmap = np.zeros((H, W), dtype=np.int32)
depth[40:60, 40:60] = cam_height - (TABLE_HEIGHT + 0.03)
segmap[40:60, 40:60] = 1
depth[140:160, 140:160] = cam_height - (TABLE_HEIGHT + 0.03)
segmap[140:160, 140:160] = 2

occupancy = build_workspace_occupancy(depth, segmap, K, T_world_cam, TABLE_HEIGHT)
assert isinstance(occupancy, WorkspaceOccupancy)
assert len(occupancy.voxel_to_segids) > 0, 'expected some occupied voxels'

seg_ids_seen = set()
for occupants in occupancy.voxel_to_segids.values():
    seg_ids_seen |= occupants
assert seg_ids_seen == {1, 2}, f'expected both seg_ids represented, got {seg_ids_seen}'
# The two synthetic objects are spatially separated -- no voxel should be
# claimed by both at once.
assert all(len(v) == 1 for v in occupancy.voxel_to_segids.values())

# Degenerate case: an all-background segmap yields an empty occupancy
# rather than raising.
empty_segmap = np.zeros((H, W), dtype=np.int32)
empty_occ = build_workspace_occupancy(depth, empty_segmap, K, T_world_cam, TABLE_HEIGHT)
assert len(empty_occ.voxel_to_segids) == 0

print('All workspace_occupancy build checks passed.')
```

- [ ] **Step 2: Write the failing test** (confirm it fails with
      `ModuleNotFoundError: No module named 'sim_grasp.workspace_occupancy'`
      before writing the implementation)

**Step 3: Implement `workspace_occupancy.py`**

```python
"""3D voxel occupancy of everything currently on the table, and a
neighbor-object collision check against it -- extends
sim_grasp.feasibility's cheap-geometric-pre-filter philosophy (table
collision, approach angle) to real neighbor geometry, using a full 3D
voxel grid rather than a 2.5D top-down heightmap (placement_planner.py's
BinHeightmap) since a gripper can reach under part of a tall neighbor,
which a top-down height value alone can't represent.

Not motion planning -- a cheap, denser-sampled geometric pre-filter, same
spirit as feasibility.py.
"""
from dataclasses import dataclass

import numpy as np

from sim_grasp.feasibility import _hand_boxes
from sim_grasp.frames import transform_points
from sim_grasp.pointcloud import depth_to_pointcloud


@dataclass
class WorkspaceOccupancy:
    voxel_size: float
    voxel_to_segids: dict   # {(int, int, int): frozenset[int]}


def _densified_box_points(x0, x1, y0, y1, z0, z1) -> np.ndarray:
    """26 points per box: 8 corners + 6 face-centers + 12 edge-midpoints --
    denser than feasibility.py's corner-only _box_corners(), for better
    coverage against thin/small neighbor objects a corner-only check
    could miss entirely."""
    xs, ys, zs = (x0, x1), (y0, y1), (z0, z1)
    xm, ym, zm = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    pts = [(x, y, z) for x in xs for y in ys for z in zs]           # 8 corners
    pts += [(x, ym, zm) for x in xs]                                 # 2 face centers
    pts += [(xm, y, zm) for y in ys]                                 # 2 face centers
    pts += [(xm, ym, z) for z in zs]                                 # 2 face centers
    pts += [(x, y, zm) for x in xs for y in ys]                      # 4 edge midpoints
    pts += [(x, ym, z) for x in xs for z in zs]                      # 4 edge midpoints
    pts += [(xm, y, z) for y in ys for z in zs]                      # 4 edge midpoints
    return np.array(pts)


def _densified_gripper_sample_points(opening: float = 0.08) -> np.ndarray:
    return np.vstack([_densified_box_points(*box) for box in _hand_boxes(opening)])


def _voxelize(pts_world: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor(pts_world / voxel_size).astype(np.int64)


def build_workspace_occupancy(depth: np.ndarray, segmap: np.ndarray,
                              K: np.ndarray, T_world_cam: np.ndarray,
                              table_height: float,
                              voxel_size: float = 0.005) -> WorkspaceOccupancy:
    """Voxelizes every on-table object's own points (segmap > 0), one
    object at a time so each voxel can be attributed to the seg_id(s) that
    put a point there -- letting collides_with_neighbors() exclude a
    grasp's own target object without needing a separate occupancy per
    object."""
    voxel_to_segids: dict = {}
    for seg_id in sorted(int(s) for s in np.unique(segmap) if s > 0):
        pts_cam = depth_to_pointcloud(depth, K, mask=(segmap == seg_id))
        if len(pts_cam) == 0:
            continue
        pts_world = transform_points(T_world_cam, pts_cam)
        pts_world = pts_world[pts_world[:, 2] > table_height + 0.005]
        if len(pts_world) == 0:
            continue
        for row in _voxelize(pts_world, voxel_size):
            key = (int(row[0]), int(row[1]), int(row[2]))
            voxel_to_segids.setdefault(key, set()).add(seg_id)
    frozen = {k: frozenset(v) for k, v in voxel_to_segids.items()}
    return WorkspaceOccupancy(voxel_size=voxel_size, voxel_to_segids=frozen)


def collides_with_neighbors(T_world_grasp: np.ndarray, opening: float,
                            exclude_seg_id: int,
                            occupancy: WorkspaceOccupancy) -> bool:
    """True if any densified gripper sample point, at this candidate grasp
    pose, falls in a voxel occupied by a seg_id other than exclude_seg_id
    (the object currently being picked -- its own points are never a
    'collision' with itself)."""
    pts_grasp = _densified_gripper_sample_points(opening)
    pts_world = transform_points(T_world_grasp, pts_grasp)
    for row in _voxelize(pts_world, occupancy.voxel_size):
        key = (int(row[0]), int(row[1]), int(row[2]))
        occupants = occupancy.voxel_to_segids.get(key)
        if occupants and (occupants - {exclude_seg_id}):
            return True
    return False
```

- [ ] **Step 3: Implement `workspace_occupancy.py`**

- [ ] **Step 4: Run the test to verify it passes**

Run (from `mujoco_grasp_sim/`): `PYTHONPATH=. python
sim_grasp/test_workspace_occupancy.py`. Expected output (two lines, one
per part of the test):
```
All workspace_occupancy collision-logic checks passed.
All workspace_occupancy build checks passed.
```

- [ ] **Step 5: Run `test_feasibility.py` again to confirm the `_hand_boxes()`
      extraction didn't break anything**

Run: `PYTHONPATH=. python sim_grasp/test_feasibility.py`. Expected: `All
feasibility checks passed.`

- [ ] **Step 6: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/feasibility.py \
       mujoco_grasp_sim/sim_grasp/workspace_occupancy.py \
       mujoco_grasp_sim/sim_grasp/test_workspace_occupancy.py
git commit -m "Add workspace_occupancy: 3D voxel occupancy + neighbor-collision check"
```

---

## Task 2: `sim_grasp/reachability.py` — analytical position + azimuth check

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/reachability.py`
- Create: `mujoco_grasp_sim/sim_grasp/test_reachability.py`

**Interfaces:**
- Produces: `PANDA_JOINT1_RANGE`, `MAX_REACH`, `MIN_REACH`,
  `AZIMUTH_MARGIN` (module-level constants),
  `is_reachable(T_world_grasp, base_xy=(0.0, 0.0)) -> bool`. Task 3
  imports this function.
- Consumes: nothing from other tasks — pure geometry on a 4x4 array and a
  2-tuple.

**Step 1: Write the failing tests**

Create `mujoco_grasp_sim/sim_grasp/test_reachability.py`:

```python
"""Standalone checks for reachability.py -- run directly, no pytest (this
codebase has no automated test suite)."""
import numpy as np

from sim_grasp.reachability import is_reachable, MAX_REACH, MIN_REACH

BASE_XY = (0.0, 0.0)


def _pose_at(x, y, z=0.8):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T

# A target at a typical, comfortably-in-range distance and azimuth
# (matches the established object spawn region, e.g. x=0.45, y=-0.10)
# straight in front of the base must be reachable.
assert is_reachable(_pose_at(0.45, -0.10), base_xy=BASE_XY)

# A target beyond MAX_REACH must be rejected.
far_x = MAX_REACH + 0.10
assert not is_reachable(_pose_at(far_x, 0.0), base_xy=BASE_XY)

# A target closer than MIN_REACH must be rejected (the "too close to fold
# the elbow" dead zone).
close_x = MIN_REACH - 0.05
assert not is_reachable(_pose_at(close_x, 0.0), base_xy=BASE_XY)

# A target exactly at the boundary is reachable (inclusive bounds) --
# 1cm inside MAX_REACH, comfortably outside MIN_REACH.
assert is_reachable(_pose_at(MAX_REACH - 0.01, 0.0), base_xy=BASE_XY)

# A target directly BEHIND the base (azimuth = pi, outside joint1's
# +-2.8973 rad range narrowed by AZIMUTH_MARGIN) must be rejected --
# this is the azimuth check, independent of distance.
mid_reach = (MIN_REACH + MAX_REACH) / 2
assert not is_reachable(_pose_at(-mid_reach, 0.0), base_xy=BASE_XY), \
    'directly behind the base should be outside the reachable azimuth range'

# base_xy offset: the same checks must work relative to a non-origin base
# (defensive -- this project's convention keeps the base at world (0,0),
# but the function itself must not hardcode that assumption).
offset_base = (1.0, 1.0)
assert is_reachable(_pose_at(1.45, 0.90), base_xy=offset_base)
assert not is_reachable(_pose_at(1.0 + far_x, 1.0), base_xy=offset_base)

print('All reachability checks passed.')
```

- [ ] **Step 1: Write the failing test** (confirm it fails with
      `ModuleNotFoundError: No module named 'sim_grasp.reachability'`
      before writing the implementation)

**Step 2: Implement `reachability.py`**

```python
"""Cheap analytical workspace-reachability pre-filter -- position (radial
distance from the base) plus azimuth (does this direction even fall
within joint1's rotation range), no IK. A full 7-DOF orientation-
reachability check has no cheap closed form; this is deliberately
narrower in scope (see
docs/superpowers/specs/2026-08-29-neighbor-collision-reachability-prefilter-design.md).
Not motion planning -- a cheap geometric pre-filter, same spirit as
feasibility.py/workspace_occupancy.py.
"""
import numpy as np

# Real Panda robot spec (radians) -- confirmed against
# mujoco_menagerie/franka_emika_panda/panda.xml's <default class="panda">
# joint range (joint1 has no per-joint override, so it uses this class
# default verbatim). A physical robot fact, not a MuJoCo-model query, so
# this module stays model-independent like the rest of the vision-only
# pipeline.
PANDA_JOINT1_RANGE = (-2.8973, 2.8973)

# Meters. Already an established constant elsewhere in this project
# (scene_generator.py's spawn-region comment: "within arm reach (0.54 m
# from the base)").
MAX_REACH = 0.54

# Meters. An APPROXIMATE estimate of the "too close to fold the elbow"
# dead zone near the base -- unlike MAX_REACH/PANDA_JOINT1_RANGE above,
# this is not a hard physical constant; treat it as benchmark-tunable if
# it proves too conservative or not conservative enough in practice.
MIN_REACH = 0.15

# Radians. Safety margin narrowing PANDA_JOINT1_RANGE on each side, since
# a target right at the joint's hard limit leaves no room for the other
# 6 joints to also converge.
AZIMUTH_MARGIN = 0.1


def is_reachable(T_world_grasp: np.ndarray, base_xy: tuple = (0.0, 0.0)) -> bool:
    dx = T_world_grasp[0, 3] - base_xy[0]
    dy = T_world_grasp[1, 3] - base_xy[1]
    dist = float(np.hypot(dx, dy))
    if not (MIN_REACH <= dist <= MAX_REACH):
        return False
    azimuth = float(np.arctan2(dy, dx))
    lo, hi = PANDA_JOINT1_RANGE
    return (lo + AZIMUTH_MARGIN) <= azimuth <= (hi - AZIMUTH_MARGIN)
```

- [ ] **Step 2: Implement `reachability.py`**

- [ ] **Step 3: Run the test to verify it passes**

Run (from `mujoco_grasp_sim/`): `PYTHONPATH=. python
sim_grasp/test_reachability.py`. Expected: `All reachability checks
passed.`

- [ ] **Step 4: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/reachability.py \
       mujoco_grasp_sim/sim_grasp/test_reachability.py
git commit -m "Add reachability: analytical position + azimuth pre-filter"
```

---

## Task 3: Wire into `filter_feasible()` + `--filter-neighbors` + validation

**Files:**
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py`

**Interfaces:**
- Consumes: `sim_grasp.workspace_occupancy.{build_workspace_occupancy,
  collides_with_neighbors}` (Task 1), `sim_grasp.reachability.is_reachable`
  (Task 2).
- Produces: `--filter-neighbors` CLI flag. No other new public interface —
  this is the integration task.

**Step 1: Extend `filter_feasible()`'s signature**

Find:

```python
def filter_feasible(grasps_cam, scores, openings, T_world_cam, table_height):
    """Table-collision filter (runs in world frame, returns camera frame)."""
    checker = GraspFeasibilityChecker(table_height=table_height,
                                     extra_approach=EXTRA_APPROACH)
    grasps_world = {k: transform_grasps(T_world_cam, np.asarray(G))
                    for k, G in grasps_cam.items()}
    kept_world, kept_scores, stats = checker.filter(grasps_world, scores, openings)
    T_cam_world = invert_se3(T_world_cam)
    kept_cam = {k: transform_grasps(T_cam_world, G) for k, G in kept_world.items()}
    return kept_cam, kept_scores, stats
```

Replace with:

```python
def filter_feasible(grasps_cam, scores, openings, T_world_cam, table_height,
                    depth=None, segmap=None, K=None, filter_neighbors=False):
    """Table-collision filter (runs in world frame, returns camera frame).
    When filter_neighbors=True (needs depth/segmap/K), also rejects grasps
    that collide with a neighboring on-table object or fall outside the
    analytical workspace-reachability envelope -- both opt-in, off by
    default so every existing caller/benchmark baseline is unaffected."""
    checker = GraspFeasibilityChecker(table_height=table_height,
                                     extra_approach=EXTRA_APPROACH)
    grasps_world = {k: transform_grasps(T_world_cam, np.asarray(G))
                    for k, G in grasps_cam.items()}
    kept_world, kept_scores, stats = checker.filter(grasps_world, scores, openings)

    if filter_neighbors and depth is not None and segmap is not None and K is not None:
        from sim_grasp.reachability import is_reachable
        from sim_grasp.workspace_occupancy import (
            build_workspace_occupancy, collides_with_neighbors)
        occupancy = build_workspace_occupancy(depth, segmap, K, T_world_cam, table_height)
        n_before_extra = sum(len(v) for v in kept_world.values())
        out_world, out_scores = {}, {}
        for seg_id, G in kept_world.items():
            keep = []
            for i, T in enumerate(G):
                opening = 0.08
                if openings and seg_id in openings and len(openings[seg_id]) > i:
                    opening = float(openings[seg_id][i])
                if not is_reachable(T):
                    continue
                if collides_with_neighbors(T, opening, int(seg_id), occupancy):
                    continue
                keep.append(i)
            if keep:
                out_world[seg_id] = G[keep]
                out_scores[seg_id] = np.asarray(kept_scores[seg_id])[keep]
        n_after_extra = sum(len(v) for v in out_world.values())
        stats = dict(stats)
        stats['n_rejected_neighbors_or_unreachable'] = n_before_extra - n_after_extra
        stats['n_after'] = n_after_extra
        kept_world, kept_scores = out_world, out_scores

    T_cam_world = invert_se3(T_world_cam)
    kept_cam = {k: transform_grasps(T_cam_world, G) for k, G in kept_world.items()}
    return kept_cam, kept_scores, stats
```

- [ ] **Step 1 done**

**Step 2: Add the `--filter-neighbors` flag**

Find:

```python
    ap.add_argument('--recenter', action='store_true',
                    help='shift each executed grasp along its finger-closing '
                         'axis onto the target object cloud — counters the '
                         'dominant closed_on_air failure (lateral CGN offset)')
```

Add immediately after it:

```python
    ap.add_argument('--filter-neighbors', action='store_true',
                    help='reject grasps that collide with a neighboring '
                         'on-table object (3D voxel occupancy) or fall '
                         'outside the analytical workspace-reachability '
                         'envelope (position + azimuth, no IK) -- opt-in, '
                         'off by default')
```

- [ ] **Step 2 done**

**Step 3: Pass `depth`/`segmap`/`K`/`filter_neighbors` at both call sites**

Find (the initial single-shot capture):

```python
    if not args.no_feasibility and pred.num_grasps > 0:
        grasps_cam, scores, feas_stats = filter_feasible(
            grasps_cam, scores, pred.gripper_openings, T_world_cam, cfg.table_height)
        print(f"[feasibility] kept {feas_stats['n_after']}/{feas_stats['n_before']} "
              f"({feas_stats['n_rejected']} table-colliding/underhand rejected)")
```

Replace with:

```python
    if not args.no_feasibility and pred.num_grasps > 0:
        grasps_cam, scores, feas_stats = filter_feasible(
            grasps_cam, scores, pred.gripper_openings, T_world_cam, cfg.table_height,
            depth=depth, segmap=segmap, K=K, filter_neighbors=args.filter_neighbors)
        print(f"[feasibility] kept {feas_stats['n_after']}/{feas_stats['n_before']} "
              f"({feas_stats['n_rejected']} table-colliding/underhand rejected)")
        if args.filter_neighbors:
            print(f"[feasibility]   + {feas_stats['n_rejected_neighbors_or_unreachable']} "
                  'rejected by neighbor-collision/reachability pre-filter')
```

Find (the `--pick-all` per-round loop):

```python
                g_r, s_r = pred_r.grasps_cam, pred_r.scores
                if not args.no_feasibility and pred_r.num_grasps > 0:
                    g_r, s_r, _ = filter_feasible(g_r, s_r, pred_r.gripper_openings,
                                                  T_wc, cfg.table_height)
```

Replace with:

```python
                g_r, s_r = pred_r.grasps_cam, pred_r.scores
                if not args.no_feasibility and pred_r.num_grasps > 0:
                    g_r, s_r, _ = filter_feasible(g_r, s_r, pred_r.gripper_openings,
                                                  T_wc, cfg.table_height,
                                                  depth=depth_r, segmap=segmap_r, K=K_r,
                                                  filter_neighbors=args.filter_neighbors)
```

- [ ] **Step 3 done**

**Step 4: Run the full existing test suite (no regressions)**

Run each of these from `mujoco_grasp_sim/` (`PYTHONPATH=. python
sim_grasp/<name>.py`): `test_placement_planner`, `test_executor_ease`,
`test_executor_place_orientation`, `test_color_utils`,
`test_resolve_real_label`, `test_scene_generator_paths`,
`test_feasibility`, `test_instruction_parser`,
`test_spatial_relation_resolver`, `test_subprocess_utils`,
`test_workspace_occupancy`, `test_reachability`. All must print their
`All ... passed.` line with no traceback.

- [ ] **Step 4 done**

**Step 5: Live smoke test — confirm zero behavior change without the flag**

Run (from `mujoco_grasp_sim/`, with `cgn_torch` active):

```bash
MUJOCO_GL=osmesa GRASPGEN_PYTHON=/home/vivek/miniconda3/envs/graspgen_torch/bin/python \
  python run_sim_grasp_test.py --pick-all --camera fused --backend graspgen \
  --seed 0 --no-vis
```

Confirm the `[feasibility] kept X/Y` counts match this same seed's numbers
from before this plan (no `+ N rejected by neighbor-collision/...` line
should print at all, since the flag isn't passed).

Then run the same command with `--filter-neighbors` appended, confirm the
new `[feasibility]   + N rejected by neighbor-collision/reachability
pre-filter` line prints and the run still completes without error.

- [ ] **Step 5: Both smoke test runs confirmed**

**Step 6: Benchmark A/B validation (per the design's Approach C — required,
not optional)**

Run both, from `mujoco_grasp_sim/`:

```bash
python benchmark.py --seeds 0-4 --mode pick-all --camera fused --backend graspgen \
  --graspgen-python /home/vivek/miniconda3/envs/graspgen_torch/bin/python \
  --tag neighbor_filter_off
python benchmark.py --seeds 0-4 --mode pick-all --camera fused --backend graspgen \
  --graspgen-python /home/vivek/miniconda3/envs/graspgen_torch/bin/python \
  --tag neighbor_filter_on
```

(`benchmark.py` doesn't have a `--filter-neighbors` passthrough flag yet —
if it doesn't accept unknown flags, add a one-line
`ap.add_argument('--filter-neighbors', action='store_true')` +
`if args.filter_neighbors: cmd += ['--filter-neighbors']` to
`benchmark.py` first, mirroring its existing `--recenter`/`--clean-depth`
passthrough pattern.)

Compare the two `summary.json` outputs' binned/knocked-off-table/failure
counts. Record the real numbers — do not declare this task done on "ran
without error" alone.

- [ ] **Step 6: Benchmark comparison recorded with real numbers**

**Step 7: Commit**

```bash
git add mujoco_grasp_sim/run_sim_grasp_test.py mujoco_grasp_sim/benchmark.py
git commit -m "Wire neighbor-collision + reachability pre-filter behind --filter-neighbors"
```

---

## Task 4: Documentation (per this repo's AGENTS.md workflow)

**Files:**
- Modify: `ROADMAP.md`

**Step 1: Record the result**

Per `AGENTS.md`'s mandatory documentation-sync workflow: this is a new
capability landing. Add a dated bullet under `ROADMAP.md` P1 (the two
long-standing TODO lines — "Filtering: neighbor-object collision check...,
workspace-reachability pre-filter" — already exist there; convert them
from `- [ ]` to `- [x]` and replace with a real dated entry) describing:
what was built (the two new modules, the opt-in `--filter-neighbors`
flag), and the Task 3 Step 6 benchmark A/B numbers (real counts, with the
same honesty about small-sample statistical weight this project's other
P1 entries already use where applicable).

- [ ] **Step 1: Update `ROADMAP.md`** with the real benchmark numbers from
      Task 3 Step 6 (not a placeholder)

**Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "Document neighbor-collision + reachability pre-filter in ROADMAP.md"
```

---

## Plan Self-Review Notes

- **Spec coverage**: Task 1 covers the design's `workspace_occupancy.py`
  section (including the `_hand_boxes()` DRY extraction the spec implied
  but didn't spell out at the code level — added here as a plan-level
  decision); Task 2 covers `reachability.py`; Task 3 covers the opt-in
  wiring, the required smoke tests (zero behavior change without the
  flag), and the design's mandatory benchmark A/B validation; Task 4
  covers this repo's `AGENTS.md`-mandated documentation sync converting
  the two long-standing ROADMAP.md TODO lines into a real dated entry.
- **Type consistency checked**: `WorkspaceOccupancy`'s two fields,
  `build_workspace_occupancy`'s and `collides_with_neighbors`'s exact
  signatures, and `is_reachable`'s signature are identical everywhere
  they're used across Tasks 1-3.
- **No placeholders**: every step contains complete, runnable code — no
  "add the check here" shortcuts; the exact before/after code blocks for
  every `run_sim_grasp_test.py` edit are spelled out in full in Task 3.
