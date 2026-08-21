# Intelligent Bin Placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded, single fixed bin-drop-point/release-height
placement logic in `mujoco_grasp_sim` with a vision-only (no MuJoCo internal
state) placement planner that finds a free, non-overlapping slot in the bin
and a dynamic release height, fixing object-stacking and tall-object-crushing
in both `run_sim_grasp_test.py` (`--pick-all` and single `--execute`) and
`interactive_pick.py`.

**Architecture:** A new `sim_grasp/placement_planner.py` module computes an
`ObjectFootprint` (size/yaw/height) from the object's own segmap mask and a
`BinHeightmap` (top-down occupancy grid) from the bin's region, both from the
depth/segmap capture every entry point already takes before each pick. An
`OccupancyPlacementPlanner` searches the heightmap for a free, oriented slot.
`executor.GraspExecutor.place()` is changed to take an explicit
`(x, y, release_z, yaw)` target instead of deriving Z from a fixed constant.

**Tech Stack:** Python, NumPy, OpenCV (`cv2.minAreaRect`, already a project
dependency), no new dependencies.

## Global Constraints

- Vision-only: no `model.geom_size`, `data.body().xpos`, or other MuJoCo
  internal-state queries anywhere in the new placement logic — only
  depth/segmap/intrinsics/extrinsics, matching the sim-to-real-portability
  requirement from the design.
- Follow this codebase's existing test convention exactly: standalone
  `test_*.py` scripts in `mujoco_grasp_sim/sim_grasp/`, run directly with
  `python <file>.py` (or as a module, see each task), using plain `assert`
  statements ending with a `print('All ... checks passed.')` line — **no
  pytest** (this codebase has no automated test suite; see the other
  `sim_grasp/test_*.py` files for the exact style to match).
- No re-tuning of unrelated, already-empirically-tuned constants
  (`PLACE_HOVER`, `RETRACT_DIST`, `LIFT_DIST`, etc.) — only `PLACE_RELEASE`'s
  role changes (from "added to a fixed drop_pos" to "no longer used inside
  `place()`; callers now compute the equivalent dynamically").
- All new/changed public function and class names below are exact — later
  tasks import them by these exact names.

---

### Task 1: Object footprint computation

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/placement_planner.py`
- Test: `mujoco_grasp_sim/sim_grasp/test_placement_planner.py`

**Interfaces:**
- Consumes: `depth_to_pointcloud(depth, K, mask=None, z_range=(0.05,4.0))` from
  `sim_grasp/pointcloud.py` (existing, returns `(N,3)` float32 in OpenCV
  camera frame); `transform_points(T, pts)` from `sim_grasp/frames.py`
  (existing, applies a 4x4 rigid transform to an `(N,3)` array).
- Produces: `ObjectFootprint` dataclass and `compute_object_footprint(...)`
  function, used by Task 3 and Task 5/6.

- [ ] **Step 1: Write the failing test**

Create `mujoco_grasp_sim/sim_grasp/test_placement_planner.py` with this
content (this step only adds the footprint-related assertions; later tasks
append more to the same file):

```python
"""Standalone checks for placement_planner.py -- run directly, no pytest
(this codebase has no automated test suite)."""
import numpy as np

from sim_grasp.placement_planner import compute_object_footprint

# Synthetic scene: a flat camera looking straight down (+Z world = -Z cam
# is not needed here -- we only need a depth/segmap/K/T_world_cam combo that
# puts a known box at a known world position/size).
H, W = 100, 100
fx = fy = 200.0
cx = cy = 50.0
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

# Camera directly above the origin, looking straight down: world (x, y, z)
# maps to camera frame (x_cam, y_cam, z_cam) = (x, -y, cam_height - z)
# (OpenCV convention: +Z forward/down, +Y down in image). Build T_world_cam
# such that transform_points(T_world_cam, pts_cam) recovers this world point.
cam_height = 1.0
T_world_cam = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, cam_height],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

depth = np.zeros((H, W), dtype=np.float32)
segmap = np.zeros((H, W), dtype=np.int32)

# A 20x10 pixel rectangle of segid=1 at depth=0.8 (i.e. world z = 1.0-0.8=0.2)
# centered at pixel (60, 50) -> world x=(60-50)*0.8/200=0.04, world y =
# -(50-50)*0.8/200=0.0 (since world_y = -cam_y).
depth[45:55, 50:70] = 0.8
segmap[45:55, 50:70] = 1

footprint = compute_object_footprint(depth, segmap, seg_id=1, K=K,
                                      T_world_cam=T_world_cam)
assert footprint is not None, 'footprint should be found for a valid mask'
assert abs(footprint.z_bottom - 0.2) < 1e-3, f'z_bottom={footprint.z_bottom}'
assert abs(footprint.z_top - 0.2) < 1e-3, f'z_top={footprint.z_top}'
# 20 px wide (x) -> 20*0.8/200 = 0.08 m; 10 px tall (y) -> 10*0.8/200 = 0.04 m
w, h = sorted(footprint.size_xy)
assert abs(h - 0.08) < 0.01, f'size_xy={footprint.size_xy}'
assert abs(w - 0.04) < 0.01, f'size_xy={footprint.size_xy}'

# Empty/degenerate mask -> None, not a crash
empty_segmap = np.zeros((H, W), dtype=np.int32)
assert compute_object_footprint(depth, empty_segmap, seg_id=1, K=K,
                                 T_world_cam=T_world_cam) is None

print('All placement_planner footprint checks passed.')
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `mujoco_grasp_sim/`): `python sim_grasp/test_placement_planner.py`
Expected: `ModuleNotFoundError: No module named 'sim_grasp.placement_planner'`
(or `ImportError: cannot import name 'compute_object_footprint'`)

- [ ] **Step 3: Write minimal implementation**

Create `mujoco_grasp_sim/sim_grasp/placement_planner.py`:

```python
"""Vision-only intelligent bin placement.

Computes WHERE (x, y, yaw) and how deep (release_z) to release a held
object in the bin, using only depth/segmap/intrinsics/extrinsics -- no
MuJoCo internal-state queries -- so the same code ports to a real
RealSense camera later. Mirrors the sim_grasp.grasp_predictor.GraspPredictor
ABC pattern used for pluggable grasp backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from sim_grasp.frames import transform_points
from sim_grasp.pointcloud import depth_to_pointcloud

MIN_FOOTPRINT_POINTS = 10


@dataclass
class ObjectFootprint:
    """World-frame footprint of an object, measured from its segmap mask
    while it sits on the table (before it is picked)."""
    center_xy: tuple      # world (x, y) center of the XY point cloud extent
    size_xy: tuple        # world (width, height) of the min-area XY rect
    yaw: float            # radians; orientation of size_xy[0]'s axis
    z_bottom: float        # world Z of the lowest point (resting surface)
    z_top: float           # world Z of the highest point


def compute_object_footprint(depth: np.ndarray, segmap: np.ndarray,
                              seg_id: int, K: np.ndarray,
                              T_world_cam: np.ndarray) -> 'ObjectFootprint | None':
    """Returns None if the object's mask yields too few valid depth points
    (degenerate/empty mask) rather than raising."""
    pts_cam = depth_to_pointcloud(depth, K, mask=(segmap == seg_id))
    if len(pts_cam) < MIN_FOOTPRINT_POINTS:
        return None
    pts_world = transform_points(T_world_cam, pts_cam)

    xy = pts_world[:, :2].astype(np.float32)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(xy)

    return ObjectFootprint(
        center_xy=(float(cx), float(cy)),
        size_xy=(float(w), float(h)),
        yaw=float(np.deg2rad(angle_deg)),
        z_bottom=float(pts_world[:, 2].min()),
        z_top=float(pts_world[:, 2].max()),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python sim_grasp/test_placement_planner.py`
Expected: `All placement_planner footprint checks passed.`

- [ ] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/placement_planner.py mujoco_grasp_sim/sim_grasp/test_placement_planner.py
git commit -m "Add ObjectFootprint + compute_object_footprint (vision-only)"
```

---

### Task 2: Bin heightmap computation

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/placement_planner.py`
- Modify: `mujoco_grasp_sim/sim_grasp/test_placement_planner.py` (append)

**Interfaces:**
- Consumes: `depth_to_pointcloud`, `transform_points` (same as Task 1).
- Produces: `BinHeightmap` dataclass and `build_bin_heightmap(...)`
  function, used by Task 3 and Task 5/6.

- [ ] **Step 1: Write the failing test**

Append to `mujoco_grasp_sim/sim_grasp/test_placement_planner.py`:

```python
from sim_grasp.placement_planner import build_bin_heightmap

# Separate, self-contained synthetic camera for the heightmap tests (wider
# frame than Task 1's footprint camera, so world coordinates around the bin
# stay in-bounds in pixel space): 300x300, cam directly above the origin
# looking straight down, cam_height=1.0.
H2 = W2 = 300
fx2 = fy2 = 150.0
cx2 = cy2 = 150.0
K2 = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]], dtype=np.float32)
cam_height2 = 1.0
T_world_cam2 = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, cam_height2],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def world_to_px2(x, y, z):
    d = cam_height2 - z
    u = cx2 + x * fx2 / d
    v = cy2 - y * fy2 / d
    return int(round(u)), int(round(v))


# Bin: 0.1m half-extent square centered at (0.5, 0.5). Background (world
# z=0, the bare bin floor) everywhere, plus one 0.03m-tall "object" patch
# (seg_id=2) at world (0.52, 0.50).
depth2 = np.full((H2, W2), cam_height2, dtype=np.float32)   # world z=0 everywhere
segmap2 = np.zeros((H2, W2), dtype=np.int32)
u0, v0 = world_to_px2(0.52, 0.50, 0.03)
assert 0 <= u0 < W2 and 0 <= v0 < H2, 'test setup bug: patch center out of bounds'
depth2[v0 - 5:v0 + 5, u0 - 5:u0 + 5] = cam_height2 - 0.03
segmap2[v0 - 5:v0 + 5, u0 - 5:u0 + 5] = 2

heightmap = build_bin_heightmap(depth2, segmap2, K2, T_world_cam2,
                                 bin_center=(0.5, 0.5), bin_inner_half=0.1,
                                 exclude_seg_id=None, cell_size=0.01)
assert heightmap.heights.shape[0] > 0 and heightmap.heights.shape[1] > 0
assert abs(heightmap.floor_z - 0.0) < 0.01, f'floor_z={heightmap.floor_z}'
assert float(np.nanmax(heightmap.heights)) > 0.02, 'the bump should show up'

# Excluding seg_id=2 should make the heightmap look flat/empty everywhere.
heightmap_excl = build_bin_heightmap(depth2, segmap2, K2, T_world_cam2,
                                      bin_center=(0.5, 0.5), bin_inner_half=0.1,
                                      exclude_seg_id=2, cell_size=0.01)
assert float(np.nanmax(heightmap_excl.heights)) < 0.01, \
    'excluding the bump object should leave a flat heightmap'

print('All placement_planner heightmap checks passed.')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python sim_grasp/test_placement_planner.py`
Expected: `ImportError: cannot import name 'build_bin_heightmap'`

- [ ] **Step 3: Write minimal implementation**

Append to `mujoco_grasp_sim/sim_grasp/placement_planner.py`:

```python
@dataclass
class BinHeightmap:
    """Top-down occupancy grid of the bin interior. `heights[row, col]` is
    the highest observed world Z in that cell; `floor_z` is the
    self-calibrated 'empty bin' reference height."""
    heights: np.ndarray          # (rows, cols) float32
    origin_xy: tuple             # world (x, y) of cell [0, 0]'s lower-left corner
    cell_size: float
    floor_z: float


def build_bin_heightmap(depth: np.ndarray, segmap: np.ndarray, K: np.ndarray,
                         T_world_cam: np.ndarray, bin_center: tuple,
                         bin_inner_half: float, exclude_seg_id: 'int | None' = None,
                         cell_size: float = 0.005) -> BinHeightmap:
    mask = np.ones_like(segmap, dtype=bool)
    if exclude_seg_id is not None:
        mask &= (segmap != exclude_seg_id)

    pts_cam = depth_to_pointcloud(depth, K, mask=mask)
    pts_world = transform_points(T_world_cam, pts_cam) if len(pts_cam) else pts_cam

    bx, by = bin_center
    n = int(np.ceil(2 * bin_inner_half / cell_size))
    origin_x, origin_y = bx - bin_inner_half, by - bin_inner_half
    heights = np.full((n, n), np.nan, dtype=np.float32)

    if len(pts_world):
        in_bin = ((np.abs(pts_world[:, 0] - bx) < bin_inner_half) &
                   (np.abs(pts_world[:, 1] - by) < bin_inner_half))
        pw = pts_world[in_bin]
        if len(pw):
            cols = np.clip(((pw[:, 0] - origin_x) / cell_size).astype(int), 0, n - 1)
            rows = np.clip(((pw[:, 1] - origin_y) / cell_size).astype(int), 0, n - 1)
            np.maximum.at(heights, (rows, cols), pw[:, 2])

    valid = ~np.isnan(heights)
    floor_z = float(np.nanpercentile(heights, 10)) if np.any(valid) else 0.0
    heights = np.where(valid, heights, floor_z).astype(np.float32)

    return BinHeightmap(heights=heights, origin_xy=(origin_x, origin_y),
                        cell_size=cell_size, floor_z=floor_z)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python sim_grasp/test_placement_planner.py`
Expected: `All placement_planner heightmap checks passed.`

- [ ] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/placement_planner.py mujoco_grasp_sim/sim_grasp/test_placement_planner.py
git commit -m "Add BinHeightmap + build_bin_heightmap"
```

---

### Task 3: Free-space + orientation search, and release-Z helper

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/placement_planner.py`
- Modify: `mujoco_grasp_sim/sim_grasp/test_placement_planner.py` (append)

**Interfaces:**
- Consumes: `ObjectFootprint` (Task 1), `BinHeightmap` (Task 2).
- Produces: `PlacementPose` dataclass, `PlacementPlanner` ABC,
  `OccupancyPlacementPlanner` class with `.plan(footprint, heightmap) ->
  PlacementPose | None`, and `compute_release_z(pose, T_world_grasp,
  footprint, safety_buffer=0.005) -> float`. Used by Task 5/6.

- [ ] **Step 1: Write the failing test**

Append to `mujoco_grasp_sim/sim_grasp/test_placement_planner.py`:

```python
from sim_grasp.placement_planner import (
    OccupancyPlacementPlanner, ObjectFootprint, BinHeightmap, compute_release_z)

# An empty 24x24cm bin (matches the project's default SceneConfig), cell
# size 5mm -> 48x48 grid, all at floor height 0.75 (table height).
n = 48
empty_heights = np.full((n, n), 0.75, dtype=np.float32)
empty_map = BinHeightmap(heights=empty_heights, origin_xy=(0.33, -0.42),
                          cell_size=0.005, floor_z=0.75)

small_footprint = ObjectFootprint(center_xy=(0.0, 0.0), size_xy=(0.03, 0.03),
                                   yaw=0.0, z_bottom=0.75, z_top=0.80)

planner = OccupancyPlacementPlanner(bin_center=(0.45, -0.30), bin_inner_half=0.12)
pose = planner.plan(small_footprint, empty_map)
assert pose is not None, 'a 3cm object must fit in an empty 24cm bin'
assert abs(pose.release_z - 0.75) < 1e-6, f'release_z={pose.release_z}'
assert abs(pose.x - 0.45) < 0.12 and abs(pose.y - (-0.30)) < 0.12

# Occupy the whole left half of the bin (col < n/2) at height 0.80 (5cm
# stack) -- planner must choose a slot on the right (unoccupied) side.
occupied_heights = empty_heights.copy()
occupied_heights[:, : n // 2] = 0.80
occupied_map = BinHeightmap(heights=occupied_heights, origin_xy=(0.33, -0.42),
                            cell_size=0.005, floor_z=0.75)
pose2 = planner.plan(small_footprint, occupied_map)
assert pose2 is not None
right_half_x_min = 0.33 + (n // 2) * 0.005
assert pose2.x >= right_half_x_min - 0.01, \
    f'expected a slot on the unoccupied side, got x={pose2.x}'

# A footprint bigger than the whole bin can never fit -> fallback path still
# returns SOME pose (never crashes), not None, per the design's fallback.
huge_footprint = ObjectFootprint(center_xy=(0, 0), size_xy=(0.5, 0.5),
                                 yaw=0.0, z_bottom=0.75, z_top=0.9)
pose3 = planner.plan(huge_footprint, empty_map)
assert pose3 is None, 'a footprint that cannot fit at all must return None'

# compute_release_z: grasp_offset = T_world_grasp z - footprint.z_bottom
T_world_grasp = np.eye(4)
T_world_grasp[2, 3] = 0.95   # hand origin 0.20m above the object's bottom (0.75)
release_z = compute_release_z(pose, T_world_grasp, small_footprint, safety_buffer=0.005)
assert abs(release_z - (0.75 + 0.20 + 0.005)) < 1e-6, f'release_z={release_z}'

print('All placement_planner search checks passed.')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python sim_grasp/test_placement_planner.py`
Expected: `ImportError: cannot import name 'OccupancyPlacementPlanner'`

- [ ] **Step 3: Write minimal implementation**

Append to `mujoco_grasp_sim/sim_grasp/placement_planner.py`:

```python
_OCCUPIED_MARGIN = 0.008    # 8mm above floor counts as "occupied"
_CLEARANCE_MARGIN = 0.003   # 3mm extra around the footprint when checking cells
_SEARCH_STRIDE = 0.01       # 1cm stride between candidate slot centers
_CANDIDATE_YAW_OFFSETS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


@dataclass
class PlacementPose:
    x: float
    y: float
    yaw: float           # radians
    release_z: float     # world Z the object's BOTTOM should end up at


def _aabb_half_extent(size_xy: tuple, yaw: float) -> tuple:
    """Half-extents (world X, world Y) of `size_xy`'s rectangle rotated by
    `yaw`, i.e. its axis-aligned bounding box -- a deliberate simplification
    over exact oriented-rectangle collision (fine for mildly rectangular
    objects like these)."""
    w, h = size_xy
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    return (w / 2 * c + h / 2 * s, w / 2 * s + h / 2 * c)


class PlacementPlanner(ABC):
    """Implement this to plug in a different placement strategy."""

    @abstractmethod
    def plan(self, footprint: ObjectFootprint,
             heightmap: BinHeightmap) -> 'PlacementPose | None':
        """Returns a world (x, y, yaw, release_z) target, or None if the
        footprint cannot fit in the bin at any orientation."""


class OccupancyPlacementPlanner(PlacementPlanner):
    def __init__(self, bin_center: tuple, bin_inner_half: float,
                 occupied_margin: float = _OCCUPIED_MARGIN,
                 clearance_margin: float = _CLEARANCE_MARGIN,
                 search_stride: float = _SEARCH_STRIDE):
        self.bin_center = bin_center
        self.bin_inner_half = bin_inner_half
        self.occupied_margin = occupied_margin
        self.clearance_margin = clearance_margin
        self.search_stride = search_stride

    def plan(self, footprint: ObjectFootprint,
             heightmap: BinHeightmap) -> 'PlacementPose | None':
        bx, by = self.bin_center
        occupied_z = heightmap.floor_z + self.occupied_margin
        rows, cols = heightmap.heights.shape
        xs = heightmap.origin_xy[0] + (np.arange(cols) + 0.5) * heightmap.cell_size
        ys = heightmap.origin_xy[1] + (np.arange(rows) + 0.5) * heightmap.cell_size
        grid_x, grid_y = np.meshgrid(xs, ys)
        occupied_mask = heightmap.heights > occupied_z
        occ_x, occ_y = grid_x[occupied_mask], grid_y[occupied_mask]

        any_orientation_fits = False
        best = None  # (clearance, x, y, yaw)
        for yaw_offset in _CANDIDATE_YAW_OFFSETS:
            yaw = footprint.yaw + yaw_offset
            half_ex, half_ey = _aabb_half_extent(footprint.size_xy, yaw)
            half_ex += self.clearance_margin
            half_ey += self.clearance_margin
            x_min, x_max = bx - self.bin_inner_half + half_ex, bx + self.bin_inner_half - half_ex
            y_min, y_max = by - self.bin_inner_half + half_ey, by + self.bin_inner_half - half_ey
            if x_min > x_max or y_min > y_max:
                continue
            any_orientation_fits = True
            for cx in np.arange(x_min, x_max + 1e-9, self.search_stride):
                for cy in np.arange(y_min, y_max + 1e-9, self.search_stride):
                    in_box = ((np.abs(grid_x - cx) <= half_ex) &
                              (np.abs(grid_y - cy) <= half_ey))
                    if np.any(heightmap.heights[in_box] > occupied_z):
                        continue
                    clearance = (float(np.min(np.hypot(occ_x - cx, occ_y - cy)))
                                 if len(occ_x) else float('inf'))
                    if best is None or clearance > best[0]:
                        best = (clearance, cx, cy, yaw)

        if best is not None:
            _, x, y, yaw = best
            return PlacementPose(x=float(x), y=float(y), yaw=float(yaw),
                                 release_z=heightmap.floor_z)
        if not any_orientation_fits:
            return None   # the footprint cannot fit in this bin at all

        # Fallback: bin has no fully-clear spot -- least-bad option.
        half_ex, half_ey = _aabb_half_extent(footprint.size_xy, footprint.yaw)
        x_min, x_max = bx - self.bin_inner_half + half_ex, bx + self.bin_inner_half - half_ex
        y_min, y_max = by - self.bin_inner_half + half_ey, by + self.bin_inner_half - half_ey
        best_fallback = None
        for cx in np.arange(x_min, x_max + 1e-9, self.search_stride):
            for cy in np.arange(y_min, y_max + 1e-9, self.search_stride):
                in_box = ((np.abs(grid_x - cx) <= half_ex) &
                          (np.abs(grid_y - cy) <= half_ey))
                cell_heights = heightmap.heights[in_box]
                if len(cell_heights) == 0:
                    continue
                max_h = float(np.max(cell_heights))
                if best_fallback is None or max_h < best_fallback[0]:
                    best_fallback = (max_h, cx, cy)
        if best_fallback is None:
            return None
        max_h, x, y = best_fallback
        return PlacementPose(x=float(x), y=float(y), yaw=footprint.yaw,
                             release_z=max_h)


def compute_release_z(pose: PlacementPose, T_world_grasp: np.ndarray,
                      footprint: ObjectFootprint,
                      safety_buffer: float = 0.005) -> float:
    """Hand-origin target Z for release: the chosen slot's floor height,
    plus how far below the gripper the object's bottom sat when grasped
    (measured, not assumed), plus a small safety buffer."""
    grasp_offset = float(T_world_grasp[2, 3]) - footprint.z_bottom
    return pose.release_z + grasp_offset + safety_buffer
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python sim_grasp/test_placement_planner.py`
Expected: `All placement_planner search checks passed.`

- [ ] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/placement_planner.py mujoco_grasp_sim/sim_grasp/test_placement_planner.py
git commit -m "Add OccupancyPlacementPlanner free-space search + compute_release_z"
```

---

### Task 4: Wire dynamic release pose into `GraspExecutor.place()`

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/executor.py:68-74` (constants),
  `mujoco_grasp_sim/sim_grasp/executor.py:303-333` (`place()` method)
- Test: `mujoco_grasp_sim/sim_grasp/test_executor_place_orientation.py`

**Interfaces:**
- Consumes: nothing new (uses `TOPDOWN_HAND_R`, `PLACE_HOVER` already in
  `executor.py`; `scipy.spatial.transform.Rotation` already imported as `R`).
- Produces: `_candidate_hand_orientations(R_cur, yaw) -> tuple[np.ndarray,
  np.ndarray, np.ndarray]` (module-level, importable for tests), and the new
  `GraspExecutor.place(x, y, release_z, yaw=0.0) -> dict` signature that
  Task 5 and Task 6 call.

- [ ] **Step 1: Write the failing test**

Create `mujoco_grasp_sim/sim_grasp/test_executor_place_orientation.py`:

```python
"""Standalone check for executor._candidate_hand_orientations -- run
directly, no pytest (this codebase has no automated test suite)."""
import numpy as np

from sim_grasp.executor import _candidate_hand_orientations, TOPDOWN_HAND_R

R_cur = np.eye(3)

cands = _candidate_hand_orientations(R_cur, yaw=0.0)
assert len(cands) == 3
assert np.allclose(cands[0], R_cur), 'first candidate must be R_cur as-is'
assert np.allclose(cands[1], TOPDOWN_HAND_R), 'yaw=0 must leave TOPDOWN_HAND_R unchanged'
assert np.allclose(cands[2], TOPDOWN_HAND_R), 'last-resort candidate is always TOPDOWN_HAND_R'

# A 90deg yaw must rotate TOPDOWN_HAND_R's in-plane (x/y) columns but leave
# its z column (approach direction, world -Z) unchanged.
cands90 = _candidate_hand_orientations(R_cur, yaw=np.pi / 2)
yawed = cands90[1]
assert np.allclose(yawed[:, 2], TOPDOWN_HAND_R[:, 2]), 'yaw must not tilt the approach axis'
assert not np.allclose(yawed[:, 0], TOPDOWN_HAND_R[:, 0]), 'yaw must rotate the closing axis'
assert np.allclose(cands90[0], R_cur), 'R_cur candidate must never be yawed'
assert np.allclose(cands90[2], TOPDOWN_HAND_R), 'last-resort candidate must never be yawed'

print('All executor orientation checks passed.')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python sim_grasp/test_executor_place_orientation.py`
Expected: `ImportError: cannot import name '_candidate_hand_orientations'`

- [ ] **Step 3: Write minimal implementation**

In `mujoco_grasp_sim/sim_grasp/executor.py`, replace lines 66-74 (the
existing `PLACE_HOVER`/`PLACE_RELEASE`/`TOPDOWN_HAND_R` block):

```python
# Place sequence (heights of the HAND ORIGIN above the drop point;
# fingertips are ~0.10 m below the hand origin)
PLACE_HOVER = 0.24     # transit / retract height
PLACE_RELEASE = 0.15   # legacy fixed release height, kept only as a fallback
                       # constant for callers whose vision-based placement
                       # planner fails (see placement_planner.py) — fingertips
                       # are ~0.10 m below the hand origin, so the object
                       # drops ~5 cm; at 0.17 it dropped ~7 cm and bounced out
                       # of the 5 cm bin walls (taxonomy: missed_bin x4)
_HOVER_ABOVE_RELEASE = PLACE_HOVER - PLACE_RELEASE  # 0.09 m transit clearance,
                                                     # now anchored to whatever
                                                     # release_z callers pass in
TOPDOWN_HAND_R = np.array([[1.0, 0.0, 0.0],    # canonical hand-down pose,
                           [0.0, -1.0, 0.0],   # fallback orientation for the
                           [0.0, 0.0, -1.0]])  # place IK


def _candidate_hand_orientations(R_cur: np.ndarray, yaw: float) -> tuple:
    """Ordered hand-orientation candidates for place(): (1) current hand
    orientation as-is, (2) canonical top-down rotated by `yaw` about world
    Z -- a top-down grasp preserves the object's on-table yaw through the
    pick, so this reliably rotates the placed object -- (3) canonical
    top-down with yaw=0, as a last resort matching the pre-existing
    unconditional fallback."""
    Rz_yaw = R.from_euler('z', yaw).as_matrix()
    return (R_cur, Rz_yaw @ TOPDOWN_HAND_R, TOPDOWN_HAND_R)
```

Then replace the `place()` method (lines 303-333 of the original file) with:

```python
    def place(self, x: float, y: float, release_z: float, yaw: float = 0.0) -> dict:
        """Carry the held object above (x, y), lower until the hand origin
        reaches `release_z`, open the fingers, retract. Tries, in order:
        (1) the current hand orientation as-is, (2) a canonical top-down
        orientation rotated by `yaw`, (3) a canonical top-down orientation
        with no yaw (last resort) -- see _candidate_hand_orientations."""
        data = self.data
        q_now = data.qpos[self.ik.qpos_idx].copy()
        R_cur = data.xmat[self.ik.hand_bid].reshape(3, 3).copy()

        plan = None
        for R_hand in _candidate_hand_orientations(R_cur, yaw):
            T_pre = np.eye(4)
            T_pre[:3, :3] = R_hand
            T_pre[:3, 3] = [x, y, release_z + _HOVER_ABOVE_RELEASE]
            ik_pre = self.ik.solve(T_pre, q_now)
            if not ik_pre.converged:
                continue
            T_rel = T_pre.copy()
            T_rel[2, 3] = release_z
            ik_rel = self.ik.solve(T_rel, ik_pre.qpos)
            if ik_rel.converged:
                plan = (ik_pre, ik_rel)
                break
        if plan is None:
            # bin unreachable from this grasp pose: release in place so the
            # object drops back on the table and can be retried next round
            self._step_to(data.ctrl[:7].copy(), 0.5, gripper_ctrl=GRIPPER_OPEN)
            self._hold(0.3)
            return {'placed': False, 'stage': 'ik_place'}

        ik_pre, ik_rel = plan
        self._step_to(ik_pre.qpos, 2.2, gripper_ctrl=GRIPPER_CLOSED)   # transit
        self._step_to(ik_rel.qpos, 1.0, gripper_ctrl=GRIPPER_CLOSED)   # lower
        self._hold(0.2)
        self._step_to(ik_rel.qpos, 0.6, gripper_ctrl=GRIPPER_OPEN)     # release
        self._hold(0.4)
        self._step_to(ik_pre.qpos, 0.8, gripper_ctrl=GRIPPER_OPEN)     # retract
        return {'placed': True, 'stage': 'place_done'}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python sim_grasp/test_executor_place_orientation.py`
Expected: `All executor orientation checks passed.`

Also run the codebase's existing executor-adjacent standalone tests to
confirm nothing else broke:
`python sim_grasp/test_color_utils.py && python sim_grasp/test_resolve_real_label.py`
Expected: both print their existing "All ... checks passed." lines.

- [ ] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/executor.py mujoco_grasp_sim/sim_grasp/test_executor_place_orientation.py
git commit -m "GraspExecutor.place() takes an explicit (x, y, release_z, yaw) target"
```

---

### Task 5: Wire the planner into `run_sim_grasp_test.py` (both call sites)

**Files:**
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py:543` (pick-all setup),
  `mujoco_grasp_sim/run_sim_grasp_test.py:620-645` (pick-all per-round
  placement), `mujoco_grasp_sim/run_sim_grasp_test.py:716-744` (single
  `--execute` placement)

**Interfaces:**
- Consumes: `compute_object_footprint`, `build_bin_heightmap`,
  `OccupancyPlacementPlanner`, `compute_release_z` (Tasks 1-3);
  `GraspExecutor.place(x, y, release_z, yaw=0.0)` (Task 4);
  `executor.PLACE_RELEASE` (existing, now used only as the legacy fallback
  constant, per Task 4).
- Produces: nothing new for later tasks (this is a leaf integration point).

This task has no isolated unit test (it wires vision + physics together,
consistent with this codebase's existing convention of validating pick/place
integration via manual `run_sim_grasp_test.py` runs, not automated tests —
see `CLAUDE.md`: "There is no automated test suite... `run_sim_grasp_test.py`
runs are the correctness checks"). Validation is Task 7.

- [ ] **Step 1: Add the import and one shared planner instance**

In `mujoco_grasp_sim/run_sim_grasp_test.py`, near the top where other
`sim_grasp` imports live (find the existing `from sim_grasp...` import
lines near the top of the file and add alongside them):

```python
from sim_grasp.placement_planner import (
    compute_object_footprint, build_bin_heightmap, OccupancyPlacementPlanner,
    compute_release_z)
from sim_grasp.executor import PLACE_RELEASE
```

- [ ] **Step 2: Replace the pick-all loop's placement call**

Find this block (around line 543, inside the `if args.pick_all:` branch,
right after `executor = GraspExecutor(...)`):

```python
        label_of = {name: i + 1 for i, name in enumerate(gen.object_names)}
        drop = gen.bin_drop_point()
```

Replace with:

```python
        label_of = {name: i + 1 for i, name in enumerate(gen.object_names)}
        drop = gen.bin_drop_point()   # kept only as the legacy fallback target
        placement_planner = OccupancyPlacementPlanner(cfg.bin_center, cfg.bin_inner_half)
```

Then find this block (around lines 620-645, inside the `for rnd in
range(...)` loop, right after `T_world_grasp = T_wc @ np.asarray(g_r[sid][i])`
and its `if args.recenter:` block):

```python
            T_world_grasp = T_wc @ np.asarray(g_r[sid][i])
            shift = 0.0
            if args.recenter:
                d_o, seg_o, K_o = obs
                pts_cam = depth_to_pointcloud(d_o, K_o, mask=(seg_o == int(sid)))
                T_world_grasp, shift = recenter_grasp(
                    T_world_grasp, transform_points(T_wc, pts_cam),
                    table_z=cfg.table_height)
                if shift:
                    print(f'[recenter] grasp shifted {shift * 1e3:+.1f} mm '
                          'along the closing axis')
```

Replace with (adds an unconditional footprint/heightmap computation right
after the existing recenter block, using the same `obs` tuple):

```python
            T_world_grasp = T_wc @ np.asarray(g_r[sid][i])
            shift = 0.0
            if args.recenter:
                d_o, seg_o, K_o = obs
                pts_cam = depth_to_pointcloud(d_o, K_o, mask=(seg_o == int(sid)))
                T_world_grasp, shift = recenter_grasp(
                    T_world_grasp, transform_points(T_wc, pts_cam),
                    table_z=cfg.table_height)
                if shift:
                    print(f'[recenter] grasp shifted {shift * 1e3:+.1f} mm '
                          'along the closing axis')

            d_o, seg_o, K_o = obs
            footprint = compute_object_footprint(d_o, seg_o, int(sid), K_o, T_wc)
            place_pose = None
            if footprint is not None:
                heightmap = build_bin_heightmap(
                    d_o, seg_o, K_o, T_wc, cfg.bin_center, cfg.bin_inner_half,
                    exclude_seg_id=int(sid))
                place_pose = placement_planner.plan(footprint, heightmap)
            if place_pose is None:
                print('[placement] footprint/slot search failed for object '
                      f'{int(sid)} — falling back to the fixed bin drop point')
```

Then find this line (originally around line 643):

```python
            if res['success']:
                entry['place'] = executor.place(drop)
```

Replace with:

```python
            if res['success']:
                if place_pose is not None:
                    release_z = compute_release_z(place_pose, T_world_grasp, footprint)
                    entry['place'] = executor.place(
                        place_pose.x, place_pose.y, release_z, place_pose.yaw)
                else:
                    entry['place'] = executor.place(
                        drop[0], drop[1], drop[2] + PLACE_RELEASE)
```

- [ ] **Step 3: Replace the single `--execute` path's placement call**

Find this block (around lines 716-731, inside the `elif args.execute:`
branch, in the `for attempt, (sid, T_cam_grasp, score) in
enumerate(ranked, 1):` loop, right after the `if args.recenter:` block):

```python
            shift = 0.0
            if args.recenter:
                pts_cam = depth_to_pointcloud(depth, K, mask=(segmap == int(sid)))
                T_world_grasp, shift = recenter_grasp(
                    T_world_grasp, transform_points(T_world_cam, pts_cam),
                    table_z=cfg.table_height)
                if shift:
                    print(f'[recenter] grasp shifted {shift * 1e3:+.1f} mm '
                          'along the closing axis')
```

Replace with:

```python
            shift = 0.0
            if args.recenter:
                pts_cam = depth_to_pointcloud(depth, K, mask=(segmap == int(sid)))
                T_world_grasp, shift = recenter_grasp(
                    T_world_grasp, transform_points(T_world_cam, pts_cam),
                    table_z=cfg.table_height)
                if shift:
                    print(f'[recenter] grasp shifted {shift * 1e3:+.1f} mm '
                          'along the closing axis')

            footprint = compute_object_footprint(depth, segmap, int(sid), K, T_world_cam)
            place_pose = None
            if footprint is not None:
                heightmap = build_bin_heightmap(
                    depth, segmap, K, T_world_cam, cfg.bin_center,
                    cfg.bin_inner_half, exclude_seg_id=int(sid))
                place_pose = OccupancyPlacementPlanner(
                    cfg.bin_center, cfg.bin_inner_half).plan(footprint, heightmap)
```

Then find (originally around line 744):

```python
            if res['success']:
                res['place'] = executor.place(gen.bin_drop_point())
                res['in_bin'] = body in gen.objects_in_bin()
```

Replace with:

```python
            if res['success']:
                if place_pose is not None:
                    release_z = compute_release_z(place_pose, T_world_grasp, footprint)
                    res['place'] = executor.place(
                        place_pose.x, place_pose.y, release_z, place_pose.yaw)
                else:
                    drop = gen.bin_drop_point()
                    res['place'] = executor.place(
                        drop[0], drop[1], drop[2] + PLACE_RELEASE)
                res['in_bin'] = body in gen.objects_in_bin()
```

- [ ] **Step 4: Smoke-test both paths**

Run (from `mujoco_grasp_sim/`, headless is fine):
```bash
python run_sim_grasp_test.py --execute --no-vis
python run_sim_grasp_test.py --pick-all --no-vis
```
Expected: both complete without a Python traceback, and the console log
shows `[pick-all] round N: picking object ... -> placed in bin` (or a
`[placement] footprint/slot search failed ...` fallback line, which is also
an acceptable non-crashing outcome) instead of any `AttributeError`/
`TypeError` from the changed `executor.place(...)` call signature.

- [ ] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/run_sim_grasp_test.py
git commit -m "Wire OccupancyPlacementPlanner into run_sim_grasp_test.py (pick-all + execute)"
```

---

### Task 6: Wire the planner into `interactive_pick.py`

**Files:**
- Modify: `mujoco_grasp_sim/interactive_pick.py:195-206` (placement call)

**Interfaces:**
- Consumes: same as Task 5.
- Produces: nothing new for later tasks.

No isolated unit test for the same reason as Task 5 (physics/vision
integration point); validated manually in Task 7.

- [ ] **Step 1: Add the import**

Near the top of `mujoco_grasp_sim/interactive_pick.py`, alongside the other
`from sim_grasp...` imports:

```python
from sim_grasp.placement_planner import (
    compute_object_footprint, build_bin_heightmap, OccupancyPlacementPlanner,
    compute_release_z)
from sim_grasp.executor import PLACE_RELEASE
```

- [ ] **Step 2: Replace the placement call**

Find this block (around lines 195-206):

```python
    executor = GraspExecutor(model, data, camera_module=rec_cam, record_gif=True,
                             record_dir=save_dir / '_gif_frames',
                             on_frame=viewer.show_frame)
    # snapshot the settled state so each retry attempt starts identically
    qpos0, qvel0, ctrl0 = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    res = None
    for attempt, i in enumerate(order, 1):
        data.qpos[:], data.qvel[:], data.ctrl[:] = qpos0, qvel0, ctrl0
        mujoco.mj_forward(model, data)
        T_cam_grasp = grasps_cam[real_label][i]
        score = float(s[i])
        T_world_grasp = T_world_cam @ np.asarray(T_cam_grasp)
        print(f'[execute] attempt {attempt}/{len(order)}: object {real_label} '
              f'({body}), score {score:.3f} — watch the window...')
        res = executor.execute(T_world_grasp, target_body=body)
        res.update(object=real_label, score=score)
        if res['success']:
            res['place'] = executor.place(gen.bin_drop_point())
            res['in_bin'] = body in gen.objects_in_bin()
```

Replace with:

```python
    executor = GraspExecutor(model, data, camera_module=rec_cam, record_gif=True,
                             record_dir=save_dir / '_gif_frames',
                             on_frame=viewer.show_frame)
    footprint = compute_object_footprint(depth, new_segmap, real_label, K, T_world_cam)
    place_pose = None
    if footprint is not None:
        heightmap = build_bin_heightmap(
            depth, new_segmap, K, T_world_cam, cfg.bin_center, cfg.bin_inner_half,
            exclude_seg_id=real_label)
        place_pose = OccupancyPlacementPlanner(
            cfg.bin_center, cfg.bin_inner_half).plan(footprint, heightmap)
    if place_pose is None:
        print('[placement] footprint/slot search failed — falling back to '
              'the fixed bin drop point')
    # snapshot the settled state so each retry attempt starts identically
    qpos0, qvel0, ctrl0 = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    res = None
    for attempt, i in enumerate(order, 1):
        data.qpos[:], data.qvel[:], data.ctrl[:] = qpos0, qvel0, ctrl0
        mujoco.mj_forward(model, data)
        T_cam_grasp = grasps_cam[real_label][i]
        score = float(s[i])
        T_world_grasp = T_world_cam @ np.asarray(T_cam_grasp)
        print(f'[execute] attempt {attempt}/{len(order)}: object {real_label} '
              f'({body}), score {score:.3f} — watch the window...')
        res = executor.execute(T_world_grasp, target_body=body)
        res.update(object=real_label, score=score)
        if res['success']:
            if place_pose is not None and footprint is not None:
                release_z = compute_release_z(place_pose, T_world_grasp, footprint)
                res['place'] = executor.place(
                    place_pose.x, place_pose.y, release_z, place_pose.yaw)
            else:
                drop = gen.bin_drop_point()
                res['place'] = executor.place(drop[0], drop[1], drop[2] + PLACE_RELEASE)
            res['in_bin'] = body in gen.objects_in_bin()
```

- [ ] **Step 3: Smoke-test**

Run `python interactive_pick.py` (or its documented equivalent entry point —
check `mujoco_grasp_sim/README.md`'s interactive-pick section for the exact
invocation/flags used in this repo), click an object, confirm the console
shows either a successful `[execute] ... place` result or the
`[placement] footprint/slot search failed ...` fallback line, with no
traceback.

- [ ] **Step 4: Commit**

```bash
git add mujoco_grasp_sim/interactive_pick.py
git commit -m "Wire OccupancyPlacementPlanner into interactive_pick.py"
```

---

### Task 7: Validate the fix and record real numbers

**Files:**
- Modify: `mujoco_grasp_sim/README.md` (or the repo root `README.md`'s
  "Progress at a glance" table, whichever already documents placement/bin
  results — check both before editing) and `ROADMAP.md` (append a dated
  entry, matching the existing convention of dated milestone entries).

**Interfaces:** none (documentation + validation task).

- [ ] **Step 1: Visual sanity runs**

Run at least 3 different seeds and visually confirm (via `execution.gif` in
each run's `output/<run>/` directory, or `--no-vis` console logs plus
`metrics.json`'s `pick_all.in_bin` list) that all 3 objects land in
distinct, non-overlapping positions in the bin, and that no run shows a
lost-grasp immediately after a successful pick (the signature of the old
stacking/crushing bugs):

```bash
python run_sim_grasp_test.py --pick-all --camera fused --seed 0
python run_sim_grasp_test.py --pick-all --camera fused --seed 1
python run_sim_grasp_test.py --pick-all --camera fused --seed 2
```

- [ ] **Step 2: Benchmark comparison**

Run the project's existing multi-seed benchmark before/after this change
(if not already captured on a pre-change checkout, run it now on `main` at
the commit just before this branch started, then again on this branch, to
get a real before/after comparison — matching `ROADMAP.md`'s existing
"never trust a single run" convention):

```bash
python benchmark.py --seeds 0-9 --mode pick-all --camera fused --tag placement_intelligent
python analyze_failures.py output/bench_placement_intelligent
```

Record the resulting objects-binned success rate and failure taxonomy
breakdown.

- [ ] **Step 3: Record the result**

Append a new dated entry to `ROADMAP.md`'s existing milestones list (match
the exact format of neighboring entries, e.g. `| YYYY-MM-DD | Milestone |
Result |`), and if `README.md`'s "Progress at a glance" table/chart already
tracks bin-placement-specific numbers, update it too. Use the ACTUAL numbers
from Step 2's `benchmark.py`/`analyze_failures.py` output — do not estimate.

- [ ] **Step 4: Commit**

```bash
git add ROADMAP.md README.md
git commit -m "Record intelligent-placement benchmark results"
```
