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
