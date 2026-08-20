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
