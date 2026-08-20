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

# ============================================================================
# Bin heightmap tests
# ============================================================================

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
