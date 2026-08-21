"""Standalone checks for spatial_relation_resolver.py -- run directly, no
pytest. No real SAM 3 subprocess: near-relation tests use a fake
PromptSelector duck-typing .select()'s interface."""
import numpy as np

from sim_grasp.spatial_relation_resolver import resolve, _camera_view_axis
from sim_grasp.placement_planner import BinHeightmap, ObjectFootprint

BIN_CENTER = (0.45, -0.30)
BIN_INNER_HALF = 0.12

# ---- _camera_view_axis --------------------------------------------------
# Camera's local +X (image-right) aligned with world +X.
T_cam_x_right = np.eye(4)
axis, sign = _camera_view_axis(T_cam_x_right)
assert axis == 'x' and sign == 1.0

# Camera's local +X aligned with world -Y.
T_cam_neg_y = np.eye(4)
T_cam_neg_y[:3, 0] = [0.0, -1.0, 0.0]
axis, sign = _camera_view_axis(T_cam_neg_y)
assert axis == 'y' and sign == -1.0

print('All spatial_relation_resolver camera-axis checks passed.')

# ---- relation -> sub-region planner -------------------------------------
n = 48
empty_heights = np.full((n, n), 0.75, dtype=np.float32)
empty_map = BinHeightmap(heights=empty_heights, origin_xy=(0.33, -0.42),
                         cell_size=0.005, floor_z=0.75)
small_footprint = ObjectFootprint(center_xy=(0.0, 0.0), size_xy=(0.03, 0.03),
                                  yaw=0.0, z_bottom=0.75, z_top=0.80)

# "none" -> no wrapper needed.
assert resolve('none', None, BIN_CENTER, BIN_INNER_HALF, T_cam_x_right) is None

# "center" -> a valid sub-region planner whose plan() stays within
# bin_inner_half/2 of bin_center on both axes.
planner = resolve('center', None, BIN_CENTER, BIN_INNER_HALF, T_cam_x_right)
assert planner is not None
pose = planner.plan(small_footprint, empty_map)
assert pose is not None
assert abs(pose.x - BIN_CENTER[0]) <= BIN_INNER_HALF / 2 + 1e-6
assert abs(pose.y - BIN_CENTER[1]) <= BIN_INNER_HALF / 2 + 1e-6

# "right_of" with camera-right = world +X -> placement biased to x > bin_center[0].
planner = resolve('right_of', 'bin', BIN_CENTER, BIN_INNER_HALF, T_cam_x_right)
pose = planner.plan(small_footprint, empty_map)
assert pose is not None and pose.x > BIN_CENTER[0], f'x={pose.x}'

# "left_of" with the same camera -> placement biased to x < bin_center[0].
planner = resolve('left_of', 'bin', BIN_CENTER, BIN_INNER_HALF, T_cam_x_right)
pose = planner.plan(small_footprint, empty_map)
assert pose is not None and pose.x < BIN_CENTER[0], f'x={pose.x}'

print('All spatial_relation_resolver region checks passed.')

# ---- "near" with a fake PromptSelector ----------------------------------
class _FakeSelectionResult:
    def __init__(self, masks, scores):
        self.masks, self.scores = masks, scores
    @property
    def is_empty(self):
        return len(self.scores) == 0

class _FakePromptSelector:
    """Returns one full-frame mask; depth/K/T_world_cam below are rigged
    so that mask's world centroid lands at a known point inside the bin."""
    def __init__(self, mask):
        self._mask = mask
    def select(self, rgb, prompt=None, work_dir='.'):
        return _FakeSelectionResult(masks=np.array([self._mask]),
                                    scores=np.array([0.95], dtype=np.float32))

H, W = 20, 20
fx = fy = 100.0
cx = cy = 10.0
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
T_world_cam_topdown = np.array([
    [1.0, 0.0, 0.0, BIN_CENTER[0]],
    [0.0, -1.0, 0.0, BIN_CENTER[1]],
    [0.0, 0.0, -1.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)
depth = np.full((H, W), 0.25, dtype=np.float32)   # world z = 1.0 - 0.25 = 0.75
mask = np.zeros((H, W), dtype=bool)
mask[9:11, 9:11] = True   # centered at the principal point -> world (0.45, -0.30)
rgb = np.zeros((H, W, 3), dtype=np.uint8)

fake_selector = _FakePromptSelector(mask)
planner = resolve('near', 'blue cube', BIN_CENTER, BIN_INNER_HALF,
                  T_world_cam_topdown, rgb=rgb, depth=depth, K=K,
                  prompt_selector=fake_selector)
assert planner is not None
pose = planner.plan(small_footprint, empty_map)
assert pose is not None
assert abs(pose.x - BIN_CENTER[0]) < 0.02 and abs(pose.y - BIN_CENTER[1]) < 0.02, \
    f'expected a placement near the detected centroid, got ({pose.x}, {pose.y})'

# "near" with no match (empty selection) -> falls back to None.
class _EmptyFakeSelector:
    def select(self, rgb, prompt=None, work_dir='.'):
        return _FakeSelectionResult(masks=np.zeros((0, H, W), dtype=bool),
                                    scores=np.zeros((0,), dtype=np.float32))

planner = resolve('near', 'nonexistent object', BIN_CENTER, BIN_INNER_HALF,
                  T_world_cam_topdown, rgb=rgb, depth=depth, K=K,
                  prompt_selector=_EmptyFakeSelector())
assert planner is None

print('All spatial_relation_resolver near-relation checks passed.')
