"""Standalone regression test for GraspFeasibilityChecker's extra_approach
handling (short-object finger-table collision fix — see
docs/research/2026-08-21-short-object-finger-table-collision.md).
Run directly, no pytest (this codebase has no automated test suite)."""
import numpy as np

from sim_grasp.feasibility import GraspFeasibilityChecker, _gripper_sample_points

TABLE_HEIGHT = 0.75
MARGIN = 0.005
TABLE_Z = TABLE_HEIGHT + MARGIN
EXTRA_APPROACH = 0.012

# Top-down grasp: grasp +Z (approach) points straight down in world (-Z).
# Rx(pi): X stays +X, Y -> -Y, Z -> -Z.
_TOPDOWN_R = np.array([[1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0]])


def _grasp_at_margin(margin_above_table: float) -> np.ndarray:
    """Top-down grasp whose lowest (finger-tip) corner sits
    margin_above_table meters above TABLE_Z at the originally predicted
    pose (before any extra_approach advance)."""
    # world_z of a grasp-frame point p = origin_z - p[2] (topdown flips Z),
    # so the corner with the LARGEST local z (finger tips, 0.112) maps to
    # the LOWEST world z -- that's the table-facing corner.
    farthest_corner_z_grasp_frame = _gripper_sample_points()[:, 2].max()  # 0.112
    T = np.eye(4)
    T[:3, :3] = _TOPDOWN_R
    T[2, 3] = TABLE_Z + margin_above_table + farthest_corner_z_grasp_frame
    T[0, 3], T[1, 3] = 0.4, 0.0
    return T


# Case 1: 3mm of margin at the predicted pose -- clears today's check, but a
# further 12mm descent (EXTRA_APPROACH) pushes the lowest corner well below
# the table plane. This is the short-object bug.
grasp_thin_margin = _grasp_at_margin(0.003)

# Case 2: 5cm of margin at the predicted pose -- must stay feasible even
# after the extra_approach advance (the fix must not over-reject).
grasp_ample_margin = _grasp_at_margin(0.05)

# Case 3: upward-approach grasp (+Z points up in world) -- must be rejected
# regardless of extra_approach.
grasp_upward = np.eye(4)
grasp_upward[:3, 3] = [0.4, 0.0, TABLE_Z + 0.5]

# 1) Backward compatibility: extra_approach=0.0 reproduces today's behavior.
checker_legacy = GraspFeasibilityChecker(table_height=TABLE_HEIGHT, extra_approach=0.0)
assert checker_legacy.is_feasible(grasp_thin_margin), (
    'extra_approach=0.0 must reproduce pre-fix behavior: a grasp with only '
    '3mm of margin at the predicted pose is feasible when the deepening is '
    'not modeled')

# 2) Regression test for the actual bug: with extra_approach configured to
# match the executor's real EXTRA_APPROACH, the same thin-margin grasp must
# now be rejected.
checker_fixed = GraspFeasibilityChecker(table_height=TABLE_HEIGHT,
                                        extra_approach=EXTRA_APPROACH)
assert not checker_fixed.is_feasible(grasp_thin_margin), (
    'a grasp that only clears the table by 3mm at the predicted pose must '
    'be rejected once the 12mm extra_approach descent is accounted for')

# 3) The fix must not over-reject grasps with ample table clearance.
assert checker_fixed.is_feasible(grasp_ample_margin), (
    'a grasp with 5cm of table clearance must remain feasible even after '
    'the extra_approach advance'
)

# 4) Upward-approach rejection is unaffected by extra_approach.
assert not checker_legacy.is_feasible(grasp_upward), (
    'an upward-approaching grasp must be rejected (legacy checker)')
assert not checker_fixed.is_feasible(grasp_upward), (
    'an upward-approaching grasp must be rejected (extra_approach-aware checker)')

print('All feasibility checks passed.')
