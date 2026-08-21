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
