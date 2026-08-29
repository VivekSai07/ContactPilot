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
