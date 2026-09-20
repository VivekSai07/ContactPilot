"""Standalone check for EndEffectorController implementations -- run
directly, no pytest (this codebase has no automated test suite)."""
import numpy as np

from sim_grasp.end_effector import ParallelGripperController, ShadowHandController

# -- ParallelGripperController: must reproduce today's exact scalar behavior.
parallel = ParallelGripperController()
assert np.allclose(parallel.ctrl_for(255.0), [255.0])
assert np.allclose(parallel.ctrl_for(0.0), [0.0])
assert np.allclose(parallel.ctrl_for(127.5), [127.5])

# -- ShadowHandController: interpolates the full 18-actuator vector.
shadow = ShadowHandController()
open_ctrl = shadow.ctrl_for(255.0)
closed_ctrl = shadow.ctrl_for(0.0)
assert open_ctrl.shape == (18,)
assert closed_ctrl.shape == (18,)
assert not np.allclose(open_ctrl, closed_ctrl), \
    'open and closed postures must actually differ'

mid_ctrl = shadow.ctrl_for(127.5)
# Midpoint must be the midpoint of open/closed, not a third arbitrary posture.
assert np.allclose(mid_ctrl, (open_ctrl + closed_ctrl) / 2, atol=1e-6)

# -- is_grasping: both controllers work off the same lifted-height signal
# already computed by GraspExecutor.execute() -- confirm the signature
# accepts it without needing hand-specific contact sensing for v1.
assert parallel.is_grasping(object_raised_m=0.1) is True
assert parallel.is_grasping(object_raised_m=0.0) is False
assert shadow.is_grasping(object_raised_m=0.1) is True
assert shadow.is_grasping(object_raised_m=0.0) is False

print('All end_effector checks passed.')
