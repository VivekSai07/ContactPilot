"""Standalone check for GraspExecutor._step_to's joint-interpolation math
when a gripper_ctrl is also being set -- run directly, no pytest (this
codebase has no automated test suite).

Regression test for a real, severe bug: _step_to's outer loop uses `n` as
the total physics-step count (n = max(1, duration/timestep)), consumed
every iteration via `t = (i + 1) / n`. The end-effector-wiring change
reused the SAME name `n` for the actuator count inside the loop
(`n = self.end_effector.n_actuators`), silently shadowing the step count
from the second iteration onward -- with a 1-actuator parallel gripper,
`n` collapses to 1, so `t` jumps to 2.0, 3.0, ... instead of a fraction in
[0, 1], and `_ease(t, smooth=False)` returns `t` unchanged (linear, no
clamping) -- meaning `data.ctrl[:7]` gets a wild extrapolation FAR beyond
q_target on every step after the first, instead of smoothly approaching
and holding at q_target. This broke every pick/place motion silently
(no crash, no exception -- the arm just ends up somewhere nonsensical),
which is exactly why it wasn't caught by unit tests that only check
gripper ctrl values or IK convergence in isolation.
"""
import numpy as np

from sim_grasp import SceneConfig, SceneGenerator
from sim_grasp.executor import GraspExecutor

cfg = SceneConfig(seed=0)
gen = SceneGenerator(cfg)
model, data = gen.generate()
executor = GraspExecutor(model, data)

q_start = data.qpos[executor.ik.qpos_idx].copy()
# A real, non-trivial joint target -- if the bug were present, q_target
# would be overshot by roughly 2x-3x on the very next steps.
q_target = q_start + 0.3

# Run with gripper_ctrl set on every step (exactly how execute()/place()
# call _step_to for every motion in the real pick sequence).
executor._step_to(q_target, duration=0.5, gripper_ctrl=0.0)

final_arm_ctrl = data.ctrl[:7].copy()
assert np.allclose(final_arm_ctrl, q_target, atol=1e-6), (
    f'_step_to must converge data.ctrl[:7] to q_target when gripper_ctrl is '
    f'set -- got {final_arm_ctrl}, expected {q_target} (a large discrepancy '
    f'here is exactly the interpolation-fraction bug this test guards against)'
)

# No intermediate overshoot either -- interpolation is monotonic between
# q_start and q_target, so nothing should ever exceed q_target's own
# per-joint magnitude relative to q_start by more than the target delta.
max_delta = np.max(np.abs(q_target - q_start))
assert np.all(np.abs(final_arm_ctrl - q_start) <= max_delta + 1e-6), \
    'final ctrl must not overshoot past q_target'

# Sanity: the gripper actuator itself still ends at the commanded value.
assert np.isclose(data.ctrl[7], 0.0), f'expected gripper ctrl 0.0, got {data.ctrl[7]}'

print('All step_to gripper-interpolation checks passed.')
