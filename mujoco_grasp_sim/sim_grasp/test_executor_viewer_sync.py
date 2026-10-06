"""Execution viewer receives physics updates during both motion and holds."""
import numpy as np

from sim_grasp.executor import GraspExecutor
from sim_grasp.scene_generator import SceneConfig, SceneGenerator


class FakeViewer:
    def __init__(self):
        self.times = []

    def sync(self):
        self.times.append(float(data.time))


gen = SceneGenerator(SceneConfig(seed=0))
model, data = gen.generate()
viewer = FakeViewer()
executor = GraspExecutor(model, data, viewer=viewer)
dt = model.opt.timestep
executor._hold(3 * dt)
assert len(viewer.times) == 3
assert np.all(np.diff(viewer.times) > 0)
executor._step_to(data.ctrl[:7].copy(), 4 * dt)
assert len(viewer.times) == 7
assert np.all(np.diff(viewer.times) > 0)

class ClosedViewer:
    def is_running(self):
        return False

    def sync(self):
        raise AssertionError('closed viewer must not be synced')


executor.viewer = ClosedViewer()
executor._hold(dt)

print('Executor viewer sync checks passed.')
