"""Debug one grasp execution with side-view rendering + state prints."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_grasp import SceneConfig, SceneGenerator, CameraModule
from sim_grasp.executor import GraspExecutor, DiffIK

cfg = SceneConfig(seed=11)
cfg.n_objects_range = (4, 4)
cfg.calibration_file = 'calibration_result.yaml'
gen = SceneGenerator(cfg)
model, data = gen.generate()

metrics = json.loads(Path('output/test_execute/metrics.json').read_text())
T_world_grasp = np.array(metrics['best_grasp']['T_world_grasp'])
body = 'obj_2'
bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)

print('--- target object ---')
print('obj pos:', np.round(data.xpos[bid], 4))
gid = [i for i in range(model.ngeom) if model.geom_bodyid[i] == bid][0]
print('geom type:', model.geom(gid).type, 'size:', np.round(model.geom(gid).size, 4))

print('--- grasp ---')
print('T_world_grasp:\n', np.round(T_world_grasp, 4))
approach = T_world_grasp[:3, 2]
print('approach (world):', np.round(approach, 3))
print('grasp origin z:', round(T_world_grasp[2, 3], 4),
      ' TCP z:', round(T_world_grasp[2, 3] + 0.1034 * approach[2], 4))

# side-view renderer
renderer = mujoco.Renderer(model, height=480, width=640)
camv = mujoco.MjvCamera()
camv.lookat[:] = [T_world_grasp[0, 3], T_world_grasp[1, 3], 0.80]
camv.distance, camv.azimuth, camv.elevation = 0.7, 180, -10
frames = []

class SpyExecutor(GraspExecutor):
    def _maybe_record(self):
        if self.data.time - self._last_frame_t >= 0.10:
            renderer.update_scene(self.data, camera=camv)
            frames.append(renderer.render().copy())
            self._last_frame_t = self.data.time

ex = SpyExecutor(model, data, camera_module=None, record_gif=True)
ex.record = True
res = ex.execute(T_world_grasp, target_body=body)
print('--- result ---'); print(res)

hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
print('final hand pos:', np.round(data.xpos[hand_bid], 4))
print('final obj pos :', np.round(data.xpos[bid], 4))
f1 = model.joint('finger_joint1').qposadr[0]
print('finger qpos:', round(data.qpos[f1], 5), round(data.qpos[f1 + 1], 5))

import imageio.v2 as iio
iio.mimsave('output/test_execute/debug_side.gif', frames, fps=10, loop=0)
n = len(frames)
strip = np.hstack([frames[i] for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]])
iio.imwrite('output/test_execute/debug_strip.png', strip)
print(f'saved {n} side-view frames')
