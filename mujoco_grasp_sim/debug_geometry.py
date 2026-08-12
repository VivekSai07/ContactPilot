"""Verify grasp-frame conventions: at the close moment, where are the fingertips
relative to the target object? Detects any systematic offset between CGN's
grasp frame and the Menagerie hand frame."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_grasp import SceneConfig, SceneGenerator
from sim_grasp.executor import GraspExecutor, ARM_JOINTS

cfg = SceneConfig(seed=11)
cfg.n_objects_range = (4, 4)
cfg.calibration_file = 'calibration_result.yaml'
gen = SceneGenerator(cfg)
model, data = gen.generate()

metrics = json.loads(Path('output/test_execute4/metrics.json').read_text())
T_g = np.array(metrics['best_grasp']['T_world_grasp'])
body = 'obj_2'
bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
gid = [i for i in range(model.ngeom) if model.geom_bodyid[i] == bid][0]

print('object pos   :', np.round(data.xpos[bid], 4))
print('geom type/size:', model.geom(gid).type[0], np.round(model.geom(gid).size, 4))
print('grasp origin :', np.round(T_g[:3, 3], 4))
print('approach     :', np.round(T_g[:3, 2], 3))
print('closing axis :', np.round(T_g[:3, 0], 3))
tcp = T_g[:3, 3] + 0.1034 * T_g[:3, 2]
print('grasp TCP    :', np.round(tcp, 4))
print('TCP -> object:', np.round(data.xpos[bid] - tcp, 4),
      ' |lateral dist|:', round(np.linalg.norm((data.xpos[bid] - tcp)[:2]), 4))

# run to the close stage, then inspect actual fingertip positions
ex = GraspExecutor(model, data, record_gif=False)
ik_pre, T_hand_pre = ex._hand_targets(
    np.block([[T_g[:3, :3], (T_g[:3, 3] - 0.10 * T_g[:3, 2]).reshape(3, 1)],
              [np.zeros((1, 3)), np.ones((1, 1))]]),
    data.qpos[ex.ik.qpos_idx].copy())
T_hand_grasp = T_hand_pre.copy()
T_hand_grasp[:3, 3] = T_g[:3, 3] + 0.012 * T_g[:3, 2]
ik_grasp = ex.ik.solve(T_hand_grasp, ik_pre.qpos)
print('ik pre/grasp err mm:', round(ik_pre.pos_err * 1e3, 2), round(ik_grasp.pos_err * 1e3, 2))

ex._step_to(ik_pre.qpos, 2.0, gripper_ctrl=255)
ex._step_to(ik_grasp.qpos, 1.2, gripper_ctrl=255)
ex._hold(0.3)

# fingertip world positions just before closing
lf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_finger')
rf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_finger')
hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
print('\n--- at close moment ---')
print('hand body pos:', np.round(data.xpos[hand], 4), '(target was', np.round(T_hand_grasp[:3, 3], 4), ')')
print('left finger  :', np.round(data.xpos[lf], 4))
print('right finger :', np.round(data.xpos[rf], 4))
print('object now   :', np.round(data.xpos[bid], 4))
mid = 0.5 * (data.xpos[lf] + data.xpos[rf])
print('finger midpoint:', np.round(mid, 4))
print('midpoint -> object:', np.round(data.xpos[bid] - mid, 4))

# render side view at this exact moment
renderer = mujoco.Renderer(model, height=480, width=640)
camv = mujoco.MjvCamera()
camv.lookat[:] = data.xpos[bid]
camv.distance, camv.azimuth, camv.elevation = 0.45, 135, -15
renderer.update_scene(data, camera=camv)
import imageio.v2 as iio
iio.imwrite('output/close_moment.png', renderer.render())
camv.azimuth = 45
renderer.update_scene(data, camera=camv)
iio.imwrite('output/close_moment2.png', renderer.render())
print('saved output/close_moment*.png')
