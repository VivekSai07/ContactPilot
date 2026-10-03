"""Checks the bin list refactor (P10 SP1): default = one bin (unchanged),
second_bin_center set = bins A + B, per-bin occupancy queries. Needs
MuJoCo, no GPU. Run directly, no pytest."""
import mujoco
import numpy as np

from sim_grasp.scene_generator import BinSpec, SceneConfig, SceneGenerator

# default: exactly one bin, legacy geometry
cfg = SceneConfig(seed=0)
assert cfg.bins() == [BinSpec('A', (0.45, -0.30), 0.12, '')]

# two bins: A = food at the legacy spot, B = non_food mirrored
cfg2 = SceneConfig(seed=0, second_bin_center=(0.45, 0.30), spawn_y=(-0.09, 0.09))
assert cfg2.bins() == [BinSpec('A', (0.45, -0.30), 0.12, 'food'),
                       BinSpec('B', (0.45, 0.30), 0.12, 'non_food')]

gen = SceneGenerator(cfg2)
model, data = gen.generate()
xml = gen.scene_xml_path.read_text()
for g in ('bin_floor', 'bin_wall_xp', 'bin_b_floor', 'bin_b_wall_xp',
          'bin_b_wall_xm', 'bin_b_wall_yp', 'bin_b_wall_ym'):
    assert f'name="{g}"' in xml, g
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) >= 0, g
fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'bin_b_floor')
assert np.allclose(model.geom_pos[fid][:2], [0.45, 0.30])

# after settling nothing is in a bin
assert gen.objects_in_bins() == {} and gen.objects_in_bin() == []


def teleport(name, x, y):
    jadr = model.joint(f'{name}_joint').qposadr[0]
    data.qpos[jadr:jadr + 3] = [x, y, cfg2.table_height + 0.03]
    mujoco.mj_forward(model, data)


teleport('obj_0', 0.45, 0.30)                 # into bin B
teleport('obj_1', 0.45, -0.30)                # into bin A
assert gen.objects_in_bins() == {'obj_0': 'B', 'obj_1': 'A'}
assert gen.objects_in_bin() == ['obj_1']       # legacy query = bin A only
assert np.allclose(gen.bin_drop_point('B'), [0.45, 0.30, cfg2.table_height + 0.02])
assert np.allclose(gen.bin_drop_point(), [0.45, -0.30, cfg2.table_height + 0.02])
print('All bin-list checks passed.')
