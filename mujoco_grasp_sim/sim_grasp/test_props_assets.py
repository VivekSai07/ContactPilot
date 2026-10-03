"""Checks the DOWNLOADED prop assets (needs scripts/download_props.py run
once, plus MuJoCo; no GPU): every manifest prop resolves, has the
expected native size, and compiles + renders textured in MuJoCo.
Run directly, no pytest (MUJOCO_GL=osmesa on headless/WSL2)."""
import mujoco
import numpy as np

from sim_grasp.props import (load_manifest, obj_bounds, prop_body_xml,
                             prop_files, prop_scale)

entries = load_manifest()
for e in entries:
    files = prop_files(e.model_id)
    assert len(files['collision']) == 32, (e.model_id, len(files['collision']))
    lo, hi = obj_bounds(files['visual'])
    ext = hi - lo
    assert 0.07 < ext[2] < 0.18, (e.model_id, ext)          # upright, 8-17 cm tall
    assert min(ext[0], ext[1]) * prop_scale(ext) <= 0.0551, e.model_id
    # (origin height is NOT asserted: spawn uses -lo_z, so an off-base origin is fine)

# Crayola 24 matches the measurement taken during design (7.4 x 3.0 x 11.7 cm)
lo, hi = obj_bounds(prop_files('Crayola_Crayons_24_count')['visual'])
assert np.allclose(hi - lo, [0.074, 0.030, 0.117], atol=0.002), hi - lo

# One prop compiles, gets the manifest mass, and renders with texture
e = next(x for x in entries if x.model_id == 'Crayola_Crayons_24_count')
files = prop_files(e.model_id)
lo, hi = obj_bounds(files['visual'])
body, assets = prop_body_xml('obj_0', e, files, prop_scale(hi - lo), lo, hi)
xml = f"""<mujoco><asset>{assets}</asset><worldbody>
<light pos="0 0 2" dir="0 0 -1" directional="true"/>
<geom type="plane" size="1 1 .1" rgba=".8 .7 .55 1"/>{body}
<camera name="c" pos="0 -0.4 0.25" xyaxes="1 0 0 0 0.5 0.87"/></worldbody></mujoco>"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'obj_0')
assert abs(model.body_mass[bid] - e.mass_kg) < 1e-6, model.body_mass[bid]
data.qpos[2] = 0.01
for _ in range(1000):
    mujoco.mj_step(model, data)
assert -0.005 < data.qpos[2] < 0.02, data.qpos[:3]          # rests on the plane
r = mujoco.Renderer(model, 240, 320)
r.update_scene(data, 'c')
img = r.render()
assert img.std() > 20, img.std()                            # not a flat image
print(f'All {len(entries)} prop assets present, sized, and loadable.')
