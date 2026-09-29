"""Standalone checks for props.py (P10 SP1) -- pure logic, no MuJoCo, no
downloaded assets needed. Run directly, no pytest."""
import math
import tempfile
from pathlib import Path

import numpy as np

from sim_grasp.props import (
    CATEGORIES, H_MAX, W_MAX, PropEntry, box_inertia, footprint_radius,
    load_manifest, obj_bounds, parse_manifest, prop_body_xml, prop_files,
    prop_scale, sample_balanced, sort_outcome, validate_props_count,
)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    raise AssertionError(f'{fn.__name__}{a} did not raise {exc.__name__}')


# --- manifest ---------------------------------------------------------------
entries = load_manifest()
assert len(entries) == 12
assert sum(e.category == 'food' for e in entries) == 6
assert sum(e.category == 'non_food' for e in entries) == 6
_raises(ValueError, parse_manifest, {'not': 'a list'})
_raises(ValueError, parse_manifest, [{'model_id': 'a', 'category': 'toy', 'mass_kg': 0.1}])
_raises(ValueError, parse_manifest, [{'model_id': 'a', 'category': 'food', 'mass_kg': 0.9}])
_raises(ValueError, parse_manifest, [{'model_id': 'a', 'category': 'food', 'mass_kg': 0.1},
                                     {'model_id': 'a', 'category': 'food', 'mass_kg': 0.1}])
_raises(ValueError, parse_manifest, [{'model_id': 'a', 'category': 'food'}])

# --- scale: every shortlisted prop ends graspable (native extents, metres) ---
native = [(0.139, 0.063, 0.164), (0.122, 0.049, 0.169), (0.112, 0.050, 0.164),
          (0.089, 0.042, 0.153), (0.091, 0.064, 0.161), (0.151, 0.032, 0.084),
          (0.076, 0.030, 0.160), (0.070, 0.040, 0.134), (0.099, 0.055, 0.145),
          (0.079, 0.048, 0.156), (0.074, 0.030, 0.117), (0.147, 0.043, 0.127)]
for w, d, h in native:
    s = prop_scale((w, d, h))
    assert 0 < s <= 1.0
    assert h * s <= H_MAX + 1e-9, (w, d, h, s)
    assert min(w, d) * s <= W_MAX + 1e-9, (w, d, h, s)
assert math.isclose(prop_scale((0.139, 0.063, 0.164)), 0.14 / 0.164)
assert prop_scale((0.05, 0.03, 0.10)) == 1.0          # already small: never upscale

# --- inertia (solid box closed form) + footprint radius ---------------------
ixx, iyy, izz = box_inertia(0.12, (0.06, 0.04, 0.12))
assert math.isclose(ixx, 0.12 / 12 * (0.04**2 + 0.12**2))
assert math.isclose(iyy, 0.12 / 12 * (0.06**2 + 0.12**2))
assert math.isclose(izz, 0.12 / 12 * (0.06**2 + 0.04**2))
assert math.isclose(footprint_radius((0.151, 0.032, 0.084), 1.0),
                    0.5 * math.hypot(0.151, 0.032))

# --- balanced sampling ------------------------------------------------------
for n, counts in ((4, {2}), (5, {2, 3}), (6, {3}), (2, {1})):
    got = sample_balanced(entries, n, np.random.default_rng(7))
    assert len(got) == n and len({e.model_id for e in got}) == n
    assert sum(e.category == 'food' for e in got) in counts
a = [e.model_id for e in sample_balanced(entries, 4, np.random.default_rng(1))]
b = [e.model_id for e in sample_balanced(entries, 4, np.random.default_rng(1))]
assert a == b                                          # deterministic per seed

# --- count validation -------------------------------------------------------
for ok in (2, 3, 4, 5, 6):
    validate_props_count(ok)
for bad in (0, 1, 7):
    _raises(ValueError, validate_props_count, bad)

# --- prop_files: partial download fails loudly, naming the fix --------------
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp) / 'X'
    d.mkdir()
    (d / 'model.obj').write_text('v 0 0 0\nv 0.1 0.05 0.2\n')
    e = _raises(FileNotFoundError, prop_files, 'X', gso_dir=Path(tmp))
    assert 'download_props.py' in str(e) and 'texture.png' in str(e)
    (d / 'texture.png').write_bytes(b'png')
    e = _raises(FileNotFoundError, prop_files, 'X', gso_dir=Path(tmp))
    assert 'model_collision_' in str(e)
    for i in (0, 1, 10, 2):
        (d / f'model_collision_{i}.obj').write_text('v 0 0 0\n')
    files = prop_files('X', gso_dir=Path(tmp))
    assert [Path(p).name for p in files['collision']] == [
        'model_collision_0.obj', 'model_collision_1.obj',
        'model_collision_2.obj', 'model_collision_10.obj']   # numeric order
    assert Path(files['visual']).is_absolute()
    lo, hi = obj_bounds(files['visual'])
    assert np.allclose(lo, [0, 0, 0]) and np.allclose(hi, [0.1, 0.05, 0.2])

# --- MJCF snippet -----------------------------------------------------------
files = {'visual': '/abs/v.obj', 'texture': '/abs/t.png',
         'collision': ['/abs/c0.obj', '/abs/c1.obj']}
entry = PropEntry('Some_Box', 'food', 0.2)
body, assets = prop_body_xml('obj_2', entry, files, 0.5,
                             np.array([-0.05, -0.02, 0.0]), np.array([0.05, 0.02, 0.16]))
assert '<body name="obj_2"' in body and '<freejoint name="obj_2_joint"/>' in body
assert 'mesh="obj_2_vis"' in body and 'material="obj_2_mat"' in body
assert 'contype="0" conaffinity="0" group="2"' in body
assert body.count('group="3"') == 2 and 'mesh="obj_2_col_1"' in body
assert '<inertial pos="0.00000 0.00000 0.04000" mass="0.2000"' in body
assert 'file="/abs/t.png"' in assets and 'name="obj_2_tex"' in assets
assert 'scale="0.50000 0.50000 0.50000"' in assets
assert assets.count('<mesh ') == 3

# --- sort outcome: wrong bin is never "sorted" ------------------------------
correct, wrong = sort_outcome({'obj_0': 'A', 'obj_1': 'A', 'obj_3': 'B'},
                              {'obj_0': 'food', 'obj_1': 'non_food',
                               'obj_2': 'food', 'obj_3': 'non_food'},
                              {'A': 'food', 'B': 'non_food'})
assert correct == ['obj_0', 'obj_3'] and wrong == ['obj_1']
assert CATEGORIES == ('food', 'non_food')

print('All props.py checks passed.')
