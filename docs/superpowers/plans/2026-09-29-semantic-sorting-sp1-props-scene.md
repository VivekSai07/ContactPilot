# Semantic Sorting SP1 — Props Scene + Second Bin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--scene props` mode — 4 textured Google Scanned Objects props (food / non-food) and a second bin — where pick-all places each object into the bin for its ground-truth category and reports a correct-bin rate, then pass the SP1 benchmark gate.

**Architecture:** A new pure module `sim_grasp/props.py` (manifest, scaling, inertia, MJCF snippet, balanced sampling, sort outcome — no MuJoCo import) feeds a props branch in `SceneGenerator`. The single hard-coded bin becomes a `BinSpec` list (bin A unchanged, bin B mirrored). `run_sim_grasp_test.py` routes each object to `target_bin_for(body)` (ground truth in SP1, replaced by perception in SP3) and builds heightmap/planner per bin. With no new flags, every code path and the generated default scene are unchanged (hash-guarded).

**Tech Stack:** Python 3.10 (`cgn_torch` env), MuJoCo, numpy < 2, stdlib `urllib` for downloads; standalone assert-script tests (no pytest).

**Spec:** `docs/superpowers/specs/2026-09-29-semantic-sorting-scene-graph-design.md` (sections "Sub-project 1 design" and "Sub-project 1 gate").

## Global Constraints

- numpy must stay < 2; no new pip dependencies (OBJ bounds are parsed by hand, not trimesh — `trimesh` is not guaranteed in every env).
- Tests are standalone scripts: `cd mujoco_grasp_sim && PYTHONPATH=. python sim_grasp/test_<name>.py` — top-level asserts, print one "passed" line, no pytest. Scripts that render need `MUJOCO_GL=osmesa` on WSL2/headless.
- Default behaviour byte-for-byte unchanged: `SceneConfig(seed=s)` must produce the identical `_generated_scene.xml` and settled `qpos` as before this plan (guarded by Task 1's hash test, re-run in every later task).
- Categories are exactly `('food', 'non_food')`; bin A = food at `(0.45, -0.30)`, bin B = non_food at `(0.45, 0.30)`, both `inner_half = 0.12`.
- Prop scaling constants: `H_MAX = 0.14`, `W_MAX = 0.055`; mass clamp `[0.05, 0.40]` kg; props count range 2–6, default 4.
- Prop assets are never committed: they live in `mujoco_grasp_sim/assets/props/gso/` (gitignored), fetched from `kevinzakka/mujoco_scanned_objects` pinned at commit `6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c`.
- Commit messages: plain text, **no `Co-Authored-By` or any AI-attribution line** (standing repo rule). All work on branch `semantic-sorting-scene-graph`.
- Every fix site gets a one-line *why* comment (AGENTS.md).

## Review Focus

- Tall thin props (Epson box ≈ 3 × 7.6 × 16 cm native) tipping or bouncing at spawn → every prop must end the settle on the table and in no bin. Pinned by Task 4's 10-seed settle test.
- Same `--seed` twice → identical props, categories and positions (benchmarks depend on it). Pinned by Task 4's determinism test.
- Partially downloaded assets (e.g. `texture.png` missing) → a `FileNotFoundError` naming `scripts/download_props.py`, never a raw MuJoCo XML parse error. Pinned by Task 1's `prop_files` test.
- An object placed in the *wrong* bin must not count as sorted, even though the legacy `in_bin` (any bin) is true. Pinned by Task 1's `sort_outcome` test and Task 6's `wrong_bin` classifier test.
- `--scene props` with `--n-objects 7`, `--instruction`, `--prompt`, or an unsupported camera → a one-line `sys.exit` message, no traceback. Pinned by Task 1's `validate_props_count` test and Task 5's CLI exit checks.

---

### Task 1: Default-scene guard + `sim_grasp/props.py` pure logic + manifest

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/test_scene_default_unchanged.py`
- Create: `mujoco_grasp_sim/sim_grasp/props.py`
- Create: `mujoco_grasp_sim/assets/props/manifest.json`
- Create: `mujoco_grasp_sim/sim_grasp/test_props.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces (`sim_grasp/props.py`):
  - constants `CATEGORIES = ('food', 'non_food')`, `H_MAX = 0.14`, `W_MAX = 0.055`, `MASS_RANGE = (0.05, 0.40)`, `PROPS_N_RANGE = (2, 6)`, `PROPS_DIR`, `MANIFEST_PATH`, `GSO_DIR`, `DOWNLOAD_HINT`
  - `@dataclass(frozen=True) PropEntry(model_id: str, category: str, mass_kg: float)`
  - `parse_manifest(obj) -> list[PropEntry]`, `load_manifest(path=MANIFEST_PATH) -> list[PropEntry]`
  - `obj_bounds(path) -> tuple[np.ndarray, np.ndarray]` (lo, hi, metres)
  - `prop_scale(extents) -> float`, `box_inertia(mass, extents) -> tuple[float, float, float]`, `footprint_radius(extents, scale) -> float`
  - `sample_balanced(entries, n, rng) -> list[PropEntry]`, `validate_props_count(n) -> None`
  - `prop_files(model_id, gso_dir=GSO_DIR) -> dict` with keys `'visual'`, `'texture'` (str) and `'collision'` (list[str])
  - `prop_body_xml(name, entry, files, scale, lo, hi) -> tuple[str, str]` (body_xml, assets_xml)
  - `sort_outcome(object_bins, object_categories, bin_categories) -> tuple[list[str], list[str]]` (correct, wrong)

- [ ] **Step 1: Capture the pre-change default-scene hashes**

Run (from `mujoco_grasp_sim/`, before touching any code):

```bash
cd mujoco_grasp_sim
PYTHONPATH=. python - <<'EOF'
import hashlib, numpy as np
from sim_grasp.scene_generator import SceneConfig, SceneGenerator
for seed in (0, 3):
    gen = SceneGenerator(SceneConfig(seed=seed)); model, data = gen.generate()
    print(seed, hashlib.sha256(gen.scene_xml_path.read_bytes()).hexdigest(),
          hashlib.sha256(np.round(data.qpos, 6).tobytes()).hexdigest())
EOF
```

Expected: two lines `seed xml_sha256 qpos_sha256`. Copy the four hashes into Step 2.

- [ ] **Step 2: Write the guard test with those hashes**

`mujoco_grasp_sim/sim_grasp/test_scene_default_unchanged.py`:

```python
"""Guard: P10 (props scene + second bin) must not change the default boxes
scene. Hashes were captured on main before any P10 code change -- the
generated XML and the settled qpos (which also pins the rng consumption
order) must stay identical. Run directly, no pytest."""
import hashlib

import numpy as np

from sim_grasp.scene_generator import SceneConfig, SceneGenerator

EXPECTED = {
    0: ('<xml sha256 for seed 0 from Step 1>', '<qpos sha256 for seed 0 from Step 1>'),
    3: ('<xml sha256 for seed 3 from Step 1>', '<qpos sha256 for seed 3 from Step 1>'),
}

for seed, (xml_h, qpos_h) in EXPECTED.items():
    gen = SceneGenerator(SceneConfig(seed=seed))
    model, data = gen.generate()
    got_xml = hashlib.sha256(gen.scene_xml_path.read_bytes()).hexdigest()
    got_q = hashlib.sha256(np.round(data.qpos, 6).tobytes()).hexdigest()
    assert got_xml == xml_h, f'seed {seed}: default scene XML changed'
    assert got_q == qpos_h, f'seed {seed}: default settled qpos changed (rng order?)'

print('Default boxes scene unchanged (seeds 0, 3).')
```

Replace the four `<...>` strings with the literal hashes printed in Step 1 (they are machine-specific captures, not placeholders to keep).

- [ ] **Step 3: Run the guard — must pass before any code change**

Run: `cd mujoco_grasp_sim && PYTHONPATH=. python sim_grasp/test_scene_default_unchanged.py`
Expected: `Default boxes scene unchanged (seeds 0, 3).`

- [ ] **Step 4: Write the manifest**

`mujoco_grasp_sim/assets/props/manifest.json`:

```json
[
  {"model_id": "Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box", "category": "food", "mass_kg": 0.22},
  {"model_id": "ReadytoUse_Rolled_Fondant_Pure_White_24_oz_box", "category": "food", "mass_kg": 0.40},
  {"model_id": "Nestl_Skinny_Cow_Heavenly_Crisp_Candy_Bar_Chocolate_Raspberry_6_pack_462_oz_total", "category": "food", "mass_kg": 0.13},
  {"model_id": "Nestle_Nips_Hard_Candy_Peanut_Butter", "category": "food", "mass_kg": 0.12},
  {"model_id": "Nescafe_Momento_Mocha_Specialty_Coffee_Mix_8_ct", "category": "food", "mass_kg": 0.18},
  {"model_id": "Polar_Herring_Fillets_Smoked_Peppered_705_oz_total", "category": "food", "mass_kg": 0.20},
  {"model_id": "Epson_273XL_Ink_Cartridge_Magenta", "category": "non_food", "mass_kg": 0.05},
  {"model_id": "Dell_Series_9_Color_Ink_Cartridge_MK993_High_Yield", "category": "non_food", "mass_kg": 0.08},
  {"model_id": "Brother_LC_1053PKS_Ink_Cartridge_CyanMagentaYellow_1pack", "category": "non_food", "mass_kg": 0.06},
  {"model_id": "Just_For_Men_ShampooIn_Haircolor_Jet_Black_60", "category": "non_food", "mass_kg": 0.12},
  {"model_id": "Crayola_Crayons_24_count", "category": "non_food", "mass_kg": 0.10},
  {"model_id": "Crayola_Bonus_64_Crayons", "category": "non_food", "mass_kg": 0.30}
]
```

- [ ] **Step 5: Write the failing test**

`mujoco_grasp_sim/sim_grasp/test_props.py`:

```python
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
```

- [ ] **Step 6: Run it to verify it fails**

Run: `cd mujoco_grasp_sim && PYTHONPATH=. python sim_grasp/test_props.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim_grasp.props'`

- [ ] **Step 7: Implement `sim_grasp/props.py`**

```python
"""Semantic prop library (ROADMAP P10, sub-project 1): textured Google
Scanned Objects props, listed in a committed manifest, meshes downloaded
locally by scripts/download_props.py into assets/props/gso/ (gitignored).

Pure logic only -- no MuJoCo import -- so everything here is testable
standalone (same philosophy as placement_planner.py). SceneGenerator's
props branch is the only consumer that touches MuJoCo.
"""
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CATEGORIES = ('food', 'non_food')
MASS_RANGE = (0.05, 0.40)
# Scaled-size caps: current boxes reach 0.11 m tall with a <= 0.056 m
# graspable side; the Panda opens 0.08 m, so keep one side well below it.
H_MAX = 0.14
W_MAX = 0.055
PROPS_N_RANGE = (2, 6)   # >6 props no longer fit the narrowed spawn strip

_THIS_DIR = Path(__file__).resolve().parent
PROPS_DIR = _THIS_DIR.parent / 'assets' / 'props'
MANIFEST_PATH = PROPS_DIR / 'manifest.json'
GSO_DIR = PROPS_DIR / 'gso'
DOWNLOAD_HINT = 'run `python scripts/download_props.py` from mujoco_grasp_sim/'


@dataclass(frozen=True)
class PropEntry:
    model_id: str
    category: str
    mass_kg: float


def parse_manifest(obj) -> list[PropEntry]:
    """Validates a decoded manifest (list of {model_id, category, mass_kg})."""
    if not isinstance(obj, list) or not obj:
        raise ValueError('prop manifest must be a non-empty JSON list')
    seen, out = set(), []
    for i, e in enumerate(obj):
        if not isinstance(e, dict) or set(e) != {'model_id', 'category', 'mass_kg'}:
            raise ValueError(f'manifest entry {i} must have exactly '
                             f'model_id/category/mass_kg: {e!r}')
        if e['category'] not in CATEGORIES:
            raise ValueError(f'manifest entry {i}: category {e["category"]!r} '
                             f'not one of {CATEGORIES}')
        m = float(e['mass_kg'])
        if not MASS_RANGE[0] <= m <= MASS_RANGE[1]:
            raise ValueError(f'manifest entry {i}: mass_kg {m} outside {MASS_RANGE}')
        if e['model_id'] in seen:
            raise ValueError(f'manifest entry {i}: duplicate model_id {e["model_id"]!r}')
        seen.add(e['model_id'])
        out.append(PropEntry(str(e['model_id']), e['category'], m))
    return out


def load_manifest(path=MANIFEST_PATH) -> list[PropEntry]:
    return parse_manifest(json.loads(Path(path).read_text(encoding='utf-8')))


def obj_bounds(path) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned (lo, hi) of an OBJ's vertices, parsed by hand so no
    trimesh dependency is needed (GSO meshes are metres, z up)."""
    verts = [line.split()[1:4] for line in Path(path).read_text().splitlines()
             if line.startswith('v ')]
    if not verts:
        raise ValueError(f'no vertices in {path}')
    v = np.asarray(verts, dtype=np.float64)
    return v.min(axis=0), v.max(axis=0)


def prop_scale(extents) -> float:
    """Uniform scale (never > 1) so the prop is <= H_MAX tall and its short
    horizontal side is <= W_MAX -- uniform keeps the texture undistorted."""
    w, d, h = (float(x) for x in extents)
    return min(1.0, H_MAX / h, W_MAX / min(w, d))


def box_inertia(mass: float, extents) -> tuple[float, float, float]:
    """Solid-box principal inertia. GSO MJCF sets no mass, and MuJoCo's
    default density over 32 overlapping hulls gives e.g. the Crunch box
    2.56 kg instead of ~0.22 kg -- so we always write an explicit <inertial>."""
    w, d, h = (float(x) for x in extents)
    k = mass / 12.0
    return k * (d * d + h * h), k * (w * w + h * h), k * (w * w + d * d)


def footprint_radius(extents, scale: float) -> float:
    """Half-diagonal of the scaled XY footprint: the prop can't reach
    further than this from its centre at any yaw."""
    return 0.5 * math.hypot(float(extents[0]), float(extents[1])) * scale


def sample_balanced(entries: list[PropEntry], n: int, rng) -> list[PropEntry]:
    """n props without replacement, ceil(n/2) from an rng-chosen category
    and floor(n/2) from the other, shuffled -- every scene has work for
    both bins."""
    first = CATEGORIES[int(rng.integers(2))]
    second = CATEGORIES[1] if first == CATEGORIES[0] else CATEGORIES[0]
    picked = []
    for cat, k in ((first, (n + 1) // 2), (second, n // 2)):
        pool = [e for e in entries if e.category == cat]
        if k > len(pool):
            raise ValueError(f'need {k} {cat} props, manifest has {len(pool)}')
        idx = rng.choice(len(pool), size=k, replace=False)
        picked += [pool[int(i)] for i in idx]
    return [picked[int(i)] for i in rng.permutation(len(picked))]


def validate_props_count(n: int) -> None:
    if not PROPS_N_RANGE[0] <= n <= PROPS_N_RANGE[1]:
        raise ValueError(f'props scene supports {PROPS_N_RANGE[0]}-'
                         f'{PROPS_N_RANGE[1]} objects, got {n}')


def prop_files(model_id: str, gso_dir=GSO_DIR) -> dict:
    """Absolute paths of one prop's visual mesh, texture and collision
    hulls. Fails with the fix spelled out rather than letting MuJoCo emit
    an opaque XML error on a partial download."""
    d = Path(gso_dir) / model_id
    for fname in ('model.obj', 'texture.png'):
        if not (d / fname).is_file():
            raise FileNotFoundError(f'prop asset missing: {d / fname} -- {DOWNLOAD_HINT}')
    cols = sorted(d.glob('model_collision_*.obj'),
                  key=lambda p: int(re.search(r'_(\d+)\.obj$', p.name).group(1)))
    if not cols:
        raise FileNotFoundError(f'no model_collision_*.obj in {d} -- {DOWNLOAD_HINT}')
    return {'visual': str((d / 'model.obj').resolve()),
            'texture': str((d / 'texture.png').resolve()),
            'collision': [str(p.resolve()) for p in cols]}


def prop_body_xml(name: str, entry: PropEntry, files: dict, scale: float,
                  lo, hi) -> tuple[str, str]:
    """(body_xml, assets_xml) for one prop instance. Asset names are
    prefixed with the object name so the same GSO model could appear twice;
    file paths are absolute because the scene's meshdir belongs to panda.xml."""
    s = f'{scale:.5f} {scale:.5f} {scale:.5f}'
    assets = [f'<texture type="2d" name="{name}_tex" file="{files["texture"]}"/>',
              f'<material name="{name}_mat" texture="{name}_tex" specular="0.5" shininess="0.5"/>',
              f'<mesh name="{name}_vis" file="{files["visual"]}" scale="{s}"/>']
    assets += [f'<mesh name="{name}_col_{i}" file="{p}" scale="{s}"/>'
               for i, p in enumerate(files['collision'])]
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    ext = (hi - lo) * scale
    c = (hi + lo) / 2 * scale
    ixx, iyy, izz = box_inertia(entry.mass_kg, ext)
    # visual mesh renders (group 2) but never collides; the 32 V-HACD hulls
    # collide but sit in group 3, which mujoco.Renderer hides by default
    geoms = [f'<geom type="mesh" mesh="{name}_vis" material="{name}_mat" '
             f'contype="0" conaffinity="0" group="2"/>']
    geoms += [f'<geom type="mesh" mesh="{name}_col_{i}" group="3"/>'
              for i in range(len(files['collision']))]
    body = (f'<body name="{name}" pos="0 0 0">'
            f'<freejoint name="{name}_joint"/>'
            f'<inertial pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" mass="{entry.mass_kg:.4f}" '
            f'diaginertia="{ixx:.6e} {iyy:.6e} {izz:.6e}"/>'
            + ''.join(geoms) + '</body>')
    return body, '\n    '.join(assets)


def sort_outcome(object_bins: dict, object_categories: dict,
                 bin_categories: dict) -> tuple[list[str], list[str]]:
    """(correct, wrong) object names among those currently in any bin.
    A wrong-bin object is binned but NOT sorted."""
    correct = sorted(o for o, b in object_bins.items()
                     if bin_categories[b] == object_categories[o])
    wrong = sorted(o for o in object_bins if o not in correct)
    return correct, wrong
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `cd mujoco_grasp_sim && PYTHONPATH=. python sim_grasp/test_props.py && PYTHONPATH=. python sim_grasp/test_scene_default_unchanged.py`
Expected: `All props.py checks passed.` then `Default boxes scene unchanged (seeds 0, 3).`

- [ ] **Step 9: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/props.py mujoco_grasp_sim/sim_grasp/test_props.py \
        mujoco_grasp_sim/sim_grasp/test_scene_default_unchanged.py \
        mujoco_grasp_sim/assets/props/manifest.json
git commit -m "Add prop manifest and pure prop logic for the semantic props scene"
```

---

### Task 2: Prop download script, attribution, and real-asset check

**Files:**
- Create: `mujoco_grasp_sim/scripts/download_props.py`
- Create: `mujoco_grasp_sim/assets/props/README.md`
- Modify: `.gitignore` (append one line)
- Create: `mujoco_grasp_sim/sim_grasp/test_props_assets.py`

**Interfaces:**
- Consumes: `load_manifest`, `prop_files`, `obj_bounds`, `prop_scale`, `prop_body_xml`, `PropEntry`, `GSO_DIR`, `MANIFEST_PATH` (Task 1).
- Produces: downloaded assets under `mujoco_grasp_sim/assets/props/gso/<model_id>/` (`model.obj`, `texture.png`, `model_collision_0..31.obj`) for every manifest entry.

- [ ] **Step 1: Write the failing asset test**

`mujoco_grasp_sim/sim_grasp/test_props_assets.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd mujoco_grasp_sim && MUJOCO_GL=osmesa PYTHONPATH=. python sim_grasp/test_props_assets.py`
Expected: FAIL with `FileNotFoundError: prop asset missing: .../assets/props/gso/Crunch_.../model.obj -- run `python scripts/download_props.py` ...`

- [ ] **Step 3: Write the download script**

`mujoco_grasp_sim/scripts/download_props.py`:

```python
"""Download the P10 semantic-scene props listed in
assets/props/manifest.json from the MuJoCo port of Google Scanned Objects
(kevinzakka/mujoco_scanned_objects -- meshes/textures CC-BY 4.0, MJCF MIT)
into assets/props/gso/<model_id>/. Idempotent: files already present with
the right size are skipped. Stdlib only, so it runs in any Python env.

Usage (from mujoco_grasp_sim/):
    python scripts/download_props.py
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # mujoco_grasp_sim/
PROPS = ROOT / 'assets' / 'props'
# pinned so every machine gets byte-identical props (repo last changed 2022-07)
REF = '6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c'
API = ('https://api.github.com/repos/kevinzakka/mujoco_scanned_objects/'
       'contents/models/{}?ref=' + REF)


def _wanted(name: str) -> bool:
    return name in ('model.obj', 'texture.png') or (
        name.startswith('model_collision_') and name.endswith('.obj'))


def main() -> int:
    manifest = json.loads((PROPS / 'manifest.json').read_text(encoding='utf-8'))
    for e in manifest:
        mid = e['model_id']
        dst = PROPS / 'gso' / mid
        dst.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(API.format(mid), timeout=60) as r:
                listing = json.load(r)
        except Exception as ex:
            print(f'[props] ERROR listing {mid}: {ex}', file=sys.stderr)
            return 1
        files = [f for f in listing if _wanted(f['name'])]
        n_new = 0
        for f in files:
            out = dst / f['name']
            if out.exists() and out.stat().st_size == f['size']:
                continue
            urllib.request.urlretrieve(f['download_url'], out)
            n_new += 1
        print(f'[props] {mid}: {len(files)} files ({n_new} downloaded)')
    print(f'[props] done -> {PROPS / "gso"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 4: Attribution README + gitignore**

`mujoco_grasp_sim/assets/props/README.md`:

```markdown
# Semantic props (ROADMAP P10)

`manifest.json` lists the 12 box-shaped household props used by
`run_sim_grasp_test.py --scene props` (6 `food`, 6 `non_food`), with a
per-prop mass. The meshes/textures themselves are **not committed** —
fetch them once (idempotent):

    cd mujoco_grasp_sim
    python scripts/download_props.py

They land in `gso/<model_id>/` (gitignored).

## Attribution

3D models from **Google Scanned Objects** (Downs et al., 2022,
"Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household
Items"), licensed **CC-BY 4.0**, via the MuJoCo conversion
[kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects)
(MJCF files MIT), pinned at commit `6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c`.
Props are uniformly scaled at load time (see `sim_grasp/props.py`).
```

Append to the repo-root `.gitignore`:

```
mujoco_grasp_sim/assets/props/gso/
```

- [ ] **Step 5: Download, then run the test**

Run:
```bash
cd mujoco_grasp_sim
python scripts/download_props.py
python scripts/download_props.py      # second run must download 0 files
MUJOCO_GL=osmesa PYTHONPATH=. python sim_grasp/test_props_assets.py
git status --short                   # gso/ must NOT appear
```
Expected: first run 12 lines `... 34 files (34 downloaded)`, second run `(0 downloaded)` on every line, test prints `All 12 prop assets present, sized, and loadable.`, and `git status` lists only the new script/README/test/.gitignore.

- [ ] **Step 6: Commit**

```bash
git add .gitignore mujoco_grasp_sim/scripts/download_props.py \
        mujoco_grasp_sim/assets/props/README.md mujoco_grasp_sim/sim_grasp/test_props_assets.py
git commit -m "Add pinned, idempotent download script for the semantic props"
```

---

### Task 3: Bins as a list — `BinSpec`, bin B geometry, per-bin queries

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/scene_generator.py` (SceneConfig fields ~line 105; bin XML block in `_build_scene_xml` ~lines 371-388; `bin_drop_point`/`objects_in_bin` ~lines 517-534)
- Create: `mujoco_grasp_sim/sim_grasp/test_scene_generator_bins.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (`sim_grasp/scene_generator.py`):
  - `@dataclass(frozen=True) BinSpec(name: str, center: tuple, inner_half: float, category: str)`
  - `SceneConfig` new fields: `scene_mode: str = 'boxes'`, `second_bin_center: tuple | None = None`, `bin_categories: tuple = ('food', 'non_food')`, `props_manifest: str | None = None`
  - `SceneConfig.bins() -> list[BinSpec]` — `[A]` when `second_bin_center is None` (category `''`), else `[A(food), B(non_food)]`
  - `SceneGenerator.objects_in_bins() -> dict[str, str]` (object name → `'A'`/`'B'`), `SceneGenerator.bin_drop_point(bin_name: str = 'A') -> np.ndarray`
  - bin B geoms named `bin_b_floor`, `bin_b_wall_xp`, `bin_b_wall_xm`, `bin_b_wall_yp`, `bin_b_wall_ym`

- [ ] **Step 1: Write the failing test**

`mujoco_grasp_sim/sim_grasp/test_scene_generator_bins.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd mujoco_grasp_sim && PYTHONPATH=. python sim_grasp/test_scene_generator_bins.py`
Expected: FAIL with `ImportError: cannot import name 'BinSpec'`

- [ ] **Step 3: Add `BinSpec` and the new `SceneConfig` fields**

In `scene_generator.py`, directly above `@dataclass class SceneConfig:` add:

```python
@dataclass(frozen=True)
class BinSpec:
    """One open-top place bin. category is '' in the legacy one-bin scene."""
    name: str
    center: tuple
    inner_half: float
    category: str
```

In `SceneConfig`, after `bin_wall_half_thickness: float = 0.006` add:

```python
    # [P10] Second bin + semantic props. Defaults reproduce the legacy
    # single-bin boxes scene exactly (see test_scene_default_unchanged.py).
    scene_mode: str = 'boxes'                     # 'boxes' | 'props'
    second_bin_center: tuple | None = None        # e.g. (0.45, 0.30)
    bin_categories: tuple = ('food', 'non_food')  # bin A, bin B
    props_manifest: str | None = None
```

and, after the `seed: int | None = None` line, add the method:

```python
    def bins(self) -> 'list[BinSpec]':
        """Bin A is always the legacy bin; bin B exists only when
        second_bin_center is set, mirroring A's size."""
        if self.second_bin_center is None:
            return [BinSpec('A', tuple(self.bin_center), self.bin_inner_half, '')]
        return [BinSpec('A', tuple(self.bin_center), self.bin_inner_half,
                        self.bin_categories[0]),
                BinSpec('B', tuple(self.second_bin_center), self.bin_inner_half,
                        self.bin_categories[1])]
```

- [ ] **Step 4: Extract the bin XML into a helper (legacy string preserved)**

Add this module-level function right after `_make_mesh_object` (before the `SceneGenerator` section banner):

```python
_BIN_GEOM_PREFIX = {'A': 'bin', 'B': 'bin_b'}   # bin A keeps its legacy geom names


def _bin_xml(prefix: str, center: tuple, cfg: 'SceneConfig') -> str:
    """Floor slab + 4 walls, static, on the tabletop. Formatting copied
    verbatim from the original inline block so bin A's XML is unchanged."""
    bx, by = center
    bi, wt = cfg.bin_inner_half, cfg.bin_wall_half_thickness
    wh = cfg.bin_wall_height / 2
    bo = bi + 2 * wt                       # outer half-extent
    bz = cfg.table_height
    bin_rgba = '0.50 0.55 0.62 1'
    return '\n    '.join([
        f'<geom name="{prefix}_floor" type="box" size="{bo:.4f} {bo:.4f} 0.004" '
        f'pos="{bx} {by} {bz + 0.004:.4f}" rgba="{bin_rgba}"/>',
        f'<geom name="{prefix}_wall_xp" type="box" size="{wt:.4f} {bo:.4f} {wh:.4f}" '
        f'pos="{bx + bi + wt:.4f} {by} {bz + 0.008 + wh:.4f}" rgba="{bin_rgba}"/>',
        f'<geom name="{prefix}_wall_xm" type="box" size="{wt:.4f} {bo:.4f} {wh:.4f}" '
        f'pos="{bx - bi - wt:.4f} {by} {bz + 0.008 + wh:.4f}" rgba="{bin_rgba}"/>',
        f'<geom name="{prefix}_wall_yp" type="box" size="{bo:.4f} {wt:.4f} {wh:.4f}" '
        f'pos="{bx} {by + bi + wt:.4f} {bz + 0.008 + wh:.4f}" rgba="{bin_rgba}"/>',
        f'<geom name="{prefix}_wall_ym" type="box" size="{bo:.4f} {wt:.4f} {wh:.4f}" '
        f'pos="{bx} {by - bi - wt:.4f} {bz + 0.008 + wh:.4f}" rgba="{bin_rgba}"/>'])
```

In `_build_scene_xml`, replace the whole block from the comment `# Place bin: floor slab + 4 walls, static, sitting on the tabletop` through the closing `...rgba="{bin_rgba}"/>'])` of `bin_xml` with:

```python
        # Place bin(s): legacy bin A, plus bin B in the props scene
        bin_xml = '\n    '.join(_bin_xml(_BIN_GEOM_PREFIX[b.name], b.center, cfg)
                                for b in cfg.bins())
```

- [ ] **Step 5: Per-bin queries**

Replace the existing `bin_drop_point` and `objects_in_bin` methods with:

```python
    def bin_drop_point(self, bin_name: str = 'A') -> np.ndarray:
        """World point above which the executor releases objects."""
        b = next(b for b in self.cfg.bins() if b.name == bin_name)
        bx, by = b.center
        return np.array([bx, by, self.cfg.table_height + 0.02])

    def _objects_in_region(self, center: tuple, inner_half: float) -> list[str]:
        cfg, out = self.cfg, []
        bx, by = center
        tol = inner_half + 0.02
        for name in self.object_names:
            jadr = self.model.joint(f'{name}_joint').qposadr[0]
            x, y, z = self.data.qpos[jadr:jadr + 3]
            if (abs(x - bx) < tol and abs(y - by) < tol
                    and cfg.table_height - 0.01 < z < cfg.table_height + 0.20):
                out.append(name)
        return out

    def objects_in_bin(self) -> list[str]:
        """Names of objects currently inside the place bin (bin A)."""
        return self._objects_in_region(self.cfg.bin_center, self.cfg.bin_inner_half)

    def objects_in_bins(self) -> dict[str, str]:
        """{object name: bin name} for every object inside any bin.
        SIM ORACLE (reads qpos) -- P10 SP2 replaces this with vision-only
        scene-graph node locations so the same logic runs on the real robot."""
        out = {}
        for b in self.cfg.bins():
            for name in self._objects_in_region(b.center, b.inner_half):
                out[name] = b.name
        return out
```

- [ ] **Step 6: Run the new test, the guard, and the existing scene test**

Run:
```bash
cd mujoco_grasp_sim
PYTHONPATH=. python sim_grasp/test_scene_generator_bins.py
PYTHONPATH=. python sim_grasp/test_scene_default_unchanged.py
PYTHONPATH=. python sim_grasp/test_scene_generator_paths.py
```
Expected: `All bin-list checks passed.`, `Default boxes scene unchanged (seeds 0, 3).`, `All scene_generator path checks passed (...)`.

- [ ] **Step 7: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/scene_generator.py mujoco_grasp_sim/sim_grasp/test_scene_generator_bins.py
git commit -m "Model place bins as a list and add an optional second bin"
```

---

### Task 4: Props mode in `SceneGenerator` + settle/determinism/camera-coverage checks

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/scene_generator.py` (`ObjectSpec` ~line 124; `SceneConfig` — add `for_props`; `SceneGenerator.__init__` ~line 205; `_sample_objects` / `_sample_xy_positions` ~lines 292-322; `generate()` ~lines 455-490)
- Create: `mujoco_grasp_sim/sim_grasp/test_scene_props.py`

**Interfaces:**
- Consumes: `load_manifest`, `sample_balanced`, `validate_props_count`, `prop_files`, `obj_bounds`, `prop_scale`, `footprint_radius`, `prop_body_xml`, `MANIFEST_PATH` (Task 1); `BinSpec`, `SceneConfig.bins()`, `objects_in_bins()` (Task 3).
- Produces:
  - `ObjectSpec` new fields `category: str = ''`, `assets_xml: str = ''`, `model_id: str = ''`
  - `SceneConfig.for_props(seed=None, manifest=None, n_objects=4) -> SceneConfig`
  - `SceneGenerator.object_categories: dict[str, str]`, `SceneGenerator.object_props: dict[str, str]` (object → GSO model id; both empty in boxes mode)
  - `SceneGenerator._sample_xy_positions(n, radii=None)`
  - the camera-coverage decision `PROPS_CAMERAS` value used by Task 5 (recorded in Step 6)

- [ ] **Step 1: Write the failing test**

`mujoco_grasp_sim/sim_grasp/test_scene_props.py`:

```python
"""Props scene checks (P10 SP1): props load textured, every prop settles
on the table (none tipped into a bin / off the table) across 10 seeds,
the same seed reproduces the same scene, and the lookat camera sees both
bins. Needs downloaded props + MuJoCo, no GPU (MUJOCO_GL=osmesa on
headless/WSL2). Run directly, no pytest."""
import numpy as np

from sim_grasp.camera import CameraModule
from sim_grasp.scene_generator import SceneConfig, SceneGenerator

# 1. settles cleanly on 10 seeds, balanced categories, masses from manifest
for seed in range(10):
    gen = SceneGenerator(SceneConfig.for_props(seed=seed))
    model, data = gen.generate()
    assert len(gen.object_names) == 4
    cats = sorted(gen.object_categories.values())
    assert cats == ['food', 'food', 'non_food', 'non_food'], (seed, cats)
    assert len(set(gen.object_props.values())) == 4, seed
    on_table = gen.objects_on_table()
    assert sorted(on_table) == sorted(gen.object_names), (seed, on_table)
    assert gen.objects_in_bins() == {}, (seed, gen.objects_in_bins())
    assert gen._max_object_speed() < 0.05, (seed, gen._max_object_speed())

# 2. determinism: same seed -> same props and same settled poses
g1 = SceneGenerator(SceneConfig.for_props(seed=4)); _, d1 = g1.generate()
g2 = SceneGenerator(SceneConfig.for_props(seed=4)); _, d2 = g2.generate()
assert g1.object_props == g2.object_props
assert np.allclose(d1.qpos, d2.qpos, atol=1e-6)

# 3. n_objects override + observation camera sees textured props and bin B
gen = SceneGenerator(SceneConfig.for_props(seed=0, n_objects=5))
model, data = gen.generate()
assert len(gen.object_names) == 5
cam = CameraModule(model, data, cam_name=gen.cfg.cam_name, width=640, height=480)
rgb, depth, segmap, K, T_world_cam = cam.capture(gen.object_body_ids)
assert set(np.unique(segmap).astype(int)) >= {1, 2, 3, 4, 5}
T_cam_world = np.linalg.inv(T_world_cam)
for b in gen.cfg.bins():
    for dx in (-b.inner_half, b.inner_half):
        for dy in (-b.inner_half, b.inner_half):
            p = T_cam_world @ np.array([b.center[0] + dx, b.center[1] + dy,
                                        gen.cfg.table_height, 1.0])
            u, v, _ = K @ (p[:3] / p[2])
            assert p[2] > 0 and 0 <= u < 640 and 0 <= v < 480, (b.name, u, v)
cam.close()
print('All props-scene checks passed (10 seeds settle, deterministic, both bins in view).')
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd mujoco_grasp_sim && MUJOCO_GL=osmesa PYTHONPATH=. python sim_grasp/test_scene_props.py`
Expected: FAIL with `AttributeError: type object 'SceneConfig' has no attribute 'for_props'`

- [ ] **Step 3: `ObjectSpec` fields and `SceneConfig.for_props`**

Replace the `ObjectSpec` dataclass with:

```python
@dataclass
class ObjectSpec:
    name: str
    xml: str            # the <body>...</body> snippet
    spawn_half_height: float
    color_name: str = ''
    category: str = ''      # [P10] ground-truth category (props scene only)
    assets_xml: str = ''    # [P10] <texture>/<material>/<mesh> assets for this body
    model_id: str = ''      # [P10] Google Scanned Objects model id
```

Add to `SceneConfig`, right after the `bins()` method from Task 3:

```python
    @classmethod
    def for_props(cls, seed: int | None = None, manifest: str | None = None,
                  n_objects: int = 4) -> 'SceneConfig':
        """[P10] Semantic props preset: 4 GSO props, bin B mirrored at
        (0.45, +0.30) -- the only in-reach spot outside the spawn region,
        since bin A already sits at reachability.MAX_REACH -- and a spawn
        strip narrowed so props (<= 7.7 cm half-diagonal) clear both bins."""
        from sim_grasp.props import MANIFEST_PATH, validate_props_count
        validate_props_count(n_objects)
        return cls(seed=seed, scene_mode='props',
                   n_objects_range=(n_objects, n_objects),
                   second_bin_center=(0.45, 0.30),
                   spawn_y=(-0.09, 0.09),
                   props_manifest=str(manifest or MANIFEST_PATH))
```

- [ ] **Step 4: Generator state, props sampling, size-aware spacing**

In `SceneGenerator.__init__`, after `self.object_body_ids: dict[int, int] = {}   # mj body id -> seg label (1..N)` add:

```python
        self.object_categories: dict[str, str] = {}   # [P10] obj name -> category
        self.object_props: dict[str, str] = {}        # [P10] obj name -> GSO model id
        self._spawn_radii: list[float] | None = None  # [P10] props-only spawn spacing
```

At the top of `_sample_objects`, before `n = int(...)`, add:

```python
        if self.cfg.scene_mode == 'props':
            return self._sample_props()
```

Add this method right after `_sample_objects`:

```python
    def _sample_props(self) -> tuple[list[ObjectSpec], list[str]]:
        """[P10] Balanced food/non-food GSO props, upright in their scanned
        pose (origin at the base, so spawn_half_height is just -lo_z)."""
        from sim_grasp.props import (footprint_radius, load_manifest, obj_bounds,
                                     prop_body_xml, prop_files, prop_scale,
                                     sample_balanced, validate_props_count)
        n = int(self.rng.integers(self.cfg.n_objects_range[0],
                                  self.cfg.n_objects_range[1] + 1))
        validate_props_count(n)
        entries = sample_balanced(load_manifest(self.cfg.props_manifest), n, self.rng)
        specs, assets, radii = [], [], []
        for i, e in enumerate(entries):
            name = f'obj_{i}'
            files = prop_files(e.model_id)
            lo, hi = obj_bounds(files['visual'])
            scale = prop_scale(hi - lo)
            body, assets_xml = prop_body_xml(name, e, files, scale, lo, hi)
            specs.append(ObjectSpec(name=name, xml=body,
                                    spawn_half_height=float(-lo[2] * scale),
                                    category=e.category, assets_xml=assets_xml,
                                    model_id=e.model_id))
            assets.append(assets_xml)
            radii.append(footprint_radius(hi - lo, scale))
        self._spawn_radii = radii
        return specs, assets
```

Replace `_sample_xy_positions` with:

```python
    def _sample_xy_positions(self, n: int, radii: list[float] | None = None) -> np.ndarray:
        """Rejection-sample XY spawn positions keeping min spacing (avoids
        catastrophic initial penetration between objects). With radii
        (props scene) a pair also needs r_i + r_j + 1 cm: a fixed 9-12 cm
        spacing would let two 15 cm props spawn interpenetrating. radii=None
        consumes the rng exactly as before, so box scenes are unchanged."""
        cfg = self.cfg
        positions = []
        for k in range(n):
            for _attempt in range(300):
                xy = np.array([self.rng.uniform(*cfg.spawn_x),
                               self.rng.uniform(*cfg.spawn_y)])
                if all(np.linalg.norm(xy - p) >= (
                        cfg.min_object_spacing if radii is None
                        else max(cfg.min_object_spacing, radii[k] + radii[j] + 0.01))
                       for j, p in enumerate(positions)):
                    positions.append(xy)
                    break
            else:
                # workspace saturated — accept closest-effort placement
                positions.append(xy)
        return np.array(positions)
```

- [ ] **Step 5: Wire it into `generate()`**

In `generate()`, after `self.object_colors = {s.name: s.color_name for s in specs}` add:

```python
        self.object_categories = {s.name: s.category for s in specs if s.category}
        self.object_props = {s.name: s.model_id for s in specs if s.model_id}
```

and change `xy = self._sample_xy_positions(len(specs))` to:

```python
        xy = self._sample_xy_positions(len(specs), radii=self._spawn_radii)
```

- [ ] **Step 6: Run the tests; record the calibrated-camera decision**

Run:
```bash
cd mujoco_grasp_sim
MUJOCO_GL=osmesa PYTHONPATH=. python sim_grasp/test_scene_props.py
PYTHONPATH=. python sim_grasp/test_scene_default_unchanged.py
PYTHONPATH=. python sim_grasp/test_scene_generator_bins.py
```
Expected: `All props-scene checks passed (...)`, `Default boxes scene unchanged (seeds 0, 3).`, `All bin-list checks passed.`

If a seed fails the settle assertion (a prop tipped into a bin or off the table), do NOT loosen the assertion: print that seed's `gen.object_props` and final object poses, and fix the root cause (spawn strip, spacing, or a specific prop's mass) — then re-run.

If the **lookat** camera-coverage assertion fails (a bin B corner projects outside the 640×480 image), stop and report to the user: the spec's bin-B placement assumes lookat sees it, so moving bin B or the camera is a design change, not an implementation detail.

Then check whether the **calibrated** camera also sees bin B:

```bash
MUJOCO_GL=osmesa PYTHONPATH=. python - <<'EOF'
import numpy as np
from pathlib import Path
from sim_grasp.camera import CameraModule
from sim_grasp.scene_generator import SceneConfig, SceneGenerator
cfg = SceneConfig.for_props(seed=0); cfg.calibration_file = str(Path('calibration_result.yaml').resolve())
gen = SceneGenerator(cfg); model, data = gen.generate()
cam = CameraModule(model, data, cam_name=cfg.cam_name, width=640, height=480)
_, _, _, K, T = cam.capture(gen.object_body_ids); Ti = np.linalg.inv(T)
ok = True
for b in cfg.bins():
    for dx in (-b.inner_half, b.inner_half):
        for dy in (-b.inner_half, b.inner_half):
            p = Ti @ np.array([b.center[0]+dx, b.center[1]+dy, cfg.table_height, 1.0])
            u, v, _ = K @ (p[:3] / p[2]); ok &= bool(p[2] > 0 and 0 <= u < 640 and 0 <= v < 480)
print('CALIBRATED SEES BOTH BINS:', ok)
EOF
```

Record the result: if `True`, Task 5 uses `PROPS_CAMERAS = ('calibrated', 'lookat', 'fused')`; if `False`, `PROPS_CAMERAS = ('lookat', 'fused')`. Write the outcome into the commit message body.

- [ ] **Step 7: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/scene_generator.py mujoco_grasp_sim/sim_grasp/test_scene_props.py
git commit -m "Add the semantic props scene mode to SceneGenerator" \
           -m "Calibrated camera sees both bins: <True|False from Step 6>."
```

---

### Task 5: `--scene props` CLI, oracle routing, per-bin placement, metrics

**Files:**
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py` (imports near line 37; `--camera` arg ~line 260; new `--scene` arg; validation block ~line 331; scene config ~line 352; scene prints ~line 383; metrics dict ~line 573; pick-all block ~lines 602-840; single-execute block ~lines 880-950)
- Modify: `mujoco_grasp_sim/README.md` (new "Semantic props scene" section)

**Interfaces:**
- Consumes: `SceneConfig.for_props`, `SceneConfig.bins()`, `BinSpec`, `gen.objects_in_bins()`, `gen.bin_drop_point(name)`, `gen.object_categories`, `gen.object_props` (Tasks 3-4); `validate_props_count`, `sort_outcome` (Task 1); `PROPS_CAMERAS` decision (Task 4 Step 6).
- Produces (for Task 6): props-mode `metrics.json` keys — top level `scene: "props"`, `routing: "oracle"`, `props: {obj: model_id}`, `categories: {obj: category}`; per pick-all round `category`, `target_bin`, `landed_bin`; `pick_all.in_correct_bin: list[str]`, `pick_all.in_wrong_bin: list[str]`; single-execute results carry `target_bin`, `landed_bin`.

- [ ] **Step 1: Imports and arguments**

Next to the existing `from sim_grasp.placement_planner import (...)` import add:

```python
from sim_grasp.props import sort_outcome, validate_props_count
```

Change the `--camera` argument's `default='calibrated',` to `default=None,` and append to its help string: `' Default: calibrated (boxes) / lookat (--scene props).'`

After the `--n-objects` argument add:

```python
    ap.add_argument('--scene', choices=['boxes', 'props'], default='boxes',
                    help='"boxes" = the 3 coloured boxes (default); "props" = '
                         'textured Google Scanned Objects food / non-food props + '
                         'a second bin, each object placed in the bin for its '
                         'category (ROADMAP P10 -- run scripts/download_props.py '
                         'once first)')
```

- [ ] **Step 2: Validation (after the existing `--instruction` exclusivity checks)**

```python
    # [P10] set from the Task 4 coverage check (does the calibrated camera see bin B?)
    PROPS_CAMERAS = ('lookat', 'fused')
    if args.camera is None:
        # props default avoids a camera that can't see bin B; boxes keep calibrated
        args.camera = 'lookat' if args.scene == 'props' else 'calibrated'
    if args.scene == 'props':
        if args.instruction is not None or args.prompt or args.click or args.box:
            sys.exit('[scene] --scene props does not support --instruction/--prompt/'
                     '--click/--box: they resolve objects by colour name and assume '
                     'a single bin.')
        if args.camera not in PROPS_CAMERAS:
            sys.exit(f'[scene] --scene props needs --camera '
                     f'{" or ".join(PROPS_CAMERAS)} (the {args.camera} camera does '
                     'not see the second bin).')
        if args.n_objects is not None:
            try:
                validate_props_count(args.n_objects)
            except ValueError as e:
                sys.exit(f'[scene] {e}')
```

Set `PROPS_CAMERAS` to exactly the tuple recorded in Task 4 Step 6.

- [ ] **Step 3: Scene construction and startup print**

Replace `cfg = SceneConfig(seed=args.seed)` with:

```python
    cfg = (SceneConfig.for_props(seed=args.seed) if args.scene == 'props'
           else SceneConfig(seed=args.seed))
```

Replace the `print(f"[scene] object colors: " ...)` statement (2 lines) with:

```python
    if cfg.scene_mode == 'props':
        print('[scene] props: ' + ', '.join(
            f'{n}={gen.object_props[n]} ({gen.object_categories[n]})'
            for n in gen.object_names))
    else:
        print(f"[scene] object colors: "
              f"{', '.join(f'{n}={gen.object_colors[n]}' for n in gen.object_names)}")
```

Right after the `metrics = {...}` dict literal closes, add:

```python
    if cfg.scene_mode == 'props':
        metrics.update(scene='props', routing='oracle',
                       props=gen.object_props, categories=gen.object_categories)
```

- [ ] **Step 4: Routing helper (after `label_to_body = {...}`)**

```python
    bins_by_name = {b.name: b for b in cfg.bins()}
    bin_categories = {b.name: b.category for b in cfg.bins()}

    def target_bin_for(body: str):
        """Bin this object should go in. P10 SP1 routes by GROUND-TRUTH
        category on purpose -- it's the upper bound SP3's perceived routing
        is measured against; boxes mode always uses bin A."""
        if cfg.scene_mode != 'props':
            return bins_by_name['A']
        cat = gen.object_categories[body]
        return next(b for b in bins_by_name.values() if b.category == cat)
```

- [ ] **Step 5: Pick-all loop — per-bin planners, remaining, placement, outcome**

Replace:
```python
        drop = gen.bin_drop_point()   # kept only as the legacy fallback target
        placement_planner = OccupancyPlacementPlanner(cfg.bin_center, cfg.bin_inner_half)
```
with:
```python
        # one free-space planner per bin; the legacy drop point is resolved per
        # target bin further down (fallback stays inside the object's own bin)
        planners = {name: OccupancyPlacementPlanner(b.center, b.inner_half)
                    for name, b in bins_by_name.items()}
```

Replace `in_bin_now = set(gen.objects_in_bin())` with:
```python
            in_bin_now = set(gen.objects_in_bins())   # any bin: no re-sorting of a wrong-bin object
```

In the placement block (after `footprint = compute_object_footprint(d_o, seg_o, int(sid), K_o, T_wc)`), add `tb = target_bin_for(body)` on the line after `footprint = ...`, then change:
- `d_o, seg_o, K_o, T_wc, cfg.bin_center, cfg.bin_inner_half,` → `d_o, seg_o, K_o, T_wc, tb.center, tb.inner_half,`
- `place_pose = placement_planner.plan(footprint, heightmap)` → `place_pose = planners[tb.name].plan(footprint, heightmap)`

(`resolve_relation(...)` keeps `cfg.bin_center` — `--instruction` is boxes-only, where `tb` is bin A.)

Replace the `else:` fallback branch of the placement:
```python
                else:
                    entry['place'] = executor.place(
                        drop[0], drop[1], drop[2] + PLACE_RELEASE)
                entry['in_bin'] = body in gen.objects_in_bin()
```
with:
```python
                else:
                    drop = gen.bin_drop_point(tb.name)
                    entry['place'] = executor.place(
                        drop[0], drop[1], drop[2] + PLACE_RELEASE)
                landed = gen.objects_in_bins().get(body)
                entry['in_bin'] = landed is not None
                if cfg.scene_mode == 'props':
                    entry.update(category=gen.object_categories[body],
                                 target_bin=tb.name, landed_bin=landed)
                    if landed is not None and landed != tb.name:
                        print(f'[pick-all]   WRONG BIN: {body} ({entry["category"]}) '
                              f'landed in {landed}, target {tb.name}')
```

Replace the end-of-loop summary line `in_bin = gen.objects_in_bin()` with:
```python
        final_bins = gen.objects_in_bins()
        in_bin = ([n for n in gen.object_names if n in final_bins]
                  if cfg.scene_mode == 'props' else gen.objects_in_bin())
```

and immediately before `(save_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))` in the pick-all block (after `metrics['pick_all'] = {...}`) add:
```python
        if cfg.scene_mode == 'props':
            correct, wrong = sort_outcome(final_bins, gen.object_categories, bin_categories)
            metrics['pick_all'].update(in_correct_bin=correct, in_wrong_bin=wrong)
            print(f'[pick-all] SORTED: {len(correct)}/{n_total} in the correct bin; '
                  f'wrong bin: {wrong if wrong else "none"}')
```

Check nothing stale is left:
```bash
grep -n "placement_planner\.\|placement_planner =\|objects_in_bin()" mujoco_grasp_sim/run_sim_grasp_test.py
```
Expected: only the boxes-mode `gen.objects_in_bin()` in the final `in_bin = (...)` expression — no `placement_planner` lines.

- [ ] **Step 6: Single `--execute` path — same routing**

Move `body = label_to_body[int(sid)]` from after the placement block to directly before `footprint = compute_object_footprint(depth, segmap, int(sid), K, T_world_cam)`, and add `tb = target_bin_for(body)` after it. Then change:
- `depth, segmap, K, T_world_cam, cfg.bin_center,` / `cfg.bin_inner_half, exclude_seg_id=int(sid))` → `depth, segmap, K, T_world_cam, tb.center,` / `tb.inner_half, exclude_seg_id=int(sid))`
- `place_pose = OccupancyPlacementPlanner(` / `cfg.bin_center, cfg.bin_inner_half).plan(footprint, heightmap)` → `place_pose = OccupancyPlacementPlanner(` / `tb.center, tb.inner_half).plan(footprint, heightmap)`
- `drop = gen.bin_drop_point()` → `drop = gen.bin_drop_point(tb.name)`
- `res['in_bin'] = body in gen.objects_in_bin()` → 

```python
                landed = gen.objects_in_bins().get(body)
                res['in_bin'] = landed is not None
                if cfg.scene_mode == 'props':
                    res.update(target_bin=tb.name, landed_bin=landed)
```

- [ ] **Step 7: README section**

Add to `mujoco_grasp_sim/README.md`, after the "Interactive live pick" section:

````markdown
### Semantic props scene (`--scene props`, ROADMAP P10)

Textured Google Scanned Objects props (6 food, 6 non-food; 4 per scene,
always 2 + 2) and a second bin: food → bin A at (0.45, −0.30), non-food →
bin B at (0.45, +0.30). In this first stage each object is routed by its
**ground-truth** category (`"routing": "oracle"` in `metrics.json`) — the
upper bound the later perception-based sorting is measured against.

```bash
python scripts/download_props.py                         # once; idempotent
python run_sim_grasp_test.py --scene props --pick-all --camera fused --backend graspgen
python benchmark.py --scene props --seeds 0-9 --mode pick-all --camera fused --backend graspgen --tag props_oracle
```

`metrics.json` gains `pick_all.in_correct_bin` / `in_wrong_bin`; `in_bin`
still means "in any bin". Not supported with `--scene props`:
`--instruction`, `--prompt/--click/--box`, and `interactive_pick.py`.
````

- [ ] **Step 8: Regression — existing tests + guard**

Run:
```bash
cd mujoco_grasp_sim
for t in sim_grasp/test_*.py; do MUJOCO_GL=osmesa PYTHONPATH=. python "$t" > /dev/null || echo "FAIL $t"; done; echo done
```
Expected: only `done` (no `FAIL` lines).

- [ ] **Step 9: CLI exit checks (no GPU work happens before these exit)**

Run each; each must print one `[scene] ...` line and exit non-zero without a traceback:
```bash
python run_sim_grasp_test.py --scene props --pick-all --instruction "pick the red one" --no-vis
python run_sim_grasp_test.py --scene props --execute --prompt "the red box" --no-vis
python run_sim_grasp_test.py --scene props --n-objects 7 --no-vis
python run_sim_grasp_test.py --scene props --camera calibrated --no-vis   # only if calibrated is NOT in PROPS_CAMERAS
```

- [ ] **Step 10: Live smoke tests (GPU)**

Run:
```bash
export MUJOCO_GL=osmesa GRASPGEN_PYTHON=~/miniconda3/envs/graspgen_torch/bin/python
python run_sim_grasp_test.py --scene props --seed 0 --execute --top-k 5 --backend graspgen --camera fused --no-vis --save-dir output/p10_smoke_exec
python run_sim_grasp_test.py --scene props --seed 0 --pick-all --backend graspgen --camera fused --no-vis --save-dir output/p10_smoke_pickall
python run_sim_grasp_test.py --seed 0 --pick-all --backend graspgen --camera fused --no-vis --save-dir output/p10_boxes_regress
```
Expected:
- exec run: `[scene] props: obj_0=... (food|non_food), ...`, `PICK SUCCESS`, and in `output/p10_smoke_exec/metrics.json` the successful attempt has `target_bin == landed_bin`.
- pick-all props run: a `[pick-all] SORTED: X/4 in the correct bin` line; open `observation.png` and `execution.gif` and confirm textured props and two bins are visible.
- boxes run: `[pick-all] DONE: 3/3 objects in the bin` (matches the GraspGen/fused baseline in `ROADMAP.md` P1) and no `SORTED` line.

Record the three outcomes (numbers, not adjectives) for Task 7.

- [ ] **Step 11: Commit**

```bash
git add mujoco_grasp_sim/run_sim_grasp_test.py mujoco_grasp_sim/README.md
git commit -m "Route objects to per-category bins in the --scene props mode"
```

---

### Task 6: Benchmark `--scene` + `wrong_bin` failure category

**Files:**
- Modify: `mujoco_grasp_sim/benchmark.py` (`run_one` ~lines 39-92; argparse ~line 96-114; summary ~line 145)
- Modify: `mujoco_grasp_sim/analyze_failures.py` (taxonomy docstring ~line 17; `analyze_run` rounds loop ~line 70-79)
- Create: `mujoco_grasp_sim/sim_grasp/test_analyze_failures_wrong_bin.py`

**Interfaces:**
- Consumes: Task 5's props metrics keys (`target_bin`, `landed_bin`, `pick_all.in_correct_bin`).
- Produces: `benchmark.py --scene {boxes,props}`; per-run `in_correct_bin: int` in `summary.json`; summary line `[bench] objects in correct bin: X/Y (Z%)`; `analyze_failures` category `wrong_bin`.

- [ ] **Step 1: Write the failing test**

`mujoco_grasp_sim/sim_grasp/test_analyze_failures_wrong_bin.py`:

```python
"""analyze_failures must classify a props-scene object that was binned
into the WRONG bin as 'wrong_bin' (P10 SP1), and must not flag a
correctly-sorted or a legacy boxes-mode round. Pure dicts, no MuJoCo.
Run directly from mujoco_grasp_sim/ with PYTHONPATH=., no pytest."""
from analyze_failures import analyze_run

ok_pick = {'success': True}
m = {'pick_all': {'rounds': [
    {'round': 1, 'body': 'obj_0', 'score': 0.9, 'pick': ok_pick,
     'place': {'stage': 'placed'}, 'in_bin': True,
     'category': 'food', 'target_bin': 'A', 'landed_bin': 'B'},
    {'round': 2, 'body': 'obj_1', 'score': 0.8, 'pick': ok_pick,
     'place': {'stage': 'placed'}, 'in_bin': True,
     'category': 'non_food', 'target_bin': 'B', 'landed_bin': 'B'},
    {'round': 3, 'body': 'obj_2', 'score': 0.7, 'pick': ok_pick,
     'place': {'stage': 'placed'}, 'in_bin': True},        # boxes-mode round
], 'fell_off_table': []}}

ev = analyze_run(m, 'seed_0')
assert [e['category'] for e in ev] == ['wrong_bin'], ev
assert ev[0]['body'] == 'obj_0' and 'B' in ev[0]['detail'] and 'A' in ev[0]['detail']
print('analyze_failures wrong_bin checks passed.')
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd mujoco_grasp_sim && PYTHONPATH=. python sim_grasp/test_analyze_failures_wrong_bin.py`
Expected: FAIL with `AssertionError: []`

- [ ] **Step 3: Classifier**

In `analyze_failures.py`'s module docstring taxonomy, after the `missed_bin` line add:

```
    wrong_bin            placed in a bin, but not the one for its category (P10 props scene)
```

In `analyze_run`, extend the rounds loop's `elif` chain — after:
```python
        elif not r.get('in_bin', True):
            add('missed_bin', 'placed but landed outside the bin', **where)
```
add:
```python
        elif r.get('target_bin') and r.get('landed_bin') not in (None, r['target_bin']):
            add('wrong_bin', f"landed in bin {r['landed_bin']}, target bin "
                f"{r['target_bin']}", **where)
```

- [ ] **Step 4: Benchmark passthrough + summary**

In `benchmark.py` argparse (after `--filter-neighbors`) add:

```python
    ap.add_argument('--scene', choices=['boxes', 'props'], default='boxes',
                    help='passed through to run_sim_grasp_test.py (P10)')
```

In `run_one`, after the `if args.filter_neighbors:` block add:

```python
    if args.scene != 'boxes':          # keep the boxes command line byte-identical
        cmd += ['--scene', args.scene]
```

In the `elif args.mode == 'pick-all':` block of `run_one`, after `out['total'] = pa.get('objects_total')` add:

```python
        if 'in_correct_bin' in pa:
            out['in_correct_bin'] = len(pa['in_correct_bin'])
```

In `main()`'s pick-all summary, after the `objects binned` print add:

```python
        if any('in_correct_bin' in r for r in ok):
            corr = sum(r.get('in_correct_bin', 0) for r in ok)
            print(f'[bench] objects in correct bin: {corr}/{total} '
                  f'({100 * corr / max(total, 1):.0f}%)')
```

- [ ] **Step 5: Run the test + a 1-seed benchmark smoke**

Run:
```bash
cd mujoco_grasp_sim
PYTHONPATH=. python sim_grasp/test_analyze_failures_wrong_bin.py
python benchmark.py --scene props --seeds 0 --mode pick-all --camera fused --backend graspgen --graspgen-python "$GRASPGEN_PYTHON" --tag p10_bench_smoke
python analyze_failures.py output/bench_p10_bench_smoke
```
Expected: `analyze_failures wrong_bin checks passed.`; the benchmark prints both `objects binned: X/4` and `objects in correct bin: Y/4 (...)`; `analyze_failures` runs without error.

- [ ] **Step 6: Commit**

```bash
git add mujoco_grasp_sim/benchmark.py mujoco_grasp_sim/analyze_failures.py \
        mujoco_grasp_sim/sim_grasp/test_analyze_failures_wrong_bin.py
git commit -m "Report correct-bin rate and wrong_bin failures for the props scene"
```

---

### Task 7: SP1 gate benchmark + documentation sync

**Files:**
- Modify: `ROADMAP.md` (P10 section, SP1 bullet)
- Modify: `docs/superpowers/specs/2026-09-29-semantic-sorting-scene-graph-design.md` (append "Implementation notes")
- Modify: `CLAUDE.md` (MuJoCo sim commands block — one line)
- Modify: `/home/vivek/.claude/projects/-home-vivek-ContactPilot/memory/scene-graph-semantic-sorting.md` (SP1 outcome)

**Interfaces:**
- Consumes: everything above.
- Produces: the recorded gate verdict that decides whether SP2 may start.

- [ ] **Step 1: Run the gate (≈15-25 min)**

```bash
cd mujoco_grasp_sim
export MUJOCO_GL=osmesa GRASPGEN_PYTHON=~/miniconda3/envs/graspgen_torch/bin/python
python benchmark.py --scene props --seeds 0-9 --mode pick-all --camera fused \
    --backend graspgen --graspgen-python "$GRASPGEN_PYTHON" --tag props_oracle
python analyze_failures.py output/bench_props_oracle
```

Pass = `objects in correct bin` ≥ **34/40**, `knocked off table` ≤ **2**, `runs completed` **10/10** (0 crashed). Copy the exact `[bench]` summary lines and the `analyze_failures` totals.

- [ ] **Step 2: Reference CGN run (not gated)**

```bash
python benchmark.py --scene props --seeds 0-9 --mode pick-all --camera fused --backend cgn --tag props_oracle_cgn
```

Copy its `[bench]` summary lines.

- [ ] **Step 3: If the gate FAILS — stop here**

Do not mark SP1 done and do not start SP2. Record the numbers and the dominant `analyze_failures` categories in the ROADMAP bullet (Step 4) as a failed gate, commit the docs, and report to the user with the failure breakdown and the proposed adjustment (curation / `H_MAX` / `W_MAX` / masses) — the adjustment is a new task, decided with the user.

- [ ] **Step 4: ROADMAP**

In `ROADMAP.md` P10, change the SP1 bullet's `- [ ]` to `- [x]` (only on a passing gate) and append a dated sub-bullet with the real numbers, in the file's existing style, e.g.:

```markdown
  - **Gate result (YYYY-MM-DD):** GraspGen/fused, seeds 0-9, pick-all:
    **X/40 in the correct bin (Z%)**, binned (any bin) B/40, knocked off
    table K, crashes 0 (3-box baseline: 30/30). CGN reference: X'/40.
    Failure taxonomy: <analyze_failures totals>. Calibrated camera sees
    bin B: <True|False> (hence `--scene props` default camera = lookat).
```

Update the section header tag to `[SP1 DONE YYYY-MM-DD, SP2 not started]` on a pass.

- [ ] **Step 5: Spec implementation notes + CLAUDE.md + memory**

Append to the design spec:

```markdown
## Implementation notes (SP1)

- `--camera` default became `None` → resolved to `calibrated` (boxes) or
  `lookat` (props), so `--scene props` works without extra flags; boxes
  behaviour is unchanged.
- `ObjectSpec.model_id` was added alongside the spec's `category` /
  `assets_xml` fields (needed for `SceneGenerator.object_props`).
- Calibrated camera sees bin B: <True|False> → `PROPS_CAMERAS = <tuple>`.
- <any other divergence found during implementation, with why>
```

In `CLAUDE.md`'s MuJoCo sim command block, after the `--instruction` line add:

```
python run_sim_grasp_test.py --scene props --pick-all --camera fused   # P10: GSO food/non-food props, 2 bins (run scripts/download_props.py once)
```

Update the memory file's body with one line: SP1 gate result (numbers + date) and whether SP2 is unblocked.

- [ ] **Step 6: Final full regression + commit**

Run:
```bash
cd mujoco_grasp_sim
for t in sim_grasp/test_*.py; do MUJOCO_GL=osmesa PYTHONPATH=. python "$t" > /dev/null || echo "FAIL $t"; done; echo done
```
Expected: only `done`.

```bash
git add ROADMAP.md CLAUDE.md docs/superpowers/specs/2026-09-29-semantic-sorting-scene-graph-design.md
git commit -m "Record the P10 sub-project 1 gate result"
```
