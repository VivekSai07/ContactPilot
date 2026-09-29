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
