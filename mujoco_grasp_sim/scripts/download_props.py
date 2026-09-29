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
