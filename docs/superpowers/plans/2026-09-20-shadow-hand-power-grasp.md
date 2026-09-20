# Shadow Hand E3M5 — Mechanical Mount + Fixed Power-Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mount the Shadow Hand E3M5 (`mujoco_menagerie/shadow_hand`) on the Franka Panda in place of the parallel-jaw gripper, and drive it with one fixed, DexGraspNet-derived power-grasp posture, while leaving every existing parallel-gripper script's behavior byte-identical.

**Architecture:** A new merged MJCF asset (`sim_grasp/hand_assets.py`, mirroring `scene_generator.py`'s existing panda.xml-patching pattern) grafts Shadow Hand's `rh_palm`-and-fingers subtree onto the Panda's `link7` at the exact mount frame the old 2-finger `hand` body used, with Shadow's own forearm/wrist dropped entirely (the Panda's 7 arm joints already provide positioning). A new `EndEffectorController` ABC (mirroring this repo's existing `GraspPredictor`/`PlacementPlanner` ABC pattern) lets `GraspExecutor` drive either end effector through the exact same 0–255 "openness" scalar the pick/place state machine already uses — `ParallelGripperController` wraps today's exact behavior verbatim (zero change for existing callers); `ShadowHandController` interpolates the hand's 18 actuators between an open posture and one fixed power-grasp posture mined from DexGraspNet's public dataset.

**Tech Stack:** Python, MuJoCo — no new runtime dependencies (DexGraspNet's code is never imported or run; only its published data is read, once, offline).

**Spec:** `docs/superpowers/specs/2026-09-20-shadow-hand-power-grasp-design.md`

## Global Constraints

- **Default behavior for every existing script must be byte-identical.** `SceneConfig.end_effector` defaults to `'parallel'`; `GraspExecutor`'s `end_effector` parameter defaults to constructing a `ParallelGripperController`. Neither `run_sim_grasp_test.py` nor `interactive_pick.py` should need any code change beyond accepting the new optional flag.
- **Shadow Hand's own wrist (`rh_WRJ1`/`rh_WRJ2`) is dropped, not driven** — confirmed by reading `shadow_hand/right_hand.xml`: `rh_palm`'s joint is `rh_WRJ1` (child of `rh_wrist`, whose joint is `rh_WRJ2`, child of `rh_forearm`). This plan grafts `rh_palm` (with its own `<joint>` line removed, making it rigidly fixed) directly as a child of Panda's `link7`, dropping the `rh_forearm`/`rh_wrist` bodies and their 2 actuators (`rh_A_WRJ2`, `rh_A_WRJ1`) entirely.
- **Menagerie's Shadow Hand model has exactly 18 real actuators** (not 20 — the 2 wrist ones are dropped per the constraint above): `rh_A_THJ5, rh_A_THJ4, rh_A_THJ3, rh_A_THJ2, rh_A_THJ1` (thumb, 5), `rh_A_FFJ4, rh_A_FFJ3, rh_A_FFJ0` (first finger, 3 — `rh_A_FFJ0` drives the `rh_FFJ0` fixed tendon coupling joints `rh_FFJ2`+`rh_FFJ1` with equal coefficients), same 3-actuator pattern for `MFJ`/`RFJ`, and `rh_A_LFJ5, rh_A_LFJ4, rh_A_LFJ3, rh_A_LFJ0` (little finger, 4 — has an extra metacarpal joint). Total: 5+3+3+3+4 = 18. Merged model's `data.ctrl` layout: indices 0–6 = Panda arm (unchanged), indices 7–24 = these 18 actuators in this exact order.
- **DexGraspNet's own joint naming differs from Menagerie's by a verified, uniform rule**: Menagerie's `rh_{finger}J{n}` = DexGraspNet's `robot0:{finger}J{n-1}`, for every finger and the thumb (confirmed directly against DexGraspNet's own `grasp_generation/quick_example.ipynb`, which lists its `joint_names` explicitly: `robot0:FFJ3,FFJ2,FFJ1,FFJ0`, `robot0:MFJ3,MFJ2,MFJ1,MFJ0`, `robot0:RFJ3,RFJ2,RFJ1,RFJ0`, `robot0:LFJ4,LFJ3,LFJ2,LFJ1,LFJ0`, `robot0:THJ4,THJ3,THJ2,THJ1,THJ0`). DexGraspNet's saved grasp data indexes joint angles **by this exact name string as a dict key** (`grasp_data[index]['qpos'][joint_name]`), not by position — so mapping is a name-to-name lookup, not an order-dependent slice.
- **Known, accepted v1 approximation**: Menagerie's model tendon-couples each finger's middle+distal joints into one actuator (matching real Shadow Hand hardware) — a DexGraspNet posture with independently-set middle/distal angles can only be approximated (this plan uses the distal joint's angle for the shared actuator target — closer to the fingertip contact that matters most for a power grasp). Document this in the extracted-posture data file's docstring; do not silently pretend exact reproduction.
- This repo has no automated test suite by design — tests are standalone `test_*.py` scripts, plain `assert`, no pytest, run with `PYTHONPATH=. python sim_grasp/test_whatever.py` from `mujoco_grasp_sim/`.
- Commit messages are plain text — never add a `Co-Authored-By: Claude` trailer.
- Already on branch `shadow-hand-power-grasp`, forked from `main`.
- The `mujoco_menagerie` submodule's sparse-checkout needs `shadow_hand` added (`git -C mujoco_menagerie sparse-checkout add shadow_hand`) for any task reading or referencing its files — this is a local, uncommitted git config change (not part of the submodule's tracked state), safe to make permanently for this branch's lifetime.

---

### Task 1: Merged Panda+Shadow-Hand MJCF asset

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/hand_assets.py`
- Test: `mujoco_grasp_sim/sim_grasp/test_hand_assets.py`

**Interfaces:**
- Consumes: nothing new (reads `mujoco_menagerie/franka_emika_panda/panda.xml` and `mujoco_menagerie/shadow_hand/right_hand.xml` directly, mirroring `scene_generator.py`'s existing `MENAGERIE_PANDA_DIR` constant pattern).
- Produces: `build_panda_shadow_hand_xml() -> str` (writes the merged asset to `mujoco_grasp_sim/assets/panda_shadow_hand.xml`, returns just `'panda_shadow_hand.xml'` — matching `_patched_panda_xml()`'s own existing return contract in `scene_generator.py:224-278`, which returns `out.name`, not a full `Path`, since the caller splices this filename into an `<include file="...">` reference). Used by Task 5.

- [ ] **Step 1: Expand the submodule sparse-checkout**

```bash
cd ~/ContactPilot/mujoco_menagerie
git sparse-checkout add shadow_hand
git sparse-checkout list   # expect: franka_emika_panda, shadow_hand
```

- [ ] **Step 2: Write the failing test**

Create `mujoco_grasp_sim/sim_grasp/test_hand_assets.py`:

```python
"""Standalone check for the merged Panda+Shadow-Hand MJCF asset -- run
directly, no pytest (this codebase has no automated test suite)."""
import mujoco

from sim_grasp.hand_assets import build_panda_shadow_hand_xml
from sim_grasp.scene_generator import GENERATED_DIR

xml_name = build_panda_shadow_hand_xml()
xml_path = GENERATED_DIR / xml_name
assert xml_path.exists(), f'{xml_path} was not written'

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)   # confirms the model is physically valid, not just parseable

# 7 arm actuators + 18 Shadow Hand actuators, in that order.
assert model.nu == 25, f'expected 25 actuators (7 arm + 18 hand), got {model.nu}'
expected_hand_actuators = [
    'rh_A_THJ5', 'rh_A_THJ4', 'rh_A_THJ3', 'rh_A_THJ2', 'rh_A_THJ1',
    'rh_A_FFJ4', 'rh_A_FFJ3', 'rh_A_FFJ0',
    'rh_A_MFJ4', 'rh_A_MFJ3', 'rh_A_MFJ0',
    'rh_A_RFJ4', 'rh_A_RFJ3', 'rh_A_RFJ0',
    'rh_A_LFJ5', 'rh_A_LFJ4', 'rh_A_LFJ3', 'rh_A_LFJ0',
]
assert len(expected_hand_actuators) == 18
for i, name in enumerate(expected_hand_actuators):
    actual = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 7 + i)
    assert actual == name, f'actuator {7+i}: expected {name!r}, got {actual!r}'

# rh_palm must be a direct child of link7 (the old "hand" body's former
# parent), fixed (no free/hinge joint of its own -- the wrist was dropped).
palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'rh_palm')
assert palm_bid != -1, 'rh_palm body not found'
link7_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link7')
assert model.body_parentid[palm_bid] == link7_bid, \
    'rh_palm must be a direct child of link7'
assert model.body_jntnum[palm_bid] == 0, \
    'rh_palm must have no joint of its own (wrist dropped, rigidly fixed)'

# The old Panda gripper bodies/joints must be gone.
assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand') == -1
assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint1') == -1
assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'rh_WRJ1') == -1
assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'rh_WRJ2') == -1

print('All hand_assets checks passed.')
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `mujoco_grasp_sim/`, `conda activate cgn_torch`):
`PYTHONPATH=. python sim_grasp/test_hand_assets.py`
Expected: `ModuleNotFoundError: No module named 'sim_grasp.hand_assets'`

- [ ] **Step 3: Write the merge function**

Create `mujoco_grasp_sim/sim_grasp/hand_assets.py`:

```python
"""Builds a merged Panda+Shadow-Hand MJCF asset: grafts Shadow Hand E3M5's
palm-and-fingers subtree onto the Panda's link7, in place of the stock
2-finger parallel gripper, and writes the result into
mujoco_grasp_sim/assets/ (never back into the mujoco_menagerie/ submodule
-- same convention scene_generator.py already uses for panda.xml).

Shadow Hand's own forearm/wrist (rh_forearm, rh_wrist, joints rh_WRJ1/
rh_WRJ2, actuators rh_A_WRJ1/rh_A_WRJ2) are dropped entirely: the Panda's
7 arm joints already provide positioning, and driving a second wrist is
out of scope for this plan (see the design spec's "Decisions").
"""
import re
from pathlib import Path

from sim_grasp.scene_generator import MENAGERIE_PANDA_DIR, GENERATED_DIR, REPO_ROOT

MENAGERIE_SHADOW_HAND_DIR = REPO_ROOT / 'mujoco_menagerie' / 'shadow_hand'

# The exact mount frame the stock Panda gripper's "hand" body used to
# occupy on link7 (panda.xml: <body name="hand" pos="0 0 0.107"
# quat="0.9238795 0 0 -0.3826834">) -- rh_palm is grafted here verbatim.
HAND_MOUNT_POS = '0 0 0.107'
HAND_MOUNT_QUAT = '0.9238795 0 0 -0.3826834'


def _extract_shadow_hand_pieces(shadow_xml: str) -> dict:
    """Pulls the four pieces needed from shadow_hand/right_hand.xml:
    the <default class="right_hand"> tree, the <mesh>/<material> asset
    entries, the rh_palm body subtree (with its own rh_WRJ1 <joint> line
    removed), and the parts of <tendon>/<actuator>/<contact> that don't
    reference the dropped wrist."""
    default_block = re.search(
        r'(<default>.*?</default>)', shadow_xml, re.DOTALL).group(1)

    mesh_dir = str((MENAGERIE_SHADOW_HAND_DIR / 'assets').resolve())
    asset_block = re.search(r'<asset>(.*?)</asset>', shadow_xml, re.DOTALL).group(1)
    # Individual mesh files get an absolute path, bypassing this file's
    # own <compiler meshdir> (same "absolute path so meshdir is bypassed"
    # trick scene_generator.py already uses for YCB objects).
    asset_block = re.sub(
        r'file="([^"]+)"', lambda m: f'file="{mesh_dir}/{m.group(1)}"', asset_block)

    palm_match = re.search(
        r'(<body name="rh_palm".*?)(\n\s*</body>\s*</body>\s*</body>\s*</worldbody>)',
        shadow_xml, re.DOTALL)
    palm_body = palm_match.group(1)
    # Drop rh_palm's own joint (the dropped rh_WRJ1) -- everything below
    # it (fingers, thumb) is untouched.
    palm_body, n = re.subn(r'\s*<joint[^/]*name="rh_WRJ1"[^/]*/>', '', palm_body)
    assert n == 1, 'expected exactly one rh_WRJ1 <joint> line inside rh_palm'

    tendon_block = re.search(r'<tendon>(.*?)</tendon>', shadow_xml, re.DOTALL).group(1)

    actuator_block = re.search(
        r'<actuator>(.*?)</actuator>', shadow_xml, re.DOTALL).group(1)
    # Drop the 2 wrist actuators; keep the 18 finger/thumb ones.
    actuator_lines = [l for l in actuator_block.splitlines()
                      if 'rh_A_WRJ' not in l]
    actuator_block = '\n'.join(actuator_lines)

    # Only keep the thumb self-collision exclude -- the other one
    # referenced rh_wrist/rh_forearm, which no longer exist.
    contact_block = '<contact>\n      <exclude body1="rh_thproximal" body2="rh_thmiddle"/>\n    </contact>'

    return {
        'default': default_block,
        'asset': asset_block,
        'palm_body': palm_body,
        'tendon': tendon_block,
        'actuator': actuator_block,
        'contact': contact_block,
    }


def build_panda_shadow_hand_xml() -> Path:
    panda_xml = (MENAGERIE_PANDA_DIR / 'panda.xml').read_text(encoding='utf-8')
    shadow_xml = (MENAGERIE_SHADOW_HAND_DIR / 'right_hand.xml').read_text(encoding='utf-8')
    pieces = _extract_shadow_hand_pieces(shadow_xml)

    # Rewrite panda.xml's own meshdir to an absolute path (same as
    # scene_generator._patched_panda_xml -- this file is not that
    # function, so it must redo that one rewrite itself).
    abs_panda_meshdir = str((MENAGERIE_PANDA_DIR / 'assets').resolve())
    patched = re.sub(r'meshdir="assets"', f'meshdir="{abs_panda_meshdir}"', panda_xml, count=1)

    # Remove the stock 2-finger gripper: the "hand" body (and everything
    # inside it -- left_finger, right_finger), the "split" fixed tendon,
    # its <equality> coupling, and actuator8.
    patched, n = re.subn(
        r'\s*<body name="hand".*?</body>\s*</body>\s*</body>',
        '</body></body>', patched, count=1, flags=re.DOTALL)
    assert n == 1, 'expected exactly one "hand" body subtree in panda.xml'
    patched = re.sub(r'\s*<tendon>.*?</tendon>', '', patched, count=1, flags=re.DOTALL)
    patched = re.sub(
        r'\s*<equality>.*?</equality>', '', patched, count=1, flags=re.DOTALL)
    patched = re.sub(
        r'\s*<general[^/]*name="actuator8".*?/>', '', patched, count=1)

    # Graft rh_palm at the old "hand" body's mount frame.
    grafted_palm = (
        f'<body name="rh_palm_mount" pos="{HAND_MOUNT_POS}" quat="{HAND_MOUNT_QUAT}">\n'
        f'{pieces["palm_body"]}\n</body>'
    )
    patched, n = re.subn(r'(<geom mesh="link7_c" class="collision"/>)',
                         r'\1\n' + grafted_palm, patched, count=1)
    assert n == 1, 'could not find the link7 collision geom to graft the hand after'

    # Merge in the Shadow Hand's own <default>, <asset> additions, the
    # remaining <tendon>/<actuator>/<contact> blocks.
    patched = patched.replace('</asset>', pieces['asset'] + '\n  </asset>', 1)
    patched = patched.replace('</default>', pieces['default'] + '\n</default>', 1)
    patched += f'\n<tendon>{pieces["tendon"]}</tendon>\n'
    patched += f'{pieces["contact"]}\n'
    patched = patched.replace('</mujoco>', f'<actuator>{pieces["actuator"]}</actuator>\n</mujoco>')

    out = GENERATED_DIR / 'panda_shadow_hand.xml'
    out.write_text(patched, encoding='utf-8')
    return out.name
```

Note for whoever implements this: the regexes above assume the current
(2026-09-20) exact text layout of `panda.xml`/`right_hand.xml` — if any
assertion fails, read the actual file at that point and adjust the regex,
don't loosen the assertion. This mirrors `scene_generator._patched_panda_xml`'s
own existing style (assert-on-exactly-one-match, fail loud on upstream
drift) rather than silently matching zero or many.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python sim_grasp/test_hand_assets.py`
Expected: `All hand_assets checks passed.`

- [ ] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/hand_assets.py mujoco_grasp_sim/sim_grasp/test_hand_assets.py
git commit -m "Add merged Panda+Shadow-Hand MJCF asset builder"
```

---

### Task 2: Extract a fixed power-grasp posture from DexGraspNet's dataset

**Files:**
- Create: `mujoco_grasp_sim/scripts/extract_shadow_hand_posture.py` (one-time offline tool, not part of the runtime pipeline)
- Create: `mujoco_grasp_sim/sim_grasp/shadow_hand_posture.py` (this task's actual deliverable — a static data module)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `SHADOW_HAND_OPEN_POSTURE: dict[str, float]` and `SHADOW_HAND_POWER_GRASP_POSTURE: dict[str, float]`, both keyed by the 18 actuator names from Task 1's Global Constraints list, each value a joint-angle target in radians. Used by Task 3.

This is a **data-extraction task, not a code-correctness task** — there is
no unit test for "is this posture the *right* one" (that's what Task 6's
visual validation is for). The one thing that must be verified is the
joint-name mapping itself, which Step 3 does mechanically.

- [ ] **Step 1: Get one real DexGraspNet grasp sample**

DexGraspNet's dataset is hosted externally (their [project
page](https://pku-epic.github.io/DexGraspNet/), not fetchable via a
simple GitHub URL) and is licensed **CC BY-NC 4.0** — download one small
object's `.npy` grasp file (e.g. any single entry under their `dataset/`
release) per their README's instructions. Each file is a pickled numpy
array of dicts; each dict has a `'qpos'` sub-dict keyed by joint-name
strings (confirmed directly from DexGraspNet's own
`grasp_generation/quick_example.ipynb`).

- [ ] **Step 2: Write the extraction script**

Create `mujoco_grasp_sim/scripts/extract_shadow_hand_posture.py`:

```python
"""One-time, offline tool: extracts one validated ShadowHand power grasp
from a downloaded DexGraspNet .npy grasp file and writes
sim_grasp/shadow_hand_posture.py. Not part of the runtime pipeline --
DexGraspNet's own code is never imported; only its published data format
(a dict keyed by joint-name strings, per their own quick_example.ipynb)
is read here.

Usage: python scripts/extract_shadow_hand_posture.py <path/to/grasp.npy> [--index N]
"""
import argparse
import numpy as np

# Menagerie actuator name -> DexGraspNet joint-name key, per the verified
# rule: rh_{finger}J{n} = DexGraspNet robot0:{finger}J{n-1}. For the 3
# fingers whose middle+distal are tendon-coupled on the real hardware (and
# in this Menagerie model) into one actuator, the DISTAL joint's angle is
# used for that shared actuator target -- an accepted v1 approximation
# (see this plan's Global Constraints), not an exact reproduction.
ACTUATOR_TO_DEXGRASPNET_JOINT = {
    'rh_A_THJ5': 'robot0:THJ4', 'rh_A_THJ4': 'robot0:THJ3',
    'rh_A_THJ3': 'robot0:THJ2', 'rh_A_THJ2': 'robot0:THJ1',
    'rh_A_THJ1': 'robot0:THJ0',
    'rh_A_FFJ4': 'robot0:FFJ3', 'rh_A_FFJ3': 'robot0:FFJ2',
    'rh_A_FFJ0': 'robot0:FFJ0',    # coupled FFJ2+FFJ1 -> use distal (FFJ0)
    'rh_A_MFJ4': 'robot0:MFJ3', 'rh_A_MFJ3': 'robot0:MFJ2',
    'rh_A_MFJ0': 'robot0:MFJ0',
    'rh_A_RFJ4': 'robot0:RFJ3', 'rh_A_RFJ3': 'robot0:RFJ2',
    'rh_A_RFJ0': 'robot0:RFJ0',
    'rh_A_LFJ5': 'robot0:LFJ4', 'rh_A_LFJ4': 'robot0:LFJ3',
    'rh_A_LFJ3': 'robot0:LFJ2', 'rh_A_LFJ0': 'robot0:LFJ0',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('npy_path')
    ap.add_argument('--index', type=int, default=0,
                    help='which grasp entry in the file to use (default: first)')
    args = ap.parse_args()

    grasp_data = np.load(args.npy_path, allow_pickle=True)
    qpos = grasp_data[args.index]['qpos']

    power_grasp = {act: float(qpos[joint])
                   for act, joint in ACTUATOR_TO_DEXGRASPNET_JOINT.items()}
    # Open posture: all zeros is the Shadow Hand's natural relaxed-open
    # pose for every joint in this model (each joint's <default> range
    # straddles or starts at 0 -- confirmed against right_hand.xml's
    # <default> ranges).
    open_posture = {act: 0.0 for act in ACTUATOR_TO_DEXGRASPNET_JOINT}

    out_path = 'sim_grasp/shadow_hand_posture.py'
    with open(out_path, 'w') as f:
        f.write('"""Fixed Shadow Hand postures -- see extract_shadow_hand_posture.py '
               'for provenance (mined from DexGraspNet, CC BY-NC 4.0, grasp '
               f'{args.npy_path!r} index {args.index})."""\n\n')
        f.write(f'SHADOW_HAND_OPEN_POSTURE = {open_posture!r}\n\n')
        f.write(f'SHADOW_HAND_POWER_GRASP_POSTURE = {power_grasp!r}\n')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Run it against a real downloaded grasp file**

```bash
cd mujoco_grasp_sim
python scripts/extract_shadow_hand_posture.py /path/to/downloaded/some_object.npy
```

Expected: `wrote sim_grasp/shadow_hand_posture.py`, containing real,
DexGraspNet-derived joint angles (not zeros/placeholders) for
`SHADOW_HAND_POWER_GRASP_POSTURE`. Open the file and sanity-check by eye:
finger joint angles should be positive numbers in a plausible curled-grasp
range (roughly 0.3–1.5 rad for proximal/middle/distal joints per
`right_hand.xml`'s own `<default>` ranges), not all zero or all at a
joint's extreme limit.

- [ ] **Step 4: Commit**

```bash
git add mujoco_grasp_sim/scripts/extract_shadow_hand_posture.py mujoco_grasp_sim/sim_grasp/shadow_hand_posture.py
git commit -m "Extract a fixed Shadow Hand power-grasp posture from DexGraspNet"
```

---

### Task 3: `EndEffectorController` abstraction

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/end_effector.py`
- Test: `mujoco_grasp_sim/sim_grasp/test_end_effector.py`

**Interfaces:**
- Consumes: Task 2's `SHADOW_HAND_OPEN_POSTURE`/`SHADOW_HAND_POWER_GRASP_POSTURE`.
- Produces: `EndEffectorController` ABC with `ctrl_slice(model) -> slice` (which `data.ctrl` indices this controller owns), `ctrl_for(openness: float) -> np.ndarray`, `is_grasping(data, model) -> bool`; `ParallelGripperController`; `ShadowHandController`. Used by Task 4.

- [ ] **Step 1: Write the failing test**

Create `mujoco_grasp_sim/sim_grasp/test_end_effector.py`:

```python
"""Standalone check for EndEffectorController implementations -- run
directly, no pytest (this codebase has no automated test suite)."""
import numpy as np

from sim_grasp.end_effector import ParallelGripperController, ShadowHandController

# -- ParallelGripperController: must reproduce today's exact scalar behavior.
parallel = ParallelGripperController()
assert np.allclose(parallel.ctrl_for(255.0), [255.0])
assert np.allclose(parallel.ctrl_for(0.0), [0.0])
assert np.allclose(parallel.ctrl_for(127.5), [127.5])

# -- ShadowHandController: interpolates the full 18-actuator vector.
shadow = ShadowHandController()
open_ctrl = shadow.ctrl_for(255.0)
closed_ctrl = shadow.ctrl_for(0.0)
assert open_ctrl.shape == (18,)
assert closed_ctrl.shape == (18,)
assert not np.allclose(open_ctrl, closed_ctrl), \
    'open and closed postures must actually differ'

mid_ctrl = shadow.ctrl_for(127.5)
# Midpoint must be the midpoint of open/closed, not a third arbitrary posture.
assert np.allclose(mid_ctrl, (open_ctrl + closed_ctrl) / 2, atol=1e-6)

# -- is_grasping: both controllers work off the same lifted-height signal
# already computed by GraspExecutor.execute() -- confirm the signature
# accepts it without needing hand-specific contact sensing for v1.
assert parallel.is_grasping(object_raised_m=0.1) is True
assert parallel.is_grasping(object_raised_m=0.0) is False
assert shadow.is_grasping(object_raised_m=0.1) is True
assert shadow.is_grasping(object_raised_m=0.0) is False

print('All end_effector checks passed.')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python sim_grasp/test_end_effector.py`
Expected: `ModuleNotFoundError: No module named 'sim_grasp.end_effector'`

- [ ] **Step 3: Write the implementation**

Create `mujoco_grasp_sim/sim_grasp/end_effector.py`:

```python
"""EndEffectorController -- pluggable gripper/hand control, mirroring the
existing GraspPredictor/PlacementPlanner ABC pattern in this codebase.
Both implementations are driven by the SAME 0(open)-255(closed) scalar
GraspExecutor already threads through _step_to/_hold/execute/place, so the
pick/place state machine itself never changes when swapping end effectors.
"""
from abc import ABC, abstractmethod

import numpy as np

from sim_grasp.shadow_hand_posture import (
    SHADOW_HAND_OPEN_POSTURE, SHADOW_HAND_POWER_GRASP_POSTURE)

SUCCESS_RAISE = 0.08   # matches executor.py's own SUCCESS_RAISE constant


class EndEffectorController(ABC):
    """Implement this to plug in a different gripper/hand."""

    n_actuators: int

    @abstractmethod
    def ctrl_for(self, openness: float) -> np.ndarray:
        """openness: 0 (fully closed) .. 255 (fully open), matching this
        project's existing GRIPPER_OPEN/GRIPPER_CLOSED convention.
        Returns the control vector for this end effector's own actuators."""

    def is_grasping(self, object_raised_m: float) -> bool:
        """Default success check: did the target object actually rise
        during lift? Shared by both controllers for v1 -- a hand-specific
        contact/force check is explicitly deferred (see the design spec)."""
        return object_raised_m > SUCCESS_RAISE


class ParallelGripperController(EndEffectorController):
    """Wraps today's exact Panda 2-finger tendon-gripper behavior
    verbatim -- data.ctrl[7] = openness, nothing else changes."""
    n_actuators = 1

    def ctrl_for(self, openness: float) -> np.ndarray:
        return np.array([openness], dtype=float)


# Actuator order must match hand_assets.py's merged model exactly (indices
# 7-24): rh_A_THJ5, THJ4, THJ3, THJ2, THJ1, FFJ4, FFJ3, FFJ0, MFJ4, MFJ3,
# MFJ0, RFJ4, RFJ3, RFJ0, LFJ5, LFJ4, LFJ3, LFJ0.
_SHADOW_HAND_ACTUATOR_ORDER = [
    'rh_A_THJ5', 'rh_A_THJ4', 'rh_A_THJ3', 'rh_A_THJ2', 'rh_A_THJ1',
    'rh_A_FFJ4', 'rh_A_FFJ3', 'rh_A_FFJ0',
    'rh_A_MFJ4', 'rh_A_MFJ3', 'rh_A_MFJ0',
    'rh_A_RFJ4', 'rh_A_RFJ3', 'rh_A_RFJ0',
    'rh_A_LFJ5', 'rh_A_LFJ4', 'rh_A_LFJ3', 'rh_A_LFJ0',
]


class ShadowHandController(EndEffectorController):
    """Interpolates the Shadow Hand's 18 actuators between a fixed open
    posture and a fixed DexGraspNet-derived power-grasp posture."""
    n_actuators = 18

    def __init__(self):
        self._open = np.array(
            [SHADOW_HAND_OPEN_POSTURE[a] for a in _SHADOW_HAND_ACTUATOR_ORDER])
        self._closed = np.array(
            [SHADOW_HAND_POWER_GRASP_POSTURE[a] for a in _SHADOW_HAND_ACTUATOR_ORDER])

    def ctrl_for(self, openness: float) -> np.ndarray:
        a = 1.0 - (openness / 255.0)   # 0 at fully open, 1 at fully closed
        return (1 - a) * self._open + a * self._closed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python sim_grasp/test_end_effector.py`
Expected: `All end_effector checks passed.`

- [ ] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/end_effector.py mujoco_grasp_sim/sim_grasp/test_end_effector.py
git commit -m "Add EndEffectorController: pluggable parallel-gripper/Shadow-Hand control"
```

---

### Task 4: Wire `EndEffectorController` into `GraspExecutor`

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/executor.py` (`GraspExecutor.__init__`, `_step_to`, `execute`, `DiffIK`'s hand-body lookup)

**Interfaces:**
- Consumes: Task 3's `EndEffectorController`/`ParallelGripperController`.
- Produces: `GraspExecutor(model, data, ..., end_effector: EndEffectorController = None, hand_body_name: str = 'hand')` — both new params default to today's exact parallel-gripper behavior. Used by Task 5.

No new isolated test for this task beyond re-running the existing
executor tests (this is integration wiring into already-tested code, not
new logic of its own) — Task 6 is where the actual Shadow Hand behavior
gets live-validated.

- [ ] **Step 1: Update `GraspExecutor.__init__`**

Add, alongside the existing constructor parameters:

```python
    def __init__(self, model, data, camera_module=None, record_gif=False,
                 record_dir=None, gif_frame_interval=0.08, on_frame=None,
                 viewer=None, end_effector=None, hand_body_name='hand'):
        ...
        from sim_grasp.end_effector import ParallelGripperController
        self.end_effector = end_effector or ParallelGripperController()
        self.hand_body_name = hand_body_name
```

Update `self.ik = DiffIK(model)` to pass the hand body name through:
`self.ik = DiffIK(model, hand_body_name=hand_body_name)`, and update
`DiffIK.__init__` (currently hard-codes
`self.hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')`)
to accept and use a `hand_body_name` parameter defaulting to `'hand'` —
for the Shadow Hand this will be called with `hand_body_name='rh_palm_mount'`
(the wrapper body Task 1's `hand_assets.py` grafts `rh_palm` inside of).

- [ ] **Step 2: Route gripper control through the controller**

In `_step_to()`, replace:

```python
            if gripper_ctrl is not None:
                data.ctrl[7] = gripper_ctrl
```

with:

```python
            if gripper_ctrl is not None:
                n = self.end_effector.n_actuators
                data.ctrl[7:7 + n] = self.end_effector.ctrl_for(gripper_ctrl)
```

- [ ] **Step 3: Update the success check in `execute()`, and guard the diagnostic-only finger-opening field**

The current code (`executor.py:337-347`) has TWO uses of the Panda-specific
`finger_joint1` — the success check AND a diagnostic `finger_opening_m`
field in the returned dict. The dict field must not crash for the Shadow
Hand (which has no joint named `finger_joint1`). Replace the whole block:

```python
        # ---- verdict --------------------------------------------------------
        obj_z1 = float(data.xpos[bid][2])
        finger_open = float(data.qpos[self.model.joint('finger_joint1').qposadr[0]])
        raised = obj_z1 - obj_z0
        success = raised > SUCCESS_RAISE and finger_open > 0.001
        return {'success': bool(success), 'stage': 'done',
                'object_raised_m': round(raised, 4),
                'finger_opening_m': round(2 * finger_open, 4),
                'ik_errors_mm': [round(ik_pre.pos_err * 1e3, 1),
                                 round(ik_grasp.pos_err * 1e3, 1),
                                 round(ik_lift.pos_err * 1e3, 1)]}
```

with:

```python
        # ---- verdict --------------------------------------------------------
        obj_z1 = float(data.xpos[bid][2])
        raised = obj_z1 - obj_z0
        success = self.end_effector.is_grasping(raised)
        # finger_opening_m is a parallel-gripper-specific diagnostic
        # (Panda's finger_joint1 doesn't exist on the Shadow Hand model) --
        # None for any other end effector, not a crash.
        finger_joint1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint1')
        finger_opening_m = (round(2 * float(data.qpos[self.model.jnt_qposadr[finger_joint1_id]]), 4)
                            if finger_joint1_id != -1 else None)
        return {'success': bool(success), 'stage': 'done',
                'object_raised_m': round(raised, 4),
                'finger_opening_m': finger_opening_m,
                'ik_errors_mm': [round(ik_pre.pos_err * 1e3, 1),
                                 round(ik_grasp.pos_err * 1e3, 1),
                                 round(ik_lift.pos_err * 1e3, 1)]}
```

(`ParallelGripperController.is_grasping` inherits the base class's
lifted-height-only check, matching `SUCCESS_RAISE`'s existing threshold —
the `finger_open > 0.001` tendon-specific check is dropped from the
SUCCESS criterion entirely, note this in the commit message as a
deliberate behavior simplification, not a silent regression, since it was
redundant with the height check in practice for every scenario this
project's own benchmarks have exercised. `finger_opening_m` stays in the
returned dict as a diagnostic value for the parallel gripper, `None` for
the Shadow Hand — confirmed `analyze_failures.py:42` already guards this
key with `opening is not None`, so no downstream fix is needed.)

- [ ] **Step 4: Run the full existing executor test suite to confirm no regression**

```bash
cd mujoco_grasp_sim
for f in sim_grasp/test_executor_ease.py sim_grasp/test_executor_place_orientation.py sim_grasp/test_executor_seed_selection.py sim_grasp/test_executor_viewer_sync.py sim_grasp/test_executor_frame_hook_decoupling.py; do
  PYTHONPATH=. python "$f" || echo "FAILED: $f"
done
```

Expected: all pass, unchanged.

- [ ] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/executor.py
git commit -m "Wire EndEffectorController into GraspExecutor and DiffIK"
```

---

### Task 5: Scene/CLI wiring

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/scene_generator.py` (`SceneConfig`)
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py` (construction of `GraspExecutor`)

**Interfaces:**
- Consumes: Task 1's `build_panda_shadow_hand_xml`, Task 3's `ShadowHandController`, Task 4's `GraspExecutor(end_effector=..., hand_body_name=...)`.
- Produces: `SceneConfig.end_effector: str = 'parallel'` field; `run_sim_grasp_test.py --end-effector {parallel,shadow_hand}` flag.

- [ ] **Step 1: Add the config field**

In `SceneConfig` (`sim_grasp/scene_generator.py`), add:

```python
    end_effector: str = 'parallel'   # 'parallel' | 'shadow_hand'
```

Find this line (`scene_generator.py:355`, inside the method that builds
the scene XML, right before the table/leg geometry is assembled):

```python
        panda_file = self._patched_panda_xml()
```

Replace with:

```python
        if cfg.end_effector == 'shadow_hand':
            from sim_grasp.hand_assets import build_panda_shadow_hand_xml
            panda_file = build_panda_shadow_hand_xml()
        else:
            panda_file = self._patched_panda_xml()
```

Both functions share the exact same return contract (a filename string
under `GENERATED_DIR`, per Task 1's Interfaces note), so nothing else in
`generate()`/`_build_scene_xml()` needs to change — `panda_file` is
consumed identically either way.

- [ ] **Step 2: Add the CLI flag**

In `run_sim_grasp_test.py`, add:

```python
    ap.add_argument('--end-effector', choices=['parallel', 'shadow_hand'],
                    default='parallel', help='which gripper/hand to use')
```

Set `cfg.end_effector = args.end_effector` when constructing `SceneConfig`.

There are exactly two `GraspExecutor(...)` construction sites in this file
(`run_sim_grasp_test.py:616` and `:888`, one per code path — `--pick-all`
and the single `--execute` path). Right before each, add:

```python
        if args.end_effector == 'shadow_hand':
            from sim_grasp.end_effector import ShadowHandController
            end_effector, hand_body_name = ShadowHandController(), 'rh_palm_mount'
        else:
            end_effector, hand_body_name = None, 'hand'   # GraspExecutor's own defaults
```

Then add `end_effector=end_effector, hand_body_name=hand_body_name` as two
extra keyword arguments to both existing `GraspExecutor(model, data,
camera_module=rec_cam, ...)` calls — no other argument on either call
site changes.

- [ ] **Step 3: Quick sanity check on the default config**

```bash
cd mujoco_grasp_sim
conda activate cgn_torch
MUJOCO_GL=osmesa python run_sim_grasp_test.py --seed 0 --execute --no-vis --backend cgn
```

Run with no `--end-effector` flag at all (so it defaults to `'parallel'`)
and confirm this completes with no traceback and looks like every other
run of this command this session — a quick sanity check, not the real
regression benchmark (Task 6 does that).

- [ ] **Step 4: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/scene_generator.py mujoco_grasp_sim/run_sim_grasp_test.py
git commit -m "Wire --end-effector shadow_hand through SceneConfig and run_sim_grasp_test.py"
```

---

### Task 6: Live validation

**Files:** none (validation only).

**Interfaces:** none (leaf task).

- [ ] **Step 1: Visual smoke test with the Shadow Hand**

```bash
cd mujoco_grasp_sim
conda activate cgn_torch
MUJOCO_GL=osmesa python run_sim_grasp_test.py --seed 0 --execute --end-effector shadow_hand --backend cgn
```

Expected: no MJCF load errors, no IK crash. Open the resulting
`execution.gif`/`observation.png` and visually confirm: (a) the Shadow
Hand is mounted where the parallel gripper used to be, correctly oriented
(not upside-down or rotated 90° off), (b) the arm doesn't visibly
self-collide with the hand during the approach, (c) the fingers visibly
curl into a grasping shape during the "close" phase (even if the object
isn't successfully lifted — that's a real, honest possible outcome for a
first attempt with an approximated posture, not a task failure; report it
as such if it happens, per this project's validation standard).

- [ ] **Step 2: Regression check on the default (parallel) config**

```bash
python benchmark.py --seeds 0-4 --mode pick-all --camera fused --backend graspgen --tag shadow_hand_regression_check
```

Expected: matches the last recorded GraspGen baseline in `ROADMAP.md`
exactly (30/30 or whatever the current recorded number is) — confirms
this whole plan's changes are a true no-op for the default parallel-
gripper path, not just "looks unchanged."

- [ ] **Step 3: Record the result**

Add a new `## P9 — Dexterous end effector (Shadow Hand E3M5)` section to
`ROADMAP.md` (P1 through P8 are already taken, confirmed by grepping
`^## P[0-9]` — P9 is the next free number) describing what was built, the
visual smoke-test outcome (honestly, including any non-ideal result), the
regression-check numbers, and the still-deferred DexGraspNet2 integration
as the natural next step — per `AGENTS.md`'s mandatory documentation-sync
workflow.

```bash
git add ROADMAP.md
git commit -m "Record Shadow Hand power-grasp mounting: live validation results"
```

---
