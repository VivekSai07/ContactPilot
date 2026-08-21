# Reasoning Layer Phase 1 (Option A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `run_sim_grasp_test.py --pick-all --instruction "..."` control
both pick order and placement destination from a natural-language
instruction, per the approved design in
`docs/superpowers/specs/2026-08-21-reasoning-layer-phase1-design.md`.

**Architecture:** Three new, independently-testable pieces, none modifying
already-shipped code: `sim_grasp/instruction_parser.py` (NIM API call +
JSON-schema validation into a `Step` list), `sim_grasp/spatial_relation_resolver.py`
(turns one step's spatial relation into a smaller, differently-positioned
`OccupancyPlacementPlanner` instance, or `None` for no bias — `plan()` and
`OccupancyPlacementPlanner.__init__` are never touched), and a
`--instruction` CLI flag wired into `run_sim_grasp_test.py`'s existing
`--pick-all` round loop.

**Tech Stack:** Python, NumPy, `urllib`/`json` (stdlib only for the NIM
call — no new dependency), existing `PromptSelector`/SAM 3 subprocess,
existing `OccupancyPlacementPlanner`.

## Global Constraints

- This codebase has **no pytest suite** — tests are standalone `test_*.py`
  scripts run directly (`python sim_grasp/test_name.py`, from
  `mujoco_grasp_sim/` with `PYTHONPATH=.`), plain `assert` statements,
  ending with a `print('All ... checks passed.')` line.
- `conda activate cgn_torch` before running anything in this repo.
- Branch: `reasoning-layer-phase1` (already created, holds the design spec
  commit — these tasks land as additional commits on the same branch).
- Do not modify `sim_grasp/placement_planner.py`,
  `sim_grasp/prompt_selector.py`, or `sim_grasp/executor.py` — every new
  piece consumes these unchanged (per the design's explicit goal of never
  touching already-shipped, already-reviewed P5/P7 code).
- `--instruction` only applies to `--pick-all`; reject it at
  argument-parsing time (before any scene/model setup) if `--pick-all` is
  not also given.
- No live NIM API calls inside any `test_*.py` file — `instruction_parser`'s
  pure JSON-validation logic must be separable from its HTTP call so it's
  testable without a network/API key.
- No real SAM 3 subprocess calls inside any `test_*.py` file —
  `spatial_relation_resolver`'s tests use a fake/stub object duck-typing
  `PromptSelector.select()`'s interface, not the real class.
- When `--instruction` is not given, every existing code path must be
  byte-for-byte unchanged (zero regression risk for every benchmark/test
  that doesn't pass it).

---

## Task 1: `sim_grasp/instruction_parser.py` — NIM call + JSON validation

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/instruction_parser.py`
- Create: `mujoco_grasp_sim/sim_grasp/test_instruction_parser.py`

**Interfaces:**
- Produces: `Step` (dataclass: `step: int`, `pick_target: str`,
  `place_relation: str`, `place_reference: str | None`),
  `parse_instruction(text: str, api_key: str | None = None) -> list[Step]`
  (raises `ValueError` on any schema violation, `RuntimeError` on API/
  network failure). Task 3 imports both names.
- Consumes: nothing from other tasks.

**Background:** the exact system prompt and JSON schema below are already
empirically validated (8/8 valid JSON+schema, ~0.7s latency) in
`docs/research/bakeoff_instruction_parser.py` — copy them verbatim, do not
redesign. `place_relation` is one of exactly:
`{"left_of", "right_of", "near", "center", "none"}`.

- [ ] **Step 1: Write the failing tests**

Create `mujoco_grasp_sim/sim_grasp/test_instruction_parser.py`:

```python
"""Standalone checks for instruction_parser.py -- run directly, no pytest
(this codebase has no automated test suite). No live NIM API calls here:
only the pure JSON-validation logic is exercised."""
import json

from sim_grasp.instruction_parser import Step, _parse_and_validate

VALID_TWO_STEP = json.dumps({"steps": [
    {"step": 1, "pick_target": "blue cube", "place_relation": "left_of", "place_reference": "bin"},
    {"step": 2, "pick_target": "red cube", "place_relation": "near", "place_reference": "blue cube"},
]})

steps = _parse_and_validate(VALID_TWO_STEP)
assert len(steps) == 2
assert steps[0] == Step(step=1, pick_target="blue cube",
                        place_relation="left_of", place_reference="bin")
assert steps[1] == Step(step=2, pick_target="red cube",
                        place_relation="near", place_reference="blue cube")

VALID_NONE_RELATION = json.dumps({"steps": [
    {"step": 1, "pick_target": "yellow block", "place_relation": "none", "place_reference": None},
]})
steps = _parse_and_validate(VALID_NONE_RELATION)
assert steps == [Step(step=1, pick_target="yellow block",
                      place_relation="none", place_reference=None)]

# Not valid JSON at all.
try:
    _parse_and_validate("not json at all")
    assert False, "expected ValueError for unparseable JSON"
except ValueError:
    pass

# Top-level array instead of {"steps": [...]} (the exact quirk the bake-off
# notes hit with response_format=json_object -- must be rejected, not
# silently misinterpreted).
try:
    _parse_and_validate(json.dumps([{"step": 1, "pick_target": "x",
                                     "place_relation": "none", "place_reference": None}]))
    assert False, "expected ValueError for a top-level array"
except ValueError:
    pass

# Missing the "steps" key.
try:
    _parse_and_validate(json.dumps({"wrong_key": []}))
    assert False, "expected ValueError for a missing 'steps' key"
except ValueError:
    pass

# Empty steps array.
try:
    _parse_and_validate(json.dumps({"steps": []}))
    assert False, "expected ValueError for an empty steps array"
except ValueError:
    pass

# A step missing a required key.
try:
    _parse_and_validate(json.dumps({"steps": [
        {"step": 1, "pick_target": "x", "place_relation": "none"}]}))  # no place_reference
    assert False, "expected ValueError for a step missing place_reference"
except ValueError:
    pass

# An invalid place_relation value.
try:
    _parse_and_validate(json.dumps({"steps": [
        {"step": 1, "pick_target": "x", "place_relation": "on_top_of",
         "place_reference": "bin"}]}))
    assert False, "expected ValueError for an invalid place_relation"
except ValueError:
    pass

# Wrong type for "step" (must be an int, not a string).
try:
    _parse_and_validate(json.dumps({"steps": [
        {"step": "1", "pick_target": "x", "place_relation": "none",
         "place_reference": None}]}))
    assert False, "expected ValueError for a non-int step"
except ValueError:
    pass

print('All instruction_parser checks passed.')
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. python sim_grasp/test_instruction_parser.py` from
`mujoco_grasp_sim/`. Expected: `ModuleNotFoundError: No module named
'sim_grasp.instruction_parser'` (or `ImportError` for `_parse_and_validate`).

- [ ] **Step 3: Implement `instruction_parser.py`**

```python
"""Instruction parser: turns a free-text pick-and-place instruction into
an ordered list of Steps via NVIDIA's OpenAI-compatible NIM endpoint
(meta/llama-3.1-8b-instruct -- the only bake-off candidate confirmed to
support schema-enforced JSON output, see
docs/research/2026-08-20-reasoning-layer-reflectvlm.md and
docs/research/bakeoff_instruction_parser.py for the empirical validation).

SYSTEM_PROMPT below is kept in sync BY HAND with
docs/research/bakeoff_instruction_parser.py's copy -- that script is
deliberately standalone/stdlib-only (no cgn_torch env needed to rerun the
bake-off), so it cannot import this module without pulling in
sim_grasp/__init__.py's full mujoco/cv2 import chain. Update both files
together if the parsing contract changes.
"""
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-8b-instruct"
VALID_RELATIONS = {"left_of", "right_of", "near", "center", "none"}

SYSTEM_PROMPT = """You are a robotic task planner. Translate a human's \
pick-and-place instruction into a structured execution sequence.

Output ONLY a JSON object, nothing else -- no chat filler, no markdown code \
fences, no explanation. The object must have exactly one key, "steps", whose \
value is a JSON array with one element per pick-and-place action described \
in the instruction (an instruction describing N objects to pick produces N \
elements, in the order they should be executed).

Each element of "steps" must have exactly these keys:
  "step": integer, 1-indexed
  "pick_target": string, a short description of the object to pick
  "place_relation": one of "left_of", "right_of", "near", "center", "none"
  "place_reference": string naming what the relation is relative to (e.g.
    "bin", or another object's description), or null if place_relation is
    "none"

If the instruction asks to stack one object on another, use place_relation
"near" with place_reference set to the object it should go near (stacking
is not supported; treat it as "put it next to that object" instead).

Example for a two-object instruction:
{"steps": [
  {"step": 1, "pick_target": "blue cube", "place_relation": "left_of", "place_reference": "bin"},
  {"step": 2, "pick_target": "red cube", "place_relation": "near", "place_reference": "blue cube"}
]}

Each distinct object mentioned in the instruction gets exactly ONE step, in
the order it should be picked. Never repeat the same pick_target in two
different steps -- if only one object is mentioned, output exactly one step,
even if the instruction also describes where to place it:
{"steps": [
  {"step": 1, "pick_target": "red cube", "place_relation": "left_of", "place_reference": "bin"}
]}
"""


@dataclass
class Step:
    step: int
    pick_target: str
    place_relation: str
    place_reference: 'str | None'


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE per line, '#' comments) -- avoids a
    python-dotenv dependency, copied from
    docs/research/bakeoff_instruction_parser.py for the same reason
    SYSTEM_PROMPT is copied rather than imported."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _parse_and_validate(raw: str) -> list[Step]:
    """Pure JSON-validation logic, no network access -- kept separate from
    parse_instruction() so it's testable without an API key."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f'model response is not valid JSON: {e}') from e
    if not isinstance(obj, dict):
        raise ValueError(f'expected a JSON object with a "steps" key, got '
                         f'{type(obj).__name__}')
    if 'steps' not in obj:
        raise ValueError('model response is missing the "steps" key')
    steps_raw = obj['steps']
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ValueError('"steps" must be a non-empty JSON array')

    required_keys = {'step', 'pick_target', 'place_relation', 'place_reference'}
    steps = []
    for i, s in enumerate(steps_raw):
        if not isinstance(s, dict) or set(s.keys()) != required_keys:
            raise ValueError(f'step {i} has the wrong keys: {s!r} '
                             f'(expected exactly {sorted(required_keys)})')
        if not isinstance(s['step'], int):
            raise ValueError(f'step {i}: "step" must be an int, got '
                             f'{type(s["step"]).__name__}')
        if not isinstance(s['pick_target'], str):
            raise ValueError(f'step {i}: "pick_target" must be a string')
        if s['place_relation'] not in VALID_RELATIONS:
            raise ValueError(f'step {i}: "place_relation" {s["place_relation"]!r} '
                             f'not one of {sorted(VALID_RELATIONS)}')
        if s['place_reference'] is not None and not isinstance(s['place_reference'], str):
            raise ValueError(f'step {i}: "place_reference" must be a string or null')
        steps.append(Step(step=s['step'], pick_target=s['pick_target'],
                          place_relation=s['place_relation'],
                          place_reference=s['place_reference']))
    return steps


def parse_instruction(text: str, api_key: 'str | None' = None) -> list[Step]:
    """Parses a free-text instruction into an ordered Step list via the
    NIM API. Raises ValueError (bad schema) or RuntimeError (network/API
    failure) -- callers should let both propagate and abort the run, per
    this project's 'fail loudly, no silent fallback' decision for this
    call (see docs/superpowers/specs/2026-08-21-reasoning-layer-phase1-design.md)."""
    if api_key is None:
        _load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')
        api_key = os.environ.get('NVIDIA_API_KEY')
    if not api_key:
        raise RuntimeError('NVIDIA_API_KEY not found in the environment or .env -- '
                           'required for --instruction')

    payload = json.dumps({
        'model': MODEL,
        'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': text}],
        'temperature': 0.2,
        'max_tokens': 1024,
        'stream': False,
        'response_format': {'type': 'json_object'},
    }).encode('utf-8')
    req = urllib.request.Request(
        NIM_URL, data=payload, method='POST',
        headers={'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=90.0) as resp:
            body = json.loads(resp.read().decode('utf-8'))
    except urllib.error.URLError as e:
        raise RuntimeError(f'NIM API call failed: {e}') from e
    content = body['choices'][0]['message']['content']
    return _parse_and_validate(content)
```

- [ ] **Step 4: Run the test again to verify it passes**

Run: `PYTHONPATH=. python sim_grasp/test_instruction_parser.py` from
`mujoco_grasp_sim/`. Expected: `All instruction_parser checks passed.`

- [ ] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/instruction_parser.py \
       mujoco_grasp_sim/sim_grasp/test_instruction_parser.py
git commit -m "Add instruction_parser: NIM-backed instruction -> Step list parsing"
```

---

## Task 2: `sim_grasp/spatial_relation_resolver.py` — relation → placement bias

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/spatial_relation_resolver.py`
- Create: `mujoco_grasp_sim/sim_grasp/test_spatial_relation_resolver.py`

**Interfaces:**
- Consumes: `sim_grasp.placement_planner.OccupancyPlacementPlanner`
  (unmodified), `sim_grasp.pointcloud.depth_to_pointcloud`,
  `sim_grasp.frames.transform_points` (both unmodified, already used
  elsewhere in this codebase).
- Produces:
  `resolve(place_relation: str, place_reference: str | None, bin_center:
  tuple[float, float], bin_inner_half: float, T_world_cam: np.ndarray, rgb:
  'np.ndarray | None' = None, depth: 'np.ndarray | None' = None, K:
  'np.ndarray | None' = None, prompt_selector=None, work_dir: 'str |
  Path' = '.') -> 'OccupancyPlacementPlanner | None'`. Task 3 imports this
  name and calls it with the real `PromptSelector` instance.

**Background:** `OccupancyPlacementPlanner` only accepts a scalar
`bin_inner_half` (a square region) — it cannot represent a true half-bin
rectangle. Every relation therefore resolves to a smaller **square**
sub-region of half-size `bin_inner_half / 2`, differing only in where
that square's center sits (see the design spec's "Per-relation
sub-region" section for the exact geometry and why no clipping is needed
for `left_of`/`right_of`/`center` but is needed for `near`).

- [ ] **Step 1: Write the failing tests**

Create `mujoco_grasp_sim/sim_grasp/test_spatial_relation_resolver.py`:

```python
"""Standalone checks for spatial_relation_resolver.py -- run directly, no
pytest. No real SAM 3 subprocess: near-relation tests use a fake
PromptSelector duck-typing .select()'s interface."""
import numpy as np

from sim_grasp.spatial_relation_resolver import resolve, _camera_view_axis
from sim_grasp.placement_planner import BinHeightmap, ObjectFootprint

BIN_CENTER = (0.45, -0.30)
BIN_INNER_HALF = 0.12

# ---- _camera_view_axis --------------------------------------------------
# Camera's local +X (image-right) aligned with world +X.
T_cam_x_right = np.eye(4)
axis, sign = _camera_view_axis(T_cam_x_right)
assert axis == 'x' and sign == 1.0

# Camera's local +X aligned with world -Y.
T_cam_neg_y = np.eye(4)
T_cam_neg_y[:3, 0] = [0.0, -1.0, 0.0]
axis, sign = _camera_view_axis(T_cam_neg_y)
assert axis == 'y' and sign == -1.0

print('All spatial_relation_resolver camera-axis checks passed.')

# ---- relation -> sub-region planner -------------------------------------
n = 48
empty_heights = np.full((n, n), 0.75, dtype=np.float32)
empty_map = BinHeightmap(heights=empty_heights, origin_xy=(0.33, -0.42),
                         cell_size=0.005, floor_z=0.75)
small_footprint = ObjectFootprint(center_xy=(0.0, 0.0), size_xy=(0.03, 0.03),
                                  yaw=0.0, z_bottom=0.75, z_top=0.80)

# "none" -> no wrapper needed.
assert resolve('none', None, BIN_CENTER, BIN_INNER_HALF, T_cam_x_right) is None

# "center" -> a valid sub-region planner whose plan() stays within
# bin_inner_half/2 of bin_center on both axes.
planner = resolve('center', None, BIN_CENTER, BIN_INNER_HALF, T_cam_x_right)
assert planner is not None
pose = planner.plan(small_footprint, empty_map)
assert pose is not None
assert abs(pose.x - BIN_CENTER[0]) <= BIN_INNER_HALF / 2 + 1e-6
assert abs(pose.y - BIN_CENTER[1]) <= BIN_INNER_HALF / 2 + 1e-6

# "right_of" with camera-right = world +X -> placement biased to x > bin_center[0].
planner = resolve('right_of', 'bin', BIN_CENTER, BIN_INNER_HALF, T_cam_x_right)
pose = planner.plan(small_footprint, empty_map)
assert pose is not None and pose.x > BIN_CENTER[0], f'x={pose.x}'

# "left_of" with the same camera -> placement biased to x < bin_center[0].
planner = resolve('left_of', 'bin', BIN_CENTER, BIN_INNER_HALF, T_cam_x_right)
pose = planner.plan(small_footprint, empty_map)
assert pose is not None and pose.x < BIN_CENTER[0], f'x={pose.x}'

print('All spatial_relation_resolver region checks passed.')

# ---- "near" with a fake PromptSelector ----------------------------------
class _FakeSelectionResult:
    def __init__(self, masks, scores):
        self.masks, self.scores = masks, scores
    @property
    def is_empty(self):
        return len(self.scores) == 0

class _FakePromptSelector:
    """Returns one full-frame mask; depth/K/T_world_cam below are rigged
    so that mask's world centroid lands at a known point inside the bin."""
    def __init__(self, mask):
        self._mask = mask
    def select(self, rgb, prompt=None, work_dir='.'):
        return _FakeSelectionResult(masks=np.array([self._mask]),
                                    scores=np.array([0.95], dtype=np.float32))

H, W = 20, 20
fx = fy = 100.0
cx = cy = 10.0
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
T_world_cam_topdown = np.array([
    [1.0, 0.0, 0.0, BIN_CENTER[0]],
    [0.0, -1.0, 0.0, BIN_CENTER[1]],
    [0.0, 0.0, -1.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)
depth = np.full((H, W), 0.25, dtype=np.float32)   # world z = 1.0 - 0.25 = 0.75
mask = np.zeros((H, W), dtype=bool)
mask[9:11, 9:11] = True   # centered at the principal point -> world (0.45, -0.30)
rgb = np.zeros((H, W, 3), dtype=np.uint8)

fake_selector = _FakePromptSelector(mask)
planner = resolve('near', 'blue cube', BIN_CENTER, BIN_INNER_HALF,
                  T_world_cam_topdown, rgb=rgb, depth=depth, K=K,
                  prompt_selector=fake_selector)
assert planner is not None
pose = planner.plan(small_footprint, empty_map)
assert pose is not None
assert abs(pose.x - BIN_CENTER[0]) < 0.02 and abs(pose.y - BIN_CENTER[1]) < 0.02, \
    f'expected a placement near the detected centroid, got ({pose.x}, {pose.y})'

# "near" with no match (empty selection) -> falls back to None.
class _EmptyFakeSelector:
    def select(self, rgb, prompt=None, work_dir='.'):
        return _FakeSelectionResult(masks=np.zeros((0, H, W), dtype=bool),
                                    scores=np.zeros((0,), dtype=np.float32))

planner = resolve('near', 'nonexistent object', BIN_CENTER, BIN_INNER_HALF,
                  T_world_cam_topdown, rgb=rgb, depth=depth, K=K,
                  prompt_selector=_EmptyFakeSelector())
assert planner is None

print('All spatial_relation_resolver near-relation checks passed.')
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. python sim_grasp/test_spatial_relation_resolver.py` from
`mujoco_grasp_sim/`. Expected: `ModuleNotFoundError` for
`sim_grasp.spatial_relation_resolver`.

- [ ] **Step 3: Implement `spatial_relation_resolver.py`**

```python
"""Turns one instruction Step's spatial relation into a smaller,
differently-positioned OccupancyPlacementPlanner (or None for no bias).

OccupancyPlacementPlanner only accepts a scalar bin_inner_half (a square
region) -- it cannot represent a true half-bin rectangle. Every relation
therefore resolves to a smaller SQUARE sub-region of half-size
bin_inner_half / 2, differing only in where that square's center sits.
This keeps placement_planner.py completely untouched -- see
docs/superpowers/specs/2026-08-21-reasoning-layer-phase1-design.md for
the full rationale and geometry.
"""
from pathlib import Path

import numpy as np

from sim_grasp.frames import transform_points
from sim_grasp.placement_planner import OccupancyPlacementPlanner
from sim_grasp.pointcloud import depth_to_pointcloud


def _camera_view_axis(T_world_cam: np.ndarray) -> 'tuple[str, float]':
    """World axis ('x' or 'y') most aligned with the camera's local +X
    (image-right, OpenCV convention -- see frames.py) direction, and the
    sign of that alignment (+1 if image-right points toward +axis)."""
    cam_right_world = T_world_cam[:3, 0]
    x_comp, y_comp = float(cam_right_world[0]), float(cam_right_world[1])
    if abs(x_comp) >= abs(y_comp):
        return 'x', (1.0 if x_comp >= 0 else -1.0)
    return 'y', (1.0 if y_comp >= 0 else -1.0)


def resolve(place_relation: str, place_reference: 'str | None',
           bin_center: 'tuple[float, float]', bin_inner_half: float,
           T_world_cam: np.ndarray,
           rgb: 'np.ndarray | None' = None,
           depth: 'np.ndarray | None' = None,
           K: 'np.ndarray | None' = None,
           prompt_selector=None,
           work_dir='.') -> 'OccupancyPlacementPlanner | None':
    if place_relation == 'none':
        return None

    half = bin_inner_half / 2.0
    bx, by = bin_center

    if place_relation == 'center':
        return OccupancyPlacementPlanner(bin_center=(bx, by), bin_inner_half=half)

    if place_relation in ('left_of', 'right_of'):
        axis, sign = _camera_view_axis(T_world_cam)
        direction = sign if place_relation == 'right_of' else -sign
        if axis == 'x':
            center = (bx + direction * half, by)
        else:
            center = (bx, by + direction * half)
        return OccupancyPlacementPlanner(bin_center=center, bin_inner_half=half)

    if place_relation == 'near':
        if prompt_selector is None or rgb is None or depth is None or K is None:
            print(f'[spatial-relation] "near {place_reference}" needs rgb/depth/K/'
                 'a PromptSelector -- falling back to unbiased placement')
            return None
        result = prompt_selector.select(rgb, prompt=place_reference, work_dir=work_dir)
        if result.is_empty:
            print(f'[spatial-relation] no match for "near {place_reference}" -- '
                 'falling back to unbiased placement')
            return None
        idx = int(np.argmax(result.scores))
        mask = result.masks[idx]
        pts_cam = depth_to_pointcloud(depth, K, mask=mask)
        if len(pts_cam) == 0:
            print(f'[spatial-relation] "near {place_reference}" matched a mask with '
                 'no valid depth -- falling back to unbiased placement')
            return None
        pts_world = transform_points(T_world_cam, pts_cam)
        cx, cy = float(pts_world[:, 0].mean()), float(pts_world[:, 1].mean())
        # Clamp so the sub-region stays fully inside the full bin's own bounds
        # -- unlike left_of/right_of/center, an arbitrary detected centroid
        # near an edge needs this explicit clamp (see design spec).
        cx = float(np.clip(cx, bx - bin_inner_half + half, bx + bin_inner_half - half))
        cy = float(np.clip(cy, by - bin_inner_half + half, by + bin_inner_half - half))
        return OccupancyPlacementPlanner(bin_center=(cx, cy), bin_inner_half=half)

    raise ValueError(f'unknown place_relation: {place_relation!r}')
```

- [ ] **Step 4: Run the test again to verify it passes**

Run: `PYTHONPATH=. python sim_grasp/test_spatial_relation_resolver.py` from
`mujoco_grasp_sim/`. Expected: all three `All ... checks passed.` lines.

- [ ] **Step 5: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/spatial_relation_resolver.py \
       mujoco_grasp_sim/sim_grasp/test_spatial_relation_resolver.py
git commit -m "Add spatial_relation_resolver: relation -> sub-region placement bias"
```

---

## Task 3: `--instruction` CLI wiring in `run_sim_grasp_test.py`

**Files:**
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py`

**Interfaces:**
- Consumes: `sim_grasp.instruction_parser.{Step, parse_instruction}` (Task
  1), `sim_grasp.spatial_relation_resolver.resolve` (Task 2),
  `sim_grasp.prompt_selector.{PromptSelector, resolve_real_label}`
  (existing, unmodified — same import already used by the `--prompt` path
  around line 344).
- Produces: the `--instruction TEXT` CLI flag; no new importable names
  (this is the integration task, nothing downstream depends on it).

**Step 1: Add the CLI flag and its validation**

Find the `ap.add_argument('--pick-object', ...)` block (around line 242)
and add immediately after it:

```python
    ap.add_argument('--instruction', type=str, default=None,
                    help='natural-language pick-and-place instruction '
                         '(e.g. "pick the blue cube first and put it on '
                         'the left, then the red one on the right"). '
                         'Requires --pick-all and NVIDIA_API_KEY (see '
                         '.env.example). Parsed once via NIM before the '
                         'round loop starts.')
```

Find the existing mutual-exclusion checks right after `args = ap.parse_args()`
(around line 279: `if args.pick_all: args.execute = True`) and add:

```python
    if args.instruction is not None and not args.pick_all:
        sys.exit('[instruction] --instruction requires --pick-all')
    if args.instruction is not None and (args.prompt or args.click or args.box):
        sys.exit('[instruction] --instruction and --prompt/--click/--box are '
                 'mutually exclusive — the instruction itself selects objects.')
```

- [ ] **Step 1 done**

**Step 2: Parse the instruction once, before the round loop**

Find the `if args.pick_all:` block that starts the pick-all section (the
one that constructs `executor = GraspExecutor(...)`, around line 550) and
add, immediately before `placement_planner = OccupancyPlacementPlanner(...)`:

```python
        steps = None
        if args.instruction:
            from sim_grasp.instruction_parser import parse_instruction
            steps = parse_instruction(args.instruction)
            print(f'[instruction] parsed {len(steps)} step(s):')
            for s in steps:
                print(f'  [{s.step}] pick {s.pick_target!r} -> '
                     f'{s.place_relation} {s.place_reference!r}')
        step_idx = 0            # index into `steps` of the currently-active step
        step_miss_count = 0     # consecutive rounds the active step failed to match
        instr_selector = None
        if steps is not None:
            from sim_grasp.prompt_selector import PromptSelector, resolve_real_label
            instr_selector = PromptSelector(sam3_python=args.sam3_python)
```

(`parse_instruction` raising `ValueError`/`RuntimeError` here propagates
and aborts the run — no try/except added, per the "fail loudly" decision.)

- [ ] **Step 2 done**

**Step 3: Track the current round's RGB alongside the existing `obs` tuple**

Find `obs = (depth, segmap, K)   # observation behind the current grasps`
(right before the round loop's `for rnd in range(1, max_rounds + 1):`) and
change it to also track `rgb`:

```python
        obs = (depth, segmap, K)   # observation behind the current grasps
        cur_rgb = rgb              # this round's RGB (for --instruction SAM 3 matching)
```

Inside the loop's `else:` branch (the re-observe path, where `obs = (depth_r,
segmap_r, K_r)` is set), add right after it:

```python
                obs = (depth_r, segmap_r, K_r)
                cur_rgb = rgb_r
```

- [ ] **Step 3 done**

**Step 4: Resolve the current step's target instead of ranking all remaining objects**

Find the block:

```python
            in_bin_now = set(gen.objects_in_bin())
            remaining = [n for n in gen.objects_on_table()
                         if n not in in_bin_now and fail_count.get(n, 0) < 3]
            if args.pick_object is not None:        # P3: only the chosen one
                remaining = [n for n in remaining
                             if label_of[n] == args.pick_object]
            if not remaining:
                print(f'[pick-all] round {rnd}: nothing left to pick')
                break
            allowed = {label_of[n] for n in remaining}
```

Replace it with:

```python
            in_bin_now = set(gen.objects_in_bin())
            remaining = [n for n in gen.objects_on_table()
                         if n not in in_bin_now and fail_count.get(n, 0) < 3]
            if args.pick_object is not None:        # P3: only the chosen one
                remaining = [n for n in remaining
                             if label_of[n] == args.pick_object]
            if not remaining:
                print(f'[pick-all] round {rnd}: nothing left to pick')
                break

            active_step = None
            if steps is not None:
                while step_idx < len(steps):
                    active_step = steps[step_idx]
                    result = instr_selector.select(cur_rgb, prompt=active_step.pick_target)
                    matched_label = None
                    if not result.is_empty:
                        idx = int(np.argmax(result.scores))
                        matched_label = resolve_real_label(obs[1], result.masks[idx])
                    if matched_label is not None and matched_label in {label_of[n] for n in remaining}:
                        allowed = {matched_label}
                        step_miss_count = 0
                        break
                    step_miss_count += 1
                    print(f'[instruction] step {active_step.step} '
                         f'({active_step.pick_target!r}) not matched — '
                         f'attempt {step_miss_count}/3')
                    if step_miss_count < 3:
                        allowed = set()   # retry the same step next round
                        break
                    step_miss_count = 0
                    step_idx += 1         # give up on this step, try the next
                    active_step = None
                else:
                    print(f'[pick-all] round {rnd}: no instruction steps left to resolve')
                    break
                if not allowed:
                    continue   # retry the same (still-active) step next round
            else:
                allowed = {label_of[n] for n in remaining}
```

This mirrors the existing `fail_count`-based 3-attempt retry budget:
`step_miss_count` increments per round the active step's `pick_target`
doesn't resolve to an object still on the table; after 3 misses the
pointer permanently advances to the next step (a transient miss — one
round's occlusion — gets retried on the next round's fresh observation
first). With no `--instruction`, `steps is None` and this is exactly
today's `allowed = {label_of[n] for n in remaining}` — unchanged.

- [ ] **Step 4 done**

**Step 5: Bias placement by the active step's spatial relation**

Find the placement block:

```python
            place_pose = None
            if footprint is not None:
                try:
                    heightmap = build_bin_heightmap(
                        d_o, seg_o, K_o, T_wc, cfg.bin_center, cfg.bin_inner_half,
                        exclude_seg_id=int(sid))
                    place_pose = placement_planner.plan(footprint, heightmap)
                except ValueError as e:
                    print(f'[placement] heightmap build failed: {e}')
```

Replace it with:

```python
            place_pose = None
            if footprint is not None:
                try:
                    heightmap = build_bin_heightmap(
                        d_o, seg_o, K_o, T_wc, cfg.bin_center, cfg.bin_inner_half,
                        exclude_seg_id=int(sid))
                    if active_step is not None and active_step.place_relation != 'none':
                        from sim_grasp.spatial_relation_resolver import resolve as resolve_relation
                        scoped_planner = resolve_relation(
                            active_step.place_relation, active_step.place_reference,
                            cfg.bin_center, cfg.bin_inner_half, T_wc,
                            rgb=cur_rgb, depth=d_o, K=K_o,
                            prompt_selector=instr_selector, work_dir=save_dir)
                        if scoped_planner is not None:
                            place_pose = scoped_planner.plan(footprint, heightmap)
                    if place_pose is None:
                        place_pose = placement_planner.plan(footprint, heightmap)
                except ValueError as e:
                    print(f'[placement] heightmap build failed: {e}')
```

(When `steps is None`, `active_step` is always `None`, so this is exactly
today's single `placement_planner.plan(...)` call — unchanged.)

- [ ] **Step 5 done**

**Step 6: Run the full existing test suite (no regressions)**

Run each of these from `mujoco_grasp_sim/` (`PYTHONPATH=. python
sim_grasp/<name>.py`): `test_placement_planner`, `test_executor_ease`,
`test_executor_place_orientation`, `test_color_utils`,
`test_resolve_real_label`, `test_scene_generator_paths`,
`test_feasibility`, `test_instruction_parser`,
`test_spatial_relation_resolver`. All must print their `All ... passed.`
line with no traceback.

- [ ] **Step 6 done**

**Step 7: Live smoke test**

Run (from `mujoco_grasp_sim/`, with `cgn_torch` active and
`NVIDIA_API_KEY` set in `.env`):

```bash
MUJOCO_GL=osmesa GRASPGEN_PYTHON=/home/vivek/miniconda3/envs/graspgen_torch/bin/python \
  python run_sim_grasp_test.py --pick-all --camera fused --backend graspgen \
  --seed 0 --no-vis \
  --instruction "Pick up any cube and place it in the center of the bin."
```

Confirm: the `[instruction] parsed N step(s):` line prints, at least one
`[pick-all] round ...` line shows a pick proceeding, and the run finishes
with a `[pick-all] DONE:` summary (not a crash). Then run a second smoke
test with a two-step, relation-bearing instruction (e.g. `"Pick the first
cube and put it on the left, then pick another cube and put it on the
right."`) and visually confirm via `output/<run>/execution.gif` that the
two placements land on visibly different sides of the bin.

- [ ] **Step 7 done**

**Step 8: Commit**

```bash
git add mujoco_grasp_sim/run_sim_grasp_test.py
git commit -m "Wire --instruction into --pick-all: instruction-driven pick order + placement bias"
```

---

## Task 4: Documentation update — results, ROADMAP, spec reconciliation

**Files:**
- Modify: `ROADMAP.md` (P8 section)
- Modify: `docs/research/2026-08-20-reasoning-layer-reflectvlm.md` ("Next
  step (not yet done)" section)
- Modify: `docs/superpowers/specs/2026-08-21-reasoning-layer-phase1-design.md`
  (reconcile against whatever changed during implementation, if anything)

**Step 1: Record what actually got built and its live-smoke-test results**

- `ROADMAP.md` P8: change the `[PLANNED, not started]` header tag to
  reflect Phase 1 being implemented, and add a dated bullet (matching the
  style of existing dated P1/P7 bullets) summarizing: the three new
  modules, the `--instruction` flag, and the two live smoke tests from
  Task 3 Step 7 (quote their actual printed `[instruction] parsed N
  step(s)` output and the `[pick-all] DONE:` summary, and confirm by eye
  from `execution.gif` whether the two-step relation-bearing smoke test's
  placements landed on visibly different sides of the bin — state this
  plainly, don't guess).
- `docs/research/2026-08-20-reasoning-layer-reflectvlm.md`: replace the
  "Next step (not yet done)" section with a short "Phase 1 implemented
  (2026-08-21)" note pointing at the spec and plan files, so the doc
  doesn't keep telling readers the architecture is unfleshed once it
  isn't.
- `docs/superpowers/specs/2026-08-21-reasoning-layer-phase1-design.md`:
  if implementation diverged from the spec in any way (e.g. a signature
  changed, an edge case behaved differently than described), add a short
  "Implementation notes (2026-08-21)" section at the end documenting the
  divergence and why — do not silently edit the original design
  sections. If nothing diverged, state that explicitly in the same
  section instead of leaving the spec looking unreviewed.

- [ ] **Step 1: Update all three docs with real results (no placeholder
      numbers/guesses)**

**Step 2: Commit**

```bash
git add ROADMAP.md docs/research/2026-08-20-reasoning-layer-reflectvlm.md \
       docs/superpowers/specs/2026-08-21-reasoning-layer-phase1-design.md
git commit -m "Document reasoning-layer Phase 1 results in ROADMAP.md and research docs"
```

---

## Plan Self-Review Notes

- **Spec coverage**: Task 1 covers the design's `instruction_parser.py`
  section; Task 2 covers `spatial_relation_resolver.py` (including the
  square-region-only geometry fix and the `near`-relation clamp); Task 3
  covers the `--pick-all` round-loop wiring, the step-retry budget, and
  both required live smoke tests. Task 4 covers the user's explicit
  request to update documentation (results, architecture status) once
  implementation is done. All three "Testing" bullets from the spec are
  covered (schema validation, resolver unit tests with a fake selector,
  one live end-to-end smoke test — Task 3 adds a second smoke test for
  the relation-bearing case, strictly more coverage than the spec
  required).
- **Type consistency checked**: `Step`'s four fields, `parse_instruction`'s
  signature, and `resolve`'s full signature are identical everywhere they
  appear across Tasks 1-3.
- **No placeholders**: every step above contains complete, runnable code —
  no "add error handling here" or "similar to Task N" placeholders.
