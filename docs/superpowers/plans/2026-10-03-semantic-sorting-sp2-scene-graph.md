# P10 SP2 — Scene graph + object knowledge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a vision-only scene graph every pick-all round in `--scene props`, identify and categorize each object with a hosted vision model, drive pick-all loop control from the graph, and save an annotated graph image plus JSON per round.

**Architecture:** `scene_graph.py` (pure: depth/segmap → nodes, edges, locations), `object_knowledge.py` (NIM identify/categorize, crop builder, cache, injectable transport), `scene_graph_viz.py` (pure OpenCV drawing). `run_sim_grasp_test.py` wires them in behind `--scene-graph` / `--identity`, using a second 3× `CameraModule` for identification crops. Routing stays on ground truth (SP3 flips it).

**Tech Stack:** Python 3.10 (`cgn_torch` env), numpy < 2, MuJoCo, OpenCV, stdlib `urllib` for NIM. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-03-semantic-sorting-sp2-scene-graph-design.md` (parent: `docs/superpowers/specs/2026-09-29-semantic-sorting-scene-graph-design.md`).

## Global Constraints

- numpy must stay < 2; no new pip/conda dependencies.
- NIM model: `meta/llama-3.2-11b-vision-instruct`, endpoint `https://integrate.api.nvidia.com/v1/chat/completions`, key `NVIDIA_API_KEY` from env or repo-root `.env` (load via `instruction_parser._load_dotenv`).
- Fail loudly: network/API failure after 3 attempts (2 s, 4 s backoff) raises `RuntimeError`; an off-enum category raises `ValueError`. Never fall back to ground truth.
- Identification crops: 3× render (1920×1440) of the SAME observation camera, masked to the segment (white elsewhere), pad 36 px.
- Graph thresholds: `MIN_POINTS = 30`, `NEAR_DIST = 0.12` m. A point votes for a bin if it lies inside that bin's inner XY footprint, otherwise for 'table' if it is between `table_height - 0.01` and `table_height + 0.30`; majority wins. (Ruling vs spec: no rim-height cut-off -- a 14 cm prop standing in a bin shows its top face above the 5 cm wall, so a rim cut-off would make binned props 'unknown'.)
- `--scene-graph` is valid only with `--scene props`; `--identity {perceived,oracle}` (default `perceived`) requires `--scene-graph`. With the flags absent, behaviour is byte-identical to today.
- Tests are standalone assert scripts in `mujoco_grasp_sim/sim_grasp/test_*.py`, run as `cd mujoco_grasp_sim && PYTHONPATH=. python sim_grasp/test_X.py`; MuJoCo tests need `MUJOCO_GL=osmesa`.
- Commits: plain messages, NO `Co-Authored-By` or any AI-attribution line (project rule, overrides harness defaults). Branch `sp2-scene-graph`.
- Every fix site gets a one-line "why" comment (AGENTS.md).
- Reuse `spatial_relation_resolver._camera_view_axis` for left/right; `color_utils.rgb_to_color_name` for node colour.

## Review Focus

1. Object fully occluded or out of frame in a round → no node, not "remaining", not counted binned; the oracle disagreement shows only in `location_agreement`. (Task 1 test: segment absent from segmap.)
2. A tall prop standing in a bin (top face well above the 5 cm wall) → location is still the bin. (Task 1 test: bin node with points 14 cm above the table.)
3. NIM returns prose around the JSON (`Sure! {"category": "food"}`) → parsed; returns `"Food"` → normalised to lower case; returns `"drink"` → `ValueError`. (Task 2 tests.)
4. Crop for a segment touching the image border → padding clamps, no crash, non-empty crop. (Task 2 test.)
5. Same seg_id seen in later rounds → zero new NIM calls. (Task 2 cache test; Task 5 smoke checks `nim_calls == 2 * n_objects`.)

---

### Task 0: Bring the grip fix into this branch

**Files:** none edited (merge only).

- [ ] **Step 1: Merge** `git fetch origin && git merge --no-edit origin/props-grasp-reliability` (PR #30, gripper stiffness ×5 — needed for the gate numbers to be comparable to SP1's 37/40).
- [ ] **Step 2: Verify** `grep -n "GRIPPER_STIFFNESS_SCALE = 5.0" mujoco_grasp_sim/sim_grasp/scene_generator.py` prints one line.

### Task 1: `scene_graph.py` — nodes, locations, edges

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/scene_graph.py`
- Test: `mujoco_grasp_sim/sim_grasp/test_scene_graph.py`

**Interfaces:**
- Consumes: `pointcloud.depth_to_pointcloud(depth, K, rgb=None, mask=None)`, `spatial_relation_resolver._camera_view_axis(T_world_cam) -> (axis, sign)`, `color_utils.rgb_to_color_name(rgb01) -> str`, `scene_generator.BinSpec(name, center, inner_half, category)`.
- Produces: `Node`, `Edge`, `SceneGraph` (with `.to_json() -> dict`, `.table_nodes() -> list[int]`), `build(depth, segmap, K, T_world_cam, rgb, bins, table_height) -> SceneGraph`, constants `MIN_POINTS`, `NEAR_DIST`.

- [ ] **Step 1: Write the failing test** `sim_grasp/test_scene_graph.py`:

```python
"""Standalone checks for scene_graph.build (no MuJoCo/network)."""
import json
import numpy as np

from sim_grasp.scene_generator import BinSpec
from sim_grasp.scene_graph import build, MIN_POINTS, NEAR_DIST

H, W = 120, 160
K = np.array([[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]])
# camera 1 m above the table looking straight down; image-right = world +x
T = np.eye(4); T[:3, :3] = np.diag([1.0, -1.0, -1.0]); T[:3, 3] = [0.5, 0.0, 1.75]
TABLE = 0.75
bins = [BinSpec('A', (0.80, 0.0), 0.08, 'food'), BinSpec('B', (0.20, 0.0), 0.08, 'non_food')]


def pix(x, y):          # world (x, y) on the table -> pixel (u, v)
    return int(round(K[0, 0] * (x - 0.5) / 1.0 + W / 2)), int(round(K[1, 1] * -(y - 0.0) / 1.0 + H / 2))


depth = np.full((H, W), 1.0, np.float32)        # tabletop everywhere
segmap = np.zeros((H, W), np.float32)
rgb = np.full((H, W, 3), 200, np.uint8)

def blob(label, x, y, half_px, z_top, colour):
    u, v = pix(x, y)
    segmap[v - half_px:v + half_px, u - half_px:u + half_px] = label
    depth[v - half_px:v + half_px, u - half_px:u + half_px] = 1.75 - z_top
    rgb[v - half_px:v + half_px, u - half_px:u + half_px] = colour

blob(1, 0.45, 0.0, 4, TABLE + 0.02, (255, 0, 0))        # table, left
blob(2, 0.53, 0.0, 4, TABLE + 0.02, (0, 0, 255))        # table, right, near #1
blob(3, 0.80, 0.0, 4, TABLE + 0.14, (0, 255, 0))        # tall prop standing in bin A: top face above the 5 cm wall
u, v = pix(0.30, 0.10); segmap[v, u:u + 3] = 4             # 3-pixel sliver
# label 5 deliberately absent (occluded object)

g = build(depth, segmap, K, T, rgb, bins, TABLE)
assert set(g.nodes) == {1, 2, 3, 4}, g.nodes.keys()          # no node for an absent label
assert g.nodes[1].location == 'table' and g.nodes[2].location == 'table'
assert g.nodes[3].location == 'A'                            # bin footprint, regardless of height
assert g.nodes[4].location == 'unknown'                      # < MIN_POINTS
assert g.nodes[1].colour_name == 'red' and g.nodes[2].colour_name == 'blue'
assert np.allclose(g.nodes[1].centroid_world[:2], [0.45, 0.0], atol=0.01)
x0, y0, x1, y1 = g.nodes[1].pixel_bbox
assert x0 < pix(0.45, 0)[0] < x1 and y0 < pix(0.45, 0)[1] < y1
rels = {(e.src, e.dst, e.relation) for e in g.edges}
assert (1, 2, 'left_of') in rels and (2, 1, 'right_of') in rels
assert (1, 2, 'near') in rels and (2, 1, 'near') in rels       # 8 cm < NEAR_DIST
assert not any(3 in (e.src, e.dst) or 4 in (e.src, e.dst) for e in g.edges)  # edges: table nodes only
assert g.table_nodes() == [1, 2]
js = json.loads(json.dumps(g.to_json()))
assert js['nodes']['1']['location'] == 'table' and js['nodes']['1']['identity'] is None
assert {'src': 1, 'dst': 2, 'relation': 'left_of'} in js['edges']
assert MIN_POINTS == 30 and NEAR_DIST == 0.12
print('All scene_graph checks passed.')
```

- [ ] **Step 2: Run it** `cd mujoco_grasp_sim && PYTHONPATH=. python sim_grasp/test_scene_graph.py` → expect `ModuleNotFoundError: No module named 'sim_grasp.scene_graph'`.

- [ ] **Step 3: Implement** `sim_grasp/scene_graph.py`:

```python
"""[P10 SP2] Vision-only scene graph over one capture: a node per segmented
object (world centroid/extents, colour, location table|bin|unknown) and
left_of/right_of/near edges between table objects. Pure -- no MuJoCo, no
network -- so it runs unchanged on a real camera. Identity/category are
filled in later by object_knowledge."""
from dataclasses import dataclass, field

import numpy as np

from sim_grasp.color_utils import rgb_to_color_name
from sim_grasp.pointcloud import depth_to_pointcloud
from sim_grasp.spatial_relation_resolver import _camera_view_axis

MIN_POINTS = 30          # fewer points -> location 'unknown' (sliver/occluded)
NEAR_DIST = 0.12         # m, centroid distance for a 'near' edge


@dataclass
class Node:
    seg_id: int
    centroid_world: np.ndarray
    extents: np.ndarray
    colour_name: str
    location: str
    pixel_bbox: tuple
    identity: 'str | None' = None
    category: 'str | None' = None


@dataclass
class Edge:
    src: int
    dst: int
    relation: str


@dataclass
class SceneGraph:
    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)

    def table_nodes(self) -> list:
        return sorted(s for s, n in self.nodes.items() if n.location == 'table')

    def to_json(self) -> dict:
        return {'nodes': {str(s): {'seg_id': n.seg_id,
                                   'centroid_world': [round(float(v), 4) for v in n.centroid_world],
                                   'extents': [round(float(v), 4) for v in n.extents],
                                   'colour_name': n.colour_name, 'location': n.location,
                                   'pixel_bbox': [int(v) for v in n.pixel_bbox],
                                   'identity': n.identity, 'category': n.category}
                          for s, n in sorted(self.nodes.items())},
                'edges': [{'src': e.src, 'dst': e.dst, 'relation': e.relation}
                          for e in self.edges]}


def _location(pts_world, bins, table_height) -> str:
    # bin = XY footprint only: a tall prop's top face sits above the bin wall
    in_any = np.zeros(len(pts_world), bool)
    votes = {}
    for b in bins:
        inside = ((np.abs(pts_world[:, 0] - b.center[0]) <= b.inner_half)
                  & (np.abs(pts_world[:, 1] - b.center[1]) <= b.inner_half))
        votes[b.name] = int(inside.sum())
        in_any |= inside
    z = pts_world[:, 2]
    votes['table'] = int((~in_any & (z >= table_height - 0.01) & (z <= table_height + 0.30)).sum())
    best = max(votes.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else 'unknown'


def build(depth, segmap, K, T_world_cam, rgb, bins, table_height) -> SceneGraph:
    seg = np.asarray(segmap).reshape(depth.shape)
    g = SceneGraph()
    for sid in sorted(int(s) for s in np.unique(seg) if s > 0):
        mask = seg == sid
        vs, us = np.nonzero(mask)
        bbox = (int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1)
        pc, cols = depth_to_pointcloud(depth, K, rgb=rgb, mask=mask)
        if len(pc) < MIN_POINTS:
            g.nodes[sid] = Node(sid, np.zeros(3), np.zeros(3), 'unknown', 'unknown', bbox)
            continue
        pw = pc @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
        g.nodes[sid] = Node(sid, pw.mean(axis=0), pw.max(axis=0) - pw.min(axis=0),
                            rgb_to_color_name(cols.mean(axis=0)),
                            _location(pw, bins, table_height), bbox)
    # edges between table objects only; bin membership is a node attribute
    table = g.table_nodes()
    axis, sign = _camera_view_axis(T_world_cam)
    k = 0 if axis == 'x' else 1
    order = sorted(table, key=lambda s: sign * g.nodes[s].centroid_world[k])
    for a, b in zip(order, order[1:]):          # immediate neighbours keep the image readable
        g.edges += [Edge(a, b, 'left_of'), Edge(b, a, 'right_of')]
    for i, a in enumerate(table):
        for b in table[i + 1:]:
            if np.linalg.norm(g.nodes[a].centroid_world[:2] - g.nodes[b].centroid_world[:2]) < NEAR_DIST:
                g.edges += [Edge(a, b, 'near'), Edge(b, a, 'near')]
    return g
```

- [ ] **Step 4: Run** the test → `All scene_graph checks passed.` If `depth_to_pointcloud(..., mask=...)` returns only `pc` when `rgb` is None, keep passing `rgb` as above (it returns `(pc, colors)` when rgb is given).
- [ ] **Step 5: Commit** `git add mujoco_grasp_sim/sim_grasp/scene_graph.py mujoco_grasp_sim/sim_grasp/test_scene_graph.py && git commit -m "Add the vision-only scene graph builder (P10 SP2)"`

### Task 2: `object_knowledge.py` — identify, categorize, crop, cache

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/object_knowledge.py`
- Test: `mujoco_grasp_sim/sim_grasp/test_object_knowledge.py`

**Interfaces:**
- Consumes: `instruction_parser._load_dotenv(path)`.
- Produces: `MODEL`, `identify(crop_rgb, transport=None) -> str`, `categorize(name, categories, transport=None) -> str`, `identification_crop(rgb_hi, segmap_hi, seg_id, pad=36) -> np.ndarray`, `class KnowledgeCache` with `.lookup(seg_id, make_crop, categories, oracle_name=None) -> (identity, category)`, `.items()` (seg_id → (identity, category)), `.calls: int`, `.seconds: float`. `transport(body: dict) -> str` returns the message content; default posts to NIM.

- [ ] **Step 1: Write the failing test** `sim_grasp/test_object_knowledge.py`:

```python
"""Standalone checks for object_knowledge (fake transport, no network)."""
import numpy as np

import sim_grasp.object_knowledge as ok

replies = []
def fake(body):
    replies.append(body)
    return fake.next.pop(0)

# categorize: JSON inside prose, case-normalised, enum-validated
fake.next = ['Sure! {"category": "Food"}']
assert ok.categorize('box of cookies', ('food', 'non_food'), transport=fake) == 'food'
assert replies[-1]['response_format'] == {'type': 'json_object'}
fake.next = ['{"category": "drink"}']
try:
    ok.categorize('juice', ('food', 'non_food'), transport=fake); raise AssertionError('no raise')
except ValueError:
    pass

# identify sends an image and strips quotes/whitespace
fake.next = ['  "Box of crayons."  ']
assert ok.identify(np.zeros((10, 10, 3), np.uint8), transport=fake) == 'Box of crayons.'
content = replies[-1]['messages'][0]['content']
assert content[1]['image_url']['url'].startswith('data:image/png;base64,')
assert replies[-1]['model'] == ok.MODEL == 'meta/llama-3.2-11b-vision-instruct'

# retries then raises RuntimeError
calls = {'n': 0}
def flaky(body):
    calls['n'] += 1
    raise OSError('boom')
ok._SLEEP = lambda s: None
try:
    ok.identify(np.zeros((4, 4, 3), np.uint8), transport=flaky); raise AssertionError('no raise')
except RuntimeError:
    pass
assert calls['n'] == 3

# crop: masked to white, padded, clamped at the border
rgb = np.full((50, 60, 3), 7, np.uint8); seg = np.zeros((50, 60), np.float32)
seg[0:10, 0:8] = 2                                   # touches the top-left border
c = ok.identification_crop(rgb, seg, 2, pad=36)
assert c.shape[0] == 46 and c.shape[1] == 44         # 0..10+36, 0..8+36
assert (c[0, 0] == 7).all() and (c[-1, -1] == 255).all()

# cache: second lookup of the same seg_id makes no calls; oracle skips identify
fake.next = ['box of cookies', '{"category": "food"}']
cache = ok.KnowledgeCache(transport=fake)
r1 = cache.lookup(1, lambda: np.zeros((5, 5, 3), np.uint8), ('food', 'non_food'))
n = len(replies)
r2 = cache.lookup(1, lambda: (_ for _ in ()).throw(AssertionError('crop rebuilt')), ('food', 'non_food'))
assert r1 == r2 == ('box of cookies', 'food') and len(replies) == n and cache.calls == 2
fake.next = ['{"category": "non_food"}']
r3 = cache.lookup(2, lambda: (_ for _ in ()).throw(AssertionError('no crop in oracle')),
                  ('food', 'non_food'), oracle_name='Crayola Bonus 64 Crayons')
assert r3 == ('Crayola Bonus 64 Crayons', 'non_food') and cache.calls == 3
assert dict(cache.items()) == {1: ('box of cookies', 'food'), 2: ('Crayola Bonus 64 Crayons', 'non_food')}
print('All object_knowledge checks passed.')
```

- [ ] **Step 2: Run it** → expect `ModuleNotFoundError`.
- [ ] **Step 3: Implement** `sim_grasp/object_knowledge.py`:

```python
"""[P10 SP2] Object identity + commonsense category from hosted NIM:
identify(crop) names the object, categorize(name) maps it onto the bin
categories (schema-enforced enum, so the model can never invent a bin).
Fails loudly, like instruction_parser (no silent fallback to sim truth)."""
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from sim_grasp.instruction_parser import _load_dotenv

NIM_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'
MODEL = 'meta/llama-3.2-11b-vision-instruct'   # 2026-10-03 model test: 10/12 on 3x masked crops
_SLEEP = time.sleep
_BACKOFF = (2.0, 4.0)

IDENTIFY_PROMPT = ('This is a cropped photo of one household product on a table. '
                   'Name the product in a short phrase (e.g. "box of chocolate candy bars"). '
                   'Answer with the phrase only.')


def _nim_transport(body: dict) -> str:
    if not os.environ.get('NVIDIA_API_KEY'):
        _load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')
    key = os.environ.get('NVIDIA_API_KEY')
    if not key:
        raise RuntimeError('NVIDIA_API_KEY not found in the environment or .env -- '
                           'required for --scene-graph')
    req = urllib.request.Request(NIM_URL, data=json.dumps(body).encode(), method='POST',
                                 headers={'Authorization': f'Bearer {key}',
                                          'Content-Type': 'application/json',
                                          'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=90.0) as r:
        return json.loads(r.read())['choices'][0]['message']['content']


def _call(body: dict, transport) -> str:
    transport = transport or _nim_transport
    last = None
    for attempt in range(3):
        try:
            return transport(body)
        except (OSError, urllib.error.URLError, KeyError, ValueError) as e:
            # the hosted endpoint returned transient 404/410s in the model test
            last = e
            if attempt < 2:
                _SLEEP(_BACKOFF[attempt])
    raise RuntimeError(f'NIM call to {MODEL} failed after 3 attempts: {last}')


def identify(crop_rgb: np.ndarray, transport=None) -> str:
    ok, png = cv2.imencode('.png', cv2.cvtColor(np.ascontiguousarray(crop_rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError('could not PNG-encode the identification crop')
    b64 = base64.b64encode(png.tobytes()).decode()
    body = {'model': MODEL, 'temperature': 0.0, 'max_tokens': 60, 'messages': [{
        'role': 'user', 'content': [
            {'type': 'text', 'text': IDENTIFY_PROMPT},
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}]}]}
    return _call(body, transport).strip().strip('"').strip()


def categorize(name: str, categories: tuple, transport=None) -> str:
    opts = ' or '.join(f'{{"category": "{c}"}}' for c in categories)
    body = {'model': MODEL, 'temperature': 0.0, 'max_tokens': 30,
            'response_format': {'type': 'json_object'},
            'messages': [{'role': 'system', 'content':
                          f'Classify the product. Reply JSON {opts}. '
                          'food = anything people eat or drink.'},
                         {'role': 'user', 'content': name}]}
    text = _call(body, transport)
    try:
        cat = json.loads(text[text.index('{'):text.rindex('}') + 1])['category']
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f'categorize: no valid JSON category in {text!r}') from e
    cat = str(cat).strip().lower()
    if cat not in categories:
        raise ValueError(f'categorize: {cat!r} is not one of {categories}')
    return cat


def identification_crop(rgb_hi, segmap_hi, seg_id, pad: int = 36) -> np.ndarray:
    seg = np.asarray(segmap_hi).reshape(rgb_hi.shape[:2])
    mask = seg == seg_id
    vs, us = np.nonzero(mask)
    if len(vs) == 0:
        raise ValueError(f'seg_id {seg_id} not in the hi-res segmap')
    H, W = mask.shape
    y0, y1 = max(0, vs.min() - pad), min(H, vs.max() + 1 + pad)
    x0, x1 = max(0, us.min() - pad), min(W, us.max() + 1 + pad)
    crop = rgb_hi[y0:y1, x0:x1, :3].copy()
    crop[~mask[y0:y1, x0:x1]] = 255      # masked crops beat raw ones in the model test
    return crop


class KnowledgeCache:
    """One identify + one categorize per seg_id per run (seg ids are stable)."""

    def __init__(self, transport=None):
        self.transport = transport
        self._by_seg = {}
        self.calls = 0
        self.seconds = 0.0

    def lookup(self, seg_id, make_crop, categories, oracle_name=None):
        if seg_id in self._by_seg:
            return self._by_seg[seg_id]
        t0 = time.time()
        if oracle_name is None:
            name = identify(make_crop(), self.transport)
            self.calls += 1
        else:
            name = oracle_name
        cat = categorize(name, tuple(categories), self.transport)
        self.calls += 1
        self.seconds += time.time() - t0
        self._by_seg[seg_id] = (name, cat)
        return name, cat

    def items(self):
        return self._by_seg.items()
```

- [ ] **Step 4: Run** → `All object_knowledge checks passed.`
- [ ] **Step 5: Live smoke (needs `.env` key, 2 calls):** `PYTHONPATH=. python -c "import numpy as np, sim_grasp.object_knowledge as k; print(k.categorize('box of crayons', ('food','non_food')))"` → `non_food`.
- [ ] **Step 6: Commit** `git add mujoco_grasp_sim/sim_grasp/object_knowledge.py mujoco_grasp_sim/sim_grasp/test_object_knowledge.py && git commit -m "Add NIM object identification and categorization (P10 SP2)"`

### Task 3: 3× identification render

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/scene_generator.py` (the `<global offwidth=... offheight=...>` line in the scene XML template)
- Test: `mujoco_grasp_sim/sim_grasp/test_scene_props.py` (append a block)

**Interfaces:**
- Produces: in props mode the model's offscreen buffer is ≥ 1920×1440, so `CameraModule(model, data, cam_name=cfg.cam_name, width=1920, height=1440)` works. Constant `HIRES_SCALE = 3` in `scene_generator.py`.

Ruling vs spec: the spec names a `capture_hires` method; the plan uses a second `CameraModule` at 3× created ONCE per run instead (same camera, same pose, intrinsics scale automatically). Reason: one renderer per resolution, reused, avoids the OSMesa crash seen when creating many renderers in one process.

- [ ] **Step 1: Append failing test** to `test_scene_props.py`:

```python
# 4. [SP2] 3x identification render: same camera, labels align with 1x
from sim_grasp.scene_generator import HIRES_SCALE
gen = SceneGenerator(SceneConfig.for_props(seed=2)); model, data = gen.generate()
lo = CameraModule(model, data, cam_name=gen.cfg.cam_name, width=640, height=480)
hi = CameraModule(model, data, cam_name=gen.cfg.cam_name,
                  width=640 * HIRES_SCALE, height=480 * HIRES_SCALE)
_, _, seg_lo, _, _ = lo.capture(gen.object_body_ids)
rgb_hi, _, seg_hi, _, _ = hi.capture(gen.object_body_ids)
assert rgb_hi.shape[:2] == (1440, 1920)
seg_hi = seg_hi.reshape(1440, 1920); seg_lo = seg_lo.reshape(480, 640)
down = seg_hi[1::3, 1::3]
agree = (down == seg_lo)[seg_lo > 0].mean()
assert agree > 0.95, agree
print('3x identification render aligned with the 1x segmap.')
```

- [ ] **Step 2: Run** `MUJOCO_GL=osmesa PYTHONPATH=. python sim_grasp/test_scene_props.py` → fails (`ImportError: HIRES_SCALE`, or the renderer refuses a size above the 1280×960 buffer).
- [ ] **Step 3: Implement.** In `scene_generator.py` add near the other module constants:

```python
# [P10 SP2] identification crops come from a 3x render of the observation
# camera: at 640x480 a prop is ~80 px and the vision model guesses (8/12).
HIRES_SCALE = 3
```

and in the scene XML template replace the hard-coded global line with a value that is larger only in props mode (boxes mode must stay byte-identical):

```python
        off_w, off_h = ((640 * HIRES_SCALE, 480 * HIRES_SCALE)
                        if cfg.scene_mode == 'props' else (1280, 960))
```

```
    <global offwidth="{off_w}" offheight="{off_h}" azimuth="120" elevation="-20"/>
```

(compute `off_w, off_h` just before the `xml = f"""...` assignment).
- [ ] **Step 4: Run** `test_scene_props.py`, `test_scene_generator_paths.py` and `test_scene_default_unchanged.py`. The first two must pass. The last one already fails before this branch (platform-dependent qpos); confirm its failure message is unchanged (`seed 0: default settled qpos changed`).
- [ ] **Step 5: Commit** `git commit -am "Render a 3x observation for object identification in props mode (P10 SP2)"`

### Task 4: `scene_graph_viz.py` — annotated image

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/scene_graph_viz.py`
- Test: `mujoco_grasp_sim/sim_grasp/test_scene_graph_viz.py`

**Interfaces:**
- Consumes: `SceneGraph`, `Node`, `Edge` (Task 1).
- Produces: `draw_scene_graph(rgb, graph, bin_categories, gt_categories=None) -> np.ndarray` (RGB uint8, same H×W as input plus a 28 px legend strip at the bottom), `LOCATION_COLOURS`.

- [ ] **Step 1: Write the failing test:**

```python
"""Standalone checks for scene_graph_viz (pure OpenCV)."""
import numpy as np

from sim_grasp.scene_graph import Edge, Node, SceneGraph
from sim_grasp.scene_graph_viz import LOCATION_COLOURS, draw_scene_graph

rgb = np.full((120, 160, 3), 128, np.uint8)
g = SceneGraph(nodes={
    1: Node(1, np.zeros(3), np.zeros(3), 'red', 'table', (10, 10, 40, 40), 'box of cookies', 'food'),
    2: Node(2, np.zeros(3), np.zeros(3), 'blue', 'A', (90, 60, 130, 100), None, None)},
    edges=[Edge(1, 2, 'left_of'), Edge(1, 2, 'near')])
out = draw_scene_graph(rgb, g, {'A': 'food', 'B': 'non_food'}, gt_categories={1: 'non_food'})
assert out.dtype == np.uint8 and out.shape == (120 + 28, 160, 3)
assert (rgb == 128).all()                                  # input not modified
assert tuple(out[25, 10]) == LOCATION_COLOURS['table']     # left box edge (labels sit above the top edge)
assert tuple(out[80, 90]) == LOCATION_COLOURS['A']
assert (out[120:] != 128).any()                            # legend strip drawn
print('All scene_graph_viz checks passed.')
```

- [ ] **Step 2: Run** → `ModuleNotFoundError`.
- [ ] **Step 3: Implement:**

```python
"""[P10 SP2] Annotated scene-graph image: a box per node coloured by
location, '#id identity - category' labels (red when the category
disagrees with sim ground truth -- a sim-only debugging aid), left_of
arrows and near lines between table objects, and a bin->category legend."""
import cv2
import numpy as np

LOCATION_COLOURS = {'table': (255, 200, 0), 'A': (0, 200, 0), 'B': (0, 120, 255),
                    'unknown': (160, 160, 160)}
_WRONG = (255, 0, 0)
_LEGEND_H = 28


def _centre(n):
    x0, y0, x1, y1 = n.pixel_bbox
    return (x0 + x1) // 2, (y0 + y1) // 2


def draw_scene_graph(rgb, graph, bin_categories, gt_categories=None) -> np.ndarray:
    img = np.ascontiguousarray(rgb[..., :3]).copy()
    for e in graph.edges:
        if e.src not in graph.nodes or e.dst not in graph.nodes:
            continue
        p, q = _centre(graph.nodes[e.src]), _centre(graph.nodes[e.dst])
        if e.relation == 'left_of':
            cv2.arrowedLine(img, p, q, (255, 255, 255), 1, tipLength=0.15)
        elif e.relation == 'near' and e.src < e.dst:      # draw each near pair once
            cv2.line(img, p, q, (255, 255, 0), 1, lineType=cv2.LINE_AA)
    for sid, n in graph.nodes.items():
        colour = LOCATION_COLOURS.get(n.location, LOCATION_COLOURS['unknown'])
        x0, y0, x1, y1 = n.pixel_bbox
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), colour, 1)
        label = f'#{sid} {n.identity or "?"} - {n.category or "?"}'
        wrong = gt_categories is not None and n.category is not None \
            and gt_categories.get(sid) not in (None, n.category)
        cv2.putText(img, label, (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    _WRONG if wrong else colour, 1, cv2.LINE_AA)
    legend = np.full((_LEGEND_H, img.shape[1], 3), 32, np.uint8)
    text = '  '.join(f'bin {b}: {c}' for b, c in sorted(bin_categories.items()))
    cv2.putText(legend, text + '   (red label = wrong vs sim truth)', (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([img, legend])
```

- [ ] **Step 4: Run** → `All scene_graph_viz checks passed.` (If the box-edge pixel asserts fail because of anti-aliasing, the rectangle uses the default 8-connected line, which is exact; check the coordinates rather than switching to approximate asserts.)
- [ ] **Step 5: Commit** `git add mujoco_grasp_sim/sim_grasp/scene_graph_viz.py mujoco_grasp_sim/sim_grasp/test_scene_graph_viz.py && git commit -m "Draw an annotated scene-graph image per round (P10 SP2)"`

### Task 5: Wire the graph into pick-all (`run_sim_grasp_test.py`)

**Files:**
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py` (argparse near line 255/300, validation near line 352-370, pick-all loop near lines 672-760, end-of-pick-all metrics near line 911)

**Interfaces:**
- Consumes: Task 1 `build`, Task 2 `KnowledgeCache`/`identification_crop`, Task 3 `HIRES_SCALE`, Task 4 `draw_scene_graph`.
- Produces: `metrics['scene_graph'] = {'identity_mode', 'rounds': [graph json per round], 'objects': {name: {identity, category, gt_category, correct}}, 'category_accuracy', 'location_agreement', 'nim_calls', 'nim_seconds'}`; files `output/<run>/scene_graph_round_<k>.png`.

- [ ] **Step 1: Flags.** Next to `--instruction`:

```python
    ap.add_argument('--scene-graph', action='store_true',
                    help='[P10 SP2] props only: build a vision-only scene graph each '
                         'pick-all round (drives loop control), identify/categorize '
                         'objects via NIM, save scene_graph_round_<k>.png + JSON')
    ap.add_argument('--identity', choices=['perceived', 'oracle'], default=None,
                    help='[P10 SP2] with --scene-graph: perceived = vision model names '
                         'the crop (default); oracle = sim prop name, knowledge-only upper bound')
```

and after the existing `--instruction` validation:

```python
    if args.scene_graph and args.scene != 'props':
        sys.exit('[scene-graph] --scene-graph requires --scene props')
    if args.scene_graph and not args.pick_all:
        sys.exit('[scene-graph] --scene-graph requires --pick-all')
    if args.identity is not None and not args.scene_graph:
        sys.exit('[scene-graph] --identity requires --scene-graph')
    if args.scene_graph and args.identity is None:
        args.identity = 'perceived'
```

- [ ] **Step 2: Setup** inside `if args.pick_all:`, right after `label_of = {...}`:

```python
        graph_on = args.scene_graph
        if graph_on:
            from sim_grasp import scene_graph as sg
            from sim_grasp.object_knowledge import KnowledgeCache, identification_crop
            from sim_grasp.scene_graph_viz import draw_scene_graph
            from sim_grasp.scene_generator import HIRES_SCALE
            import imageio.v2 as _iio
            name_of_label = {v: k for k, v in label_of.items()}
            categories = tuple(b.category for b in cfg.bins())
            knowledge = KnowledgeCache()
            hires_cam = None            # created once on first need (one renderer per size)
            sg_rounds, sg_agree = [], []
```

- [ ] **Step 3: Per-round graph** at the top of the `for rnd ...` loop body, AFTER the existing `if cur is not None: ... else: ...` block (so `obs`, `cur_rgb`, `T_wc` hold this round's observation) and BEFORE `in_bin_now = ...`:

```python
            if graph_on:
                d_g, seg_g, K_g = obs
                graph = sg.build(d_g, seg_g, K_g, T_wc, cur_rgb, cfg.bins(),
                                 cfg.table_height)
                need = [s for s in graph.nodes if s in name_of_label]
                hi = None
                for sid in need:
                    body = name_of_label[sid]
                    oracle = (gen.object_props[body].replace('_', ' ')
                              if args.identity == 'oracle' else None)

                    def make_crop(sid=sid):
                        nonlocal hires_cam, hi
                        if hi is None:      # one 3x render per round, only if a new node needs it
                            if hires_cam is None:
                                hires_cam = CameraModule(model, data, cam_name=cfg.cam_name,
                                                         width=640 * HIRES_SCALE,
                                                         height=480 * HIRES_SCALE)
                            rgb_h, _, seg_h, _, _ = hires_cam.capture(gen.object_body_ids)
                            hi = (rgb_h, seg_h)
                        return identification_crop(hi[0], hi[1], sid)
                    graph.nodes[sid].identity, graph.nodes[sid].category = knowledge.lookup(
                        sid, make_crop, categories, oracle_name=oracle)
                oracle_bins = gen.objects_in_bins()
                oracle_table = set(gen.objects_on_table())
                for sid, node in graph.nodes.items():
                    body = name_of_label.get(sid)
                    if body is None:
                        continue
                    truth = oracle_bins.get(body) or ('table' if body in oracle_table else 'off')
                    sg_agree.append(node.location == truth)
                sg_rounds.append({'round': rnd, **graph.to_json()})
                gt = {s: gen.object_categories[name_of_label[s]] for s in graph.nodes
                      if s in name_of_label}
                _iio.imwrite(save_dir / f'scene_graph_round_{rnd}.png',
                             draw_scene_graph(cur_rgb, graph,
                                              {b.name: b.category for b in cfg.bins()}, gt))
```

This code runs inside `main()` (`def main` in `run_sim_grasp_test.py`), so `nonlocal hires_cam, hi` binds to `main`'s locals: the 3× camera is created once per run and the 3× frame once per round.

- [ ] **Step 4: Loop control from the graph.** Replace

```python
            in_bin_now = set(gen.objects_in_bins())   # any bin: no re-sorting of a wrong-bin object
            remaining = [n for n in gen.objects_on_table()
                         if n not in in_bin_now and fail_count.get(n, 0) < 3]
```

with

```python
            if graph_on:
                # [P10 SP2] vision-only loop control: graph table nodes, not qpos
                remaining = [name_of_label[s] for s in graph.table_nodes()
                             if s in name_of_label and fail_count.get(name_of_label[s], 0) < 3]
            else:
                in_bin_now = set(gen.objects_in_bins())   # any bin: no re-sorting of a wrong-bin object
                remaining = [n for n in gen.objects_on_table()
                             if n not in in_bin_now and fail_count.get(n, 0) < 3]
```

Leave the result scoring (`landed = gen.objects_in_bins()...`, `final_bins`, `sort_outcome`) on the oracle: that is measurement, not control. Routing (`target_bin_for`) is unchanged (SP3).

- [ ] **Step 5: Metrics** right after `metrics['pick_all'].update(in_correct_bin=..., in_wrong_bin=...)`:

```python
        if graph_on:
            objs = {}
            for sid, (ident, cat) in knowledge.items():
                body = name_of_label[sid]
                gt_cat = gen.object_categories[body]
                objs[body] = {'identity': ident, 'category': cat, 'gt_category': gt_cat,
                              'correct': cat == gt_cat}
            acc = sum(o['correct'] for o in objs.values()) / max(len(objs), 1)
            agree = sum(sg_agree) / max(len(sg_agree), 1)
            metrics['scene_graph'] = {
                'identity_mode': args.identity, 'rounds': sg_rounds, 'objects': objs,
                'category_accuracy': round(acc, 3), 'location_agreement': round(agree, 3),
                'nim_calls': knowledge.calls, 'nim_seconds': round(knowledge.seconds, 1)}
            print(f'[scene-graph] category accuracy {acc:.0%} ({args.identity}), '
                  f'location agreement {agree:.0%}, {knowledge.calls} NIM calls')
            if hires_cam is not None:
                hires_cam.close()
```

- [ ] **Step 6: Live smoke** (needs the `.env` key):

```bash
cd mujoco_grasp_sim && MUJOCO_GL=osmesa GRASPGEN_PYTHON=~/miniconda3/envs/graspgen_torch/bin/python \
python run_sim_grasp_test.py --seed 0 --scene props --pick-all --no-vis --camera fused \
  --backend graspgen --graspgen-python "$GRASPGEN_PYTHON" --scene-graph 2>&1 | grep -E "scene-graph|DONE|SORTED"
```

Expected: `DONE` and `SORTED` lines; a `[scene-graph] category accuracy ...` line with `8 NIM calls` for 4 objects; `scene_graph_round_1.png` exists; open it and check boxes/labels/arrows by eye. Also run `--identity oracle` once (expect `4 NIM calls`) and the flag errors (`--scene-graph` without `--scene props` exits with the message).
- [ ] **Step 7: Regression** — without `--scene-graph`, `git diff` must not change any line executed in the non-graph path except the `if graph_on:` branches; run one plain props pick-all (seed 2) and confirm it completes.
- [ ] **Step 8: Commit** `git commit -am "Drive props pick-all from the scene graph and record perceived categories (P10 SP2)"`

### Task 6: Benchmark support

**Files:**
- Modify: `mujoco_grasp_sim/benchmark.py` (`run_one` command build + result fields, argparse, aggregate print)

- [ ] **Step 1: Flags** in `main()` argparse (note `--scene`, `--backend`, `--graspgen-python`, `--filter-neighbors` already exist in this file — add only these two):

```python
    ap.add_argument('--scene-graph', action='store_true',
                    help='forward --scene-graph (props pick-all only)')
    ap.add_argument('--identity', choices=['perceived', 'oracle'], default=None,
                    help='forward --identity (needs --scene-graph)')
```

- [ ] **Step 2: Forward** in `run_one`, after the `--scene` lines:

```python
    if args.scene_graph:
        cmd += ['--scene-graph']
        if args.identity:
            cmd += ['--identity', args.identity]
```

- [ ] **Step 3: Collect** in `run_one`'s pick-all branch:

```python
        sgm = m.get('scene_graph')
        if sgm:
            objs = sgm.get('objects', {})
            out['cat_correct'] = sum(1 for o in objs.values() if o.get('correct'))
            out['cat_total'] = len(objs)
            out['loc_agreement'] = sgm.get('location_agreement')
            out['nim_calls'] = sgm.get('nim_calls')
```

- [ ] **Step 4: Aggregate** after the correct-bin print:

```python
        if any('cat_total' in r for r in ok):
            cc = sum(r.get('cat_correct', 0) for r in ok)
            ct = sum(r.get('cat_total', 0) for r in ok)
            la = [r['loc_agreement'] for r in ok if r.get('loc_agreement') is not None]
            print(f'[bench] perceived category accuracy: {cc}/{ct} '
                  f'({100 * cc / max(ct, 1):.0f}%), mean location agreement: '
                  f'{100 * sum(la) / max(len(la), 1):.0f}%')
```

- [ ] **Step 5: Smoke** `python benchmark.py --seeds 0 --mode pick-all --camera fused --backend graspgen --graspgen-python "$GRASPGEN_PYTHON" --scene props --scene-graph --tag sg_smoke` → prints the new line.
- [ ] **Step 6: Commit** `git commit -am "Report scene-graph accuracy in benchmark.py (P10 SP2)"`

### Task 7: Gate run + documentation sync

**Files:**
- Modify: `ROADMAP.md` (P10 SP2 bullet + header tag), the SP2 spec (`## Implementation notes` section), `mujoco_grasp_sim/README.md` (one flag line each for `--scene-graph`, `--identity`), `CLAUDE.md` (add `scene_graph.py` / `object_knowledge.py` / `scene_graph_viz.py` to the key-modules list).

- [ ] **Step 1: Gate** (≈ 45 min each, run sequentially):

```bash
python benchmark.py --seeds 0-9 --mode pick-all --camera fused --backend graspgen \
  --graspgen-python "$GRASPGEN_PYTHON" --scene props --scene-graph --tag sp2_gate
python benchmark.py --seeds 0-9 --mode pick-all --camera fused --backend graspgen \
  --graspgen-python "$GRASPGEN_PYTHON" --scene props --scene-graph --identity oracle --tag sp2_oracle
```

Gate (spec): correct-bin ≥ 34/40, ≤ 2 knocked off, 0 crashes; location agreement ≥ 95%; perceived category accuracy ≥ 80% (report the oracle run beside it).
- [ ] **Step 2: Docs** — quote the real `[bench]` lines in ROADMAP P10 SP2 (tag `[SP2 IMPLEMENTED <date>]` + gate met/not met), add Implementation notes to the SP2 spec (the `capture_hires` → second `CameraModule` ruling, and any other divergence), README + CLAUDE.md lines.
- [ ] **Step 3: Commit** `git add ROADMAP.md CLAUDE.md mujoco_grasp_sim/README.md docs/superpowers/specs/2026-10-03-semantic-sorting-sp2-scene-graph-design.md && git commit -m "Record the P10 SP2 gate results"`
