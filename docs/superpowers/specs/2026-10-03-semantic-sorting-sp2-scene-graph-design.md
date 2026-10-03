# P10 SP2 — Scene graph + object knowledge — Design

**Parent design:** `docs/superpowers/specs/2026-09-29-semantic-sorting-scene-graph-design.md`
(umbrella architecture and the SP2/SP3 interface contracts). This document
fills in SP2 only. SP1 is the props scene, merged as PR #28 and #30 with a
37/40 correct-bin gate result. SP3 (routing by perceived category) gets its
own spec afterwards.

## Goal

Every pick-all round builds a **scene graph from vision only**: one node per
segmented object, edges for spatial relations, and each node's identity and
food/non_food category from a hosted vision model. SP2 then:

1. **replaces the qpos oracle for loop control.** "Which objects are still
   on the table, which are in a bin" comes from graph node locations, not
   `gen.objects_in_bins()` / `gen.objects_on_table()`;
2. **measures perception without acting on it.** Routing still uses
   ground-truth categories (that switch is SP3), but every run records how
   often the perceived identity/category matches ground truth;
3. **makes the graph visible.** An annotated image per round plus the graph
   as JSON in `metrics.json` (user-chosen option A, 2026-10-03).

## Decisions

### Confirmed with the user

- **Visualisation (2026-10-03):** option A, an annotated image per round
  plus JSON. No live overlay.
- **Model selection by measurement (2026-10-03):** a throwaway test on all
  12 props' crops. Candidates came from the research note plus the
  user's links.

### From the model test (2026-10-03, review these)

- **Model: `meta/llama-3.2-11b-vision-instruct`** for both steps,
  identify (image → short name) and categorize (name → JSON enum). On
  masked 3× crops: 10/12 correct category, ~1 s median per call.
  `llama-3.2-90b-vision` also scored 10/12 but had 40–63 s stalls.
  `gemma-3-*`, `phi-3-vision`, `cosmos-reason2-8b` and `cosmos3-nano-reasoner`
  are unreachable on this key (404). `llama-3.1-8b-instruct`, which the
  umbrella design named for categorize, is retired (410), so SP2 uses one
  model for both steps. The P8 instruction parser moved to the same model
  (PR #29).
- **Identification crops come from a 3× render of the same camera**
  (1920×1440), masked to the object's segment with a white background.
  At the normal 640×480 each prop is ~80 px and both models were at chance
  (8/12 vs a 6/12 always-"non_food" baseline). At 3× they read the printed
  labels ("Just For Men hair dye", "Nescafe Memento coffee"). This models
  a higher-resolution RGB stream used only for recognition. It is the
  same camera and pose, not a new viewpoint, and is rendered only when a
  new node needs identifying.
- **Two calls, not one.** Identify-then-categorize keeps the umbrella
  design's error split (identification error vs knowledge error).
  `--identity oracle` skips identify and feeds the prop's ground-truth
  name, giving the knowledge-only upper bound.

## Components

### `sim_grasp/scene_graph.py` (pure, no MuJoCo/network)

```python
@dataclass
class Node:
    seg_id: int
    centroid_world: np.ndarray      # (3,)
    extents: np.ndarray             # (3,) world AABB size of the segment's points
    colour_name: str                # from color_utils (existing)
    location: str                   # 'table' | bin name ('A'/'B') | 'unknown'
    pixel_bbox: tuple[int, int, int, int]   # x0, y0, x1, y1 in the 640x480 image
    identity: str | None = None     # filled by object_knowledge
    category: str | None = None

@dataclass
class Edge:
    src: int; dst: int
    relation: str                   # 'left_of' | 'right_of' | 'near'

@dataclass
class SceneGraph:
    nodes: dict[int, Node]
    edges: list[Edge]
    def to_json(self) -> dict

def build(depth, segmap, K, T_world_cam, rgb, bins, table_height) -> SceneGraph
```

- **Location:** the segment's world points are classified per point as
  inside a bin's inner footprint (and below its rim), on the tabletop
  (within 3 cm above `table_height`), or neither. The majority wins. A
  segment with < 30 points gets `'unknown'`, which loop control treats as
  "not pickable this round", never "in a bin".
- **Edges, table nodes only:** `left_of`/`right_of` reuse
  `spatial_relation_resolver._camera_view_axis`, so
  "left" means the same thing in instructions and the graph. Only
  immediate neighbours in that order get an edge, which keeps the image
  readable. `near` is centroid distance < 12 cm. Bin membership is the
  node's `location`, not an edge.
- **Seg ids are stable across rounds** (the segmap label is fixed per
  object), so identity/category caching is keyed by `seg_id`.

### `sim_grasp/object_knowledge.py` (network, fails loudly)

```python
MODEL = "meta/llama-3.2-11b-vision-instruct"
def identify(crop_rgb: np.ndarray) -> str
def categorize(name: str, categories: tuple[str, ...]) -> str   # JSON enum, validated
def identification_crop(rgb_hi, segmap_hi, seg_id, pad=36) -> np.ndarray   # pure
```

- Same transport and `.env` loading as `instruction_parser.py`.
- Each call retries 3× with 2 s / 4 s backoff on network errors or 5xx,
  because this endpoint returned transient 404/410s during the model test.
  It then raises `RuntimeError`, aborting the run (P8's fail-loudly rule:
  no silent fallback to ground truth).
- `categorize` rejects any answer outside `categories` (`ValueError`, also
  fatal). The model can't invent a bin.
- A small `KnowledgeCache` (dict keyed by seg_id) means each object costs
  one identify + one categorize call per run, about 8 calls for 4 props.

### Hi-res capture (`camera.py`)

`CameraModule.capture_hires(body_to_label, scale=3)` renders RGB + segmap
at `scale×` resolution from the same camera. The props scene raises
`<global offwidth/offheight>` to 1920×1440. Boxes mode is untouched.

### Pick-all integration (`run_sim_grasp_test.py`)

New flag `--scene-graph`, valid only with `--scene props` (error
otherwise). Off by default, so existing runs don't change at all.
`--identity {perceived,oracle}` (default `perceived`) requires
`--scene-graph`.

Per round, after the existing capture:

1. `graph = scene_graph.build(...)`.
2. For nodes not yet cached: one `capture_hires` per round (only if needed),
   crop → `identify` → `categorize`, then cache.
3. Loop control uses `graph` locations: `remaining` = table nodes not
   failed 3×. The qpos oracle is still evaluated, but only to record
   `location_agreement` (graph location == oracle location) per node per
   round.
4. Routing is unchanged (ground-truth category, SP1). The perceived
   category is logged next to the ground truth, and SP3 flips the source.
5. Writes `output/<run>/scene_graph_round_<k>.png` and appends
   `graph.to_json()` to `metrics['scene_graph']['rounds']`.

### Annotated image (`visualizer.py`)

Drawn on the round's 640×480 RGB with OpenCV, like existing overlays:

- a box per node, coloured by location (table / bin A / bin B / unknown);
- a label per node: `#<seg_id> <identity> — <category>`, red if it
  disagrees with ground truth (a sim-only debugging aid, clearly marked);
- `left_of` as arrows between neighbours, `near` as thin dashed lines;
- a legend strip listing the bin → category mapping.

### Metrics and benchmark

`metrics['scene_graph']`: `identity_mode`, per-object
`{identity, category, gt_category, correct}`, `category_accuracy`,
`location_agreement`, `nim_calls`, `nim_seconds`. `benchmark.py` sums
these over seeds and prints category accuracy and location agreement.

## Error handling

- No `NVIDIA_API_KEY` → fails at startup, before any physics.
- NIM failure after retries → run aborts with the model name and HTTP
  status. `benchmark.py` already records it as a crashed run.
- Fewer than 30 points for a node → `'unknown'` location, skipped this
  round, re-checked next round. Never counted as binned.

## Testing

Standalone assert scripts, no network:

- `test_scene_graph.py`: synthetic depth/segmap with three objects
  (table, in bin A, partially out of view) checks locations, `left_of`
  order matching the resolver's convention, `near` threshold,
  `to_json` round-trip, `'unknown'` for tiny segments.
- `test_object_knowledge.py`: injected fake transport checks JSON enum
  validation, retry then raise, cache hits not re-calling, and crop
  masking/padding on a synthetic segmap.
- `test_scene_props.py` gains a `capture_hires` shape/alignment check
  (the 3× segmap labels downsample to the 1× labels).

## SP2 gate (before SP3 starts)

GraspGen/fused, seeds 0–9, `--pick-all --scene props --scene-graph`:

1. **No regression:** ≥ 34/40 correct-bin (SP1 is 37/40), ≤ 2 knocked
   off, 0 crashes, with loop control on graph locations.
2. **Location agreement** ≥ 95% of node-rounds against the qpos oracle.
3. **Perceived category accuracy** ≥ 80% of objects (the model test
   predicts ~83%), reported next to the `--identity oracle` upper bound.

## Out of scope (SP2)

Routing by perceived category (SP3). Graph use in `--instruction`. A
local VLM (Qwen3-VL) — revisit only if the hosted model misses the gate.
Clutter or stacked relations (`on_top_of`). Boxes mode. Real-robot runs.

## Implementation notes (2026-10-03)

Divergences from the design above:

- (a) A second `CameraModule` at 3x resolution is created once per run
  instead of a `capture_hires` method (avoids an OSMesa multi-renderer crash).
- (b) Bin location is the XY footprint only, with no rim-height cut-off (a
  14 cm prop's top face is above the 5 cm wall).
- (c) Node labels use "-" not an em dash (OpenCV Hershey fonts cannot draw it).
- (d) An object with no graph node in a round is not retried, so the run can
  end early (oracle scoring still counts it as left on table).
- (e) Gate results (GraspGen/fused, seeds 0-9, pick-all): MET. 36/40
  correct-bin (>= 34), 0 knocked off, 0 crashes (10/10), location agreement
  100% (>= 95%), perceived category accuracy 33/40 = 82% (>= 80%). Oracle
  identity: 38/40 = 95% accuracy, 37/40 correct-bin. 8 NIM calls/run
  perceived, 4 oracle. Routing still uses ground truth, so category errors do
  not change correct-bin in SP2. 7 of 8 perceived misses are identification
  errors (Epson ink -> "Box of cereal" 2x, Nescafe -> "Box of hair dye");
  the Fondant box is a knowledge error (named correctly, categorized
  non_food; also both oracle misses).
