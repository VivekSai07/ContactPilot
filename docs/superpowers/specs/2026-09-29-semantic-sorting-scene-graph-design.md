# Semantic Sorting with a Scene/Knowledge Graph — Design

**Scope of this document:** the umbrella architecture for the whole
feature (three sequential sub-projects), plus the full design of
**sub-project 1 (semantic prop scene + second bin + oracle-routed
sorting)**. Sub-projects 2 and 3 have their interface contracts fixed
here but get their own spec + plan once sub-project 1 passes its gate.
Research background: `docs/research/2026-09-29-scene-knowledge-graph.md`.

## Problem

ContactPilot can pick unseen objects and (P8) follow colour-based
instructions, but it has no notion of *what* an object is. The current
scene is exactly 3 plain coloured boxes spawned ≥ 9 cm apart — there is
nothing for a scene graph or commonsense knowledge to reason about: no
identities, no categories, one destination. A scene/knowledge graph only
becomes a beneficial feature once the scene contains objects whose
*identity* determines what the robot should do with them.

## Goal

Sort real-looking household props into two bins by category (e.g. food
→ bin A, non-food → bin B), where each object's category is derived
from perception (identify from pixels → commonsense category), never
from simulator ground truth — preserving the project's core
"unseen objects" premise. Success is a single measurable number:
**correct-bin rate**, broken down by failure source.

## Decisions

### Confirmed with the user (2026-09-29)

- **Direction:** semantic props + knowledge (chosen over clutter/stacking
  order, colour-only language grounding, and real-robot state tracking).
- **Task:** sort into bins by category; metric = correct-bin rate.
- **Knowledge approach A — hosted NIM models:** a NIM vision model
  identifies each object crop; the existing NIM `meta/llama-3.1-8b-instruct`
  maps the identified name to a category via schema-enforced JSON whose
  enum is exactly the bin categories. A sim-truth identity switch gives an
  upper bound. Zero local VRAM, no new conda env, fails loudly offline
  (same policy as P8). Rejected: local Qwen3-VL worker (B), and a
  B-ready pluggable interface (not needed now — YAGNI).
- **Scene modification is allowed.**
- **Architecture approved:** three sequential sub-projects (below), each
  with its own spec → plan → PR; default behaviour byte-for-byte unchanged
  when the new flags are not used.

### Made during spec writing (review these)

- **Prop source:** [Google Scanned Objects, MuJoCo port](https://github.com/kevinzakka/mujoco_scanned_objects)
  (1,030 textured household scans; assets CC-BY 4.0, MJCF MIT). Verified
  by a throwaway render: textures render through `mujoco.Renderer` and
  printed text ("CRAYONS", "CRUNCH") is legible at 640×480.
- **Categories:** `food` and `non_food` — a binary, commonsense-requiring
  split ("is this edible?") with an unambiguous ground truth.
- **Prop shortlist (12, balanced 6/6)** — filtered to box-shaped scans that
  stand upright in their native pose, 8–17 cm tall, with a horizontal side
  ≤ 6.5 cm (Panda max opening is 8 cm):

  | Category | GSO model id | native w × d × h (cm) |
  |---|---|---|
  | food | `Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box` | 13.9 × 6.3 × 16.4 |
  | food | `ReadytoUse_Rolled_Fondant_Pure_White_24_oz_box` | 12.2 × 4.9 × 16.9 |
  | food | `Nestl_Skinny_Cow_Heavenly_Crisp_Candy_Bar_Chocolate_Raspberry_6_pack_462_oz_total` | 11.2 × 5.0 × 16.4 |
  | food | `Nestle_Nips_Hard_Candy_Peanut_Butter` | 8.9 × 4.2 × 15.3 |
  | food | `Nescafe_Momento_Mocha_Specialty_Coffee_Mix_8_ct` | 9.1 × 6.4 × 16.1 |
  | food | `Polar_Herring_Fillets_Smoked_Peppered_705_oz_total` | 15.1 × 3.2 × 8.4 |
  | non_food | `Epson_273XL_Ink_Cartridge_Magenta` | 7.6 × 3.0 × 16.0 |
  | non_food | `Dell_Series_9_Color_Ink_Cartridge_MK993_High_Yield` | 7.0 × 4.0 × 13.4 |
  | non_food | `Brother_LC_1053PKS_Ink_Cartridge_CyanMagentaYellow_1pack` | 9.9 × 5.5 × 14.5 |
  | non_food | `Just_For_Men_ShampooIn_Haircolor_Jet_Black_60` | 7.9 × 4.8 × 15.6 |
  | non_food | `Crayola_Crayons_24_count` | 7.4 × 3.0 × 11.7 |
  | non_food | `Crayola_Bonus_64_Crayons` | 14.7 × 4.3 × 12.7 |

  Rejected at measurement: board games / Jenga / LEGO / Duplo / fruit-snack
  multipacks (too big), Nintendo 3DS cases and Butterfinger (lie flat:
  the only horizontal extents are ≥ 12 cm), Kotex (6.9 cm thick).
- **Second bin at `(0.45, +0.30)`** — the mirror of bin A `(0.45, −0.30)`.
  `reachability.MAX_REACH` is 0.54 m and bin A already sits at exactly
  0.54 m from the base; no other position both stays within reach and
  clears the spawn region. The Panda's joint-1 range is symmetric, so the
  mirror is equally reachable.
- **Gate threshold:** see "Sub-project 1 gate" below.

## Architecture (all three sub-projects)

```
 scene_generator ──► GSO textured props (food / non_food) + bin A + bin B      [SP1]
        │
        ▼  each pick-all round (unchanged capture): RGB + depth + segmap
 scene_graph.build()                                                            [SP2]
   nodes: seg_id → {centroid, size, colour, footprint}      ← existing geometry code
   edges: left_of / right_of / near / in(bin_A|bin_B|table) ← vision-only; replaces qpos oracle
        │
        ▼  once per NEW node (cached across rounds, keyed by seg_id)
 identify(crop)   ──► NIM vision model   → "box of chocolate candy bars"
 categorize(name) ──► NIM llama-3.1-8b   → {"category": "food"}  (JSON enum = bin categories)
   └ --identity oracle: sim prop name → same categorize() call   (upper bound)
        │
        ▼
 pick-all consumer: destination = bin_for[category] → per-bin placement planner  [SP3]
        │
        ▼
 metrics.json: correct_bin rate, split into identify_error vs knowledge_error
```

- **Identify once, not every round:** a node's category is cached after
  first sight, so a 4-object run costs ~8 NIM calls, not 8 per round.
- **The knowledge model never names a bin:** `category → bin` is fixed
  config. The LLM only answers "which of these categories?", so a wrong
  answer is still a valid category, never an invented destination.
- **Sub-project 1 routes by ground truth on purpose.** Its job is to prove
  the new props are graspable and both bins are placeable. SP2/SP3 then
  replace only the *routing source* (oracle → perceived), so the SP1
  benchmark becomes the upper bound SP3 is measured against.

### Interface contracts for SP2 / SP3 (to be specced separately)

- `sim_grasp/scene_graph.py`: pure functions over one capture
  (`depth, segmap, K, T_world_cam, rgb, bins`) → `SceneGraph` with
  `nodes: dict[int, Node]` and `edges: list[Edge]`; `Node` carries
  `seg_id, centroid_world, extents, colour_name, location ('table' | bin name),
  identity: str | None, category: str | None`. No MuJoCo imports (testable
  like `placement_planner.py`).
- `sim_grasp/object_knowledge.py`: `identify(crop_rgb) -> str` (NIM vision)
  and `categorize(name, categories) -> str` (NIM LLM, schema-enforced enum),
  both raising on failure, per P8's fail-loudly rule.
- SP3 replaces SP1's `destination_bin(obj) = bin_for[gt_category(obj)]`
  with `bin_for[graph.nodes[seg_id].category]`, and SP2 replaces
  `gen.objects_in_bins()` (sim oracle) with graph node locations.

## Sub-project 1 design

### Components

**1. Prop manifest + download script (assets not committed).**
Mirrors the CGN checkpoint pattern (assets hosted elsewhere, fetched by an
idempotent script).
- `mujoco_grasp_sim/assets/props/manifest.json` (committed): a list of
  `{"model_id", "category", "mass_kg"}` — the 12 props above. Masses (kg),
  from the net weight in the product name where stated, else estimated,
  clamped to [0.05, 0.40]: Crunch 0.22, Fondant 0.40 (0.68 clamped),
  Skinny Cow 0.13, Nips 0.12, Nescafé Momento 0.18, Polar herring 0.20,
  Epson 0.05, Dell 0.08, Brother 0.06, Just For Men 0.12, Crayola 24 0.10,
  Crayola 64 0.30.
- `mujoco_grasp_sim/assets/props/README.md` (committed): source, CC-BY 4.0
  attribution to Google Scanned Objects, how to re-download.
- `mujoco_grasp_sim/scripts/download_props.py`: for each manifest entry,
  fetch `model.obj`, `texture.png` and `model_collision_*.obj` (listed via
  the GitHub contents API) into `assets/props/gso/<model_id>/`; skip files
  already present; stdlib only (`urllib`), so it runs in any env.
- `.gitignore`: `mujoco_grasp_sim/assets/props/gso/`.

**2. `sim_grasp/props.py` — pure prop logic (no MuJoCo import).**
- `load_manifest(path) -> list[PropEntry]` — validates unique ids,
  `category ∈ {'food', 'non_food'}`, mass in range; raises `ValueError`.
- `prop_scale(extents) -> float` — uniform scale
  `s = min(1, H_MAX / h, W_MAX / min(w, d))` with `H_MAX = 0.14`,
  `W_MAX = 0.055`, so every prop ends ≤ 14 cm tall (current boxes reach
  11 cm) with a graspable side ≤ 5.5 cm (current boxes ≤ 5.6 cm). Uniform
  scaling keeps textures undistorted.
- `box_inertia(mass, extents) -> (ixx, iyy, izz)` — solid-box inertia.
  Needed because GSO MJCF sets no mass: MuJoCo's default density gives the
  Crunch box 2.56 kg (real: ~0.22 kg), and the 32 overlapping collision
  hulls double-count volume. An explicit `<inertial>` overrides both.
- `prop_body_xml(name, entry, prop_dir, scale, extents) -> (body_xml, assets_xml)`
  — emits, with every asset name prefixed by the object name (`obj_2_tex`,
  `obj_2_vis`, `obj_2_col_0…31`) and absolute file paths (the scene's
  `meshdir` belongs to panda.xml, same reason as the existing
  `_make_mesh_object`):
  - `<texture>` + `<material>` from `texture.png`;
  - visual geom: `mesh=obj_N_vis`, `material`, `contype="0" conaffinity="0" group="2"`;
  - 32 collision geoms: `group="3"` (hidden by `mujoco.Renderer`'s default
    geom groups, so depth/segmap come from the textured visual mesh);
  - `<inertial pos="0 0 h/2" mass=… diaginertia=…>`;
  - `<freejoint name="obj_N_joint"/>`.
  GSO origins sit at the base centre, so `spawn_half_height = 0`.

**3. `SceneConfig` / `SceneGenerator` changes (`scene_generator.py`).**
- New fields (defaults reproduce today exactly):
  - `scene_mode: str = 'boxes'` — `'boxes' | 'props'`.
  - `second_bin_center: tuple | None = None`.
  - `bin_categories: tuple = ('food', 'non_food')` — bin A, bin B.
  - `props_manifest: str | None = None`.
- `ObjectSpec` gains `category: str = ''` and `assets_xml: str = ''`
  (prop assets travel with the spec; the legacy `extra_asset` attribute
  path for `_make_mesh_object` is left untouched).
- `_sample_objects()` props branch: sample `n` props **without
  replacement, balanced across categories** (`ceil(n/2)` from one category
  chosen by the rng, `floor(n/2)` from the other), so every scene has work
  for both bins.
- `SceneConfig.for_props(seed, manifest)` classmethod applies the props
  preset: `scene_mode='props'`, `n_objects_range=(4, 4)`,
  `second_bin_center=(0.45, 0.30)`, `spawn_y=(-0.09, 0.09)` (spawn
  centres stay ≥ 9 cm from both bins' inner walls at y = ±0.18; the
  largest scaled prop's half-diagonal is 7.7 cm — Polar herring,
  15.1 × 3.2 cm — so no prop can spawn overlapping a bin wall at any yaw).
- **Size-aware spawn spacing (props mode only).** A fixed 12 cm spacing
  would let two 15 cm props spawn interpenetrating (explosive separation
  on the first physics step). `_sample_xy_positions(n, radii=None)` gains
  an optional per-object footprint radius (scaled half-diagonal); when
  given, a pair is accepted only if `dist ≥ max(min_object_spacing,
  r_i + r_j + 0.01)`. With `radii=None` the function is unchanged and
  consumes the rng identically, so boxes-mode scenes (and every existing
  seed's layout) are unaffected.
- Bin XML is generated by one helper `_bin_xml(prefix, center, …)`;
  bin A keeps the exact existing geom names (`bin_floor`, `bin_wall_xp`, …)
  and bin B uses `bin_b_floor`, `bin_b_wall_xp`, …, so the default scene
  XML is unchanged.
- `bins() -> list[BinSpec]` where `BinSpec(name, center, inner_half, category)`:
  `[A]` in boxes mode, `[A, B]` in props mode.
- `objects_in_bins() -> dict[str, str]` (object → bin name), sim-oracle,
  same tolerance logic as `objects_in_bin()`, with a comment marking it as
  the thing SP2 replaces. `objects_in_bin()` is unchanged.
- `object_categories: dict[str, str]` and `object_props: dict[str, str]`
  (object → GSO model id) populated in props mode.

**4. CLI + routing (`run_sim_grasp_test.py`).**
- New flag `--scene {boxes,props}` (default `boxes`). `props` builds the
  config via `SceneConfig.for_props`; `--n-objects` still overrides the
  count (validated 2–6).
- Mutually exclusive with `props`: `--instruction`, `--prompt/--click/--box`
  (both assume colour naming and a single bin) — exit with a clear message,
  like the existing exclusivity checks. `interactive_pick.py` is not
  changed.
- `--camera calibrated` with `--scene props` is only allowed once the
  plan's camera-coverage check confirms bin B is in that camera's view;
  otherwise it exits with a message pointing to `lookat`/`fused`.
- Routing (SP1 = oracle): `destination_bin(body) = bin whose category ==
  gen.object_categories[body]`. Both placement sites (single `--execute`
  and the pick-all loop) build the heightmap and
  `OccupancyPlacementPlanner` for that bin's `center`/`inner_half` instead
  of `cfg.bin_center`/`cfg.bin_inner_half`. Boxes mode keeps using bin A
  exactly as today.
- Pick-all "remaining" = objects on the table and in **no** bin (a
  wrong-bin object is not re-sorted).
- Startup print, like today's colour line:
  `[scene] props: obj_0=Crayola_Crayons_24_count (non_food), …`.

**5. Metrics + benchmark.**
- `metrics.json` in props mode adds `scene: "props"`, `routing: "oracle"`,
  per-round `category`, `target_bin`, `landed_bin`, and in `pick_all`:
  `in_correct_bin`, `in_wrong_bin` (lists). `in_bin` keeps its meaning
  (in *any* bin), so existing tooling reads props runs unchanged.
- `benchmark.py --scene {boxes,props}` passthrough; in props pick-all mode
  the summary adds `objects in correct bin: X/Y (Z%)`.
- `analyze_failures.py`: add a `wrong_bin` category (placed in a bin
  other than `target_bin`).

### Error handling

- Missing prop assets → `FileNotFoundError` naming the missing directory
  and the exact `python scripts/download_props.py` command (same style as
  the Menagerie-missing error).
- Invalid manifest → `ValueError` at scene construction, before any
  physics.
- No new silent fallbacks: if the destination bin's placement planner
  finds no free spot, the existing "place at bin centre" fallback applies
  **within that bin**, never the other bin.

### Testing

Standalone assert scripts in the existing style (`PYTHONPATH=.`, no
MuJoCo/GPU):
- `test_props.py` — manifest validation (dup id, bad category, mass range),
  `prop_scale` on the measured extents above (every shortlisted prop ends
  ≤ 0.14 m tall and ≤ 0.055 m on its short side), `box_inertia` against the
  closed form, `prop_body_xml` output (prefixed asset names, absolute
  paths, `group="2"`/`group="3"`, `contype="0"` on the visual geom,
  explicit `<inertial>`), balanced sampling (4 → 2+2, 5 → 3+2, no repeats).
- `test_scene_generator_bins.py` — `bins()` is `[A]` by default and
  `[A, B]` for `for_props`; `_bin_xml` for bin A is string-identical to the
  pre-change output; bin B geom names are prefixed.
- Default-unchanged guard: the generated `_generated_scene.xml` for
  `SceneConfig(seed=0)` is byte-identical before/after (hash recorded in
  the plan's first task).
- Live checks (MuJoCo + GPU): observation render shows textured props and
  both bins; one `--scene props --execute` places into the correct bin;
  the full existing `test_*.py` suite and a boxes `--pick-all` baseline run
  still pass unchanged.

### Sub-project 1 gate (must pass before SP2 starts)

```
python benchmark.py --scene props --seeds 0-9 --mode pick-all \
    --camera fused --backend graspgen --tag props_oracle
```

Pass = **≥ 34/40 objects (85 %) in their intended bin**, **≤ 2 knocked
off the table**, **0 crashed runs**. Reference: the 3-box scene bins
30/30 (100 %) with the same GraspGen/fused config. A CGN run with the same
seeds is recorded for reference only (not gated). On failure: classify
with `analyze_failures.py`, then adjust curation / `H_MAX` / `W_MAX` /
masses and re-run — do not start SP2 on a failing gate (P1 lesson:
non-box shapes previously dominated failures).

## Out of scope (SP1)

Scene graph, NIM identification/categorisation, perceived routing
(SP2/SP3); `--instruction`/`--prompt` in props mode; `interactive_pick.py`
props support; clutter/stacking; more than two bins or categories;
real-robot runs.
