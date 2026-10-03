# Scene graph / knowledge graph from vision

**Status:** graduated into a design on 2026-09-29 —
`docs/superpowers/specs/2026-09-29-semantic-sorting-scene-graph-design.md`,
tracked as `ROADMAP.md` P10. Sub-project 1 (semantic prop scene + second
bin) not started.

## Origin

A Google AI Mode summary of "vision → scene graph / knowledge graph":
detect objects → label attributes → predict pairwise relations → link
nodes to a commonsense KB (ConceptNet) for affordances/context. Same
origin pattern as P8 (ReflectVLM) — so, same treatment: reality-check it
against this repo and an 8 GB GPU before adopting anything.

## Feasibility against the repo (as of main @ bf1b261)

Almost every scene-graph ingredient already exists as geometry:

| Ingredient | Already here | Gap |
|---|---|---|
| nodes (instance masks) | sim segmap; SAM 3 `PromptSelector` on real RGB | — |
| colour attribute | `color_utils.rgb_to_color_name` | — |
| 3D centroid / extent | `placement_planner.compute_object_footprint` | height not exposed |
| left_of / right_of | `spatial_relation_resolver._camera_view_axis` | only used for bin placement |
| near / blocks | `workspace_occupancy.collides_with_neighbors` | not recorded as edges |
| reachable | `reachability.is_reachable` | not attached to nodes |
| in(bin) | `SceneGenerator.objects_in_bin()` | **reads sim qpos — an oracle; a real-robot gap** |

Findings that shaped the decision:
- With calibrated depth + instance masks, relations like `left_of`,
  `near`, `on_top_of` are **measured**, not guessed. A scene graph here is
  a data structure, not a model.
- The P8 instruction parser is text-only — the LLM never sees the scene.
  A serialized graph in its prompt is the cheapest way to ground
  "the tallest one" / "left of the red one" (a lighter alternative to the
  P8 Phase 2 VLM).
- The scene as it stood (exactly 3 plain boxes, ≥ 9 cm apart, no stacking,
  no identities) gives a graph almost nothing to reason about. Relational
  pick-ordering (VMRN/REGRAD, ThinkGrasp, UNOGrasp) needs clutter/stacking;
  commonsense KBs need objects with semantic identity.

## External landscape (checked 2026-09-29)

- **Learned SGG models** (RelTR, EGTR, PSG family): Visual-Genome-trained,
  2D, weak on tabletop geometry; essentially no maintained Hugging Face
  weights (top "scene graph" Hub repo: 31 downloads). Rejected.
- **3D open-vocab scene graphs** (ConceptGraphs, DovSG — RA-L 2025, Hydra):
  room-scale, multi-view, mobile-manipulation oriented. Overkill for one
  fixed eye-to-hand camera over a table. Rejected.
- **LLM planning verified against a scene graph** (VeriGraph — ICRA 2026;
  GAVEL, Sept 2026 — Qwen3-8B, single-task success 41 % → 92 % on
  BEHAVIOR-1K): the pattern that fits this repo's P8 bugs (step index,
  ambiguous matches, missing preconditions). Worth adopting later.
- **Task-oriented grasping with semantic knowledge** (GraspGPT,
  FoundationGrasp, Lan-grasp): needs part semantics (handles, blades) —
  not applicable to box-shaped props.
- **Small VLMs:** Qwen3-VL-2B/4B-Instruct (~3 M Hub downloads each, 4-bit
  variants fit 8 GB). The NIM endpoint already used by P8 also serves
  vision models: `google/gemma-3-4b-it`, `google/gemma-3-12b-it`,
  `meta/llama-3.2-11b-vision-instruct`, `meta/llama-3.2-90b-vision-instruct`,
  `nvidia/cosmos-reason2-8b`, `microsoft/phi-3-vision-128k-instruct` —
  zero local VRAM.
- **Props:** Google Scanned Objects MuJoCo port (kevinzakka/
  mujoco_scanned_objects): 1,030 textured scans, MJCF + 32 V-HACD
  collision hulls each, CC-BY 4.0 / MIT. No category labels. Throwaway
  probe: textures render through `mujoco.Renderer` with printed text
  legible at 640×480; default density gives unrealistic masses (Crunch
  box 2.56 kg vs ~0.22 kg real). The existing `_make_mesh_object` loader
  applies flat `rgba` — it would strip exactly the texture a VLM needs.

## Decision (with the user)

Semantic props + knowledge → **sort into two bins by category**, knowledge
via **hosted NIM models** (vision model identifies a crop, llama-3.1-8b
maps name → category enum). Split into three sequential sub-projects
(scene upgrade with a benchmark gate → scene graph + knowledge → sorting
consumer + metrics). Full rationale and the prop shortlist live in the
design spec, not here.

Not pursued (for now): clutter/stacking pick-order reasoning, colour-only
language grounding via a graph, local VLM worker, ConceptNet.
