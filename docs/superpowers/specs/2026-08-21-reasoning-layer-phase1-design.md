# Reasoning Layer Phase 1 (Option A) — Design

## Problem

The pick-and-place pipeline (`run_sim_grasp_test.py --pick-all`) picks
objects in grasp-score order and always places into the next free bin
slot — there is no way to say "pick the blue cube first and put it on the
left, then the red one on the right." `ROADMAP.md` P8 / the research note
(`docs/research/2026-08-20-reasoning-layer-reflectvlm.md`) settled the
model choice (`meta/llama-3.1-8b-instruct` via NVIDIA NIM, empirically
validated) and rejected ReflectVLM (VRAM/goal-image/domain-transfer
issues), but left Phase 1's actual architecture — module boundaries, data
flow, error handling, testing — unspecified. This doc is that missing
piece.

## Constraint

Phase 1 is **text-only** (Option A): a small LLM parses free-text
instructions into a structured step list; object grounding reuses the
existing SAM 3 `PromptSelector` unchanged; placement bias reuses the
existing `OccupancyPlacementPlanner` unchanged. No new vision/VLM model,
no GPU footprint beyond what's already running. Phase 2 (a
vision+language model) is out of scope here and gated on Phase 1 proving
insufficient in practice.

## Decisions (confirmed with the user)

- **Scope**: `run_sim_grasp_test.py --pick-all` only. `interactive_pick.py`
  stays click-driven, untouched.
- **Left/right convention**: camera-view left/right (primary/fused
  camera's image-left vs image-right), not robot-frame or a bare world
  axis — matches what a person looking at `observation.png` would call
  "left."
- **"Near `<object>`" resolution**: re-detected via SAM 3 on the current
  observation each time it's referenced (not remembered from our own
  earlier placement choice) — kept "vision-only" pure at the cost of a
  disambiguation edge case (see Error handling).
- **NIM API failure**: abort the run with a clear error. No silent
  fallback to score-based picking — that would silently ignore the user's
  instruction.
- **Spatial-relation → placement bias**: hard region restriction with
  fallback to the full bin (Approach A of three considered). Rejected
  alternatives: a soft scoring bias inside `plan()` itself (touches
  already-shipped P7 code, needs a tunable with no principled default);
  hard restriction with no fallback (a transiently-full preferred region
  would wrongly fail the whole placement instead of degrading).

## Architecture

Three new, independently-testable pieces. None modify already-shipped
code (`prompt_selector.py`, `placement_planner.py`, `executor.py` are all
consumed, not changed).

```
--instruction "pick the blue cube first and put it on the left, ..."
        │
        ▼
instruction_parser.parse_instruction(text) -> list[Step]
 (one NIM API call, once per run, not per-frame)
        │
        ▼
ordered [Step(step, pick_target, place_relation, place_reference), ...]
        │
        ▼
--pick-all round loop (existing structure, extended):
        │
        ├── current step's pick_target
        │       │
        │       ▼
        │   PromptSelector.select(rgb, prompt=pick_target)   [existing, P5]
        │   (redone every round against the fresh observation,
        │    same as grasp prediction already is)
        │       │
        │       ▼
        │   matched seg_id -> restrict `allowed` to just that id
        │   no match -> log warning, advance to next step
        │
        └── current step's (place_relation, place_reference)
                │
                ▼
            spatial_relation_resolver.resolve(...)
            -> sub-region OccupancyPlacementPlanner, or None
                │
                ▼
            OccupancyPlacementPlanner.plan(...)   [existing, P7, unchanged]
                │
                ▼
            PlacementPose (as today) -> executor.execute()/place()
```

### `sim_grasp/instruction_parser.py`

```python
@dataclass
class Step:
    step: int
    pick_target: str
    place_relation: str   # one of: left_of, right_of, near, center, none
    place_reference: str | None

def parse_instruction(text: str, api_key: str | None = None) -> list[Step]:
    ...
```

Sends `text` to NVIDIA's OpenAI-compatible NIM endpoint
(`https://integrate.api.nvidia.com/v1/chat/completions`) using
`meta/llama-3.1-8b-instruct` with `response_format={"type": "json_object"}`
and the same system prompt already validated in
`docs/research/bakeoff_instruction_parser.py`'s `SYSTEM_PROMPT` constant.
That script stays standalone/stdlib-only by design (so anyone can rerun
the bake-off without the `cgn_torch` env); importing it from
`sim_grasp.instruction_parser` would pull in `sim_grasp/__init__.py`'s
full import chain (mujoco, cv2, torch-adjacent modules) into what's meant
to be a lightweight one-off script. The prompt is therefore copied
verbatim into `instruction_parser.py`'s own `SYSTEM_PROMPT` constant, with
a one-line comment in both files cross-referencing the other and noting
they must be kept in sync by hand — an accepted, documented duplication
(the same trade-off this project already makes at its subprocess/conda-env
isolation boundaries elsewhere, e.g. the CGN worker). Validates the
response is a JSON object with a `"steps"` array of objects with exactly
the five keys (raises `ValueError` with a specific message otherwise —
wrong type, missing key, extra key, empty array, or `place_relation`
outside the five allowed values). Reads `api_key` from the
`NVIDIA_API_KEY` env var if not passed explicitly (loaded from `.env` via
the same minimal stdlib-only loader already written for the bake-off
script — copied, not imported, for the same reason as the prompt).

### `sim_grasp/spatial_relation_resolver.py`

```python
def resolve(place_relation: str, place_reference: str | None,
           bin_center: tuple[float, float], bin_inner_half: float,
           T_world_cam: np.ndarray,
           prompt_selector: 'PromptSelector | None' = None,
           rgb: np.ndarray | None = None,
           work_dir: str | Path = '.') -> 'OccupancyPlacementPlanner | None':
    ...
```

Returns a **new, differently-configured** `OccupancyPlacementPlanner`
instance scoped to a smaller square sub-region of the bin, or `None` for
`place_relation == "none"` (caller then uses the plain bin-wide planner
exactly as it does today — zero behavior change for uninstructed runs).
`plan()` and `OccupancyPlacementPlanner.__init__` are never modified:
`OccupancyPlacementPlanner` only accepts a scalar `bin_inner_half` (a
square region), which cannot represent a true half-bin *rectangle* — so
every relation, including `left_of`/`right_of`, resolves to a **smaller
square** of half-size `bin_inner_half / 2` (the same size the `center`
relation already uses, for one shared, simple mental model), differing
only in where that square's center sits. This is a deliberate, documented
simplification (a "right_of" instruction biases toward the right-middle
portion of the bin, not a literal full-height right strip) that keeps the
already-shipped P7 planner code completely untouched.

Per-relation sub-region (all share `half = bin_inner_half / 2`; only the
center point differs):
- `left_of` / `right_of` (reference is always `"bin"` in practice — a
  reference other than `"bin"` for these two relations is treated the
  same, since only `"bin"` splits are supported in Phase 1): center =
  `bin_center` shifted by `half` along whichever world axis (`x` or `y`)
  is most aligned with the primary camera's image-right direction —
  computed once as:
  ```python
  def _camera_view_axis(T_world_cam: np.ndarray) -> tuple[str, float]:
      """World axis ('x' or 'y') most aligned with the camera's local +X
      (image-right, OpenCV convention — see frames.py) direction, and the
      sign of that alignment (+1 if image-right points toward +axis)."""
      cam_right_world = T_world_cam[:3, 0]
      x_comp, y_comp = float(cam_right_world[0]), float(cam_right_world[1])
      if abs(x_comp) >= abs(y_comp):
          return 'x', (1.0 if x_comp >= 0 else -1.0)
      return 'y', (1.0 if y_comp >= 0 else -1.0)
  ```
  `right_of` shifts the center by `+sign * half` along `axis`; `left_of`
  shifts by `-sign * half`. No clipping needed: by construction the
  shifted square's far edge lands exactly on the full bin's own edge on
  the split axis, and is centered (well within bounds) on the other axis.
- `center`: `bin_center` itself, unshifted (half-size `bin_inner_half /
  2`, i.e. half the bin's linear size, one quarter the area).
- `near <object>`: runs `PromptSelector.select(rgb, prompt=place_reference,
  work_dir=work_dir)` on the current observation, keeps only masks whose
  world-XY centroid (via the same `depth_to_pointcloud`/`transform_points`
  path `compute_object_footprint` already uses) falls within
  `bin_inner_half` of `bin_center` (i.e. actually inside the bin, not a
  same-description object still on the table), takes the
  highest-confidence remaining match, and centers the same `half =
  bin_inner_half / 2` square on that centroid — **clamped** so the square
  stays fully inside the full bin's own bounds (`center_axis = clip(
  centroid_axis, bin_center_axis - bin_inner_half + half, bin_center_axis
  + bin_inner_half - half)` per axis; unlike `left_of`/`right_of`/`center`,
  an arbitrary detected centroid near an edge needs this explicit clamp).
  No match after filtering → logs a warning, returns `None` (caller falls
  back to the plain bin-wide planner for that object).

If the sub-region planner's `.plan()` returns `None` (region is fully
occupied), the caller falls back to the plain bin-wide
`OccupancyPlacementPlanner` for that same object, logging a warning — the
fallback lives in the `--pick-all` loop (the one already-existing planner
instance it constructs today), not inside the resolver, so the resolver's
contract stays simple: "a scoped planner, or None."

### `run_sim_grasp_test.py` changes

- New `--instruction TEXT` CLI flag (only meaningful with `--pick-all`;
  reject `--instruction` without `--pick-all` at argument-parsing time
  with a clear error, since the round-loop restructuring below assumes
  the pick-all path).
- When `--instruction` is given: call `parse_instruction` once, before the
  round loop starts. Any exception propagates and aborts the run (matches
  the "fail loudly" decision) — no try/except added around this call.
- Round loop: today's `remaining`/`allowed` computation (rank all
  not-yet-picked, not-3x-failed objects by grasp score) is replaced, when
  an instruction is active, by tracking a single pointer to the current
  step plus a per-step miss counter (mirroring the existing `fail_count`
  convention that already gives up on a grasp target after 3 failed
  attempts, so the new logic reuses a familiar retry budget rather than
  inventing a different one): each round, resolve the current step's
  `pick_target` via `PromptSelector.select` against that round's fresh
  observation. Matched → `allowed = {that seg_id}` for this round (the
  existing `rank_candidates`/`cand[0]` grasp-selection logic downstream is
  untouched — it's just handed a 1-element `allowed` set instead of the
  full remaining set). Not matched → increment that step's miss counter;
  once it reaches 3, log a warning and permanently advance the pointer to
  the next step (a transient miss — e.g. one round's occlusion — gets
  retried on the next round's fresh observation instead of eliminating
  the step immediately).
- Placement: after a successful pick under an active step, call
  `spatial_relation_resolver.resolve(...)` with that step's
  `place_relation`/`place_reference`; if it returns a planner, try that
  first, falling back to the existing plain `OccupancyPlacementPlanner` on
  `None` (either from the resolver itself or from a full sub-region).
- No instruction given: every line above is skipped; the existing
  behavior is byte-for-byte unchanged (confirms zero regression risk for
  every existing benchmark/test that doesn't pass `--instruction`).

## Error handling

- NIM call fails (missing/invalid key, network error, non-2xx response,
  malformed JSON, schema violation after `instruction_parser`'s own
  validation) → the run aborts with a clear, specific error message. No
  fallback to score-based picking.
- A step's `pick_target` doesn't resolve to any object still on the table
  within 3 consecutive rounds → log a warning naming the step and the
  description, advance to the next step. If every step fails to resolve,
  `--pick-all` naturally ends with "nothing left to pick" exactly as it
  does today with an empty `remaining` list.
- A step's `place_reference` (for `near`) doesn't resolve → log a warning,
  fall back to the plain bin-wide planner for that placement (not a run
  abort — placement degrading gracefully was always the point of
  Approach A).
- Two already-placed objects share a `place_reference`'s description
  (e.g. two same-colored cubes) → `PromptSelector.select` returns
  whichever mask SAM 3 scores highest; not disambiguated further in Phase
  1 (documented, known limitation — the scene generator's random-per-object
  RGBA makes exact color collisions rare but not impossible; revisit only
  if it causes a real observed failure).

## Testing

1. `test_instruction_parser.py`: schema validation against canned JSON
   strings (valid, malformed, missing/extra key, wrong type, empty
   `steps`, invalid `place_relation` value) — no live API calls in the
   test, matching this repo's no-pytest/standalone-script convention.
2. `test_spatial_relation_resolver.py`: synthetic heightmaps/footprints
   (same fixture style as `test_placement_planner.py`) for `left_of`,
   `right_of`, `center`, and `near` (with a stubbed/fake `PromptSelector`
   so no SAM 3 subprocess runs in the test), including the
   sub-region-full fallback-to-`None` path.
3. One live smoke test: `run_sim_grasp_test.py --pick-all --instruction
   "..." --no-vis` end-to-end on a real scene, confirming pick order and
   left/right placement match the instruction by eye (`observation.png`
   + `execution.gif`).

## Implementation notes (2026-08-21)

Implementation matched this design's module boundaries and interfaces
exactly (`instruction_parser.py`, `spatial_relation_resolver.py`, and the
`--pick-all` wiring all landed as specified, including the square-only
sub-region geometry and the 3-attempt retry budget). Three behaviors were
refined beyond what this doc's first-pass implementation did, two found
via the mandatory live smoke test (Testing item 3) and one found by task
review reading the diff against this doc:

- **Step advancement was underspecified.** This doc's round-loop
  description covered *failing* to match a step (advance after 3 misses)
  but never said what happens after a step's object is *successfully*
  picked and binned. Implementation adds: once `entry.get('in_bin')` is
  true for the active step's object, advance `step_idx` and reset the
  miss counter. Without this, a single-step instruction would retry the
  same already-satisfied step forever against whatever objects remain.
- **`pick_target` resolution needed to try more than the top SAM 3
  match.** This doc's data-flow section described resolving `pick_target`
  via `PromptSelector.select` without specifying which match to use when
  several come back (a real, common case in this project's scenes, which
  can spawn multiple identically-described objects, e.g. three green
  cuboids). Implementation tries every returned match by descending
  score, taking the first that resolves to an object still on the table,
  rather than only the single highest-scoring one — otherwise the
  top match can be an object a *previous* step already placed.
- **The `near` relation's first-pass implementation dropped this doc's
  location filter.** This doc specifies keeping only masks whose
  centroid falls inside the bin before ranking by score; the first
  implementation instead took the single top-scoring match
  unconditionally, which could resolve to an identical-looking object
  still on the table rather than the actually-placed reference (found by
  task review, not the original unit tests, which only exercised a
  single in-bin candidate). Fixed to match this doc: candidates are
  filtered to those within `bin_inner_half` of `bin_center` first, then
  the highest-scoring survivor is used. A regression test (two
  candidates, the out-of-bin one scored higher) was added to
  `test_spatial_relation_resolver.py`.

All three are refinements toward the specified behavior, not accepted
departures from it (the design's stated *intent* — one step per object,
resolved against the current table state, with `near` matching only
already-placed objects — is what the fixes actually enforce).
