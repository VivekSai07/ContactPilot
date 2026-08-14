# Design: Promptable object selection (SAM 3)

Date: 2026-08-14

## Problem

`mujoco_grasp_sim` can only target a specific object numerically
(`--pick-object SEG_ID`, reading MuJoCo's ground-truth instance segmap).
There's no way to say "grasp the red mug" — the original motivation for idea
2. This is sub-project 2 of idea 2; sub-project 1 (GraspGen backend,
`docs/superpowers/specs/2026-08-12-graspgen-backend-design.md`) is merged
and verified (GraspGen beats CGN: 15/15 vs 14/15 binned, eliminated
`closed_on_air` as the dominant failure mode).

## Goals

- Add text-prompted object selection (`--prompt "the red mug"`) and
  click/box-prompted selection (`--click X,Y`, `--box X1,Y1,X2,Y2`), all
  resolving to a segmap the existing pipeline already knows how to consume
  — zero changes to `GraspPredictor`, `feasibility.py`, `executor.py`, or
  either grasp backend (CGN or GraspGen).
- Real segmentation on the sim's rendered RGB, not MuJoCo's ground-truth
  segmap — this is what makes the feature meaningful (transfers to a real
  camera later) rather than a lookup table.
- An automated accuracy benchmark (not just manual spot-checks), despite
  scene objects getting randomized colors per seed.

## Non-goals

- Real-camera/RealSense integration — sim-only for this sub-project.
- Video/tracking prompts (SAM 3 supports this; we only need single-image).
- "Grasp all objects matching X" as a pick-all variant — out of scope;
  ambiguous multi-match requires a more specific prompt or `--prompt-index`,
  same spirit as `--pick-object`/`--grasp-index` today.
- Fine-tuning or training SAM 3 — inference only, off-the-shelf checkpoint.

## Research findings (2026-08-14)

- **Model**: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
  — Meta's SAM 3, actively maintained (updated as recently as this date).
  Unlike SAM/SAM2, it natively supports **text prompts** for open-vocabulary
  segmentation ("detect, segment, and track objects using text or visual
  prompts"), and can return multiple instances matching one concept — no
  need for a separate detector (e.g. Grounding DINO) bolted on, simplifying
  the architecture originally envisioned.
- **License**: custom Meta "SAM License" (not Apache, unlike SAM/SAM2) — no
  explicit commercial/MAU restriction found on a scan for such language, but
  it's not a standard permissive license either. Same category of
  consideration as GraspGen's NVIDIA license; confirm personal/research use
  is fine before committing to this the way it was for GraspGen.
- **Install**: plain `pip install -e .` against `torch==2.10.0+cu128` — **no
  custom CUDA extension compilation** (unlike GraspGen's `pointnet2_ops`/
  `torch_scatter`, which hit an unfixable Windows/MSVC bug). Optional
  extras (`flash-attn-3`, `cc_torch`) are compiled and skippable — do not
  install them; they're not required. Requires **Python 3.12+**, distinct
  from `cgn_torch`'s 3.10 and `graspgen_torch`'s 3.10, so still needs its
  own env regardless of the lower compile risk.
- **Checkpoints are gated**: unlike GraspGen's public HF repo, SAM 3's
  checkpoints require requesting access at
  https://huggingface.co/facebook/sam3 and waiting for approval, then
  `hf auth login`. This is a manual, human-gated prerequisite — start it
  early, independent of everything else in this plan.
- **Real API** (verified against `sam3/model/sam3_image_processor.py`
  source, not just the README):
  - `Sam3Processor.set_image(image)` → `state` (accepts PIL Image, torch
    Tensor, or numpy array).
  - `Sam3Processor.set_text_prompt(prompt: str, state: Dict)` → runs
    inference, returns `{"masks": ..., "boxes": ..., "scores": ...}` —
    **can return multiple instances** for one text prompt (the "exhaustive
    concept segmentation" capability).
  - `Sam3Processor.add_geometric_prompt(box: List, label: bool, state: Dict)`
    → box/point prompting. Box format is
    **`[center_x, center_y, width, height]`, normalized to [0,1]**
    (not corner coordinates, not pixel coordinates) — `label=True` for a
    positive/foreground box. There is no separate point-prompt method; a
    click is expressed as a small normalized box centered on the click
    pixel.
- **Ground-truth color access**: no changes needed to `scene_generator.py`
  — `model.geom_rgba[gid][:3]` (via the existing `_object_geom()` helper
  pattern already used in `run_sim_grasp_test.py` for shape-priority
  ranking) gives each object's actual spawned RGB directly from the
  compiled MuJoCo model, without needing to thread color metadata through
  `SceneGenerator`.

## Architecture

```
run_sim_grasp_test.py --prompt "TEXT" | --click X,Y | --box X1,Y1,X2,Y2
                    │  (mutually exclusive with each other and --pick-object)
                    ▼
        rgb (already captured this run)
                    │
                    ▼
        sim_grasp/prompt_selector.py (NEW)
           PromptSelector — ALWAYS subprocess (separate sam3_torch env,
           SAM3_PYTHON env var / --sam3-python override — same isolation
           pattern and fail-fast-if-unconfigured behavior as
           GraspGenPredictor/resolve_graspgen_python)
                    │
                    ▼
        sim_grasp/sam3_worker.py (NEW) — runs SAM 3, returns:
           masks (K,H,W) bool, scores (K,), boxes (K,4) pixel coords
           K=1 for click/box (single target by construction);
           K>=1 for text (may match multiple instances)
                    │
                    ▼
        if K>1: print ranked candidates (index | score | box), require
        --prompt-index I to disambiguate (NEW flag, distinct from
        --grasp-index which selects among ranked GRASPS, not selections)
                    │
                    ▼
        synthesized single-object segmap (id=1 where the chosen mask=True,
        0 elsewhere) — REPLACES MuJoCo's ground-truth segmap for this run;
        this is real segmentation on rendered RGB, same as a live camera
                    ▼
        EXISTING GraspPredictor (CGN or GraspGen, unchanged) → feasibility
        → ranking → execution → metrics.json / execution.gif
```

**Verification-only ground truth (benchmark, never the runtime pipeline):**
`sim_grasp/color_utils.py` (NEW) — `rgb_to_color_name(rgb) -> str`, nearest
named-color match in RGB space (small curated table: red, orange, yellow,
green, cyan, blue, purple, pink, brown, gray). `benchmark_prompt_selection.py`
(NEW, top-level, sibling to `benchmark.py`) generates a scene per seed, reads
each object's real spawned RGB via `model.geom_rgba`, builds a genuine
ground-truth prompt (`"the {color_name} box"`), runs `PromptSelector`, and
checks whether the resolved mask actually overlaps the intended object's
ground-truth segmap region (IoU-based) — grading only, never fed into the
pipeline itself.

## Files

- **Create:** `mujoco_grasp_sim/sim_grasp/color_utils.py` —
  `rgb_to_color_name(rgb) -> str`. No SAM 3/sam3_torch dependency —
  pure numpy, testable immediately.
- **Create:** `mujoco_grasp_sim/sim_grasp/sam3_worker.py` — subprocess
  entry point, mirrors `graspgen_worker.py`'s CLI shape:
  `python sam3_worker.py rgb.npy out.npz [--prompt TEXT] [--click X,Y] [--box X1,Y1,X2,Y2]`.
  Writes `masks` `(K,H,W)` bool, `scores` `(K,)`, `boxes` `(K,4)` pixel
  coords to the output npz.
- **Create:** `mujoco_grasp_sim/sim_grasp/prompt_selector.py` —
  `PromptSelector` class + `SelectionResult` dataclass + `resolve_sam3_python()`
  (mirrors `resolve_graspgen_python()` exactly: `SAM3_PYTHON` env var or
  `--sam3-python` override, fails fast, never falls back to `sys.executable`).
- **Modify:** `mujoco_grasp_sim/run_sim_grasp_test.py` — add
  `--prompt`, `--click`, `--box`, `--prompt-index`, `--sam3-python`;
  resolve a selection to a synthesized segmap right after camera capture,
  before grasp prediction.
- **Create:** `mujoco_grasp_sim/benchmark_prompt_selection.py` — the
  automated accuracy benchmark described above.
- **Modify:** `mujoco_grasp_sim/README.md` — document the new flags,
  `sam3_torch` env setup (including the gated-checkpoint-access
  prerequisite), and the benchmark workflow.

## Error handling

- `--prompt`/`--click`/`--box` requested but `SAM3_PYTHON` unset and
  `--sam3-python` not passed: fail fast, same pattern as
  `resolve_graspgen_python`.
- More than one of `--prompt`/`--click`/`--box` passed together: fail fast
  with a clear "mutually exclusive" message (argparse mutually-exclusive
  group).
- `--pick-object` passed together with `--prompt`/`--click`/`--box`: fail
  fast — they serve the same "pick this one object" purpose via different
  mechanisms, combining them is ambiguous, not additive.
- Text prompt matches nothing (`K=0`): exit with a clear message ("no
  object matched the prompt") rather than proceeding with an empty segmap.
- Text prompt matches multiple objects (`K>1`) and `--prompt-index` isn't
  given: print the ranked candidate list and exit, mirroring how
  `--grasp-index` already works for ranked grasps.

## Testing / verification plan

1. **Standalone SAM 3 smoke test** (after requesting HF access + env
   setup): run SAM 3's own basic usage on a sample image, confirm text and
   box prompting both produce real masks. Do this before writing any
   ContactPilot integration code — same "retire the biggest unknown first"
   principle as GraspGen's Task 2, though the compile-risk here is much
   lower since there's no custom CUDA extension.
2. `sam3_worker.py` smoke test: feed it a captured sim RGB frame with a
   text prompt, confirm real masks/scores/boxes come back.
3. `prompt_selector.py` smoke test: same, through the `PromptSelector`
   class (mirrors `GraspGenPredictor`'s direct-class smoke test in the
   GraspGen plan).
4. Single sim run: `run_sim_grasp_test.py --seed N --execute --prompt "the {known color} box"`
   for a seed where the target color is known ahead of time — confirm the
   right object gets picked.
5. **The decisive test**: `benchmark_prompt_selection.py` across several
   seeds — real ground-truth-derived prompts, checking whether the resolved
   mask actually corresponds to the intended object (IoU against the
   ground-truth segmap region). This is the accuracy number to record,
   parallel to how the GraspGen benchmark recorded a comparable pick-rate
   number in `ROADMAP.md`.

## Risks

- SAM 3's checkpoint gating (manual HF approval) is a real, un-timeboxed
  prerequisite — start the access request immediately, independent of
  everything else.
- License terms need the same quick confirmation GraspGen's did (personal/
  research use) before committing further.
- `add_geometric_prompt`'s box format (`[cx, cy, w, h]`, normalized to
  [0,1]) is easy to get backwards (corner coords vs. center+size, pixel vs.
  normalized) — verified directly against the source in this document, but
  worth double-checking against the actual installed version in case it's
  changed by the time this is implemented.
- No custom CUDA extension needed, but this hasn't been tried on this
  specific GPU/WSL2 combination yet — Task 1's smoke test is what actually
  retires this risk, not this document.
