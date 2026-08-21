# Reasoning layer: natural-language task instructions

**Status:** Phase 1 implemented (2026-08-21), tracked as `ROADMAP.md` P8.
Phase 2 not started, gated on Phase 1 proving insufficient in practice.

## Origin

Came from a Google AI Mode conversation about
[ReflectVLM](https://github.com/yunhaif/reflect-vlm) ("Reflective Planning:
Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation")
as a candidate "reasoning layer" sitting above the existing perception
(SAM3) + grasp (CGN/GraspGen) + placement pipeline.

## Reality check on ReflectVLM (fetched from the actual repo, not just the
Google AI summary)

- 13B-parameter VLM (`llava-v1.5-13b` base) + a separate diffusion dynamics
  model for the "reflection"/look-ahead step. ~26GB VRAM in fp16; even
  aggressive 4-bit quantization (~7-8GB) would consume this machine's
  **entire** 8GB budget (RTX PRO 2000 Blackwell laptop GPU), leaving zero
  headroom for GraspGen/SAM3/MuJoCo rendering running alongside it.
- Its `act()` interface needs a **goal image**, not free text — you'd have
  to synthesize a picture of the desired end state first, which is itself
  an unsolved sub-problem here.
- Pretrained checkpoints were trained/evaluated on the paper's own
  procedurally-generated peg-insertion/interlocking-assembly tasks — a
  different object/visual distribution than our box-in-bin scenes. Real
  domain-transfer risk. Fine-tuning support is listed as "**Coming soon**"
  in the repo — not available yet.
- License (MIT + Apache 2.0) is not a blocker, hardware/domain fit is.

## Reframed goal (after clarifying with the user)

ReflectVLM's actual novelty — visual look-ahead to predict "would this
placement be physically stable" — is **redundant** with the
`intelligent-bin-placement` plan already in progress (deterministic
occupancy-heightmap + free-space search, zero VRAM cost, already solves
"don't stack, don't crush").

The genuinely new capability the user wants: **natural-language task
instructions that control both pick order and placement destination**,
e.g. "pick the blue cube first and put it on the left, then the red one on
the right." Confirmed scope (via clarifying questions):
- Object selection via description → already solved by the existing SAM3
  `PromptSelector` (`--prompt "the brown box"`), no new model needed there.
- Destination via *relative*/*semantic* language ("left", "next to the blue
  cube") → genuinely new; needs to resolve into a geometric constraint fed
  into the placement planner's free-space search.
- Instructions can also reorder/select which object gets picked first, not
  just where it lands.
- NOT yet scoped: whole-scene attribute sorting (e.g. "arrange smallest to
  largest") — deferred, not asked for.

## Approaches considered

**A. (recommended) Lightweight text-only LLM parser + geometric spatial
resolver.** A small LLM (local quantized 3-8B, or even a one-shot cloud API
call — this parses once per run, not per-frame) turns the instruction into
an ordered step list: `[{object_description, spatial_relation, reference}, ...]`.
`object_description` → existing SAM3 `PromptSelector` (unchanged).
`spatial_relation` → a directional bias fed into the same
`OccupancyPlacementPlanner`/heightmap machinery from the placement-planner
work (still guarantees no-collision, just prefers a region). Minimal VRAM,
reuses everything already built, three independently-testable units
(parse → ground → place).

**B. Adopt ReflectVLM as originally proposed.** Rejected for now — see
reality check above (VRAM, goal-image requirement, domain-transfer risk,
no fine-tuning support yet). Would solve a problem (physical-stability
lookahead) this project doesn't actually have.

**C. Small general VLM (e.g. Qwen2-VL 2B/7B) doing language+vision
grounding in one shot.** Lighter than ReflectVLM, sees image + text
together (helps with attribute references like "the tallest one"). Noted
as a possible later upgrade if plain-text parsing + SAM3 grounding proves
insufficient for tricky references — not needed to start.

## Sequencing decision (2026-08-21)

Strictly two-phase, sequential — full detail in `ROADMAP.md` P8:

1. **Phase 1 = Option A** (lightweight text-only LLM parser + geometric
   spatial resolver, reusing SAM 3 + the occupancy placement planner) —
   start once the intelligent bin-placement work (P7) is verified robust
   end-to-end.
2. **Phase 2 = Option C** (upgrade to a small vision+language model like
   Qwen2-VL) — only if Phase 1's text-only grounding proves insufficient
   for tricky attribute references. Do not start this before Phase 1 is
   proven robust.

Option B (ReflectVLM as originally proposed) remains rejected.

## Phase 1 model choice: `meta/llama-3.1-8b-instruct` via NVIDIA NIM (2026-08-21)

Ran an empirical bake-off (`docs/research/bakeoff_instruction_parser.py`,
standalone/stdlib-only, no new conda-env dependency) across NVIDIA's
`build.nvidia.com` free-tier catalog against 8 representative test
instructions (single-object, multi-step, "stack" requests, attribute
sorting).

**Decisive factor**: each model's NIM card has an explicit "Capabilities"
section. Only `meta/llama-3.1-8b-instruct` is confirmed **"Structured
Output: Supported"** — `mistralai/mistral-nemotron`,
`nvidia/nemotron-3-nano-30b-a3b`, `meta/llama-3.3-70b-instruct`, and
`nvidia/llama-3.1-nemotron-nano-8b-v1` are all confirmed **"Structured
Output: Not supported"**.

Empirical results confirmed this matters in practice, not just on paper:

| Model | valid_json | valid_schema | avg latency | errors |
|---|---|---|---|---|
| `meta/llama-3.1-8b-instruct` (+ `response_format=json_object`) | 8/8 | 8/8 | **~0.7s** | 0 |
| `mistralai/mistral-nemotron` | 8/8 | 8/8 | 19.7s (worst case 124.5s) | 500 Internal Server Error |
| `nvidia/nemotron-3-nano-30b-a3b` | 7/8 | 7/8 | 6.9s | 503 Service Unavailable (x3), one response leaked its reasoning trace as plain text instead of JSON |

`response_format={"type": "json_object"}` requires a top-level JSON
**object**, not a bare array — the schema is `{"steps": [...]}`, not a
top-level array (an early attempt at a top-level array silently collapsed
multi-step instructions to a single step). One round of system-prompt
refinement (explicit "one step per distinct object, never repeat the same
pick_target" + a negative example) fixed an object-duplication quirk seen
in round 1.

**Decision**: Phase 1 uses `meta/llama-3.1-8b-instruct` via NVIDIA's
OpenAI-compatible NIM endpoint (`https://integrate.api.nvidia.com/v1`),
with `response_format={"type": "json_object"}`, and the system prompt now
committed in `docs/research/bakeoff_instruction_parser.py`. No local
model/VRAM footprint at all (0MB, matches the "cloud API" option from
`ROADMAP.md` P8). Requires `NVIDIA_API_KEY` in a local `.env` (gitignored,
see `.env.example`).

**Known minor residual quirk**: for a literal "stack X on Y" instruction,
the parser sometimes emits a spurious extra step for the *reference*
object Y (as if it also needed picking). Low risk in practice — Phase 1's
planned pipeline already skips/logs a warning for any `pick_target` that
doesn't resolve to an object still on the table, so a phantom re-pick of
an already-placed object degrades gracefully rather than breaking anything.
Not chased further with more prompt iterations (diminishing returns).

## Phase 1 implemented (2026-08-21)

Architecture fleshed out, approved, and built. See
`docs/superpowers/specs/2026-08-21-reasoning-layer-phase1-design.md` for
the full design (module boundaries, data flow, error handling, testing)
and `docs/superpowers/plans/2026-08-21-reasoning-layer-phase1.md` for the
implementation record, including two bugs found and fixed via live smoke
testing. Results summarized in `ROADMAP.md` P8. Phase 2 (Option C) remains
un-started, gated on Phase 1 proving insufficient in practice.

A follow-up usability fix landed the same day, found while manually
running `--instruction` end to end: scene objects previously got random
RGBA, making instruction text like "the red cube" unreliable to write
(and occasionally impossible, if no object was actually that color).
Fixed to a deterministic red/green/blue-first palette
(`color_utils.object_color`) — see `ROADMAP.md` P8's dated bullet for
detail.
