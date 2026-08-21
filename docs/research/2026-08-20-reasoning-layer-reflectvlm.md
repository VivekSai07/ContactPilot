# Reasoning layer: natural-language task instructions

**Status:** brainstorming in progress (approaches proposed, not yet approved).

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

## Next step (not yet done)

Flesh out Approach A's architecture in detail (exact module boundaries,
data flow, error handling, testing) and get it approved as a proper design
before writing a spec/plan under `docs/superpowers/`.
