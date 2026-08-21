# Research notes

Raw exploration of ideas/integrations for ContactPilot that haven't (yet,
or maybe ever) become a formal spec + plan under `docs/superpowers/`. Think
of this as a running notebook: half-formed ideas, feasibility checks,
things we looked at and decided against, and reasoning we want to revisit
later — not polished, not necessarily acted on.

Convention: one file per idea/topic, `YYYY-MM-DD-<topic>.md`, dated the day
the idea was first explored (update in place as the idea evolves; add a new
dated file only for a genuinely new/unrelated idea). When an idea graduates
into a real design, it gets a proper spec under `docs/superpowers/specs/`
and this file should say so (with a link) rather than duplicating it.

## Index

- [2026-08-20-reasoning-layer-reflectvlm.md](2026-08-20-reasoning-layer-reflectvlm.md) —
  natural-language task instructions (pick order + relative placement) for
  the MuJoCo pick-and-place pipeline. Status: Phase 1 implemented (see
  `ROADMAP.md` P8), Phase 2 not started.
- [2026-08-21-short-object-finger-table-collision.md](2026-08-21-short-object-finger-table-collision.md) —
  root cause + fix for fingers hitting the table on short objects during
  grasp closing. Status: fixed (see `ROADMAP.md` P1).
