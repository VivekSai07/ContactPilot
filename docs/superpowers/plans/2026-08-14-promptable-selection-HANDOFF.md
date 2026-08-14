# Handoff: Promptable object selection (sub-project 2 of idea 2)

**Read this first.** You are a Claude Code session running inside WSL2
(`~/ContactPilot`), picking up fresh work — unlike the GraspGen handoff,
this isn't a mid-task pivot. Sub-project 1 (GraspGen backend) is already
merged to `main` and verified. This is the next, independent sub-project.

## Start here

```bash
cd ~/ContactPilot
git checkout main
git pull origin main
```

You should see (among the GraspGen-backend commits already merged):
- `docs/superpowers/specs/2026-08-14-promptable-selection-design.md` — the design spec
- `docs/superpowers/plans/2026-08-14-promptable-selection.md` — the 7-task implementation plan
- this file

**Use `superpowers:using-git-worktrees` to set up an isolated worktree/branch
first** (same pattern as the GraspGen work — protects `main` from a
half-finished 7-task change), then **`superpowers:subagent-driven-development`**
to execute the plan: fresh subagent per task, task review after each,
ledger-tracked. There's no existing ledger for this plan yet — you're
starting Task 1 fresh, not resuming.

## What this sub-project is

Read the design spec first — it has the full architecture, research
findings (including the exact SAM 3 API, verified against source, not just
the README), and reasoning behind every decision. In short: add
`--prompt "the red box"` / `--click X,Y` / `--box X1,Y1,X2,Y2` to
`run_sim_grasp_test.py`, resolving to a target object via Meta SAM 3,
producing a segmap the existing grasp pipeline (CGN or GraspGen — both
already work, unchanged) already knows how to consume.

## The one thing that isn't scriptable — start it immediately

**SAM 3's checkpoints are gated on Hugging Face.** Task 1, Step 1 of the
plan is: request access at https://huggingface.co/facebook/sam3. This may
take time to be approved and blocks Step 5 of Task 1 (though Steps 2-4 —
cloning SAM 3, creating the env, installing it — don't need approval yet).
Do this first, in parallel with everything else, exactly as the plan says.

## Why WSL2 again (short version — full reasoning in the design spec)

GraspGen's compiled CUDA extensions hit a confirmed-unfixable Windows/MSVC
bug, so that work moved to WSL2. SAM 3's install is plain `pip install -e .`
with **no custom CUDA extension** — meaningfully lower risk than GraspGen's
situation — but it still needs its own environment (Python 3.12, distinct
from `cgn_torch`'s 3.10 and `graspgen_torch`'s 3.10), and starting directly
in WSL2 avoids re-risking a repeat of the GraspGen saga for no reason.

## What's already true in this WSL2 checkout (from the GraspGen work)

- `cgn_torch`-equivalent Python env for the sim side should already exist
  here from the GraspGen sub-project (check what the previous session set
  up — it wasn't necessarily named identically to the Windows-side
  `cgn_torch`; look at how `graspgen_torch` was configured there for the
  pattern).
- `graspgen_torch` env and the GraspGen backend itself are already working
  and merged — you can use `--backend graspgen` for any of this plan's
  smoke tests that need a working grasp backend (Task 5, Task 6), it's not
  limited to CGN.
- Git identity is already correct (`VivekSai07`/`viveksaisurya07@gmail.com`)
  — **do not add a Co-Authored-By trailer to any commit**, this has been
  corrected once already this project and should not regress.

## What's new for this sub-project (nothing carries over)

- A fresh `sam3_torch` conda env (Task 1) — nothing from `cgn_torch` or
  `graspgen_torch` is reusable here beyond the general pattern.
- A fresh SAM 3 clone (`~/sam3` or wherever you place it, sibling to
  `~/ContactPilot`, not inside it — same reasoning as the GraspGen clone).

## When done

Same as before: PR to `main`, decisive benchmark result
(`benchmark_prompt_selection.py`) recorded in `ROADMAP.md` with real
numbers (the plan has no predetermined pass value — write what actually
happens), final whole-branch review clean, then
`superpowers:finishing-a-development-branch`.

If you want a second opinion on the results the way the Windows-side
session independently re-verified the GraspGen numbers before trusting
them, that's a reasonable thing to ask for rather than assume — the same
"don't just trust the report" principle applies to your own work here too.
