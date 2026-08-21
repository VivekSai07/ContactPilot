# Agent instructions for ContactPilot

Read automatically by GitHub Copilot (workspace-level agent instructions).
Claude Code is pointed here from `CLAUDE.md`. Applies to every task in this
repo, for any agent — not a single-workflow document.

## Documentation Sync Workflow (mandatory — do not ask first)

Whenever a task produces one of the following, update the affected docs as
part of *finishing* the task itself — don't ask "should I update the docs?"
and don't leave it for later. Treat it the same whether the discovery came
from the agent's own testing or from the human partner manually running the
project and reporting a result back mid-session.

- **A bug found and fixed** (during implementation, live testing, or the
  human manually running something and reporting back) → add a dated
  bullet to the relevant `ROADMAP.md` section (P1, P2, ... P8, ...)
  describing what broke, the fix, and concrete before/after numbers or
  behavior (quote real command output, not a vague claim). If the bug
  relates to an existing `docs/research/*.md` investigation, update that
  note's `Status:` line too.
- **A new capability, module, or integration lands** (new CLI flag, new
  backend, new subsystem) → update `ROADMAP.md`'s relevant section header
  tag (e.g. `[PLANNED, not started]` → `[Phase 1 IMPLEMENTED ...]`) and add
  a dated bullet describing what was built and how it was validated (live
  smoke test output, a benchmark run — real evidence, not an assertion).
- **Implementation diverges from an approved `docs/superpowers/specs/*.md`
  design** → add/extend that spec's "Implementation notes" section
  documenting the divergence and why, rather than silently editing the
  original design sections or leaving the spec stale and inaccurate.
- **A new or updated `docs/research/*.md` file** → update
  `docs/research/README.md`'s index in the *same* commit. (A past, real
  gap: this was missed once and only caught by a later review — don't
  repeat it.)
- **Every fix site** gets a one-line code comment explaining *why*, not
  what the next line already shows — matches this project's existing
  convention throughout `sim_grasp/`.

This is an agent-judgment workflow (deciding *which* doc needs updating,
writing the actual content), not something a deterministic hook can do —
it lives here as a standing instruction, not a `.github/hooks/*.json`.

## Branch/commit hygiene

Every change — including docs-only changes — goes on a feature branch +
PR, never committed directly to `main` (check `git log`: even a
single-file research-notes-folder addition went through a PR). Run
`git branch --show-current` before the first commit of any new task.

## Where things live (see `CLAUDE.md` for the full project map)

- `ROADMAP.md` — single source of truth for "what's done, what changed,
  why," organized by numbered `P<n>` sections.
- `docs/research/` — informal investigation notes, indexed in
  `docs/research/README.md`.
- `docs/superpowers/specs/` — approved architecture/design docs.
- `docs/superpowers/plans/` — task-by-task implementation plans.
