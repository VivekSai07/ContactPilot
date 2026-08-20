# Remove Docker Support (Preserved on a Branch) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all Docker support from `main` (the user no longer works at the lab and doesn't want this repo supporting Docker going forward), while permanently preserving the current, working Docker setup on a dedicated branch that will never be deleted.

**Architecture:** Two-step git operation. First, snapshot the current state onto a permanent branch (`docker-support`) before touching anything — this is the preservation step, done once, no further maintenance expected. Second, on a normal short-lived feature branch, delete the 5 Docker-specific files and clean up prose references in `CLAUDE.md`/`ROADMAP.md`, then PR and merge to `main` per this repo's established branch→PR→merge convention.

**Tech Stack:** git only — no code changes beyond deleting files and editing two markdown docs.

**Spec:** No separate spec file — this was scoped as a bounded task in chat during brainstorming.

## Global Constraints

- `docker-support` is a **permanent** branch — once created and pushed, this plan makes no further commits to it, and no task in this plan (or any future one) deletes it.
- Files to remove from `main`: `Dockerfile`, `docker-compose.yml`, `DOCKER.md`, `requirements-docker.txt`, `.dockerignore`. Nothing else.
- Out of scope, do not touch: `GraspGen/docker/*` and `contact_graspnet_pytorch/README.md`'s "docker" mention — both belong to git submodules (vendor/upstream content), not this repo's own code.
- Out of scope, do not touch: `docs/superpowers/plans/2026-07-29-contact-graspnet-dependency-restructure.md` and `docs/superpowers/specs/2026-07-29-contact-graspnet-dependency-restructure-design.md` — historical, dated implementation records; this project's established convention is to document later changes via their own PR/plan rather than retroactively edit historical plans.
- Commit messages are plain — **do not add a `Co-Authored-By: Claude` trailer to any commit**.
- Every code/doc change on `main` goes through its own branch → PR → merge, per this repo's established convention — no direct commits to `main`.

---

### Task 1: Create and push the permanent `docker-support` preservation branch

**Files:** none — pure git branch operation, no file changes.

**Interfaces:**
- Produces: branch `docker-support`, pushed to `origin`, containing the exact current state of `main` (all 5 Docker files + all current doc references, untouched). Task 2 branches off `main` (not this branch) to do the actual removal.

- [ ] **Step 1: Confirm current state and create the branch**

```bash
cd ~/ContactPilot
git checkout main
git pull origin main
git status --short   # expect clean
git checkout -b docker-support
```
Expected: new branch `docker-support`, identical content to `main` at this exact commit — no changes made yet, this step only creates the branch.

- [ ] **Step 2: Push it**

```bash
git push -u origin docker-support
```
Expected: `docker-support` now exists on `origin`, tracking this branch.

- [ ] **Step 3: Verify it's a pure snapshot (no changes)**

```bash
git diff main docker-support
```
Expected: no output (branches are identical at this point — `docker-support` is a pure fork point, not yet diverged).

- [ ] **Step 4: Switch back to `main`**

```bash
git checkout main
```
No commit in this task — branch creation and push are the only actions.

---

### Task 2: Remove Docker files and clean up doc references

**Files:**
- Delete: `Dockerfile`, `docker-compose.yml`, `DOCKER.md`, `requirements-docker.txt`, `.dockerignore`
- Modify: `CLAUDE.md`
- Modify: `ROADMAP.md`

**Interfaces:** none (file removal + doc edits, no code consumes these files).

- [ ] **Step 1: Create the feature branch**

```bash
cd ~/ContactPilot
git checkout main
git checkout -b remove-docker-support
```

- [ ] **Step 2: Delete the 5 Docker files**

```bash
git rm Dockerfile docker-compose.yml DOCKER.md requirements-docker.txt .dockerignore
```
Expected: all 5 files staged for deletion, no errors (all 5 are confirmed to exist at repo root).

- [ ] **Step 3: Remove the Docker paragraph from `CLAUDE.md`**

Current (`CLAUDE.md`, right after the "Full from-scratch recreation steps..." paragraph, before "## Known constraints"):
```markdown
Commands in this file assume that env is active. Full from-scratch recreation
steps (including the CUDA wheel index caveat for RTX 5090/Blackwell, which
needs cu128 instead of cu126) are in `README.md` under "Environment".

There is also a Docker path (`Dockerfile` + `docker-compose.yml`, CUDA 12.8 /
torch cu128, covers both GTX 1650 sm_75 and RTX 5090 sm_120 with one image) —
see `DOCKER.md`. Use it for a reproducible run on a machine without the conda
env set up; it always runs `mujoco_grasp_sim` headless (`--no-vis` baked into
the entrypoint).

---

## Known constraints worth knowing before touching things
```

New:
```markdown
Commands in this file assume that env is active. Full from-scratch recreation
steps (including the CUDA wheel index caveat for RTX 5090/Blackwell, which
needs cu128 instead of cu126) are in `README.md` under "Environment".

---

## Known constraints worth knowing before touching things
```

- [ ] **Step 4: Trim the Docker clause from the RTX 5090/Blackwell bullet in `CLAUDE.md`**

Current (`CLAUDE.md`, "Known constraints" section):
```markdown
- **RTX 5090 / Blackwell (sm_120) needs cu128 torch wheels** — cu126 has no
  sm_120 kernels. The Docker image is pinned to cu128 for this reason; the
  conda env doc has separate laptop/lab instructions.
```

New:
```markdown
- **RTX 5090 / Blackwell (sm_120) needs cu128 torch wheels** — cu126 has no
  sm_120 kernels; the conda env doc has separate laptop/lab instructions.
```

- [ ] **Step 5: Remove the "P0 — Dockerized, reproducible pipeline" section from `ROADMAP.md`**

Read the current section first (`grep -n "P0 — Dockerized" ROADMAP.md` to find its exact line range, since line numbers may have shifted since this plan was written), then delete that entire section (from its `## P0 — Dockerized, reproducible pipeline` header through the line immediately before the next `##` section header). Do not remove the following section's own header or content — only the P0 Docker section itself.

- [ ] **Step 6: Verify no Docker references remain in this repo's own files**

```bash
grep -rln "docker\|Docker" --include="*.md" --include="*.yml" --include="*.yaml" . 2>/dev/null | grep -v "/.git/" | grep -v "^./GraspGen/" | grep -v "^./contact_graspnet_pytorch/" | grep -v "2026-07-29-contact-graspnet-dependency-restructure"
```
Expected: no output — every remaining hit should be either inside the `GraspGen/`/`contact_graspnet_pytorch/` submodules (excluded above) or the two historical 2026-07-29 plan/spec docs (excluded above, left alone per Global Constraints).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Remove Docker support from main (preserved on docker-support branch)"
```

---

### Task 3: Push, open PR, and verify the preservation branch is untouched

**Files:** none.

**Interfaces:** none.

- [ ] **Step 1: Push the feature branch**

```bash
git push -u origin remove-docker-support
```

- [ ] **Step 2: Open the PR**

```bash
gh pr create --title "Remove Docker support from main" --base main --head remove-docker-support --body "$(cat <<'EOF'
## Summary
- Removes all Docker support from `main` — the user no longer works at the lab and doesn't want this repo supporting Docker going forward.
- The current, working Docker setup is permanently preserved on branch `docker-support` (created from this same fork point, untouched) — it will never be deleted.
- Removed: `Dockerfile`, `docker-compose.yml`, `DOCKER.md`, `requirements-docker.txt`, `.dockerignore`.
- Cleaned up prose references in `CLAUDE.md` (the "Docker path" paragraph, and the Docker clause in the RTX 5090/Blackwell bullet) and removed `ROADMAP.md`'s "P0 — Dockerized, reproducible pipeline" section.
- Left untouched (out of scope): `GraspGen/docker/*` and `contact_graspnet_pytorch/README.md`'s unrelated "docker" path mention, both belonging to submodules; two historical 2026-07-29 plan/spec docs that discuss Docker as part of a past implementation plan (archived record, not live docs).

## Test plan
- [x] Grepped the whole repo (excluding submodules and the two historical docs) for "docker"/"Docker" — zero remaining references
- [x] Confirmed `docker-support` branch is a pure, untouched snapshot of the pre-removal state (`git diff main docker-support` at the fork point showed no differences)

EOF
)"
```
Expected: PR created, URL printed.

- [ ] **Step 3: Verify `docker-support` is still untouched**

```bash
git fetch origin docker-support
git diff origin/docker-support origin/main   # (after this PR merges — run once merged)
```
Before merge, this will show the full diff of what's being removed (expected, that's the point). After the PR merges, re-run this to confirm `docker-support` still has every file this plan removed from `main` — this is the actual preservation guarantee, worth a final look once merged.

No commit in this task beyond what Task 2 already made — this task only pushes, opens the PR, and verifies.
