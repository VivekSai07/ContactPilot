# Design: Restructure `contact_graspnet_pytorch` as a proper dependency

Date: 2026-07-29

## Problem

`contact_graspnet_pytorch/` is currently ~181 MB of plain files committed
directly into the ContactPilot repo — network source, the vendored
`Pointnet_Pointnet2_pytorch` helpers, a 26 MB checkpoint (`model.pt`), and
136 MB of test scenes (`test_data/*.npy`). It is not a git submodule; it's a
full copy, so every future edit to any of its ~20 source files bloats
ContactPilot's own history, and the four local patches documented in
`README.md`/`CLAUDE.md` exist only as silent, undiffable modifications to
that copy.

The user maintains a fork, `VivekSai07/contact_graspnet_pytorch`, 3 commits
ahead of `elchun/contact_graspnet_pytorch:main`, which already contains all
four documented local patches (byte-identical, verified by diff) plus two
additional improvements not present in ContactPilot's local copy:

- `contact_grasp_estimator.py`: runs `forward_passes` sequentially instead of
  batching them into one tensor, keeping peak VRAM roughly constant
  regardless of `forward_passes` (relevant on the 4 GB laptop GPU).
- `contact_graspnet.py`: a one-line `torch.cross(..., dim=-1)` compat fix.

The fork still carries `checkpoints/` (26 MB) and `test_data/` (136 MB) as
plain committed files (no Git LFS), so simply pointing at the fork via a
submodule would not by itself remove the size problem — the binary data has
to be relocated.

## Goals

- ContactPilot's own git history stops carrying the vendored source and
  binary assets.
- The four documented local patches (plus the two undocumented ones already
  in the fork) become normal, diffable commits on the fork instead of
  untracked local modifications.
- `checkpoints/model.pt` and `test_data/*.npy` are hosted externally (Hugging
  Face Hub) and fetched by a small download script, not committed to either
  repo going forward.
- The existing dev workflow (conda env, `test_inference_headless.py`,
  `run_sim_grasp_test.py`, Docker build/run) keeps working with only a setup-step
  addition, not a restructuring of how the code is invoked.
- ContactPilot's own git history is rewritten so the repo is genuinely small,
  not just lean going forward (justified below under Migration).

## Non-goals

- Not changing how the code is invoked (still relative-path scripts, not an
  installed pip package) — see "Approaches considered" for why.
- Not touching `mujoco_menagerie` (already a sparse clone, unaffected).
- Not adding CI in this pass.

## Current state (verified 2026-07-29)

- `contact_graspnet_pytorch/` in ContactPilot: plain tracked files, not a
  submodule (`git ls-files -s` shows regular blobs, no gitlink entry).
- Fork diff vs local vendored copy (excluding `checkpoints/`, `test_data/`,
  `.git`): only `contact_grasp_estimator.py`, `contact_graspnet.py`,
  `.gitignore`, `contact_graspnet_env.yml`, `README.md`, and
  `gripper_control_points/panda_gripper_coords.pickle` differ; the pickle
  diff is the known Windows `core.autocrlf` CRLF-corruption issue, not new
  drift. `checkpoints.py`, `visualize_saved_scene.py`,
  `visualization_utils_o3d.py`, and `test_inference_headless.py` are
  byte-identical.
- `checkpoints/contact_graspnet/checkpoints/model.pt` is 26 MB in both copies
  (the README's "266 MB" figure is stale/incorrect).
- Two HF Hub repos already created and confirmed publicly reachable
  (unauthenticated `HTTP 200`):
  - Model: `https://huggingface.co/VivekSai07/contact_graspnet_checkpoint`
  - Dataset: `https://huggingface.co/datasets/VivekSai07/test_data`
- ContactPilot's origin (`github.com/VivekSai07/ContactPilot`) has exactly
  one pushed commit (`3ada254`, "Initial commit: ContactPilot grasping
  pipeline") on `main`.

## Approaches considered

**Code linkage — git submodule (chosen) vs. `pip install git+...`:**
`pip install` would only capture the inner installable Python package per
`setup.py`; `test_inference_headless.py`, `visualize_saved_scene.py`,
`checkpoints/`, and `test_data/` all live at the fork's repo root, not inside
that package. Using pip would mean either losing easy access to those driver
scripts or restructuring the fork into a proper package layout — more scope
than this change needs. A git submodule keeps the on-disk layout ~identical
(same relative paths, same `COPY contact_graspnet_pytorch/` in the
Dockerfile), matches the pattern ContactPilot already uses for
`mujoco_menagerie`, and reduces upstream syncs to a normal submodule bump.

**Asset hosting — Hugging Face Hub (chosen) vs. Git LFS vs. GitHub Release:**
Git LFS was rejected — adds quota/bandwidth considerations for a small lab
team without shrinking the conceptual footprint much (~160 MB either way).
GitHub Release assets would work but require manual re-upload per release and
a bespoke HTTP+zip download path. HF Hub has a mature Python client
(`huggingface_hub`), free public hosting, and the user already has both repos
set up and reachable with no auth needed.

## Design

### 1. Fork-side changes (`VivekSai07/contact_graspnet_pytorch`)

1. `git rm --cached -r checkpoints test_data`; add both to `.gitignore`.
2. Add `-text` (or `binary`) for
   `gripper_control_points/panda_gripper_coords.pickle` in `.gitattributes`,
   permanently fixing the `core.autocrlf` corruption at the source (retires
   the "clone with `--config core.autocrlf=false`" workaround).
3. Add `scripts/download_assets.py`: uses `huggingface_hub.hf_hub_download` /
   `snapshot_download` to fetch:
   - `model.pt` from `VivekSai07/contact_graspnet_checkpoint` →
     `checkpoints/contact_graspnet/checkpoints/model.pt`
   - the 14 `*.npy` scenes from `datasets/VivekSai07/test_data` → `test_data/`
   Script is idempotent (skip if files already present with correct size, or
   just rely on `huggingface_hub`'s local cache/symlink behavior) and prints
   what it fetched.
4. Commit these changes, tag the resulting commit (e.g. `v1.0.0`).

### 2. ContactPilot-side changes

1. Remove `contact_graspnet_pytorch/` from tracking:
   `git rm -r --cached contact_graspnet_pytorch`.
2. Add the submodule pinned to the fork's tag:
   `git submodule add https://github.com/VivekSai07/contact_graspnet_pytorch.git contact_graspnet_pytorch`
   then `git -C contact_graspnet_pytorch checkout v1.0.0` (submodule records
   the resolved commit SHA).
3. `git add .gitmodules contact_graspnet_pytorch`.
4. Setup flow becomes:
   ```
   git submodule update --init --depth 1
   python contact_graspnet_pytorch/scripts/download_assets.py
   ```
   (replaces the current "cloned repo, weights+test data bundled" assumption).
5. **Dockerfile**: keep `COPY contact_graspnet_pytorch/ contact_graspnet_pytorch/`
   as-is (host runs `git submodule update --init` before `docker build`, same
   expectation as any submodule-based project). Add `huggingface_hub` to
   `requirements-docker.txt` and a
   `RUN python3 contact_graspnet_pytorch/scripts/download_assets.py` build
   step before the final `WORKDIR`/`ENTRYPOINT`, so the built image still
   bakes in the weights — no runtime download, `docker compose up` behavior
   unchanged. Update the Dockerfile's comment above the `COPY` line (it
   currently claims "including checkpoints/model.pt", which will no longer
   be true pre-download-step).
6. Delete the now-obsolete "Windows git `core.autocrlf` corrupts the pickle"
   gotcha from `README.md`'s "Known gotchas" (fixed at the source in the fork
   per step 1.2) and its "clone with `--config core.autocrlf=false`"
   instruction.

### 3. Migration (history rewrite)

Origin (`github.com/VivekSai07/ContactPilot`) has exactly one commit, which
is also the only commit referencing the 181 MB of vendored files, so a full
history rewrite is just amending that one commit — no `git filter-repo`/BFG
needed:

```
git rm -r --cached contact_graspnet_pytorch
git submodule add https://github.com/VivekSai07/contact_graspnet_pytorch.git contact_graspnet_pytorch
git -C contact_graspnet_pytorch checkout v1.0.0
git add .gitmodules contact_graspnet_pytorch README.md CLAUDE.md DOCKER.md Dockerfile requirements-docker.txt
git commit --amend
git push --force origin main
```

Confirmed safe: only one commit exists, on `main`, already pushed — no other
known clones/collaborators at time of writing.

### 4. Documentation updates

- `README.md`: "Layout" section documents the submodule (fork URL + pinned
  tag) instead of "cloned repo (weights + test data bundled)"; drop the
  "Local modifications" section entirely (patches now live as normal commits
  on the fork); drop the stale "266 MB" figure; add the submodule-init +
  asset-download step to the setup instructions; remove the
  `core.autocrlf=false` clone gotcha.
- `DOCKER.md`: add a one-line prerequisite noting `git submodule update --init`
  must run before `docker compose build`.
- `CLAUDE.md`: replace "Local modifications to the vendored
  `contact_graspnet_pytorch` repo" with a short "Contact-GraspNet dependency"
  section: fork URL, pinned tag, submodule + `download_assets.py` workflow,
  and a pointer to the fork for anyone wanting to see the patch history.

## Testing / verification plan

- After migration, from a clean clone: run the full setup flow (submodule
  init + asset download) and confirm `checkpoints/contact_graspnet/checkpoints/model.pt`
  and all 14 `test_data/*.npy` files land at the expected paths with correct
  sizes.
- Run `test_inference_headless.py --np_path=test_data/7.npy` and compare
  grasp count/scores/timing against the baseline numbers already recorded in
  `README.md` ("scene 7 -> 222 grasps / 8 objects, 2.82 GB peak VRAM, ~48 s")
  — this is the smoke test for the two behavior changes coming in from the
  fork (sequential `forward_passes`, `torch.cross` fix).
- Run the same scene with `--forward_passes=5` specifically to exercise the
  new sequential-batching code path and confirm peak VRAM stays roughly flat
  rather than scaling with `forward_passes`.
- `docker compose build` from scratch, confirm the image build succeeds and
  bakes in the assets (no network access needed at `docker compose up` time).
- Confirm `git clone` of the rewritten ContactPilot repo (fresh, no
  `--depth`) is now on the order of a few MB, not 181 MB+.

## Risks / open items

- Force-pushing rewrites the commit hash of `main` — anyone who already has
  a clone will need to re-clone or hard-reset. Accepted given only one known
  clone exists (confirmed with the user).
- The two behavior changes arriving from the fork (sequential
  `forward_passes`, `torch.cross` fix) are net improvements but are
  technically unreviewed-by-ContactPilot changes; the verification plan above
  is the acceptance check for them.
- HF repos are public and unauthenticated today (verified); if either is ever
  made private, `download_assets.py` and the Docker build step would need an
  HF token wired in.
