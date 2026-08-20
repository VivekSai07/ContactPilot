# GraspGen Git Submodule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert GraspGen from a manually-cloned external sibling directory (`~/GraspGen`, outside the repo) into a git submodule at `GraspGen/` inside the ContactPilot repo, pinned to the exact commit already validated end-to-end on this machine — with zero code changes, since no code in this repo references GraspGen's filesystem path.

**Architecture:** Pure git/docs change. `GraspGen/` becomes a submodule at repo root, mirroring `contact_graspnet_pytorch/`'s existing pattern exactly (both are unpatched-or-patched vendor dependencies pinned via `.gitmodules`). Every runtime touchpoint (`GRASPGEN_PYTHON`, `GraspGenPredictor`, `graspgen_worker.py`) only cares about the `graspgen_torch` conda env's interpreter — never GraspGen's source location — so this repo's Python code needs no modification at all. Only two README files and `.gitmodules` change.

**Spec:** No separate spec file — this was scoped as a bounded task in chat during brainstorming (confirmed: no path-coupling in code, no new licensing exposure since the submodule points at NVIDIA's own official repo, same as today's manual clone).

## Global Constraints

- Submodule target: `https://github.com/NVlabs/GraspGen.git` (official upstream, NOT a fork — confirmed no local patches exist anywhere in this project's history for GraspGen, unlike `contact_graspnet_pytorch` which has 6+ real patches justifying its own fork).
- Pinned commit: `2dd8852` ("Add Grasp Mixture of Experts (GraspMoE)") — the exact commit this machine's `~/GraspGen` clone is currently at, already validated (checkpoint download, `pointnet2_ops` build, the bfloat16-autocast fix, real GraspGen inference all confirmed working against this commit in prior sub-projects). Do not pin to a different commit without re-validating the backend end-to-end.
- Submodule path: `GraspGen/` at repo root — mirrors `contact_graspnet_pytorch/`'s existing top-level placement.
- Zero changes to any `.py` file in this repo. If anything seems to require a code change, STOP — that would mean the "no path coupling" assumption this plan is built on is wrong, and the plan needs to be re-scoped, not patched around.
- This machine's existing, working `graspgen_torch` conda env (with its editable install still pointing at `~/GraspGen`) is left untouched by this plan — do not `pip install -e .` against the new in-repo path, do not delete `~/GraspGen`. This plan only changes the repo's own tracked structure and documentation for *future* fresh setups.
- Commit messages are plain — **do not add a `Co-Authored-By: Claude` trailer to any commit**.

---

### Task 1: Add the GraspGen git submodule

**Files:**
- Create/modify: `.gitmodules` (git manages this automatically via `git submodule add`)
- Create: `GraspGen/` (submodule gitlink entry — the actual files are not stored in ContactPilot's git history, only a pinned commit reference)

**Interfaces:**
- Produces: `GraspGen/` submodule at repo root, pinned to commit `2dd8852`. Tasks 2-3 (README updates) reference this path.

- [ ] **Step 1: Add the submodule**

```bash
cd ~/ContactPilot
git submodule add https://github.com/NVlabs/GraspGen.git GraspGen
```
Expected: clones GraspGen into `GraspGen/`, adds a `[submodule "GraspGen"]` section to `.gitmodules`, and stages both `.gitmodules` and the new `GraspGen` gitlink entry. This will check out GraspGen's current default-branch HEAD, which is NOT yet pinned to the validated commit — fixed in Step 2.

- [ ] **Step 2: Pin to the validated commit**

```bash
cd ~/ContactPilot/GraspGen
git checkout 2dd8852
cd ~/ContactPilot
git add GraspGen
```
Expected: `git status` shows `GraspGen` staged as modified (the gitlink now points at `2dd8852`), and running `git -C GraspGen log -1 --oneline` prints `2dd8852 Add Grasp Mixture of Experts (GraspMoE)`.

- [ ] **Step 3: Verify submodule status**

```bash
git submodule status
```
Expected: two lines, one for `contact_graspnet_pytorch` (unchanged, still pinned to its tag) and one for `GraspGen` showing commit `2dd8852` with no leading `-`/`+` (meaning it's checked out and matches the pinned commit exactly, not just initialized-but-different or not-yet-initialized).

- [ ] **Step 4: Commit**

```bash
git add .gitmodules GraspGen
git commit -m "Add GraspGen as a git submodule, pinned to 2dd8852"
```

---

### Task 2: Update the root `README.md`

**Files:**
- Modify: `README.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Add `GraspGen/` to the Layout diagram**

Current (`README.md`, in the `## Layout` fenced block):
```
ContactPilot/
├── README.md                      ← this file
├── mujoco_grasp_sim/              ← MuJoCo tabletop sim (Panda + RGB-D + CGN/GraspGen) — see its README
├── mujoco_menagerie/              ← sparse clone: franka_emika_panda model only
└── contact_graspnet_pytorch/      ← git submodule (VivekSai07/contact_graspnet_pytorch, pinned tag)
    ├── checkpoints/contact_graspnet/checkpoints/model.pt   (26 MB, fetched by download_assets.py)
    ├── test_data/0.npy … 13.npy   (14 test scenes: rgb, depth[m], K, seg @ 1280×720; fetched by download_assets.py)
    ├── test_inference_headless.py (headless driver — no GUI, prints stats)
    ├── scripts/download_assets.py (fetches checkpoint + test_data from Hugging Face Hub)
    ├── results/                   (predictions land here as .npz)
    └── contact_graspnet_pytorch/  (source)
```

New:
```
ContactPilot/
├── README.md                      ← this file
├── mujoco_grasp_sim/              ← MuJoCo tabletop sim (Panda + RGB-D + CGN/GraspGen) — see its README
├── mujoco_menagerie/              ← sparse clone: franka_emika_panda model only
├── GraspGen/                      ← git submodule (NVlabs/GraspGen upstream, pinned commit, unpatched)
├── contact_graspnet_pytorch/      ← git submodule (VivekSai07/contact_graspnet_pytorch, pinned tag)
│   ├── checkpoints/contact_graspnet/checkpoints/model.pt   (26 MB, fetched by download_assets.py)
│   ├── test_data/0.npy … 13.npy   (14 test scenes: rgb, depth[m], K, seg @ 1280×720; fetched by download_assets.py)
│   ├── test_inference_headless.py (headless driver — no GUI, prints stats)
│   ├── scripts/download_assets.py (fetches checkpoint + test_data from Hugging Face Hub)
│   ├── results/                   (predictions land here as .npz)
│   └── contact_graspnet_pytorch/  (source)
```

- [ ] **Step 2: Note the second submodule where `contact_graspnet_pytorch`'s submodule setup is documented**

Current (`README.md`, `### Getting `contact_graspnet_pytorch` (submodule + assets)` section):
```markdown
`contact_graspnet_pytorch/` is a git submodule pointing at
[`VivekSai07/contact_graspnet_pytorch`](https://github.com/VivekSai07/contact_graspnet_pytorch)
(pinned to a tag). The checkpoint and test scenes are hosted on Hugging Face
Hub and fetched by a script — neither is committed to git.

```powershell
git submodule update --init --depth 1
pip install huggingface_hub
python contact_graspnet_pytorch\scripts\download_assets.py
```

Run this once after cloning ContactPilot (or after any `git submodule update`
that moves the pinned commit). The download script is idempotent — safe to
re-run. `huggingface_hub` is also included below in "Recreate the env from
scratch" for fresh conda environments.
```

New:
```markdown
`contact_graspnet_pytorch/` is a git submodule pointing at
[`VivekSai07/contact_graspnet_pytorch`](https://github.com/VivekSai07/contact_graspnet_pytorch)
(pinned to a tag). `GraspGen/` is a second git submodule, pointing directly
at [`NVlabs/GraspGen`](https://github.com/NVlabs/GraspGen) upstream (pinned
to a commit, not a fork — nothing in this project patches GraspGen's
source). The command below initializes both. The checkpoint and test
scenes for `contact_graspnet_pytorch` are hosted on Hugging Face Hub and
fetched by a script — neither is committed to git; GraspGen's own
checkpoint is fetched separately, see `mujoco_grasp_sim/README.md`'s
"GraspGen backend setup".

```powershell
git submodule update --init --depth 1
pip install huggingface_hub
python contact_graspnet_pytorch\scripts\download_assets.py
```

Run this once after cloning ContactPilot (or after any `git submodule update`
that moves either pinned commit). The download script is idempotent — safe
to re-run. `huggingface_hub` is also included below in "Recreate the env
from scratch" for fresh conda environments.
```

- [ ] **Step 3: Verify**

```bash
grep -n "GraspGen/" README.md
```
Expected: at least 3 matches (the Layout diagram line, the submodule-setup paragraph, and the existing "GraspGen — a second, selectable grasp backend" section header, which needs no change itself).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Document GraspGen as a git submodule in the root README"
```

---

### Task 3: Update `mujoco_grasp_sim/README.md`'s GraspGen backend setup section

**Files:**
- Modify: `mujoco_grasp_sim/README.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Replace the manual-clone step with the submodule-init step**

Current (`mujoco_grasp_sim/README.md`, "GraspGen backend setup" section, step 1 of the bash block):
```bash
# 1. Clone GraspGen OUTSIDE this repo (not a submodule — nothing here patches it)
cd ~ && git clone https://github.com/NVlabs/GraspGen.git
```

New:
```bash
# 1. GraspGen is a git submodule of this repo (NVlabs/GraspGen upstream,
#    pinned to a validated commit) — initialize it if you haven't already:
cd ~/ContactPilot && git submodule update --init GraspGen
```

- [ ] **Step 2: Update the `cd` paths in steps 4 and 5 to the new in-repo location**

Current (`mujoco_grasp_sim/README.md`, step 4 of the same bash block):
```bash
# 4. Install GraspGen.
#    GOTCHA: `pip install -e .` alone silently DOWNGRADES torch to
#    GraspGen's own pinned torch==2.1.0 (breaking Blackwell support), even
#    with --no-build-isolation — that flag only protects the *build* step,
#    not the final dependency-resolution/install step. Work around it:
cd ~/GraspGen
/path/to/graspgen_torch/bin/python -m pip install --no-build-isolation -e .
# The above downgrades torch — restore it:
/path/to/graspgen_torch/bin/python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
# It may also bump numpy past this project's numpy<2 requirement — restore that too:
/path/to/graspgen_torch/bin/python -m pip install --no-deps "numpy==1.26.4"

# 5. Build the pointnet2_ops CUDA extension
cd ~/GraspGen/pointnet2_ops
CUDA_HOME=/usr/local/cuda-12.8 /path/to/graspgen_torch/bin/python -m pip install --no-build-isolation --no-deps .
```

New:
```bash
# 4. Install GraspGen.
#    GOTCHA: `pip install -e .` alone silently DOWNGRADES torch to
#    GraspGen's own pinned torch==2.1.0 (breaking Blackwell support), even
#    with --no-build-isolation — that flag only protects the *build* step,
#    not the final dependency-resolution/install step. Work around it:
cd ~/ContactPilot/GraspGen
/path/to/graspgen_torch/bin/python -m pip install --no-build-isolation -e .
# The above downgrades torch — restore it:
/path/to/graspgen_torch/bin/python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
# It may also bump numpy past this project's numpy<2 requirement — restore that too:
/path/to/graspgen_torch/bin/python -m pip install --no-deps "numpy==1.26.4"

# 5. Build the pointnet2_ops CUDA extension
cd ~/ContactPilot/GraspGen/pointnet2_ops
CUDA_HOME=/usr/local/cuda-12.8 /path/to/graspgen_torch/bin/python -m pip install --no-build-isolation --no-deps .
```

- [ ] **Step 3: Verify**

```bash
grep -n "~/GraspGen\b" mujoco_grasp_sim/README.md
```
Expected: no output — every remaining GraspGen path reference should now be `~/ContactPilot/GraspGen`, not the old sibling-clone path.

- [ ] **Step 4: Commit**

```bash
git add mujoco_grasp_sim/README.md
git commit -m "Update GraspGen backend setup docs for the new submodule path"
```

---

### Task 4: Fresh-clone verification

**Files:** none — this task only verifies Tasks 1-3, no new changes.

**Interfaces:** none.

Since no code changed, this task does NOT re-verify the GraspGen backend's runtime behavior (that would require reinstalling `graspgen_torch` against the new path, which the Global Constraints explicitly say not to do on this machine). It verifies only that a brand-new clone gets a correctly-populated, correctly-pinned submodule — the thing this whole plan actually changed.

- [ ] **Step 1: Fresh clone with submodules, into a scratch directory**

```bash
cd /tmp
rm -rf contactpilot-submodule-check
git clone --recurse-submodules https://github.com/VivekSai07/ContactPilot.git contactpilot-submodule-check
```
Expected: clones cleanly, no errors, no prompts.

- [ ] **Step 2: Verify both submodules populated at the correct commits**

```bash
cd /tmp/contactpilot-submodule-check
git submodule status
ls GraspGen/pointnet2_ops   # confirm real GraspGen files are actually present, not an empty dir
```
Expected: `git submodule status` shows `contact_graspnet_pytorch` and `GraspGen` both with no leading `-` (fully initialized), `GraspGen` at commit `2dd8852`; `ls` shows real files (e.g. `setup.py`, source dirs), confirming the submodule content actually downloaded, not just a gitlink reference.

- [ ] **Step 3: Clean up the scratch clone**

```bash
rm -rf /tmp/contactpilot-submodule-check
```

No commit — this task only verifies, touches nothing in the real repo.
