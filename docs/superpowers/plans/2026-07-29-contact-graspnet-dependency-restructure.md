# Contact-GraspNet Dependency Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ~181MB plain-file copy of `contact_graspnet_pytorch/` with a git submodule pinned to the `VivekSai07/contact_graspnet_pytorch` fork, and move the checkpoint/test-data binaries out of git entirely onto Hugging Face Hub, fetched by a small script.

**Architecture:** Fork gets its checkpoint/test_data untracked and replaced with a `scripts/download_assets.py` that pulls both from Hugging Face Hub, then gets tagged. ContactPilot drops the plain copy, adds the fork as a submodule pinned to that tag, and rewrites its single existing commit (via `git commit --amend` + force-push) so the old 181MB of blobs are fully gone from history, not just untracked going forward.

**Tech Stack:** git submodules, `huggingface_hub` (Python), Hugging Face Hub (model repo + dataset repo), Docker.

## Global Constraints

- `checkpoints/` and `test_data/` must never be committed to git again, in either the fork or ContactPilot.
- The four already-patched files in the fork (`checkpoints.py`, `visualize_saved_scene.py`, `visualization_utils_o3d.py`, `test_inference_headless.py`) are byte-identical to ContactPilot's current vendored copy and must not change behavior.
- HF Hub repos are public, no-auth: model `VivekSai07/contact_graspnet_checkpoint`, dataset `VivekSai07/test_data`. Verified reachable unauthenticated (`HTTP 200`) as of 2026-07-29.
- ContactPilot's submodule must be pinned to an explicit tag (not a floating branch).
- `docker compose up` must stay fully self-contained at run time — no network access needed outside `docker compose build`.
- ContactPilot's `origin/main` currently has exactly one commit (`3ada254`); the migration is a `git commit --amend` + `git push --force`, approved by the user, not a `git filter-repo`/BFG rewrite.
- numpy must stay < 2 across the whole stack (unaffected by this change, but don't introduce anything that violates it).

---

### Task 1: Upload assets to Hugging Face Hub

**This task must be run by a human with Hugging Face credentials — `hf auth login` is interactive and cannot be scripted by an agent. Do not delegate this task to a subagent; run it yourself, then mark it done.**

**Files:** none (uploads existing local files to external HF Hub repos; no repo files change).

**Interfaces:**
- Consumes: `contact_graspnet_pytorch/checkpoints/contact_graspnet/checkpoints/model.pt` and `contact_graspnet_pytorch/test_data/*.npy` from the current ContactPilot working copy (verified correct — 26MB checkpoint, 14 test scenes).
- Produces: `model.pt` at the root of the `VivekSai07/contact_graspnet_checkpoint` model repo; `0.npy` … `13.npy` at the root of the `VivekSai07/test_data` dataset repo. Task 3's `download_assets.py` depends on exactly these paths.

- [x] **Step 1: Authenticate with Hugging Face (interactive — run yourself)**

```powershell
hf auth login
```

Follow the prompts (paste a token from https://huggingface.co/settings/tokens with write access).

- [x] **Step 2: Upload the checkpoint**

```powershell
cd D:\Projects\ContactPilot
hf upload VivekSai07/contact_graspnet_checkpoint contact_graspnet_pytorch\checkpoints\contact_graspnet\checkpoints\model.pt model.pt
```

Expected: progress bar, ends with a line like `https://huggingface.co/VivekSai07/contact_graspnet_checkpoint/blob/main/model.pt`.

- [x] **Step 3: Upload the test scenes**

```powershell
hf upload VivekSai07/test_data contact_graspnet_pytorch\test_data . --repo-type=dataset
```

Expected: progress output uploading 14 `.npy` files, ends with a repo URL.

- [x] **Step 4: Verify both repos are populated**

```powershell
curl.exe -s https://huggingface.co/api/models/VivekSai07/contact_graspnet_checkpoint
curl.exe -s https://huggingface.co/api/datasets/VivekSai07/test_data
```

Expected: the model repo's `siblings` list now includes `{"rfilename":"model.pt"}` (in addition to `.gitattributes`); the dataset repo's `siblings` list includes 14 entries named `0.npy` through `13.npy`.

---

### Task 2: Vendor the fork as a git submodule (replace the plain copy)

**Files:**
- Create: `.gitmodules`
- Replace: `contact_graspnet_pytorch/` (plain tracked directory → submodule gitlink)

**Interfaces:**
- Consumes: fork URL `https://github.com/VivekSai07/contact_graspnet_pytorch.git`.
- Produces: a `contact_graspnet_pytorch/` working-tree checkout with its own `.git`, remote `origin` = the fork. Task 3 works inside this checkout.

- [ ] **Step 1: Confirm no unrelated uncommitted changes are about to be caught up in this**

```bash
git status
```

Expected: only `contact_graspnet_pytorch/` content is tracked/clean; any unrelated untracked files (e.g. a local `CLAUDE.md`) are irrelevant to this task and left alone.

- [ ] **Step 2: Untrack the old vendored copy**

```bash
git rm -r --cached contact_graspnet_pytorch
```

Expected: prints one `rm 'contact_graspnet_pytorch/...'` line per file (hundreds of lines), exits 0.

- [ ] **Step 3: Delete the now-untracked directory from disk**

```bash
rm -rf contact_graspnet_pytorch
```

Expected: `ls contact_graspnet_pytorch` now errors with "No such file or directory".

- [ ] **Step 4: Add the fork as a submodule at the same path**

```bash
git submodule add https://github.com/VivekSai07/contact_graspnet_pytorch.git contact_graspnet_pytorch
```

Expected: prints `Cloning into 'D:/Projects/ContactPilot/contact_graspnet_pytorch'...` then `done.`; creates `.gitmodules`.

- [ ] **Step 5: Verify**

```bash
git status
cat .gitmodules
```

Expected `git status` shows `new file: .gitmodules` and `new file: contact_graspnet_pytorch` staged, plus the bulk `deleted:` entries from Step 2 still staged. Expected `.gitmodules` content:

```
[submodule "contact_graspnet_pytorch"]
	path = contact_graspnet_pytorch
	url = https://github.com/VivekSai07/contact_graspnet_pytorch.git
```

Do not commit yet — this stays staged until Task 6's final amend.

---

### Task 3: Fork-side changes — retire committed assets, fix the pickle CRLF issue at the source, add the download script, tag and push

**Files (inside the `contact_graspnet_pytorch/` submodule checkout — a separate git repo, remote `origin` = the fork):**
- Modify: `contact_graspnet_pytorch/.gitignore`
- Modify: `contact_graspnet_pytorch/.gitattributes`
- Create: `contact_graspnet_pytorch/scripts/download_assets.py`

**Interfaces:**
- Consumes: HF Hub repos populated in Task 1 (`VivekSai07/contact_graspnet_checkpoint`, `VivekSai07/test_data`).
- Produces: tag `v1.0.0` on the fork; `download_assets.py` with functions `download_checkpoint()` and `download_test_data()`, run as `python scripts/download_assets.py`. Task 4 (Dockerfile) and Task 7 (verification) both invoke this script.

- [ ] **Step 1: Confirm the submodule's remote is the fork**

```bash
git -C contact_graspnet_pytorch remote -v
```

Expected: `origin  https://github.com/VivekSai07/contact_graspnet_pytorch.git (fetch)` and `(push)`.

- [ ] **Step 2: Stop tracking the checkpoint and test data**

```bash
git -C contact_graspnet_pytorch rm -r --cached checkpoints test_data
```

Expected: prints removal of every file under `checkpoints/` and `test_data/`.

- [ ] **Step 3: Add both to `.gitignore`**

Append to `contact_graspnet_pytorch/.gitignore`:

```
# Downloaded via scripts/download_assets.py — see README for setup
checkpoints/
test_data/
```

- [ ] **Step 4: Fix the CRLF-prone pickle at the source**

Append to `contact_graspnet_pytorch/.gitattributes`:

```
gripper_control_points/panda_gripper_coords.pickle -text
```

This retires the "clone with `--config core.autocrlf=false`" workaround — the file is now marked binary regardless of the cloning host's `core.autocrlf` setting.

- [ ] **Step 5: Add the download script**

Create `contact_graspnet_pytorch/scripts/download_assets.py`:

```python
#!/usr/bin/env python3
"""Download the Contact-GraspNet checkpoint and test scenes from Hugging Face Hub.

Run this after cloning this repo (or after a `git submodule update` in
ContactPilot that changes the pinned commit) to populate checkpoints/ and
test_data/, which are no longer committed to git.
"""
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_REPO = "VivekSai07/contact_graspnet_checkpoint"
CHECKPOINT_DEST = REPO_ROOT / "checkpoints" / "contact_graspnet" / "checkpoints" / "model.pt"

TEST_DATA_REPO = "VivekSai07/test_data"
TEST_DATA_DEST = REPO_ROOT / "test_data"


def download_checkpoint() -> None:
    if CHECKPOINT_DEST.exists():
        print(f"Checkpoint already present: {CHECKPOINT_DEST}")
        return
    CHECKPOINT_DEST.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=CHECKPOINT_REPO, filename="model.pt")
    shutil.copyfile(downloaded, CHECKPOINT_DEST)
    print(f"Downloaded checkpoint to {CHECKPOINT_DEST}")


def download_test_data() -> None:
    if TEST_DATA_DEST.exists() and any(TEST_DATA_DEST.glob("*.npy")):
        print(f"Test data already present: {TEST_DATA_DEST}")
        return
    TEST_DATA_DEST.mkdir(parents=True, exist_ok=True)
    snapshot_dir = snapshot_download(repo_id=TEST_DATA_REPO, repo_type="dataset")
    for npy_file in Path(snapshot_dir).glob("*.npy"):
        shutil.copyfile(npy_file, TEST_DATA_DEST / npy_file.name)
    count = len(list(TEST_DATA_DEST.glob("*.npy")))
    print(f"Downloaded {count} test scenes to {TEST_DATA_DEST}")


if __name__ == "__main__":
    download_checkpoint()
    download_test_data()
```

- [ ] **Step 6: Install `huggingface_hub` locally to test the script (in the `cgn_torch` conda env)**

```powershell
conda activate cgn_torch
pip install huggingface_hub
```

Expected: installs cleanly (pure-Python, no numpy/torch version conflicts).

- [ ] **Step 7: Run the script and verify it fetches correctly**

```powershell
cd contact_graspnet_pytorch
python scripts\download_assets.py
```

Expected output:
```
Downloaded checkpoint to <path>\checkpoints\contact_graspnet\checkpoints\model.pt
Downloaded 14 test scenes to <path>\test_data
```

Verify sizes match the originals:

```powershell
Get-Item checkpoints\contact_graspnet\checkpoints\model.pt | Select-Object Length
(Get-ChildItem test_data\*.npy | Measure-Object Length -Sum).Sum
```

Expected: checkpoint `Length` is `26628146` bytes; test_data files sum to roughly 136 MB (matches the sizes recorded in the design doc).

- [ ] **Step 8: Commit the fork-side changes**

```bash
cd contact_graspnet_pytorch
git add .gitignore .gitattributes scripts/download_assets.py
git commit -m "Move checkpoint/test-data assets to Hugging Face Hub; fix pickle CRLF at source"
```

- [ ] **Step 9: Tag and push to the fork**

```bash
git tag v1.0.0
git push origin main
git push origin v1.0.0
```

Expected: both pushes succeed (`main` fast-forwards, tag ref created).

- [ ] **Step 10: Verify the tag landed on GitHub**

```bash
curl -s https://api.github.com/repos/VivekSai07/contact_graspnet_pytorch/tags
```

Expected: JSON array containing an entry with `"name": "v1.0.0"`.

- [ ] **Step 11: Pin the submodule checkout to the tag**

```bash
cd contact_graspnet_pytorch
git checkout v1.0.0
cd ..
```

Expected: `HEAD is now at <sha> Move checkpoint/test-data assets...` (detached HEAD message).

---

### Task 4: Update Dockerfile + requirements-docker.txt for build-time asset download

**Files:**
- Modify: `Dockerfile:42-49`
- Modify: `requirements-docker.txt:12-13`

**Interfaces:**
- Consumes: `contact_graspnet_pytorch/scripts/download_assets.py` from Task 3.
- Produces: a Docker image with `checkpoints/` and `test_data/` baked in at build time (no runtime network dependency).

- [ ] **Step 1: Add `huggingface_hub` to `requirements-docker.txt`**

Current (lines 12-13):
```
tqdm
matplotlib
```

New:
```
tqdm
matplotlib
huggingface_hub
```

- [ ] **Step 2: Update the Dockerfile's Contact-GraspNet section**

Current (lines 42-49):
```dockerfile
# Contact-GraspNet PyTorch port, including checkpoints/model.pt and the
# already line-ending-fixed gripper data files (see README "Windows clone
# gotcha" — copying from this tree avoids re-corrupting them).
COPY contact_graspnet_pytorch/ contact_graspnet_pytorch/
# Equivalent of the conda env's editable install: put the repo root on the
# path (its setup.py is too old for PEP 660 editable installs under pip 26).
# This also exposes the vendored Pointnet_Pointnet2_pytorch helpers.
ENV PYTHONPATH=/work/contact_graspnet_pytorch
```

New:
```dockerfile
# Contact-GraspNet PyTorch port (git submodule on the host — make sure
# `git submodule update --init` has run before `docker build`).
COPY contact_graspnet_pytorch/ contact_graspnet_pytorch/
# Equivalent of the conda env's editable install: put the repo root on the
# path (its setup.py is too old for PEP 660 editable installs under pip 26).
# This also exposes the vendored Pointnet_Pointnet2_pytorch helpers.
ENV PYTHONPATH=/work/contact_graspnet_pytorch

# Checkpoint + test scenes aren't committed to git (see the submodule's
# scripts/download_assets.py) — fetch them at build time so the image stays
# self-contained at `docker compose up`.
RUN python3 contact_graspnet_pytorch/scripts/download_assets.py
```

- [ ] **Step 3: Verify the submodule is initialized on the host before building**

```bash
git submodule status
```

Expected: a line starting with a space (not `-`) and the commit SHA matching the `v1.0.0` tag, confirming it's checked out (not uninitialized).

- [ ] **Step 4: Build the image**

```bash
docker compose build
```

Expected: build succeeds; near the end, the `RUN python3 contact_graspnet_pytorch/scripts/download_assets.py` layer prints the same two "Downloaded ..." lines seen in Task 3 Step 7.

- [ ] **Step 5: Sanity-check the image actually has the assets baked in**

```bash
docker compose run --rm --entrypoint bash grasp-sim -c "ls -la /work/contact_graspnet_pytorch/checkpoints/contact_graspnet/checkpoints/model.pt /work/contact_graspnet_pytorch/test_data/*.npy | head -3"
```

Expected: lists `model.pt` and at least the first few `.npy` files with nonzero sizes, confirming no runtime download is needed.

- [ ] **Step 6: Commit staged (not yet committed to `main` — see Task 6)**

No commit here; these changes stay staged alongside Task 2's changes until the final amend.

---

### Task 5: Update documentation (README.md, DOCKER.md, CLAUDE.md)

**Files:**
- Modify: `README.md:13-42` (Layout section), `README.md` (Environment section), `README.md:208-215` (Known gotchas)
- Modify: `DOCKER.md:20-26` (Build section)
- Modify: `CLAUDE.md` (Local modifications section)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Replace `README.md`'s Layout section**

Current (lines 13-42):
```markdown
## Layout

```
ContactPilot/
├── README.md                      ← this file
├── mujoco_grasp_sim/              ← MuJoCo tabletop sim (Panda + RGB-D + CGN) — see its README
├── mujoco_menagerie/              ← sparse clone: franka_emika_panda model only
└── contact_graspnet_pytorch/      ← cloned repo (weights + test data bundled)
    ├── checkpoints/contact_graspnet/checkpoints/model.pt   (266 MB, included)
    ├── test_data/0.npy … 13.npy   (14 test scenes: rgb, depth[m], K, seg @ 1280×720)
    ├── test_inference_headless.py (our headless driver — no GUI, prints stats)
    ├── results/                   (predictions land here as .npz)
    └── contact_graspnet_pytorch/  (source)
```

**Local modifications (re-apply when cloning fresh on the lab PC):**
1. `contact_graspnet_pytorch/checkpoints.py` — `torch.load` wrapped in
   `torch.serialization.safe_globals([...numpy scalars...])` with
   `weights_only=True`. **Required on PyTorch ≥ 2.6** or the checkpoint won't load.
2. `contact_graspnet_pytorch/visualize_saved_scene.py` — now takes
   `--results_path` instead of being hardcoded to scene 7.
3. `contact_graspnet_pytorch/visualization_utils_o3d.py` — bare `import mesh_utils`
   changed to package-relative (the original only worked when running from
   inside the package directory).
4. `test_inference_headless.py` (new file) — headless inference driver.

> Note on `allow_pickle=True`: the repo's `.npy`/`.npz` files store Python dicts,
> so pickle loading is required by upstream design. All such files here are
> either shipped with the repo or generated locally by our own scripts — safe.

---
```

New:
```markdown
## Layout

```
ContactPilot/
├── README.md                      ← this file
├── mujoco_grasp_sim/              ← MuJoCo tabletop sim (Panda + RGB-D + CGN) — see its README
├── mujoco_menagerie/              ← sparse clone: franka_emika_panda model only
└── contact_graspnet_pytorch/      ← git submodule (VivekSai07/contact_graspnet_pytorch, pinned tag)
    ├── checkpoints/contact_graspnet/checkpoints/model.pt   (26 MB, fetched by download_assets.py)
    ├── test_data/0.npy … 13.npy   (14 test scenes: rgb, depth[m], K, seg @ 1280×720; fetched by download_assets.py)
    ├── test_inference_headless.py (headless driver — no GUI, prints stats)
    ├── scripts/download_assets.py (fetches checkpoint + test_data from Hugging Face Hub)
    ├── results/                   (predictions land here as .npz)
    └── contact_graspnet_pytorch/  (source)
```

`contact_graspnet_pytorch/` tracks its own patches as normal commits on the
fork (checkpoint-loading fix for PyTorch ≥ 2.6, `visualize_saved_scene.py`
`--results_path` flag, a package-relative import fix, the headless driver,
constant-VRAM `forward_passes` batching, and a `torch.cross` compat fix) —
see the fork's commit history rather than a local diff.

> Note on `allow_pickle=True`: the repo's `.npy`/`.npz` files store Python dicts,
> so pickle loading is required by upstream design. All such files here are
> either shipped with the repo or generated locally by our own scripts — safe.

---
```

- [ ] **Step 2: Add a "Getting the code + assets" subsection to `README.md`'s Environment section**

Current start of the Environment section:
```markdown
## Environment

Conda env **`cgn_torch`** — Python 3.10, PyTorch 2.12.0+cu126, numpy 1.26 (**must stay < 2**).
```

New:
```markdown
## Environment

### Getting `contact_graspnet_pytorch` (submodule + assets)

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

Conda env **`cgn_torch`** — Python 3.10, PyTorch 2.12.0+cu126, numpy 1.26 (**must stay < 2**).
```

- [ ] **Step 3: Add `huggingface_hub` to "Recreate the env from scratch"**

Current:
```markdown
pip install "numpy<2" opencv-python pillow scipy trimesh pyyaml tqdm open3d plotly matplotlib pyrender pyglet
pip install -e . --no-build-isolation     # from repo root
```

New:
```markdown
pip install "numpy<2" opencv-python pillow scipy trimesh pyyaml tqdm open3d plotly matplotlib pyrender pyglet huggingface_hub
pip install -e . --no-build-isolation     # from repo root
```

This ensures a freshly recreated `cgn_torch` env can run
`contact_graspnet_pytorch/scripts/download_assets.py` without a separate
`pip install huggingface_hub` step (Task 3's implementer found this gap: the
Docker path already had `huggingface_hub` via `requirements-docker.txt`, but
the local conda-env path did not).

- [ ] **Step 4: Remove the obsolete autocrlf gotcha from "Known gotchas"**

Current (lines 210-215):
```markdown
- **Windows git `core.autocrlf=true` corrupts `gripper_control_points/panda_gripper_coords.pickle`**
  (protocol-0 ASCII pickle → git converts LF→CRLF on checkout → visualization crashes with
  `UnpicklingError: the STRING opcode argument must be quoted`). Fixed locally by stripping
  CR bytes. **When cloning fresh (lab PC!), clone with:**
  `git clone --config core.autocrlf=false https://github.com/elchun/contact_graspnet_pytorch.git`
  (On Linux lab PCs this doesn't occur — autocrlf is a Windows default.)
- **PyTorch ≥ 2.6**: stock repo fails with `Weights only load failed` — our
```

New (delete the first bullet entirely, keep the rest):
```markdown
- **PyTorch ≥ 2.6**: stock repo fails with `Weights only load failed` — our
```

- [ ] **Step 5: Verify README changes**

```bash
grep -n "266 MB" README.md
grep -n "core.autocrlf=false" README.md
grep -n "download_assets.py" README.md
grep -n "huggingface_hub" README.md
```

Expected: first two greps return nothing (stale content removed); third grep finds at least 2 matches (Layout section + new setup subsection); fourth grep finds at least 2 matches (new setup subsection + recreate-env pip line).

- [ ] **Step 6: Update `DOCKER.md`'s Build section**

Current (lines 20-26):
```markdown
## Build (one command)

From the repo root (the folder containing this file):

```bash
docker compose build
```

First build downloads ~4 GB (torch cu128) — later builds reuse cached layers.
Code-only changes to `mujoco_grasp_sim/` rebuild in seconds.
```

New:
```markdown
## Build (one command)

From the repo root (the folder containing this file), make sure the
`contact_graspnet_pytorch` submodule is checked out first:

```bash
git submodule update --init --depth 1
docker compose build
```

First build downloads ~4 GB (torch cu128) plus the ~160 MB Contact-GraspNet
checkpoint/test-data (fetched from Hugging Face Hub during the build) —
later builds reuse cached layers.
Code-only changes to `mujoco_grasp_sim/` rebuild in seconds.
```

- [ ] **Step 7: Verify DOCKER.md changes**

```bash
grep -n "submodule" DOCKER.md
```

Expected: at least one match in the Build section.

- [ ] **Step 8: Replace CLAUDE.md's "Local modifications" section**

Current section header and content in `CLAUDE.md`:
```markdown
## Local modifications to the vendored `contact_graspnet_pytorch` repo

These are hand-patches on top of the upstream `elchun/contact_graspnet_pytorch`
clone and must be re-applied if that repo is ever re-cloned fresh (e.g. on a
new lab PC):

1. `contact_graspnet_pytorch/checkpoints.py` — `torch.load` wrapped in
   `torch.serialization.safe_globals([...])` with `weights_only=True`.
   **Required on PyTorch >= 2.6**, otherwise checkpoint loading fails.
2. `visualize_saved_scene.py` — takes `--results_path` instead of being
   hardcoded to one scene.
3. `visualization_utils_o3d.py` — `import mesh_utils` changed to a
   package-relative import (bare import only worked from inside the package dir).
4. `test_inference_headless.py` — new file, not upstream; the headless
   inference driver used for day-to-day runs instead of the blocking-GUI
   `inference.py`.

Also: clone the upstream repo with `--config core.autocrlf=false`. Windows'
default `core.autocrlf=true` corrupts `gripper_control_points/panda_gripper_coords.pickle`
(a protocol-0 ASCII pickle — LF→CRLF conversion on checkout breaks unpickling).
```

New:
```markdown
## Contact-GraspNet dependency

`contact_graspnet_pytorch/` is a git submodule pointing at
[`VivekSai07/contact_graspnet_pytorch`](https://github.com/VivekSai07/contact_graspnet_pytorch),
pinned to a tag (not a floating branch). All local patches (checkpoint
loading fix for PyTorch >= 2.6, `visualize_saved_scene.py` `--results_path`
flag, a package-relative import fix, the headless inference driver, plus
constant-VRAM `forward_passes` batching and a `torch.cross` compat fix) live
as normal commits on the fork — check its history rather than looking for a
local diff.

The checkpoint (`model.pt`) and the 14 test scenes are not committed to
either repo; they're hosted on Hugging Face Hub and fetched by
`contact_graspnet_pytorch/scripts/download_assets.py`. After
`git submodule update --init`, run that script once to populate
`checkpoints/` and `test_data/` (it's idempotent).
```

- [ ] **Step 9: Verify CLAUDE.md changes**

```bash
grep -n "Local modifications" CLAUDE.md
grep -n "Contact-GraspNet dependency" CLAUDE.md
```

Expected: first grep returns nothing (old section header gone); second grep finds the new section header.

- [ ] **Step 10: Stage all documentation changes**

```bash
git add README.md DOCKER.md CLAUDE.md
```

No commit here — stays staged until Task 6's final amend.

---

### Task 6: Final migration — amend the single ContactPilot commit, force-push

**Files:** none new; finalizes everything staged in Tasks 2, 4, and 5.

**Interfaces:**
- Consumes: all staged changes from Tasks 2, 4, 5 (submodule, Dockerfile, requirements-docker.txt, README.md, DOCKER.md, CLAUDE.md).
- Produces: a rewritten `main` on `github.com/VivekSai07/ContactPilot` with the vendored 181MB removed from history entirely.

- [ ] **Step 1: Confirm everything intended is staged and nothing unintended is**

```bash
git status
```

Expected: staged changes include `.gitmodules` (new), `contact_graspnet_pytorch` (new, submodule), the bulk `deleted:` entries for the old vendored files, `Dockerfile` (modified), `requirements-docker.txt` (modified), `README.md` (modified), `DOCKER.md` (modified), `CLAUDE.md` (modified or new, depending on whether it was already tracked). No unrelated files staged.

- [ ] **Step 2: Amend the single existing commit**

```bash
git commit --amend
```

In the editor, keep or lightly revise the message, e.g.:
```
Initial commit: ContactPilot grasping pipeline

Reference contact_graspnet_pytorch as a git submodule (pinned tag v1.0.0
on the VivekSai07 fork) instead of a vendored copy; checkpoint/test-data
assets now fetched from Hugging Face Hub via download_assets.py instead
of being committed.
```

Expected: commit succeeds, produces a new commit SHA (different from `3ada254`).

- [ ] **Step 3: Force-push**

```bash
git push --force origin main
```

Expected: push succeeds; GitHub shows the rewritten history (old SHA `3ada254` no longer reachable from `main`).

- [ ] **Step 4: Verify the repo is actually small now**

```bash
git count-objects -v
```

Expected: `size-pack` on the order of a few MB, not ~180 MB (compare against the same command run before this task, if you captured it).

---

### Task 7: End-to-end verification (fresh clone, setup flow, CGN smoke tests)

**Files:** none (verification only, runs in a scratch directory).

**Interfaces:** none — this is the acceptance test for the whole plan.

- [ ] **Step 1: Simulate a fresh clone in a scratch directory**

```bash
cd "$(mktemp -d)"
git clone https://github.com/VivekSai07/ContactPilot.git contactpilot_fresh_check
cd contactpilot_fresh_check
git submodule update --init --depth 1
```

Expected: clone completes quickly (repo is now small); submodule init pulls `contact_graspnet_pytorch` at the pinned `v1.0.0` commit.

- [ ] **Step 2: Run the asset download script**

```powershell
python contact_graspnet_pytorch\scripts\download_assets.py
```

(Requires `huggingface_hub` installed — `pip install huggingface_hub` in whatever Python is on PATH for this scratch check.)

Expected:
```
Downloaded checkpoint to ...\checkpoints\contact_graspnet\checkpoints\model.pt
Downloaded 14 test scenes to ...\test_data
```

- [ ] **Step 3: Run the CGN inference smoke test against the recorded baseline**

In the `cgn_torch` conda env, from this scratch checkout:

```powershell
conda activate cgn_torch
cd contact_graspnet_pytorch
python test_inference_headless.py --np_path=test_data\7.npy
```

Expected: output is consistent with the baseline recorded in `README.md`
("scene 7 → 222 grasps / 8 objects, 2.82 GB peak VRAM, ~48 s" on a GTX
1650) — allow for normal CGN stochastic variance in grasp count, but VRAM
and object count should match.

- [ ] **Step 4: Exercise the new sequential `forward_passes` code path**

```powershell
python test_inference_headless.py --np_path=test_data\7.npy --forward_passes=5
```

Expected: completes without CUDA OOM; peak VRAM reported should be close to
the single-pass run's peak VRAM (not roughly 5x it), confirming the fork's
sequential-batching change (Task 3 background) is working as intended.

- [ ] **Step 5: Confirm Docker still builds and runs from this fresh checkout**

```bash
docker compose build
docker compose up
```

Expected: build succeeds (per Task 4's verification), and the default run
(`--seed 5 --camera lookat --pick-all`) completes and writes
`mujoco_grasp_sim/output/<timestamp>/metrics.json`.

- [ ] **Step 6: Clean up the scratch checkout**

```bash
cd ..
rm -rf contactpilot_fresh_check
```
