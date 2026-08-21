# Menagerie Submodule + Generated-XML Relocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop writing generated/patched MJCF into the vendored `mujoco_menagerie/` directory, then convert `mujoco_menagerie` into a real git submodule (matching `contact_graspnet_pytorch`/`GraspGen`) instead of a manually sparse-checked-out, directly-committed copy.

**Architecture:** `SceneGenerator` currently patches `panda.xml` (friction/mount-height) and writes a fresh per-run scene XML directly inside `mujoco_menagerie/franka_emika_panda/`, because MuJoCo resolves `meshdir` and `<include>` paths relative to the top-level XML file's own directory — co-location was the only way to keep `meshdir="assets"` resolving. We break that coupling by rewriting `meshdir` to an **absolute** path during the existing patch step, which frees both generated files to live in `mujoco_grasp_sim/assets/` instead. Once nothing writes into the vendor tree anymore, `mujoco_menagerie` can safely become a `git submodule` (sparse-checked-out to `franka_emika_panda/` only, same pattern the setup instructions already describe) without ever showing as "dirty".

**Tech Stack:** Python 3.10, MuJoCo (MJCF/XML), git submodules, `cgn_torch` conda env for the final smoke test; `graspgen_torch` conda env (via `GRASPGEN_PYTHON`) for the GraspGen backend used in the final validation run.

**Spec:** No separate written spec — this plan implements the design agreed in conversation (classified as a **bounded** change: existing flow in `mujoco_grasp_sim/sim_grasp/scene_generator.py`, no new subsystem). Key decisions, recorded here since there's no separate doc:
- Generated files move to `mujoco_grasp_sim/assets/` (flat, not a new subdirectory — explicit user instruction).
- `mujoco_menagerie` becomes a real git submodule, sparse-checked-out to `franka_emika_panda/`, matching the existing setup-instructions text in `scene_generator.py`'s `FileNotFoundError` message.
- Final task must run one live GraspGen-backend pick to confirm nothing broke (explicit user instruction) — this is a **manual verification step**, not a scripted assertion, since it depends on `GRASPGEN_PYTHON` being configured in the executor's environment.

## Global Constraints

- **numpy must stay < 2** across the whole stack (from `CLAUDE.md`) — do not introduce or upgrade any dependency here.
- This codebase has **no pytest suite** — existing tests under `sim_grasp/test_*.py` are standalone assert-based scripts run directly with `python sim_grasp/test_name.py` (see `sim_grasp/test_resolve_real_label.py`). New tests in this plan follow that same convention.
- `cgn_torch` conda env must be active for every command in this plan except the final GraspGen smoke test, which additionally needs `GRASPGEN_PYTHON` pointed at a `graspgen_torch` interpreter (see `mujoco_grasp_sim/README.md`, "GraspGen backend setup").
- Never silently fall back — the existing patch step already raises `RuntimeError` if `panda.xml`'s structure doesn't match the expected regex (see `_patched_panda_xml`); the new `meshdir` patch must follow the same fail-loudly pattern.

---

## File Structure

- **Modify:** `mujoco_grasp_sim/sim_grasp/scene_generator.py` — change `MENAGERIE_PANDA_DIR`-based output paths to a new `GENERATED_DIR` constant (`mujoco_grasp_sim/assets/`), and rewrite `<compiler meshdir="assets">` to an absolute path during patching.
- **Create:** `mujoco_grasp_sim/sim_grasp/test_scene_generator_paths.py` — standalone smoke test asserting generated files land in the new location and the model still loads (mesh resolution didn't silently break).
- **Modify:** `.gitignore` (repo root) — drop the old `mujoco_menagerie/franka_emika_panda/_generated_scene.xml` ignore line, add the two new paths under `mujoco_grasp_sim/assets/`.
- **Modify:** `.gitmodules` (repo root) — add the `mujoco_menagerie` submodule entry.
- **Delete + re-add:** `mujoco_menagerie/` — untrack the manually-vendored copy, re-add as a submodule.
- **Modify:** `CLAUDE.md`, `README.md`, `mujoco_grasp_sim/README.md` — update setup instructions (submodule init step) and drop stale "generated XML lives inside menagerie" language.

---

## Task 1: Relocate generated/patched XML output, decouple `meshdir` from co-location

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/scene_generator.py:1-15` (module docstring), `:30-34` (path constants), `:217-253` (`_patched_panda_xml`), `:421-423` (`_build_scene_xml` output write)
- Test: `mujoco_grasp_sim/sim_grasp/test_scene_generator_paths.py`

**Interfaces:**
- Consumes: `MENAGERIE_PANDA_DIR` (existing constant, unchanged — still points at `mujoco_menagerie/franka_emika_panda/`, still the *source* of `panda.xml`/`assets/`).
- Produces: new `GENERATED_DIR = PROJECT_ROOT / 'assets'` constant; `SceneGenerator._patched_panda_xml()` still returns a filename (now relative to `GENERATED_DIR`, not `MENAGERIE_PANDA_DIR`); `SceneGenerator.scene_xml_path` (existing public attribute, unchanged type — now points into `GENERATED_DIR`).

- [ ] **Step 1: Read the two functions being changed in full, to get exact current text for the edits below**

Run: `sed -n '1,40p;140,255p;355,425p' mujoco_grasp_sim/sim_grasp/scene_generator.py`

This confirms line numbers haven't drifted before editing (this plan was written against a specific snapshot of the file).

- [ ] **Step 2: Add the `GENERATED_DIR` constant and update the module docstring**

In `mujoco_grasp_sim/sim_grasp/scene_generator.py`, replace the docstring's closing paragraph (currently: `"The generated XML is written INTO the menagerie franka_emika_panda directory so that panda.xml's relative meshdir="assets" keeps resolving (MJCF resolves asset paths relative to the main model file)."`) with:

```python
"""SceneGenerator — randomized MuJoCo tabletop scenes with a Menagerie Franka Panda.

Responsibilities
----------------
1. Patch the Menagerie panda.xml (strip keyframe — its qpos size would clash
   with our added free joints — and lift link0 to tabletop height).
2. Generate an MJCF scene: pedestal + table + eye-to-hand camera + N random
   objects (primitives and, if available, meshes from assets/objects/).
3. Spawn objects without catastrophic penetration (rejection-sampled XY,
   staggered drop heights) and settle physics before observation capture.

The patched panda.xml and the generated scene XML are written into
mujoco_grasp_sim/assets/ (NOT into the vendored mujoco_menagerie/ tree —
that directory is a git submodule and must stay pristine so it never shows
as locally modified). Because MJCF resolves meshdir/<include> paths
relative to the top-level XML file's own directory, the patch step rewrites
panda.xml's <compiler meshdir="assets"> to an absolute path pointing back
at mujoco_menagerie/franka_emika_panda/assets, so mesh loading still works
regardless of where the generated files live.
"""
```

Then update the path constants block (currently `_THIS_DIR` / `PROJECT_ROOT` / `REPO_ROOT` / `MENAGERIE_PANDA_DIR` / `MESH_OBJECT_DIR`) to add one new constant:

```python
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent                              # mujoco_grasp_sim/
REPO_ROOT = PROJECT_ROOT.parent                              # repo root
MENAGERIE_PANDA_DIR = REPO_ROOT / 'mujoco_menagerie' / 'franka_emika_panda'
MESH_OBJECT_DIR = PROJECT_ROOT / 'assets' / 'objects'        # drop YCB .obj/.stl here
GENERATED_DIR = PROJECT_ROOT / 'assets'                       # patched/generated MJCF lands here
```

- [ ] **Step 3: Patch `_patched_panda_xml` to rewrite `meshdir` to an absolute path and write to `GENERATED_DIR`**

Find the existing method (currently ends with the friction-patch `re.subn` call, then `out = MENAGERIE_PANDA_DIR / '_panda_sim_patched.xml'` / `out.write_text(...)` / `return out.name`). Insert the `meshdir` rewrite immediately after the `link0` mount-height patch and its `RuntimeError` check, and before the friction-patch block, so the method reads:

```python
    def _patched_panda_xml(self) -> str:
        """Strip the keyframe (qpos-size clash with added free joints), lift
        link0 onto the pedestal at table height, and rewrite meshdir to an
        absolute path (the output file no longer lives next to panda.xml,
        so the original relative meshdir="assets" would otherwise break).
        Returns the patched filename (written into GENERATED_DIR)."""
        src = (MENAGERIE_PANDA_DIR / 'panda.xml').read_text(encoding='utf-8')
        patched = re.sub(r'<keyframe>.*?</keyframe>', '', src, flags=re.S)
        patched = patched.replace(
            '<body name="link0" childclass="panda">',
            f'<body name="link0" childclass="panda" pos="0 0 {self.cfg.table_height}">',
            1)
        if f'pos="0 0 {self.cfg.table_height}"' not in patched:
            raise RuntimeError('Failed to patch link0 mount height in panda.xml '
                               '(upstream file structure changed?)')

        # meshdir="assets" is relative to panda.xml's own directory; since
        # the patched file is now written into GENERATED_DIR (not next to
        # panda.xml), rewrite it to an absolute path so mesh loading still
        # resolves correctly regardless of where the top-level scene XML
        # that includes this file actually lives.
        abs_meshdir = str((MENAGERIE_PANDA_DIR / 'assets').resolve())
        patched, n_meshdir = re.subn(
            r'meshdir="assets"',
            f'meshdir="{abs_meshdir}"',
            patched)
        if n_meshdir != 1:
            raise RuntimeError(
                f'Expected to patch exactly one meshdir="assets" in panda.xml, '
                f'patched {n_meshdir} (upstream file structure changed?)')

        # [P1 friction audit] Compliant rubber fingertip pads grip harder
        # against rotation/slip than MuJoCo's rigid-body default
        # (1 0.005 0.0001, condim=3) — bump sliding+torsional friction on the
        # 5 fingertip pad collision boxes, the actual contact surfaces during
        # a grasp, to resist the slow sliding/torsional "walk-out" of
        # off-center grasps during lift/transit-to-bin.
        # condim="4" is REQUIRED here: MuJoCo's default condim=3 only puts
        # friction[0] (sliding) into the contact friction cone, so the
        # torsional term friction[1] is a dead value no matter how high it's
        # set without it (the original 1.0/0.01/0.004 patch was a no-op,
        # since 1.0 sliding == the unpatched default).
        patched, n_friction = re.subn(
            r'(<default class="fingertip_pad_collision_\d">\s*'
            r'<geom type="box" size="[^"]*" pos="[^"]*")/>',
            r'\1 friction="1.5 0.02 0.004" condim="4"/>',
            patched)
        if n_friction != 5:
            raise RuntimeError(
                f'Expected to patch friction on 5 fingertip pad collision '
                f'geoms in panda.xml, patched {n_friction} '
                '(upstream file structure changed?)')
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        out = GENERATED_DIR / '_panda_sim_patched.xml'
        out.write_text(patched, encoding='utf-8')
        return out.name
```

- [ ] **Step 4: Point the generated-scene writer at `GENERATED_DIR`**

Find the tail of `_build_scene_xml` (currently `out = MENAGERIE_PANDA_DIR / '_generated_scene.xml'` / `out.write_text(xml, encoding='utf-8')` / `return out`) and change it to:

```python
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        out = GENERATED_DIR / '_generated_scene.xml'
        out.write_text(xml, encoding='utf-8')
        return out
```

(`GENERATED_DIR.mkdir` is idempotent and cheap to call twice — `_patched_panda_xml` runs first within `generate()`, so this second call is a no-op in practice, but keeping it here makes the function correct standalone.)

- [ ] **Step 5: Write the smoke test**

Create `mujoco_grasp_sim/sim_grasp/test_scene_generator_paths.py`:

```python
"""Standalone check that generated/patched MJCF lands in GENERATED_DIR (not
inside the vendored mujoco_menagerie/ tree) and that the model still loads
correctly (i.e. the absolute-meshdir rewrite didn't break mesh resolution).
Run directly, no pytest (this codebase has no automated test suite)."""
from sim_grasp.scene_generator import (
    GENERATED_DIR, MENAGERIE_PANDA_DIR, SceneConfig, SceneGenerator,
)

gen = SceneGenerator(SceneConfig(seed=0))
model, data = gen.generate()

# Generated files must live in GENERATED_DIR, never inside the vendored dir.
assert gen.scene_xml_path.parent == GENERATED_DIR, (
    f'generated scene XML written to {gen.scene_xml_path.parent}, '
    f'expected {GENERATED_DIR}')
patched_panda = GENERATED_DIR / '_panda_sim_patched.xml'
assert patched_panda.is_file(), f'{patched_panda} was not written'
assert not (MENAGERIE_PANDA_DIR / '_panda_sim_patched.xml').exists(), (
    'stale patched panda.xml still present inside the vendored menagerie dir')
assert not (MENAGERIE_PANDA_DIR / '_generated_scene.xml').exists(), (
    'stale generated scene XML still present inside the vendored menagerie dir')

# Model must have actually loaded meshes (not silently fallen back to zero
# geoms) -- the panda arm alone has dozens of mesh geoms.
n_mesh_geoms = sum(1 for i in range(model.ngeom)
                    if model.geom_type[i] == 7)  # mjGEOM_MESH == 7
assert n_mesh_geoms > 10, (
    f'expected >10 mesh geoms from the Panda arm, got {n_mesh_geoms} '
    '(meshdir rewrite likely broke mesh resolution)')

print(f'All scene_generator path checks passed ({n_mesh_geoms} mesh geoms loaded).')
```

- [ ] **Step 6: Run it and confirm it fails before the fix is in place is moot (fix is already written) — instead run it now to verify the fix works**

Run (from `mujoco_grasp_sim/`, `cgn_torch` env active):
```bash
python sim_grasp/test_scene_generator_paths.py
```
Expected output: `All scene_generator path checks passed (N mesh geoms loaded).` with no assertion errors. If it raises `RuntimeError` from the meshdir patch, the regex in Step 3 didn't match — re-check `mujoco_menagerie/franka_emika_panda/panda.xml`'s `<compiler>` line hasn't changed.

- [ ] **Step 7: Run the existing full scene-generation path once more manually to be sure nothing else regressed**

Run: `python run_sim_grasp_test.py --seed 5 --no-vis --pick-object 1 --grasp-index 0`

This exercises the same `SceneGenerator.generate()` path end-to-end (including camera rendering against the newly-relocated model) without needing any grasp backend to succeed — a crash here means the scene/model itself is broken, independent of grasp prediction. Expected: runs to completion (pick may succeed or fail depending on the randomized seed/object — that's not what we're checking; a stack trace from `mujoco.MjModel.from_xml_path` or a rendering error would indicate the relocation broke something).

- [ ] **Step 8: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/scene_generator.py mujoco_grasp_sim/sim_grasp/test_scene_generator_paths.py
git commit -m "Write generated/patched menagerie MJCF into assets/, not the vendored tree"
```

---

## Task 2: Git hygiene — untrack the stale generated file, update `.gitignore`

**Files:**
- Modify: `.gitignore` (repo root)
- Delete (from git tracking, not from disk if regenerated): `mujoco_menagerie/franka_emika_panda/_panda_sim_patched.xml`

**Interfaces:**
- Consumes: Task 1's new output paths (`mujoco_grasp_sim/assets/_panda_sim_patched.xml`, `mujoco_grasp_sim/assets/_generated_scene.xml`).
- Produces: a clean `git status` after running the generator — no generated files show up as tracked or as unexpected untracked files outside `mujoco_grasp_sim/assets/`.

- [ ] **Step 1: Confirm the old tracked file is gone from disk (Task 1 already stopped writing there) and untrack it from git**

```bash
git status --short mujoco_menagerie/
git rm --cached mujoco_menagerie/franka_emika_panda/_panda_sim_patched.xml
```

Expected: `git status --short` shows nothing under `mujoco_menagerie/` before the `rm --cached` (since Task 1's test run in Step 7 already regenerated files into the new location and the old file was never rewritten) — if it *does* show a diff, do not proceed to Task 3 until you understand why (it would mean something is still writing into the vendored dir).

- [ ] **Step 2: Update `.gitignore`**

Read the current root `.gitignore` entry (`grep -n "generated_scene\|panda_sim_patched" .gitignore`) and replace the line:
```
mujoco_menagerie/franka_emika_panda/_generated_scene.xml
```
with:
```
mujoco_grasp_sim/assets/_generated_scene.xml
mujoco_grasp_sim/assets/_panda_sim_patched.xml
```

(Both generated files are now ignored — `_panda_sim_patched.xml` was previously tracked, but it's pure build output derived from `panda.xml` + this repo's patch logic, same category as `_generated_scene.xml`.)

- [ ] **Step 3: Verify ignore rules work and no unexpected files are tracked**

```bash
cd mujoco_grasp_sim && python sim_grasp/test_scene_generator_paths.py && cd ..
git status --short
```

Expected: `git status --short` shows only the `.gitignore` modification and the `git rm --cached` staged deletion — the freshly-regenerated files in `mujoco_grasp_sim/assets/` must **not** appear (confirms the new ignore rules match).

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "Stop tracking generated menagerie MJCF; ignore new assets/ output location"
```

---

## Task 3: Convert `mujoco_menagerie` to a real git submodule

**Files:**
- Modify: `.gitmodules` (repo root)
- Delete + re-create: `mujoco_menagerie/` (entire directory, replaced by a submodule checkout)

**Interfaces:**
- Consumes: Task 1/2 completed (nothing writes into `mujoco_menagerie/` anymore, so it's safe for git to manage it as a submodule without ever seeing local modifications).
- Produces: `mujoco_menagerie` submodule entry alongside the existing `contact_graspnet_pytorch`/`GraspGen` entries in `.gitmodules`; `git submodule status` lists all three.

- [ ] **Step 1: Remove the manually-vendored copy from git tracking and disk**

```bash
git rm -r --cached mujoco_menagerie
rm -rf mujoco_menagerie
git status --short
```

Expected: `mujoco_menagerie/` no longer appears in `git status` (fully untracked and removed) before proceeding.

- [ ] **Step 2: Add it as a sparse-checked-out submodule**

```bash
git submodule add --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git mujoco_menagerie
git -C mujoco_menagerie sparse-checkout init --cone
git -C mujoco_menagerie sparse-checkout set franka_emika_panda
```

Expected: `mujoco_menagerie/franka_emika_panda/panda.xml` exists again on disk, and `mujoco_menagerie/` now contains only `franka_emika_panda/` plus the small set of top-level repo files (README, LICENSE, etc. — sparse-checkout with `--cone` keeps top-level files by default, matching what was there before).

- [ ] **Step 3: Verify `panda.xml` didn't change in a way that breaks the patch regexes**

```bash
cd mujoco_grasp_sim && python sim_grasp/test_scene_generator_paths.py && cd ..
```

Expected: same pass output as Task 1 Step 6. If this raises a `RuntimeError` about a patch count mismatch, upstream `panda.xml` has drifted since the original vendored copy — inspect the diff (`git diff` won't help since the old copy is gone; compare against the `RuntimeError` message and the current `mujoco_menagerie/franka_emika_panda/panda.xml` structure directly) and adjust the regex in `_patched_panda_xml` accordingly before continuing.

- [ ] **Step 4: Commit**

```bash
git add .gitmodules mujoco_menagerie
git commit -m "Convert mujoco_menagerie from vendored copy to a sparse-checked-out git submodule"
```

---

## Task 4: Update setup docs

**Files:**
- Modify: `CLAUDE.md` (submodule list + any "sparse clone" wording)
- Modify: `README.md` ("Getting the submodules" section)
- Modify: `mujoco_grasp_sim/README.md` (any menagerie setup instructions)

**Interfaces:**
- Consumes: Task 3's final submodule command sequence (the exact commands a fresh clone needs to run).
- Produces: docs that match reality — a fresh clone following them ends up with a working `mujoco_menagerie/franka_emika_panda/` checkout.

- [ ] **Step 1: Find every place menagerie setup is documented**

```bash
grep -rn "mujoco_menagerie\|sparse-checkout\|sparse clone" CLAUDE.md README.md mujoco_grasp_sim/README.md
```

- [ ] **Step 2: Update `CLAUDE.md`**

In the "What this repo is" section, change:
```
`mujoco_menagerie/franka_emika_panda/` is a sparse clone providing only the
Panda robot model, consumed by `mujoco_grasp_sim`.
```
to:
```
`mujoco_menagerie/franka_emika_panda/` is a sparse-checked-out git submodule
(pinned commit, `franka_emika_panda/` only) providing the Panda robot model,
consumed by `mujoco_grasp_sim`.
```

In "Contact-GraspNet dependency" (or wherever submodules are enumerated), add a paragraph alongside the existing `contact_graspnet_pytorch`/`GraspGen` submodule descriptions:
```
`mujoco_menagerie/` is a third git submodule, pointing at
[`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
upstream (pinned commit, sparse-checked-out to `franka_emika_panda/` only —
the full menagerie repo covers dozens of unrelated robots). Nothing in this
repo patches its source on disk: `SceneGenerator` patches `panda.xml`
in-memory and writes the result to `mujoco_grasp_sim/assets/`, never back
into the submodule.
```

- [ ] **Step 3: Update `README.md`**

In "Getting the submodules (+ CGN assets)", extend the existing two-submodule description to three, and update the init command block to include the sparse-checkout step:

```bash
git submodule update --init --depth 1
git -C mujoco_menagerie sparse-checkout init --cone
git -C mujoco_menagerie sparse-checkout set franka_emika_panda
pip install huggingface_hub
python contact_graspnet_pytorch\scripts\download_assets.py
```

Adjust the prose above the block to mention `mujoco_menagerie` by name alongside `contact_graspnet_pytorch` and `GraspGen`.

- [ ] **Step 4: Update `mujoco_grasp_sim/README.md`**

Remove any remaining instructions that say to manually `git clone --sparse` menagerie (that's now handled by the repo-root submodule init) — point readers at the root `README.md`'s "Getting the submodules" section instead.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md mujoco_grasp_sim/README.md
git commit -m "Update setup docs for mujoco_menagerie as a git submodule"
```

---

## Task 5: End-to-end validation with the GraspGen backend

**Files:** none (verification only — no code changes in this task)

**Interfaces:**
- Consumes: everything from Tasks 1–4 (relocated generated XML, submodule menagerie, updated docs).
- Produces: a recorded pass/fail confirming the full pipeline — scene generation → camera → GraspGen subprocess → feasibility → diff-IK execution → place-in-bin — still works after all the path/submodule changes.

- [ ] **Step 1: Confirm `GRASPGEN_PYTHON` is available**

```bash
echo $GRASPGEN_PYTHON
```

If empty, set it per `mujoco_grasp_sim/README.md`'s "GraspGen backend setup" section (e.g. `export GRASPGEN_PYTHON=/path/to/miniconda3/envs/graspgen_torch/bin/python`) before continuing — this task cannot be completed without it.

- [ ] **Step 2: Run one live GraspGen pick-and-place**

From `mujoco_grasp_sim/`, `cgn_torch` env active:

```bash
python run_sim_grasp_test.py --seed 5 --execute --backend graspgen --no-vis
```

- [ ] **Step 3: Inspect the result**

Check the console output and `output/<run>/metrics.json` for the run. Confirm:
- No traceback / crash (in particular, no `FileNotFoundError` for menagerie paths, no MuJoCo mesh-loading error — these are exactly the failure modes Tasks 1–3 could have introduced).
- The GraspGen subprocess launched and returned grasp candidates (console should show grasp scores from the `graspgen_torch` worker).
- The pick either succeeds and the object lands in the bin, or fails for an ordinary reason unrelated to this change (e.g. IK-reachability retry) — a failure caused by a missing/broken menagerie asset or a bad generated-XML path is what this step exists to catch, not grasp-success rate itself.

- [ ] **Step 4: Also confirm `--pick-all` still works (exercises the re-observe loop, which re-runs `SceneGenerator.generate()` multiple times per process)**

```bash
python run_sim_grasp_test.py --pick-all --backend graspgen --no-vis
```

Expected: completes without crashing; `output/<run>/metrics.json` shows attempts for each object in the scene.

- [ ] **Step 5: Report the result**

No commit for this task (verification only). Report back: did both runs complete cleanly, and did `metrics.json` show sane numbers (no zero-object scenes, no all-failures-with-a-path-error pattern)? If either run failed in a way traceable to Tasks 1–3, stop and fix before considering this plan complete — do not silently mark it done.

---

## Self-Review Notes

- **Spec coverage:** relocation of generated XML (Task 1), meshdir decoupling (Task 1 Step 3), git untracking (Task 2), submodule conversion (Task 3), doc updates (Task 4), and the explicitly-requested GraspGen end-to-end test (Task 5) are all covered.
- **Placeholder scan:** no TBD/TODO; every step has literal commands or literal code.
- **Type/signature consistency:** `GENERATED_DIR`, `MENAGERIE_PANDA_DIR`, `SceneConfig`, `SceneGenerator`, `gen.scene_xml_path` are used identically across Task 1's implementation and its test — no renamed symbols between steps.
- **Ordering risk called out explicitly:** Task 3 (submodule conversion) depends on Task 1/2 landing first, since converting the submodule while generated files were still being written into it would immediately make the submodule appear dirty. The task order in this plan enforces that.
