# Quiet Worker Subprocess Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hide third-party warning/INFO-log noise from `cgn_worker.py`,
`graspgen_worker.py`, and `sam3_worker.py` by default, with a
`--verbose` flag on `run_sim_grasp_test.py` to restore full output, per
`docs/superpowers/specs/2026-08-21-quiet-worker-subprocess-output-design.md`.

**Architecture:** A shared `run_worker(cmd) -> int` helper replaces the
three near-identical `subprocess.run(cmd)` call sites; it captures
stdout/stderr and only prints them if the worker's exit code is nonzero,
unless `SIM_GRASP_VERBOSE` is set (then it streams live, unchanged from
today). Each of the three worker scripts gets a suppression guard at the
very top, before any other import, that filters known noisy warning
categories and raises known noisy logger namespaces to `ERROR`, unless
`SIM_GRASP_VERBOSE` is set.

**Tech Stack:** Python stdlib only (`subprocess`, `warnings`, `logging`,
`os`) — no new dependency.

## Global Constraints

- This codebase has **no pytest suite** — tests are standalone `test_*.py`
  scripts run directly (`python sim_grasp/test_name.py`, from
  `mujoco_grasp_sim/` with `PYTHONPATH=.`), plain `assert` statements,
  ending with a `print('All ... checks passed.')` line.
- `conda activate cgn_torch` before running anything in this repo.
- Branch: `quiet-worker-subprocess-output` (already created off `main`,
  holds the design spec commit).
- Never touch this project's own `[bracket]`-prefixed prints, or any
  real error/traceback — only third-party `FutureWarning`/`UserWarning`
  and third-party `INFO`-level logging are in scope.
- `SIM_GRASP_VERBOSE` is a plain environment variable (not a new
  parameter on any `GraspPredictor`/`PromptSelector` public method) — set
  once in `os.environ` by `run_sim_grasp_test.py`'s `--verbose` flag;
  every subprocess call relies on `subprocess.run`'s default behavior of
  inheriting the parent's environment, so no `env=` kwarg is added
  anywhere.

---

## Task 1: `sim_grasp/subprocess_utils.py` — shared quiet/verbose worker runner

**Files:**
- Create: `mujoco_grasp_sim/sim_grasp/subprocess_utils.py`
- Create: `mujoco_grasp_sim/sim_grasp/test_subprocess_utils.py`

**Interfaces:**
- Produces: `run_worker(cmd: list[str]) -> int` — runs `cmd` as a
  subprocess and returns its exit code. Reads `SIM_GRASP_VERBOSE` from
  `os.environ` directly (no parameter). Tasks 2 and 3 both depend on this
  exact signature.
- Consumes: nothing from other tasks.

**Step 1: Write the failing tests**

Create `mujoco_grasp_sim/sim_grasp/test_subprocess_utils.py`:

```python
"""Standalone checks for subprocess_utils.run_worker -- run directly, no
pytest (this codebase has no automated test suite). Uses trivial inline
`python -c` commands, not any real GPU/heavy-dependency worker, so this
test has no extra requirements beyond a Python interpreter."""
import contextlib
import io
import os
import sys

from sim_grasp.subprocess_utils import run_worker

_PRINT_BOTH = ("import sys; print('stdout-line'); "
              "print('stderr-line', file=sys.stderr)")

os.environ.pop('SIM_GRASP_VERBOSE', None)   # start from a known quiet state

# --- quiet mode (default): success -> nothing printed, output discarded ---
buf_out, buf_err = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
    rc = run_worker([sys.executable, '-c', _PRINT_BOTH])
assert rc == 0, f'expected exit code 0, got {rc}'
assert buf_out.getvalue() == '', f'expected no stdout on success, got {buf_out.getvalue()!r}'
assert buf_err.getvalue() == '', f'expected no stderr on success, got {buf_err.getvalue()!r}'

# --- quiet mode: failure -> captured output IS surfaced ---
buf_out, buf_err = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
    rc = run_worker([sys.executable, '-c', _PRINT_BOTH + '; sys.exit(1)'])
assert rc == 1, f'expected exit code 1, got {rc}'
combined = buf_out.getvalue() + buf_err.getvalue()
assert 'stdout-line' in combined, 'worker stdout must be surfaced on failure'
assert 'stderr-line' in combined, 'worker stderr must be surfaced on failure'

# --- verbose mode: returncode still propagates correctly ---
os.environ['SIM_GRASP_VERBOSE'] = '1'
try:
    assert run_worker([sys.executable, '-c', 'import sys; sys.exit(0)']) == 0
    assert run_worker([sys.executable, '-c', 'import sys; sys.exit(1)']) == 1
finally:
    os.environ.pop('SIM_GRASP_VERBOSE', None)

print('All subprocess_utils checks passed.')
```

- [ ] **Step 1: Write the failing test** (as above; confirm it fails —
      `ModuleNotFoundError: No module named 'sim_grasp.subprocess_utils'`
      — before writing the implementation)

**Step 2: Implement `subprocess_utils.py`**

```python
"""Shared subprocess runner for the cgn_worker.py/graspgen_worker.py/
sam3_worker.py worker scripts -- quiet by default (captures output,
surfaces it only on failure), or fully verbose if SIM_GRASP_VERBOSE is
set (see --verbose on run_sim_grasp_test.py), matching each worker
script's own suppression guard (see cgn_worker.py/graspgen_worker.py/
sam3_worker.py's top-of-file warnings/logging setup).
"""
import os
import subprocess
import sys


def run_worker(cmd: list) -> int:
    if os.environ.get('SIM_GRASP_VERBOSE'):
        return subprocess.run(cmd).returncode
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
    return r.returncode
```

- [ ] **Step 2: Implement `subprocess_utils.py`**

- [ ] **Step 3: Run the test to verify it passes**

Run (from `mujoco_grasp_sim/`): `PYTHONPATH=. python
sim_grasp/test_subprocess_utils.py`. Expected: `All subprocess_utils
checks passed.`

- [ ] **Step 4: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/subprocess_utils.py \
       mujoco_grasp_sim/sim_grasp/test_subprocess_utils.py
git commit -m "Add subprocess_utils.run_worker: quiet-by-default worker subprocess runner"
```

---

## Task 2: Wire the 3 call sites to `run_worker` + add `--verbose`

**Files:**
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py`
- Modify: `mujoco_grasp_sim/sim_grasp/graspgen_predictor.py`
- Modify: `mujoco_grasp_sim/sim_grasp/prompt_selector.py`

**Interfaces:**
- Consumes: `sim_grasp.subprocess_utils.run_worker` (Task 1, exact
  signature `run_worker(cmd: list) -> int`).
- Produces: `--verbose` CLI flag on `run_sim_grasp_test.py`. No other
  new public interface — this is pure internal rewiring.

**Step 1: Add `--verbose` and set the env var**

In `mujoco_grasp_sim/run_sim_grasp_test.py`, add `import os` to the
existing top-of-file import block:

```python
import argparse
import json
import os
import sys
import time
from pathlib import Path
```

Find the `ap.add_argument('--clean-depth', action='store_true', ...)`
block and add immediately before it:

```python
    ap.add_argument('--verbose', action='store_true',
                    help='show full third-party output (warnings, INFO '
                         'logs) from the cgn/graspgen/sam3 worker '
                         'subprocesses -- by default they are hidden and '
                         'only surfaced automatically if a worker fails')
```

Find `args = ap.parse_args()` and add immediately after it:

```python
    args = ap.parse_args()
    if args.verbose:
        os.environ['SIM_GRASP_VERBOSE'] = '1'
```

- [ ] **Step 1 done**

**Step 2: Rewire `run_sim_grasp_test.py`'s own CGN call site**

Find, inside `_subprocess_predict()`:

```python
def _subprocess_predict(payload, forward_passes, arg_configs,
                        work_dir) -> GraspPrediction:
    import subprocess
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    obs_f, out_f = work_dir / '_cgn_obs.npz', work_dir / '_cgn_out.npz'
    np.savez(obs_f, **payload)
    worker = Path(__file__).parent / 'sim_grasp' / 'cgn_worker.py'
    cmd = [sys.executable, str(worker), str(obs_f), str(out_f),
           '--forward-passes', str(forward_passes)]
    if arg_configs:
        cmd += ['--arg-configs', *arg_configs]
    r = subprocess.run(cmd)
    if r.returncode != 0 or not out_f.exists():
        raise RuntimeError(f'CGN worker failed (exit code {r.returncode})')
```

Replace the last three lines with:

```python
    from sim_grasp.subprocess_utils import run_worker
    returncode = run_worker(cmd)
    if returncode != 0 or not out_f.exists():
        raise RuntimeError(f'CGN worker failed (exit code {returncode})')
```

Also remove the now-unused `import subprocess` line at the top of
`_subprocess_predict()` — this function no longer calls `subprocess`
directly (no dead imports).

- [ ] **Step 2 done**

**Step 3: Rewire `graspgen_predictor.py`'s call site**

Find, inside `GraspGenPredictor._run()`:

```python
        r = subprocess.run(cmd)
        if r.returncode != 0 or not out_f.exists():
            raise RuntimeError(f'GraspGen worker failed (exit code {r.returncode})')
```

Replace with:

```python
        from sim_grasp.subprocess_utils import run_worker
        returncode = run_worker(cmd)
        if returncode != 0 or not out_f.exists():
            raise RuntimeError(f'GraspGen worker failed (exit code {returncode})')
```

- [ ] **Step 3 done**

**Step 4: Rewire `prompt_selector.py`'s call site**

Find, inside `PromptSelector.select()`:

```python
        r = subprocess.run(cmd)
        if r.returncode != 0 or not out_f.exists():
            raise RuntimeError(f'SAM 3 worker failed (exit code {r.returncode})')
```

Replace with:

```python
        from sim_grasp.subprocess_utils import run_worker
        returncode = run_worker(cmd)
        if returncode != 0 or not out_f.exists():
            raise RuntimeError(f'SAM 3 worker failed (exit code {returncode})')
```

- [ ] **Step 4 done**

**Step 5: Run the full existing test suite (no regressions)**

Run each of these from `mujoco_grasp_sim/` (`PYTHONPATH=. python
sim_grasp/<name>.py`): `test_placement_planner`, `test_executor_ease`,
`test_executor_place_orientation`, `test_color_utils`,
`test_resolve_real_label`, `test_scene_generator_paths`,
`test_feasibility`, `test_instruction_parser`,
`test_spatial_relation_resolver`, `test_subprocess_utils`. All must print
their `All ... passed.` line with no traceback.

- [ ] **Step 5 done**

**Step 6: Commit**

```bash
git add mujoco_grasp_sim/run_sim_grasp_test.py \
       mujoco_grasp_sim/sim_grasp/graspgen_predictor.py \
       mujoco_grasp_sim/sim_grasp/prompt_selector.py
git commit -m "Wire the 3 worker subprocess call sites through run_worker; add --verbose"
```

---

## Task 3: Worker-side suppression + live verification

**Files:**
- Modify: `mujoco_grasp_sim/sim_grasp/cgn_worker.py`
- Modify: `mujoco_grasp_sim/sim_grasp/graspgen_worker.py`
- Modify: `mujoco_grasp_sim/sim_grasp/sam3_worker.py`

**Interfaces:**
- Consumes: `SIM_GRASP_VERBOSE` env var (Task 2 sets it).
- Produces: nothing new — this task only changes what each worker prints,
  not any function signature.

**Background (confirmed by direct source inspection, not guessed):** all
of this noise is either plain `warnings.warn()` (the timm/spconv
`FutureWarning`s, the `pkg_resources`/GraspGen-tensor `UserWarning`s) or
plain stdlib `logging` (`grasp_gen.*`, `OpenGL.acceleratesupport`, and —
confirmed by reading `torch/fx/_symbolic_trace.py` directly, which calls
`logging.getLogger(__name__).warning(...)` — the `torch.fx._symbolic_trace`
"W..." line despite its glog-style prefix). No PyTorch C++/glog env var
(`TORCH_CPP_LOG_LEVEL` etc.) is needed.

**Step 1: Add the suppression guard to all three worker scripts**

In `mujoco_grasp_sim/sim_grasp/cgn_worker.py`, `graspgen_worker.py`, and
`sam3_worker.py`, insert this block as the very first code after each
file's module docstring, before any other import:

```python
import logging
import os
import warnings

if not os.environ.get('SIM_GRASP_VERBOSE'):
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    for _name in ('grasp_gen', 'OpenGL', 'torch', 'timm', 'spconv', 'sam3'):
        logging.getLogger(_name).setLevel(logging.ERROR)
```

This must come before each file's existing imports (`import argparse`,
`import sys`, `import numpy as np`, `import torch`, etc.) since the
`FutureWarning`s are raised at import time by `timm`/`spconv` themselves
— the filter has to already be registered before those modules are first
imported. `logging.getLogger(name)` is safe to call before `name`'s
package is ever imported (the logging registry is independent of import
state), so ordering only matters for the `warnings.filterwarnings` calls,
not the `logging.getLogger` ones — keeping both in the same block is
simplest.

- [ ] **Step 1: Add the guard to all 3 worker scripts**

**Step 2: Run the full existing test suite (no regressions)**

Same list as Task 2 Step 5. All must still pass — this task doesn't
change any function's behavior, only stdout/stderr/log noise, so no test
assertions should be affected.

- [ ] **Step 2 done**

**Step 3: Live verification**

Run (from `mujoco_grasp_sim/`, with `cgn_torch` active):

```bash
MUJOCO_GL=osmesa GRASPGEN_PYTHON=/home/vivek/miniconda3/envs/graspgen_torch/bin/python \
  python run_sim_grasp_test.py --pick-all --camera fused --backend graspgen \
  --seed 2 --no-vis \
  --sam3-python /home/vivek/miniconda3/envs/sam3_torch/bin/python \
  --instruction "Put a blue cube inside the bin and then pick a green cube and put it on the right."
```

Confirm the terminal shows ONLY this project's own `[bracket]`-prefixed
lines (`[run]`, `[camera]`, `[scene]`, `[graspgen]`, `[feasibility]`,
`[instruction]`, `[pick-all]`, etc.) — no `FutureWarning`, no
`UserWarning`, no `grasp_gen.*`/`OpenGL.acceleratesupport`/`torch.fx`
logging lines, no `pkg_resources` deprecation notice.

Then re-run the exact same command with `--verbose` appended and confirm
the original full third-party output (warnings + INFO logs) is present
again, unchanged from before this plan.

- [ ] **Step 3: Both verification runs confirmed**

**Step 4: Commit**

```bash
git add mujoco_grasp_sim/sim_grasp/cgn_worker.py \
       mujoco_grasp_sim/sim_grasp/graspgen_worker.py \
       mujoco_grasp_sim/sim_grasp/sam3_worker.py
git commit -m "Suppress third-party warnings/INFO logs in the 3 worker scripts unless SIM_GRASP_VERBOSE"
```

---

## Task 4: Documentation (per this repo's AGENTS.md workflow)

**Files:**
- Modify: `ROADMAP.md`
- Modify: `mujoco_grasp_sim/README.md` (if it documents `run_sim_grasp_test.py`'s
  flags — add `--verbose` to the flag list there if such a list exists)

**Step 1: Record the fix**

Per `AGENTS.md`'s mandatory documentation-sync workflow: this is a new
capability (`--verbose` flag) landing, found via the user manually
running `--instruction` and noticing the noisy output. Add a dated
bullet under the most relevant `ROADMAP.md` section (P6, "Interactive
live pick", is the closest existing home for CLI/UX polish to
`run_sim_grasp_test.py`; if no section fits cleanly, add it as a short
new bullet near P8 since that's what was being tested when this was
found) describing: the noise source (three worker subprocesses with
uncaptured output, no suppression before importing torch/timm/spconv/
sam3), the fix (`run_worker()` capture-on-quiet/surface-on-failure +
per-worker suppression guard, `--verbose` to restore full output), and
confirmation from the Task 3 Step 3 live verification (quiet by default,
full output on `--verbose`).

- [ ] **Step 1: Update `ROADMAP.md`** with real, specific detail (not a
      placeholder) reflecting what Tasks 1-3 actually built and verified

**Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "Document quiet worker subprocess output fix in ROADMAP.md"
```

---

## Plan Self-Review Notes

- **Spec coverage**: Task 1 covers the design's `run_worker()` helper;
  Task 2 covers the `--verbose` flag + all 3 call-site rewiring; Task 3
  covers the worker-side suppression guard + the design's required live
  verification (both quiet and `--verbose` runs); Task 4 covers this
  repo's `AGENTS.md`-mandated documentation sync. All spec sections have
  a corresponding task.
- **Type consistency checked**: `run_worker(cmd: list) -> int`'s
  signature is identical in Task 1's implementation, its test, and all
  three Task 2 call sites.
- **No placeholders**: every step contains complete, runnable code — no
  "add appropriate suppression" or "similar to the other worker"
  shortcuts; each of the 3 worker scripts' exact insertion point and
  exact code is spelled out in Task 3 rather than referenced by analogy.
