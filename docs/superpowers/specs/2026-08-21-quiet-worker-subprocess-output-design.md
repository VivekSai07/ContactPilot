# Quiet Worker Subprocess Output — Design

## Problem

`run_sim_grasp_test.py`'s three worker subprocesses (`cgn_worker.py`,
`graspgen_worker.py`, `sam3_worker.py`) run with no output capture — their
stdout/stderr flow straight into the terminal, interleaved with this
project's own `[bracket]` prints. None of the three suppress anything
before importing their heavy third-party dependencies (`torch`, `timm`,
`spconv`, `sam3`), so a normal successful run is flooded with
`FutureWarning`/`UserWarning` deprecation noise and INFO-level logging
(`grasp_gen.*`, `OpenGL.acceleratesupport`, `torch.fx._symbolic_trace`'s
one-time warning) that has nothing to do with this project's own
diagnostics.

## Decisions (confirmed with the user)

- **Default behavior**: quiet — noise hidden by default, with a
  `--verbose` flag on `run_sim_grasp_test.py` to restore full output for
  debugging a real failure.
- **Scope of "noise"**: third-party `FutureWarning`/`UserWarning` plus
  informational (`INFO`) logging from third-party loggers. This project's
  own `[bracket]` prints and real errors/tracebacks are never touched.

## Architecture

Two pieces, verified by direct source inspection (not guessed):

1. **Worker-side suppression** (`cgn_worker.py`, `graspgen_worker.py`,
   `sam3_worker.py`): each script's known noise is confirmed to be
   ordinary `warnings.warn()` (the timm/spconv `FutureWarning`s, the
   pkg_resources `UserWarning`, GraspGen's tensor-construction
   `UserWarning`) or ordinary `logging.Logger.warning()`/`.info()` calls
   (`grasp_gen.*`, `OpenGL.acceleratesupport`, and — confirmed by reading
   `torch/fx/_symbolic_trace.py` directly — the `torch.fx._symbolic_trace`
   "W..." line, which despite its glog-style timestamp/pid prefix is
   plain stdlib `logging`). No PyTorch C++/glog-level env var is needed.
   Each worker script gets a small guard at the very top, before any
   other import, that (unless `SIM_GRASP_VERBOSE` is set) filters those
   warning categories and raises the level of the known noisy logger
   namespaces.
2. **Capture-on-success, surface-on-failure** at the subprocess call
   site: a new shared `run_worker(cmd) -> int` helper (replacing the
   three near-identical `subprocess.run(cmd)` call sites in
   `run_sim_grasp_test.py`, `graspgen_predictor.py`, and
   `prompt_selector.py`) captures stdout/stderr when not verbose and only
   writes them out if the worker's exit code is nonzero — so a genuine
   crash always shows its full, unfiltered output automatically, without
   requiring the user to already know to re-run with `--verbose`. In
   verbose mode, output streams live exactly as it does today (no
   capture).

`SIM_GRASP_VERBOSE` is a plain environment variable, not a function
parameter threaded through `GraspGenPredictor`/`ContactGraspNetPredictor`/
`PromptSelector`'s public methods — `run_sim_grasp_test.py`'s new
`--verbose` flag just sets it once in `os.environ` before any subprocess
call happens; `subprocess.run` inherits the parent's environment by
default, so no explicit `env=` plumbing is needed at any call site.

## Testing

`run_worker()` is tested directly with trivial inline `python -c` commands
(no GPU/heavy deps needed): a success case (quiet mode captures and
discards output), a failure case (quiet mode surfaces the captured output
on nonzero exit), and a returncode-propagation check in verbose mode.
Matches this repo's no-pytest/standalone-`test_*.py` convention.

## Verification

Re-run the exact instruction from this session's live test (seed 2,
GraspGen/fused, SAM 3) and confirm the terminal shows only this project's
own `[bracket]` lines on success; re-run with `--verbose` and confirm the
original full third-party output is still available on demand.
