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
