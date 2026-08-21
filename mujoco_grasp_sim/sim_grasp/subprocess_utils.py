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
