"""SP3 routing option rejects invalid combinations before model startup."""
import subprocess
import sys
from pathlib import Path


script = Path(__file__).resolve().parents[1] / 'run_sim_grasp_test.py'
help_result = subprocess.run([sys.executable, str(script), '--help'],
                             capture_output=True, text=True)
assert '--routing' in help_result.stdout

for extra in ([], ['--scene', 'props'],
              ['--scene', 'props', '--pick-all']):
    cmd = [sys.executable, str(script), '--routing', 'scene-graph', *extra]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert '--routing requires' in (result.stderr + result.stdout), result.stderr

print('SP3 CLI guard checks passed.')
