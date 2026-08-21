"""Standalone check for executor._pick_best_seed_result -- run directly, no
pytest (this codebase has no automated test suite). Uses synthetic
IKResult objects (no live MuJoCo model needed), mirroring
test_executor_ease.py's pattern for testing pulled-out pure logic."""
import numpy as np

from sim_grasp.executor import _pick_best_seed_result, IKResult

def _r(pos_err, ori_err, converged, tag):
    # qpos payload is just a tag so we can identify which result won
    return IKResult(qpos=np.array([tag]), pos_err=pos_err, ori_err=ori_err,
                    converged=converged)

# 1. q_init converges tightly -> picked immediately, even if a later seed
#    (canonical elbow-down, tag=1) has a lower position error.
q_init_tight = _r(0.001, 0.005, converged=True, tag=0)
competitor_lower_err = _r(0.0005, 0.003, converged=True, tag=1)
picked = _pick_best_seed_result([q_init_tight, competitor_lower_err])
assert picked.qpos[0] == 0, 'q_init must win when it converges, even if a later seed is more accurate'

# 2. q_init converges loosely (still `converged=True` under the existing
#    2x-tolerance definition) but a later seed is numerically more
#    accurate -- q_init must STILL win (this is the bug this plan fixes:
#    "converged" already means "good enough", no further comparison).
q_init_loose = _r(0.0075, 0.018, converged=True, tag=0)  # within 2x tol, not 1x
competitor_more_accurate = _r(0.001, 0.002, converged=True, tag=1)
picked = _pick_best_seed_result([q_init_loose, competitor_more_accurate])
assert picked.qpos[0] == 0, \
    'q_init must win whenever it converged, regardless of a competitor being more accurate'

# 3. q_init fails to converge entirely -> fall through to the best
#    CONVERGED competitor (lowest pos_err among converged results).
q_init_failed = _r(0.05, 0.1, converged=False, tag=0)
worse_converged = _r(0.006, 0.015, converged=True, tag=1)
better_converged = _r(0.002, 0.006, converged=True, tag=2)
picked = _pick_best_seed_result([q_init_failed, worse_converged, better_converged])
assert picked.qpos[0] == 2, 'when q_init fails, the best CONVERGED competitor must win'

# 4. Nothing converges at all -> fall back to lowest pos_err overall
#    (existing "best effort" behavior, unchanged).
none_converged = [_r(0.05, 0.1, converged=False, tag=0),
                  _r(0.03, 0.08, converged=False, tag=1),
                  _r(0.09, 0.2, converged=False, tag=2)]
picked = _pick_best_seed_result(none_converged)
assert picked.qpos[0] == 1, 'with nothing converged, lowest pos_err wins as a best-effort result'

# 5. Single-result list (only q_init tried, e.g. early-exit callers) ->
#    returns that result unconditionally.
only = [_r(0.002, 0.005, converged=True, tag=0)]
picked = _pick_best_seed_result(only)
assert picked.qpos[0] == 0

print('All executor seed-selection checks passed.')
