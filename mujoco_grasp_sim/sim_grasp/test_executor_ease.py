"""Standalone check for executor._ease -- run directly, no pytest (this
codebase has no automated test suite)."""
import numpy as np

from sim_grasp.executor import _ease

# Linear (smooth=False) is unchanged: identity function.
for t in (0.0, 0.25, 0.5, 0.75, 1.0):
    assert _ease(t, smooth=False) == t, f'_ease({t}, False) should equal {t}'

# Smoothstep (smooth=True): passes through the same endpoints...
assert _ease(0.0, smooth=True) == 0.0
assert _ease(1.0, smooth=True) == 1.0
# ...is symmetric around the midpoint...
assert abs(_ease(0.5, smooth=True) - 0.5) < 1e-9
# ...and has ZERO slope at both ends (the actual fix for the slip bug --
# estimate the derivative numerically at t=0 and t=1, it must be ~0, unlike
# linear interpolation's constant slope of 1.0 everywhere).
eps = 1e-4
slope_at_start = (_ease(eps, smooth=True) - _ease(0.0, smooth=True)) / eps
slope_at_end = (_ease(1.0, smooth=True) - _ease(1.0 - eps, smooth=True)) / eps
assert slope_at_start < 0.01, f'slope at t=0 should be ~0, got {slope_at_start}'
assert slope_at_end < 0.01, f'slope at t=1 should be ~0, got {slope_at_end}'

# Monotonically increasing (no overshoot/oscillation) across the whole range.
ts = np.linspace(0, 1, 50)
values = [_ease(float(t), smooth=True) for t in ts]
assert all(b >= a for a, b in zip(values, values[1:])), \
    'smoothstep must be monotonically increasing'

print('All executor _ease checks passed.')
