"""Standalone check for color_utils.rgb_to_color_name — run directly, no
pytest (this codebase has no automated test suite)."""
from sim_grasp.color_utils import rgb_to_color_name

# Clear, unambiguous cases
assert rgb_to_color_name((0.9, 0.15, 0.15)) == 'red', 'pure red misclassified'
assert rgb_to_color_name((0.15, 0.8, 0.15)) == 'green', 'pure green misclassified'
assert rgb_to_color_name((0.15, 0.15, 0.9)) == 'blue', 'pure blue misclassified'
assert rgb_to_color_name((0.9, 0.9, 0.15)) == 'yellow', 'pure yellow misclassified'

# Returns a string for any input in the valid [0,1]^3 range, never crashes
import numpy as np
rng = np.random.default_rng(0)
for _ in range(20):
    rgb = rng.uniform(0.15, 0.95, size=3)
    name = rgb_to_color_name(rgb)
    assert isinstance(name, str) and len(name) > 0

print('All color_utils checks passed.')
