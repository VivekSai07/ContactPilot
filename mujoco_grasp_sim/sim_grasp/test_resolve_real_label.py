"""Standalone check for prompt_selector.resolve_real_label — run directly,
no pytest (this codebase has no automated test suite)."""
import numpy as np
from sim_grasp.prompt_selector import resolve_real_label

# Mask entirely inside a single real object's region
gt = np.zeros((10, 10), dtype=np.int32)
gt[2:5, 2:5] = 3
mask = np.zeros((10, 10), dtype=bool)
mask[3:4, 3:4] = True
assert resolve_real_label(gt, mask) == 3, 'single-object overlap failed'

# Mask straddling two objects — majority (by pixel count) wins
gt2 = np.zeros((10, 10), dtype=np.int32)
gt2[0:5, :] = 1
gt2[5:10, :] = 2
mask2 = np.zeros((10, 10), dtype=bool)
mask2[3:5, :] = True   # rows 3,4 -> all label 1 (20 px)
mask2[5, :] = True     # row 5 -> label 2 (10 px, minority)
assert resolve_real_label(gt2, mask2) == 1, 'majority-overlap failed'

# Mask entirely over background (no real object underneath)
gt3 = np.zeros((10, 10), dtype=np.int32)
mask3 = np.zeros((10, 10), dtype=bool)
mask3[0:2, 0:2] = True
assert resolve_real_label(gt3, mask3) is None, 'background-only case failed'

print('All resolve_real_label checks passed.')
