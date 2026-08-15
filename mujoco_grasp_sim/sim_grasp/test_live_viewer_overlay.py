"""Standalone check for live_viewer.compose_mask_overlay/draw_status_text —
run directly, no pytest (this codebase has no automated test suite), no
window needed."""
import numpy as np
from sim_grasp.live_viewer import compose_mask_overlay, draw_status_text

rgb = np.zeros((10, 10, 3), dtype=np.uint8)
mask = np.zeros((10, 10), dtype=bool)
mask[2:5, 2:5] = True

# Outside the mask: pixel unchanged
out = compose_mask_overlay(rgb, mask, color=(255, 0, 0), alpha=0.5)
assert tuple(out[0, 0]) == (0, 0, 0), 'unmasked pixel should be unchanged'

# Inside the mask: 50/50 blend of black background and red overlay color
assert tuple(out[3, 3]) == (127, 0, 0) or tuple(out[3, 3]) == (128, 0, 0), \
    f'masked pixel should be ~50% red, got {tuple(out[3, 3])}'

# alpha=0 leaves the image untouched everywhere
out0 = compose_mask_overlay(rgb, mask, color=(255, 0, 0), alpha=0.0)
assert np.array_equal(out0, rgb), 'alpha=0 should not change the image'

# alpha=1 fully replaces masked pixels with the overlay color
out1 = compose_mask_overlay(rgb, mask, color=(255, 0, 0), alpha=1.0)
assert tuple(out1[3, 3]) == (255, 0, 0), 'alpha=1 should fully replace masked pixels'
assert tuple(out1[0, 0]) == (0, 0, 0), 'alpha=1 should not touch unmasked pixels'

# draw_status_text changes some pixels (the text banner) without crashing,
# leaves the input array untouched, and preserves shape/dtype
big = np.zeros((60, 200, 3), dtype=np.uint8)
stamped = draw_status_text(big, 'Loading...')
assert stamped.shape == big.shape and stamped.dtype == big.dtype
assert not np.array_equal(stamped, big), 'status text should change some pixels'
assert np.array_equal(big, np.zeros((60, 200, 3), dtype=np.uint8)), \
    'draw_status_text must not mutate its input'

print('All compose_mask_overlay/draw_status_text checks passed.')
