"""Standalone check for prompt_selector.filter_selection_by_click — run
directly, no pytest (this codebase has no automated test suite)."""
import numpy as np
from sim_grasp.prompt_selector import SelectionResult, filter_selection_by_click

# Two candidates, only the second contains the click pixel (5,5) — keep only it
m0 = np.zeros((10, 10), dtype=bool)
m0[0:2, 0:2] = True
m1 = np.zeros((10, 10), dtype=bool)
m1[4:7, 4:7] = True  # contains (row=5, col=5)
result = SelectionResult(
    masks=np.stack([m0, m1]),
    scores=np.array([0.9, 0.5], dtype=np.float32),
    boxes=np.zeros((2, 4), dtype=np.float32),
)
kept = filter_selection_by_click(result, click=(5.0, 5.0))
assert len(kept.scores) == 1, 'expected exactly one candidate kept'
assert kept.scores[0] == 0.5, 'kept the wrong candidate'
assert bool(kept.masks[0][5, 5]), 'kept mask must contain the click pixel'

# Both candidates overlap the click pixel — keep both, order preserved
m2 = np.zeros((10, 10), dtype=bool)
m2[3:8, 3:8] = True  # also contains (5,5)
result2 = SelectionResult(
    masks=np.stack([m1, m2]),
    scores=np.array([0.7, 0.6], dtype=np.float32),
    boxes=np.zeros((2, 4), dtype=np.float32),
)
kept2 = filter_selection_by_click(result2, click=(5.0, 5.0))
assert len(kept2.scores) == 2, 'expected both overlapping candidates kept'
assert np.allclose(kept2.scores, [0.7, 0.6]), 'order must be preserved'

# No candidate contains the click pixel — result is empty
result3 = SelectionResult(
    masks=np.stack([m0]),
    scores=np.array([0.9], dtype=np.float32),
    boxes=np.zeros((1, 4), dtype=np.float32),
)
kept3 = filter_selection_by_click(result3, click=(5.0, 5.0))
assert kept3.is_empty, 'expected empty result when no mask contains the click'

# Already-empty input is returned as empty (no crash on zero candidates)
empty_in = SelectionResult(
    masks=np.zeros((0, 10, 10), dtype=bool),
    scores=np.zeros((0,), dtype=np.float32),
    boxes=np.zeros((0, 4), dtype=np.float32),
)
kept4 = filter_selection_by_click(empty_in, click=(5.0, 5.0))
assert kept4.is_empty, 'expected empty-in to stay empty'

print('All filter_selection_by_click checks passed.')
