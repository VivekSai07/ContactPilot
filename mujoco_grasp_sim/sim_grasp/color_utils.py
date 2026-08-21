"""Nearest-named-color matching for scene objects.

Scene objects get a fixed, clearly distinguishable color per spawn index
(see scene_generator.py's object_color()) rather than a random one -- so a
--instruction description like "the red cube" reliably refers to the same
object every run of the same seed. This module's `_NAMED_COLORS` table is
the single source of truth both for that fixed assignment and for
`rgb_to_color_name`, used to build ground-truth prompts for the
promptable-selection benchmark; it is never part of the runtime selection
pipeline.
"""
import numpy as np

_NAMED_COLORS = {
    'red': (0.85, 0.2, 0.2),
    'orange': (0.9, 0.5, 0.15),
    'yellow': (0.9, 0.9, 0.2),
    'green': (0.25, 0.7, 0.3),
    'cyan': (0.2, 0.8, 0.8),
    'blue': (0.2, 0.3, 0.85),
    'purple': (0.55, 0.25, 0.75),
    'pink': (0.9, 0.5, 0.75),
    'brown': (0.5, 0.35, 0.2),
    'gray': (0.55, 0.55, 0.55),
}

# Fixed per-object-index assignment order, most mutually-distinguishable
# first (red/green/blue) since the default scene spawns exactly 3 objects.
_FIXED_ORDER = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan',
               'orange', 'pink', 'brown', 'gray']


def object_color(index: int) -> 'tuple[str, str]':
    """Fixed (name, "r g b 1" rgba string) for the index-th spawned object
    in a scene -- deterministic across runs (not randomized), cycling
    through `_FIXED_ORDER` for scenes with more objects than colors. Uses
    the exact same reference RGB `rgb_to_color_name` matches against, so
    the assigned color is always named back correctly (distance 0)."""
    name = _FIXED_ORDER[index % len(_FIXED_ORDER)]
    r, g, b = _NAMED_COLORS[name]
    return name, f'{r:.3f} {g:.3f} {b:.3f} 1'


def rgb_to_color_name(rgb) -> str:
    """Nearest named color to `rgb` (any 3+-length sequence in [0,1]), by
    Euclidean distance in RGB space."""
    rgb = np.asarray(rgb, dtype=float)[:3]
    best_name, best_dist = None, float('inf')
    for name, ref in _NAMED_COLORS.items():
        d = float(np.linalg.norm(rgb - np.asarray(ref, dtype=float)))
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name
