"""Nearest-named-color matching for scene objects.

Scene objects get uniform-random RGB in [0.15, 0.95]^3 (see
scene_generator.py's _rand_rgba) — not perceptually distributed. This maps
an arbitrary RGB to the closest name in a small curated table, used only to
build ground-truth prompts for the promptable-selection benchmark. It is
never part of the runtime selection pipeline.
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
