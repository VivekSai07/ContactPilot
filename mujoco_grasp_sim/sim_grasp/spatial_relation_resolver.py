"""Turns one instruction Step's spatial relation into a smaller,
differently-positioned OccupancyPlacementPlanner (or None for no bias).

OccupancyPlacementPlanner only accepts a scalar bin_inner_half (a square
region) -- it cannot represent a true half-bin rectangle. Every relation
therefore resolves to a smaller SQUARE sub-region of half-size
bin_inner_half / 2, differing only in where that square's center sits.
This keeps placement_planner.py completely untouched -- see
docs/superpowers/specs/2026-08-21-reasoning-layer-phase1-design.md for
the full rationale and geometry.
"""
import numpy as np

from sim_grasp.frames import transform_points
from sim_grasp.placement_planner import OccupancyPlacementPlanner
from sim_grasp.pointcloud import depth_to_pointcloud


def _camera_view_axis(T_world_cam: np.ndarray) -> 'tuple[str, float]':
    """World axis ('x' or 'y') most aligned with the camera's local +X
    (image-right, OpenCV convention -- see frames.py) direction, and the
    sign of that alignment (+1 if image-right points toward +axis)."""
    cam_right_world = T_world_cam[:3, 0]
    x_comp, y_comp = float(cam_right_world[0]), float(cam_right_world[1])
    if abs(x_comp) >= abs(y_comp):
        return 'x', (1.0 if x_comp >= 0 else -1.0)
    return 'y', (1.0 if y_comp >= 0 else -1.0)


def resolve(place_relation: str, place_reference: 'str | None',
           bin_center: 'tuple[float, float]', bin_inner_half: float,
           T_world_cam: np.ndarray,
           rgb: 'np.ndarray | None' = None,
           depth: 'np.ndarray | None' = None,
           K: 'np.ndarray | None' = None,
           prompt_selector=None,
           work_dir='.') -> 'OccupancyPlacementPlanner | None':
    if place_relation == 'none':
        return None

    half = bin_inner_half / 2.0
    bx, by = bin_center

    if place_relation == 'center':
        return OccupancyPlacementPlanner(bin_center=(bx, by), bin_inner_half=half)

    if place_relation in ('left_of', 'right_of'):
        axis, sign = _camera_view_axis(T_world_cam)
        direction = sign if place_relation == 'right_of' else -sign
        if axis == 'x':
            center = (bx + direction * half, by)
        else:
            center = (bx, by + direction * half)
        return OccupancyPlacementPlanner(bin_center=center, bin_inner_half=half)

    if place_relation == 'near':
        if prompt_selector is None or rgb is None or depth is None or K is None:
            print(f'[spatial-relation] "near {place_reference}" needs rgb/depth/K/'
                 'a PromptSelector -- falling back to unbiased placement')
            return None
        result = prompt_selector.select(rgb, prompt=place_reference, work_dir=work_dir)
        if result.is_empty:
            print(f'[spatial-relation] no match for "near {place_reference}" -- '
                 'falling back to unbiased placement')
            return None
        # Only consider matches whose centroid is actually inside the bin --
        # otherwise an identical-looking object still on the table could be
        # mistaken for the already-placed reference object.
        in_bin_candidates = []
        for i in range(len(result.scores)):
            pts_cam = depth_to_pointcloud(depth, K, mask=result.masks[i])
            if len(pts_cam) == 0:
                continue
            pts_world = transform_points(T_world_cam, pts_cam)
            ccx, ccy = float(pts_world[:, 0].mean()), float(pts_world[:, 1].mean())
            if np.hypot(ccx - bx, ccy - by) <= bin_inner_half:
                in_bin_candidates.append((float(result.scores[i]), ccx, ccy))
        if not in_bin_candidates:
            print(f'[spatial-relation] no in-bin match for "near {place_reference}" '
                 '-- falling back to unbiased placement')
            return None
        _, cx, cy = max(in_bin_candidates, key=lambda c: c[0])
        # Clamp so the sub-region stays fully inside the full bin's own bounds
        # -- unlike left_of/right_of/center, an arbitrary detected centroid
        # near an edge needs this explicit clamp (see design spec).
        cx = float(np.clip(cx, bx - bin_inner_half + half, bx + bin_inner_half - half))
        cy = float(np.clip(cy, by - bin_inner_half + half, by + bin_inner_half - half))
        return OccupancyPlacementPlanner(bin_center=(cx, cy), bin_inner_half=half)

    raise ValueError(f'unknown place_relation: {place_relation!r}')
