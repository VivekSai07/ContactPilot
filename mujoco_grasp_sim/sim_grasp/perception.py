"""Perception cleanup + grasp refinement (P1 of ROADMAP.md).

Image-space depth filters applied BEFORE grasp prediction, and a cloud-based
grasp re-centering step applied AFTER it. Image-space filtering keeps the
integration trivial: callers clean the depth map and hand it to the existing
predictor / cgn_worker unchanged.

Note for sim: MuJoCo depth is noise-free, so remove_depth_speckles() is a
no-op there — it exists for the real RealSense, where flying pixels at object
boundaries are a major source of offset grasps. workspace_crop() and
recenter_grasp() help in BOTH sim and lab: the dominant sim failure mode
(taxonomy: closed_on_air) is CGN grasps laterally offset from the object so
the fingers close on air; re-centering along the closing axis fixes exactly
that.
"""

import numpy as np

# Default workspace AABB in WORLD frame (meters): the tabletop slab.
# SceneConfig: tabletop at z=0.75, table spans x in [0.20, 1.00] (center 0.60,
# half-size 0.40), y in [-0.45, 0.45]. Keep from just under the tabletop to
# 0.5 m above it so the table plane itself stays in CGN's scene cloud.
DEFAULT_WORKSPACE = {'x': (0.18, 1.02), 'y': (-0.47, 0.47), 'z': (0.70, 1.30)}


def workspace_crop(depth: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray,
                   bounds: dict | None = None) -> np.ndarray:
    """Zero out depth pixels whose 3D world point lies outside the workspace
    AABB. Removes floor, walls, robot base and far clutter so CGN's scene
    cloud only contains the tabletop it should reason about.

    :returns: cleaned copy of depth (0 = invalid, as everywhere else)
    """
    b = {**DEFAULT_WORKSPACE, **(bounds or {})}
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    valid = z > 0
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x, y, z], axis=-1)                      # (H,W,3)
    pts_w = pts_cam @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
    inside = valid
    for ax, (lo, hi) in zip(range(3), (b['x'], b['y'], b['z'])):
        inside = inside & (pts_w[..., ax] >= lo) & (pts_w[..., ax] <= hi)
    out = depth.copy()
    out[~inside] = 0.0
    return out


def remove_depth_speckles(depth: np.ndarray, max_dev: float = 0.02,
                          win: int = 5) -> np.ndarray:
    """Drop pixels deviating more than `max_dev` m from their local median —
    kills flying pixels / boundary mixing on real depth sensors. No-op on
    clean synthetic depth."""
    from scipy.ndimage import median_filter
    med = median_filter(depth, size=win)
    out = depth.copy()
    out[(depth > 0) & (np.abs(depth - med) > max_dev)] = 0.0
    return out


def clean_depth(depth, K, T_world_cam, bounds=None, speckle_dev=0.02):
    """workspace_crop + speckle removal in one call."""
    return remove_depth_speckles(
        workspace_crop(depth, K, T_world_cam, bounds), max_dev=speckle_dev)


def recenter_grasp(T_world_grasp: np.ndarray, obj_pts_world: np.ndarray,
                   max_shift: float = 0.025, max_advance: float = 0.015,
                   table_z: float | None = None) -> tuple[np.ndarray, float]:
    """Re-center a grasp on the target object's cloud in TWO ways:

    1. shift along the finger-closing axis (+X of the grasp frame) so the
       fingers close centered on the object instead of beside it;
    2. shift across the fingers (+Y) so the closing plane actually intersects
       the object — measured on seed 5: a cylinder offset 12.7 mm in Y, the
       12 mm-wide fingers swept past it on three consecutive attempts;
    3. advance along the approach axis (+Z) so the object centroid sits
       mid-finger instead of at the fingertips — measured failure mode on
       seed 5: CGN contact at pz=0.103 with the finger sweep ending at 0.112,
       i.e. half the object beyond the fingertips, fingers pinch air.

    Points are measured in a capture box slightly larger than the finger
    sweep (feasibility.py hand model: fingers at pz in [0.066, 0.112],
    half-opening <= 0.04). Both corrections are clamped; the advance is
    additionally limited so the fingertips stay >= 3 mm above `table_z`.

    :param obj_pts_world: (N,3) points of the TARGET object in world frame
    :returns: (corrected T_world_grasp, total applied shift in m).
              Unchanged pose if too few points fall in the capture box.
    """
    g = np.asarray(T_world_grasp, dtype=float)
    rel = np.asarray(obj_pts_world, dtype=float) - g[:3, 3]
    px = rel @ g[:3, 0]          # along closing line (fingers travel here)
    py = rel @ g[:3, 1]          # across fingers
    pz = rel @ g[:3, 2]          # along approach axis
    in_box = (np.abs(px) < 0.052) & (np.abs(py) < 0.03) & \
             (pz > 0.040) & (pz < 0.160)
    if in_box.sum() < 30:
        return g, 0.0

    # Center estimate: midpoint of the 5-95% silhouette extent, NOT the point
    # median — the camera only sees the front surface, so the median is
    # biased toward the camera by ~an object radius (measured: benchmark
    # shifts averaged 22 mm on successes and failures alike).
    def mid(coord):
        lo, hi = np.percentile(coord[in_box], [5, 95])
        return (lo + hi) / 2

    dx = float(np.clip(mid(px), -max_shift, max_shift))
    dy = float(np.clip(mid(py), -max_shift, max_shift))
    # target: object center at mid-finger (pz ~ 0.089); only ever advance —
    # retreating would re-create the fingertip-pinch this fixes
    dz = float(np.clip(np.median(pz[in_box]) - 0.089, 0.0, max_advance))
    if table_z is not None and g[2, 2] < 0:     # approach pointing down
        tip_z = g[2, 3] + 0.112 * g[2, 2]       # current fingertip height
        dz = min(dz, max(0.0, (tip_z - table_z - 0.003) / -g[2, 2]))

    out = g.copy()
    out[:3, 3] += dx * g[:3, 0] + dy * g[:3, 1] + dz * g[:3, 2]
    return out, float(np.linalg.norm([dx, dy, dz]))
