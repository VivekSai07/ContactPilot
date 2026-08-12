"""Point cloud reconstruction from depth + pinhole intrinsics.

All point clouds produced here are in the OPENCV camera frame
(+Z forward, +X right, +Y down — see frames.py). Use
frames.transform_points(T_world_cam, pc) to move them to the world frame.
"""

import numpy as np


def depth_to_pointcloud(depth: np.ndarray, K: np.ndarray,
                        rgb: np.ndarray | None = None,
                        mask: np.ndarray | None = None,
                        z_range: tuple = (0.05, 4.0)):
    """Back-project a metric depth image to an (N,3) point cloud.

    Pinhole model (OpenCV camera frame):
        z = depth[v, u]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

    :param depth: (H,W) float32 depth in METERS (0 = no return)
    :param K: (3,3) intrinsics
    :param rgb: optional (H,W,3) uint8 — returns per-point colors as well
    :param mask: optional (H,W) bool — restrict to these pixels
    :param z_range: keep points with z in (min, max)
    :returns: pc (N,3) float32 [, colors (N,3) float32 in 0..1]
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid = (depth > z_range[0]) & (depth < z_range[1])
    if mask is not None:
        valid &= mask.astype(bool)

    v, u = np.nonzero(valid)
    z = depth[v, u]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pc = np.stack([x, y, z], axis=1).astype(np.float32)

    if rgb is not None:
        colors = rgb[v, u].astype(np.float32) / 255.0
        return pc, colors
    return pc
