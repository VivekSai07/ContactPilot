"""Vision-only intelligent bin placement.

Computes WHERE (x, y, yaw) and how deep (release_z) to release a held
object in the bin, using only depth/segmap/intrinsics/extrinsics -- no
MuJoCo internal-state queries -- so the same code ports to a real
RealSense camera later. Mirrors the sim_grasp.grasp_predictor.GraspPredictor
ABC pattern used for pluggable grasp backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from sim_grasp.frames import transform_points
from sim_grasp.pointcloud import depth_to_pointcloud

MIN_FOOTPRINT_POINTS = 10


@dataclass
class ObjectFootprint:
    """World-frame footprint of an object, measured from its segmap mask
    while it sits on the table (before it is picked)."""
    center_xy: tuple      # world (x, y) center of the XY point cloud extent
    size_xy: tuple        # world (width, height) of the min-area XY rect
    yaw: float            # radians; orientation of size_xy[0]'s axis
    z_bottom: float        # world Z of the lowest point (resting surface)
    z_top: float           # world Z of the highest point


def compute_object_footprint(depth: np.ndarray, segmap: np.ndarray,
                              seg_id: int, K: np.ndarray,
                              T_world_cam: np.ndarray) -> 'ObjectFootprint | None':
    """Returns None if the object's mask yields too few valid depth points
    (degenerate/empty mask) rather than raising."""
    pts_cam = depth_to_pointcloud(depth, K, mask=(segmap == seg_id))
    if len(pts_cam) < MIN_FOOTPRINT_POINTS:
        return None
    pts_world = transform_points(T_world_cam, pts_cam)

    xy = pts_world[:, :2].astype(np.float32)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(xy)

    return ObjectFootprint(
        center_xy=(float(cx), float(cy)),
        size_xy=(float(w), float(h)),
        yaw=float(np.deg2rad(angle_deg)),
        z_bottom=float(pts_world[:, 2].min()),
        z_top=float(pts_world[:, 2].max()),
    )


@dataclass
class BinHeightmap:
    """Top-down occupancy grid of the bin interior. `heights[row, col]` is
    the highest observed world Z in that cell; `floor_z` is the
    self-calibrated 'empty bin' reference height."""
    heights: np.ndarray          # (rows, cols) float32
    origin_xy: tuple             # world (x, y) of cell [0, 0]'s lower-left corner
    cell_size: float
    floor_z: float


def build_bin_heightmap(depth: np.ndarray, segmap: np.ndarray, K: np.ndarray,
                         T_world_cam: np.ndarray, bin_center: tuple,
                         bin_inner_half: float, exclude_seg_id: 'int | None' = None,
                         cell_size: float = 0.005) -> BinHeightmap:
    mask = np.ones_like(segmap, dtype=bool)
    if exclude_seg_id is not None:
        mask &= (segmap != exclude_seg_id)

    pts_cam = depth_to_pointcloud(depth, K, mask=mask)
    pts_world = transform_points(T_world_cam, pts_cam) if len(pts_cam) else pts_cam

    bx, by = bin_center
    n = int(np.ceil(2 * bin_inner_half / cell_size))
    origin_x, origin_y = bx - bin_inner_half, by - bin_inner_half
    # Initialize with -inf to make maximum.at work correctly; cells that never
    # get a point will stay -inf and become NaN in the result.
    heights = np.full((n, n), -np.inf, dtype=np.float32)

    if len(pts_world):
        in_bin = ((np.abs(pts_world[:, 0] - bx) < bin_inner_half) &
                   (np.abs(pts_world[:, 1] - by) < bin_inner_half))
        pw = pts_world[in_bin]
        if len(pw):
            cols = np.clip(((pw[:, 0] - origin_x) / cell_size).astype(int), 0, n - 1)
            rows = np.clip(((pw[:, 1] - origin_y) / cell_size).astype(int), 0, n - 1)
            np.maximum.at(heights, (rows, cols), pw[:, 2])

    valid = np.isfinite(heights)
    floor_z = float(np.percentile(heights[valid], 10)) if np.any(valid) else 0.0
    heights = np.where(valid, heights, floor_z).astype(np.float32)

    return BinHeightmap(heights=heights, origin_xy=(origin_x, origin_y),
                        cell_size=cell_size, floor_z=floor_z)
