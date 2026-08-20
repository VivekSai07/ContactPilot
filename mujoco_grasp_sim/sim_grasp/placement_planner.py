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
