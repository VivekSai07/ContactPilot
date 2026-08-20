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
    # Initialize with -inf (not NaN -- np.maximum propagates NaN, which would
    # silently discard every real height measurement). Cells that never get a
    # point stay -inf and are filled with the self-calibrated floor_z below.
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
    if not np.any(valid):
        raise ValueError(
            'build_bin_heightmap: no depth points landed in the bin region '
            '-- bin is out of view or fully occluded; callers should treat '
            'this the same as a degenerate compute_object_footprint() (fall '
            'back to the legacy fixed drop point).')
    floor_z = float(np.percentile(heights[valid], 10))
    heights = np.where(valid, heights, floor_z).astype(np.float32)

    return BinHeightmap(heights=heights, origin_xy=(origin_x, origin_y),
                        cell_size=cell_size, floor_z=floor_z)


_OCCUPIED_MARGIN = 0.008    # 8mm above floor counts as "occupied"
_CLEARANCE_MARGIN = 0.003   # 3mm extra around the footprint when checking cells
_SEARCH_STRIDE = 0.01       # 1cm stride between candidate slot centers
_CANDIDATE_YAW_OFFSETS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


@dataclass
class PlacementPose:
    x: float
    y: float
    yaw: float           # radians
    release_z: float     # world Z the object's BOTTOM should end up at
    fallback: bool = False   # True if no fully-clear slot existed (least-bad pick)


def _aabb_half_extent(size_xy: tuple, yaw: float) -> tuple:
    """Half-extents (world X, world Y) of `size_xy`'s rectangle rotated by
    `yaw`, i.e. its axis-aligned bounding box -- a deliberate simplification
    over exact oriented-rectangle collision (fine for mildly rectangular
    objects like these)."""
    w, h = size_xy
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    return (w / 2 * c + h / 2 * s, w / 2 * s + h / 2 * c)


class PlacementPlanner(ABC):
    """Implement this to plug in a different placement strategy."""

    @abstractmethod
    def plan(self, footprint: ObjectFootprint,
             heightmap: BinHeightmap) -> 'PlacementPose | None':
        """Returns a world (x, y, yaw, release_z) target, or None if the
        footprint cannot fit in the bin at any orientation."""


class OccupancyPlacementPlanner(PlacementPlanner):
    def __init__(self, bin_center: tuple, bin_inner_half: float,
                 occupied_margin: float = _OCCUPIED_MARGIN,
                 clearance_margin: float = _CLEARANCE_MARGIN,
                 search_stride: float = _SEARCH_STRIDE):
        self.bin_center = bin_center
        self.bin_inner_half = bin_inner_half
        self.occupied_margin = occupied_margin
        self.clearance_margin = clearance_margin
        self.search_stride = search_stride

    def plan(self, footprint: ObjectFootprint,
             heightmap: BinHeightmap) -> 'PlacementPose | None':
        bx, by = self.bin_center
        occupied_z = heightmap.floor_z + self.occupied_margin
        rows, cols = heightmap.heights.shape
        xs = heightmap.origin_xy[0] + (np.arange(cols) + 0.5) * heightmap.cell_size
        ys = heightmap.origin_xy[1] + (np.arange(rows) + 0.5) * heightmap.cell_size
        grid_x, grid_y = np.meshgrid(xs, ys)
        occupied_mask = heightmap.heights > occupied_z
        occ_x, occ_y = grid_x[occupied_mask], grid_y[occupied_mask]

        any_orientation_fits = False
        best = None  # (clearance, x, y, yaw)
        for yaw_offset in _CANDIDATE_YAW_OFFSETS:
            yaw = footprint.yaw + yaw_offset
            half_ex, half_ey = _aabb_half_extent(footprint.size_xy, yaw)
            half_ex += self.clearance_margin
            half_ey += self.clearance_margin
            x_min, x_max = bx - self.bin_inner_half + half_ex, bx + self.bin_inner_half - half_ex
            y_min, y_max = by - self.bin_inner_half + half_ey, by + self.bin_inner_half - half_ey
            if x_min > x_max or y_min > y_max:
                continue
            any_orientation_fits = True
            for cx in np.arange(x_min, x_max + 1e-9, self.search_stride):
                for cy in np.arange(y_min, y_max + 1e-9, self.search_stride):
                    in_box = ((np.abs(grid_x - cx) <= half_ex) &
                              (np.abs(grid_y - cy) <= half_ey))
                    if np.any(heightmap.heights[in_box] > occupied_z):
                        continue
                    clearance = (float(np.min(np.hypot(occ_x - cx, occ_y - cy)))
                                 if len(occ_x) else float('inf'))
                    if best is None or clearance > best[0]:
                        best = (clearance, cx, cy, yaw)

        if best is not None:
            _, x, y, yaw = best
            return PlacementPose(x=float(x), y=float(y), yaw=float(yaw),
                                 release_z=heightmap.floor_z)
        if not any_orientation_fits:
            return None   # the footprint cannot fit in this bin at all

        # Fallback: bin has no fully-clear spot -- least-bad option.
        half_ex, half_ey = _aabb_half_extent(footprint.size_xy, footprint.yaw)
        x_min, x_max = bx - self.bin_inner_half + half_ex, bx + self.bin_inner_half - half_ex
        y_min, y_max = by - self.bin_inner_half + half_ey, by + self.bin_inner_half - half_ey
        best_fallback = None
        for cx in np.arange(x_min, x_max + 1e-9, self.search_stride):
            for cy in np.arange(y_min, y_max + 1e-9, self.search_stride):
                in_box = ((np.abs(grid_x - cx) <= half_ex) &
                          (np.abs(grid_y - cy) <= half_ey))
                cell_heights = heightmap.heights[in_box]
                if len(cell_heights) == 0:
                    continue
                max_h = float(np.max(cell_heights))
                if best_fallback is None or max_h < best_fallback[0]:
                    best_fallback = (max_h, cx, cy)
        if best_fallback is None:
            return None
        max_h, x, y = best_fallback
        print('[placement] no fully-clear slot found in the bin -- using the '
              'least-occupied region as a fallback')
        return PlacementPose(x=float(x), y=float(y), yaw=footprint.yaw,
                             release_z=max_h, fallback=True)


def compute_release_z(pose: PlacementPose, T_world_grasp: np.ndarray,
                      footprint: ObjectFootprint,
                      safety_buffer: float = 0.005) -> float:
    """Hand-origin target Z for release: the chosen slot's floor height,
    plus how far below the gripper the object's bottom sat when grasped
    (measured, not assumed), plus a small safety buffer."""
    grasp_offset = float(T_world_grasp[2, 3]) - footprint.z_bottom
    return pose.release_z + grasp_offset + safety_buffer
