"""3D voxel occupancy of everything currently on the table, and a
neighbor-object collision check against it -- extends
sim_grasp.feasibility's cheap-geometric-pre-filter philosophy (table
collision, approach angle) to real neighbor geometry, using a full 3D
voxel grid rather than a 2.5D top-down heightmap (placement_planner.py's
BinHeightmap) since a gripper can reach under part of a tall neighbor,
which a top-down height value alone can't represent.

Not motion planning -- a cheap, denser-sampled geometric pre-filter, same
spirit as feasibility.py.
"""
from dataclasses import dataclass

import numpy as np

from sim_grasp.feasibility import _hand_boxes
from sim_grasp.frames import transform_points
from sim_grasp.pointcloud import depth_to_pointcloud


@dataclass
class WorkspaceOccupancy:
    voxel_size: float
    voxel_to_segids: dict   # {(int, int, int): frozenset[int]}


def _densified_box_points(x0, x1, y0, y1, z0, z1) -> np.ndarray:
    """26 points per box: 8 corners + 6 face-centers + 12 edge-midpoints --
    denser than feasibility.py's corner-only _box_corners(), for better
    coverage against thin/small neighbor objects a corner-only check
    could miss entirely."""
    xs, ys, zs = (x0, x1), (y0, y1), (z0, z1)
    xm, ym, zm = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    pts = [(x, y, z) for x in xs for y in ys for z in zs]           # 8 corners
    pts += [(x, ym, zm) for x in xs]                                 # 2 face centers
    pts += [(xm, y, zm) for y in ys]                                 # 2 face centers
    pts += [(xm, ym, z) for z in zs]                                 # 2 face centers
    pts += [(x, y, zm) for x in xs for y in ys]                      # 4 edge midpoints
    pts += [(x, ym, z) for x in xs for z in zs]                      # 4 edge midpoints
    pts += [(xm, y, z) for y in ys for z in zs]                      # 4 edge midpoints
    return np.array(pts)


def _densified_gripper_sample_points(opening: float = 0.08) -> np.ndarray:
    return np.vstack([_densified_box_points(*box) for box in _hand_boxes(opening)])


def _voxelize(pts_world: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor(pts_world / voxel_size).astype(np.int64)


def build_workspace_occupancy(depth: np.ndarray, segmap: np.ndarray,
                              K: np.ndarray, T_world_cam: np.ndarray,
                              table_height: float,
                              voxel_size: float = 0.005) -> WorkspaceOccupancy:
    """Voxelizes every on-table object's own points (segmap > 0), one
    object at a time so each voxel can be attributed to the seg_id(s) that
    put a point there -- letting collides_with_neighbors() exclude a
    grasp's own target object without needing a separate occupancy per
    object."""
    voxel_to_segids: dict = {}
    for seg_id in sorted(int(s) for s in np.unique(segmap) if s > 0):
        pts_cam = depth_to_pointcloud(depth, K, mask=(segmap == seg_id))
        if len(pts_cam) == 0:
            continue
        pts_world = transform_points(T_world_cam, pts_cam)
        pts_world = pts_world[pts_world[:, 2] > table_height + 0.005]
        if len(pts_world) == 0:
            continue
        for row in _voxelize(pts_world, voxel_size):
            key = (int(row[0]), int(row[1]), int(row[2]))
            voxel_to_segids.setdefault(key, set()).add(seg_id)
    frozen = {k: frozenset(v) for k, v in voxel_to_segids.items()}
    return WorkspaceOccupancy(voxel_size=voxel_size, voxel_to_segids=frozen)


def collides_with_neighbors(T_world_grasp: np.ndarray, opening: float,
                            exclude_seg_id: int,
                            occupancy: WorkspaceOccupancy) -> bool:
    """True if any densified gripper sample point, at this candidate grasp
    pose, falls in a voxel occupied by a seg_id other than exclude_seg_id
    (the object currently being picked -- its own points are never a
    'collision' with itself)."""
    pts_grasp = _densified_gripper_sample_points(opening)
    pts_world = transform_points(T_world_grasp, pts_grasp)
    for row in _voxelize(pts_world, occupancy.voxel_size):
        key = (int(row[0]), int(row[1]), int(row[2]))
        occupants = occupancy.voxel_to_segids.get(key)
        if occupants and (occupants - {exclude_seg_id}):
            return True
    return False
