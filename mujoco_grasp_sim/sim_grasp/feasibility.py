"""Grasp feasibility filtering — rejects grasps that would obviously collide
with the table. NOT motion planning; a cheap geometric pre-filter.

Method
------
The Panda hand is approximated by three boxes expressed in the GRASP frame
(+Z approach, +X closing line, origin at gripper base — see frames.py):

    palm   : the hand body,   x: ±0.102, y: ±0.0315, z: [-0.01, 0.066]
    finger : two finger boxes, x: ±(open/2 + 0.012),  z: [0.066, 0.112]

For each candidate grasp we transform the box corner points into the WORLD
frame and reject the grasp if any corner dips below the tabletop plane
(z < table_height + margin). We additionally reject "underhand" grasps whose
approach direction points upward in the world frame — those are unreachable
in a tabletop eye-to-hand setup and indicate a spurious detection.
"""

import numpy as np

from sim_grasp.frames import transform_points


def _box_corners(x0, x1, y0, y1, z0, z1):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])


def _gripper_sample_points(opening: float = 0.08) -> np.ndarray:
    """Corner samples of the simplified Panda hand in the grasp frame."""
    half_open = opening / 2.0
    palm = _box_corners(-0.102, 0.102, -0.0315, 0.0315, -0.012, 0.066)
    finger_l = _box_corners(-half_open - 0.012, -half_open + 0.012, -0.012, 0.012, 0.066, 0.112)
    finger_r = _box_corners(half_open - 0.012, half_open + 0.012, -0.012, 0.012, 0.066, 0.112)
    return np.vstack([palm, finger_l, finger_r])


class GraspFeasibilityChecker:
    def __init__(self, table_height: float, margin: float = 0.005,
                 reject_upward_approach: bool = True, max_up_z: float = 0.15):
        """
        :param table_height: world z of the tabletop plane (meters)
        :param margin: extra clearance above the table (meters)
        :param reject_upward_approach: drop grasps approaching from below
        :param max_up_z: max allowed world-z component of the approach axis
        """
        self.table_z = table_height + margin
        self.reject_upward = reject_upward_approach
        self.max_up_z = max_up_z

    def is_feasible(self, T_world_grasp: np.ndarray, opening: float = 0.08) -> bool:
        # 1) approach direction sanity (grasp +Z in world coordinates)
        if self.reject_upward and T_world_grasp[2, 2] > self.max_up_z:
            return False
        # 2) table collision: all gripper sample points must stay above the top
        pts_world = transform_points(T_world_grasp, _gripper_sample_points(opening))
        return bool(pts_world[:, 2].min() > self.table_z)

    def filter(self, grasps_world: dict, scores: dict,
               gripper_openings: dict | None = None):
        """Filter {seg_id: (N,4,4)} grasp dicts. Returns (grasps, scores, stats)."""
        out_g, out_s = {}, {}
        n_in = n_out = 0
        for seg_id, G in grasps_world.items():
            keep = []
            for i, T in enumerate(G):
                opening = 0.08
                if gripper_openings and seg_id in gripper_openings \
                        and len(gripper_openings[seg_id]) > i:
                    opening = float(gripper_openings[seg_id][i])
                if self.is_feasible(T, opening):
                    keep.append(i)
            n_in += len(G)
            n_out += len(keep)
            if keep:
                out_g[seg_id] = G[keep]
                out_s[seg_id] = np.asarray(scores[seg_id])[keep]
        stats = {'n_before': int(n_in), 'n_after': int(n_out),
                 'n_rejected': int(n_in - n_out)}
        return out_g, out_s, stats
