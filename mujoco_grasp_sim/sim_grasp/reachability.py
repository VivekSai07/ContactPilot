"""Cheap analytical workspace-reachability pre-filter -- position (radial
distance from the base) plus azimuth (does this direction even fall
within joint1's rotation range), no IK. A full 7-DOF orientation-
reachability check has no cheap closed form; this is deliberately
narrower in scope (see
docs/superpowers/specs/2026-08-29-neighbor-collision-reachability-prefilter-design.md).
Not motion planning -- a cheap geometric pre-filter, same spirit as
feasibility.py/workspace_occupancy.py.
"""
import numpy as np

# Real Panda robot spec (radians) -- confirmed against
# mujoco_menagerie/franka_emika_panda/panda.xml's <default class="panda">
# joint range (joint1 has no per-joint override, so it uses this class
# default verbatim). A physical robot fact, not a MuJoCo-model query, so
# this module stays model-independent like the rest of the vision-only
# pipeline.
PANDA_JOINT1_RANGE = (-2.8973, 2.8973)

# Meters. Already an established constant elsewhere in this project
# (scene_generator.py's spawn-region comment: "within arm reach (0.54 m
# from the base)").
MAX_REACH = 0.54

# Meters. An APPROXIMATE estimate of the "too close to fold the elbow"
# dead zone near the base -- unlike MAX_REACH/PANDA_JOINT1_RANGE above,
# this is not a hard physical constant; treat it as benchmark-tunable if
# it proves too conservative or not conservative enough in practice.
MIN_REACH = 0.15

# Radians. Safety margin narrowing PANDA_JOINT1_RANGE on each side, since
# a target right at the joint's hard limit leaves no room for the other
# 6 joints to also converge.
AZIMUTH_MARGIN = 0.1


def is_reachable(T_world_grasp: np.ndarray, base_xy: tuple = (0.0, 0.0)) -> bool:
    dx = T_world_grasp[0, 3] - base_xy[0]
    dy = T_world_grasp[1, 3] - base_xy[1]
    dist = float(np.hypot(dx, dy))
    if not (MIN_REACH <= dist <= MAX_REACH):
        return False
    azimuth = float(np.arctan2(dy, dx))
    lo, hi = PANDA_JOINT1_RANGE
    return (lo + AZIMUTH_MARGIN) <= azimuth <= (hi - AZIMUTH_MARGIN)
