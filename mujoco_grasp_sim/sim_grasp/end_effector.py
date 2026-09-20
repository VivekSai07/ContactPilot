"""EndEffectorController -- pluggable gripper/hand control, mirroring the
existing GraspPredictor/PlacementPlanner ABC pattern in this codebase.
Both implementations are driven by the SAME 0(open)-255(closed) scalar
GraspExecutor already threads through _step_to/_hold/execute/place, so the
pick/place state machine itself never changes when swapping end effectors.
"""
from abc import ABC, abstractmethod

import numpy as np

from sim_grasp.shadow_hand_posture import (
    SHADOW_HAND_OPEN_POSTURE, SHADOW_HAND_POWER_GRASP_POSTURE)

SUCCESS_RAISE = 0.08   # matches executor.py's own SUCCESS_RAISE constant


class EndEffectorController(ABC):
    """Implement this to plug in a different gripper/hand."""

    n_actuators: int

    @abstractmethod
    def ctrl_for(self, openness: float) -> np.ndarray:
        """openness: 0 (fully closed) .. 255 (fully open), matching this
        project's existing GRIPPER_OPEN/GRIPPER_CLOSED convention.
        Returns the control vector for this end effector's own actuators."""

    def is_grasping(self, object_raised_m: float, finger_opening_m: float = None) -> bool:
        """Default success check: did the target object actually rise
        during lift? Shared by both controllers for v1 -- a hand-specific
        contact/force check is explicitly deferred (see the design spec).
        `finger_opening_m` is ignored here; only ParallelGripperController
        consumes it (see override below)."""
        return object_raised_m > SUCCESS_RAISE


class ParallelGripperController(EndEffectorController):
    """Wraps today's exact Panda 2-finger tendon-gripper behavior
    verbatim -- data.ctrl[7] = openness, nothing else changes."""
    n_actuators = 1

    def ctrl_for(self, openness: float) -> np.ndarray:
        return np.array([openness], dtype=float)

    def is_grasping(self, object_raised_m: float, finger_opening_m: float = None) -> bool:
        """Restores this project's original parallel-gripper success check
        (object rose AND the fingers didn't fully collapse, i.e. something
        is actually between them) -- lost when is_grasping was generalized
        to the height-only base check for the Shadow Hand's fixed posture."""
        return object_raised_m > SUCCESS_RAISE and finger_opening_m is not None and finger_opening_m > 0.001


# Actuator order must match hand_assets.py's merged model exactly (indices
# 7-24): rh_A_THJ5, THJ4, THJ3, THJ2, THJ1, FFJ4, FFJ3, FFJ0, MFJ4, MFJ3,
# MFJ0, RFJ4, RFJ3, RFJ0, LFJ5, LFJ4, LFJ3, LFJ0.
_SHADOW_HAND_ACTUATOR_ORDER = [
    'rh_A_THJ5', 'rh_A_THJ4', 'rh_A_THJ3', 'rh_A_THJ2', 'rh_A_THJ1',
    'rh_A_FFJ4', 'rh_A_FFJ3', 'rh_A_FFJ0',
    'rh_A_MFJ4', 'rh_A_MFJ3', 'rh_A_MFJ0',
    'rh_A_RFJ4', 'rh_A_RFJ3', 'rh_A_RFJ0',
    'rh_A_LFJ5', 'rh_A_LFJ4', 'rh_A_LFJ3', 'rh_A_LFJ0',
]


class ShadowHandController(EndEffectorController):
    """Interpolates the Shadow Hand's 18 actuators between a fixed open
    posture and a fixed DexGraspNet-derived power-grasp posture."""
    n_actuators = 18

    def __init__(self):
        self._open = np.array(
            [SHADOW_HAND_OPEN_POSTURE[a] for a in _SHADOW_HAND_ACTUATOR_ORDER])
        self._closed = np.array(
            [SHADOW_HAND_POWER_GRASP_POSTURE[a] for a in _SHADOW_HAND_ACTUATOR_ORDER])

    def ctrl_for(self, openness: float) -> np.ndarray:
        a = 1.0 - (openness / 255.0)   # 0 at fully open, 1 at fully closed
        return (1 - a) * self._open + a * self._closed
