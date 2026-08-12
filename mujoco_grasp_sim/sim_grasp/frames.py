"""Coordinate frame conventions used throughout this project.

============================================================================
FRAME DEFINITIONS
============================================================================

WORLD FRAME (MuJoCo)
    Z-up, right-handed. Origin on the floor at the center of the robot
    pedestal. The floor is the z=0 plane.

ROBOT BASE FRAME
    The Franka 'link0' frame. We mount link0 on a pedestal so that
        T_world_base = Trans(0, 0, TABLE_HEIGHT)
    i.e. the robot base frame is the world frame lifted to tabletop height.
    (Identical convention to a real Panda bolted to the workstation surface.)

CAMERA FRAME — MuJoCo convention (internal only)
    MuJoCo cameras look along **-Z**, with +X right and +Y up in the image.

CAMERA FRAME — OpenCV convention (used for EVERYTHING downstream)
    +Z forward (into the scene), +X right, +Y down in the image.
    Depth images, intrinsics K, point clouds and Contact-GraspNet grasps all
    live in this frame. Conversion from the MuJoCo camera frame is a fixed
    rotation:  R_cv = R_mj @ diag(1, -1, -1)   (flip Y and Z axes).

GRASP FRAME (Contact-GraspNet / Panda gripper convention)
    Origin at the gripper BASE (wrist flange side of the hand), NOT the TCP.
      +Z : approach direction (points from wrist toward the object)
      +X : finger closing line (the two fingertips lie on the X axis)
      +Y : completes the right-handed frame
    The TCP (point between the fingertips) sits at +0.1034 m along +Z.
    Grasps returned by Contact-GraspNet are T_cam_grasp (camera OpenCV frame).
    Robot execution therefore uses:
        T_base_grasp = inv(T_world_base) @ T_world_cam @ T_cam_grasp
============================================================================
"""

import numpy as np

# Fixed rotation MuJoCo-camera -> OpenCV-camera (flip Y and Z).
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])

# TCP offset along grasp +Z (Panda hand: base->point between fingertips).
PANDA_TCP_OFFSET = 0.1034


def mujoco_cam_to_cv_cam(cam_pos: np.ndarray, cam_xmat: np.ndarray) -> np.ndarray:
    """Build T_world_cam (4x4) for the *OpenCV* camera frame.

    :param cam_pos:  (3,)  camera position in world frame (mjData.cam_xpos)
    :param cam_xmat: (9,) or (3,3) world rotation of the MuJoCo camera frame
                     (mjData.cam_xmat, row-major)
    :returns: (4,4) T_world_camCV — maps OpenCV-camera-frame points to world.
    """
    R_mj = np.asarray(cam_xmat, dtype=np.float64).reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = R_mj @ _MJ_TO_CV
    T[:3, 3] = np.asarray(cam_pos, dtype=np.float64)
    return T


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Efficient inverse of a rigid transform (4x4)."""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an (N,3) point array."""
    return pts @ T[:3, :3].T + T[:3, 3]


def transform_grasps(T: np.ndarray, grasps: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an (N,4,4) array of grasp poses."""
    return np.einsum('ij,njk->nik', T, grasps)


def look_at_xyaxes(cam_pos, target, up=(0.0, 0.0, 1.0)):
    """Compute the MJCF `xyaxes` string for a camera at `cam_pos` looking at
    `target` (MuJoCo convention: camera -Z = viewing direction, +Y = image up).

    :returns: (x_axis, y_axis) world-frame unit vectors, and the xyaxes string.
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    f = target - cam_pos                      # viewing direction (world)
    f /= np.linalg.norm(f)
    z_mj = -f                                 # MuJoCo camera +Z points backward
    x_mj = np.cross(f, np.asarray(up, dtype=np.float64))  # image right
    x_mj /= np.linalg.norm(x_mj)
    y_mj = np.cross(z_mj, x_mj)               # image up (right-handed)
    xyaxes = ' '.join(f'{v:.6f}' for v in (*x_mj, *y_mj))
    return x_mj, y_mj, xyaxes
