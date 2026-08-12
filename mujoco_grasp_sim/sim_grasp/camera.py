"""CameraModule — RGB / metric depth / segmentation rendering from MuJoCo.

DEPTH CORRECTNESS (read me)
---------------------------
We use the modern `mujoco.Renderer` API. With `enable_depth_rendering()`,
`render()` returns depth ALREADY LINEARIZED TO METERS by MuJoCo itself:
it converts the raw OpenGL z-buffer via

    z_metric = near / (1 - z_buf * (1 - near/far))

(see mujoco/renderer.py). The returned value is the perpendicular distance
along the camera viewing axis (true "pinhole z"), NOT the ray length —
exactly what the pinhole back-projection x=(u-cx)z/fx expects, and exactly
what a RealSense depth image contains. No further conversion is needed.
Pixels at the far plane come back as huge values; we zero them out so they
behave like RealSense "no return" pixels.

INTRINSICS
----------
MuJoCo cameras specify a vertical FOV (fovy). For a W x H image:
    fy = H / (2 * tan(fovy / 2)),  fx = fy   (square pixels)
    cx = W / 2,  cy = H / 2

EXTRINSICS
----------
mjData.cam_xpos / cam_xmat give the MuJoCo-convention camera pose in world.
`extrinsics()` returns T_world_cam for the OPENCV camera frame (+Z forward),
which is the frame all depth/point-cloud/grasp data lives in. See frames.py.
"""

import mujoco
import numpy as np

from sim_grasp.frames import mujoco_cam_to_cv_cam


class CameraModule:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 cam_name: str = 'ext_cam', width: int = 640, height: int = 480,
                 depth_clip: float = 4.0):
        self.model, self.data = model, data
        self.cam_name = cam_name
        self.width, self.height = width, height
        self.depth_clip = depth_clip
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if self.cam_id < 0:
            raise ValueError(f'Camera "{cam_name}" not found in model')
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    # -- intrinsics ----------------------------------------------------------
    def intrinsics(self) -> np.ndarray:
        """3x3 pinhole intrinsics K for the current resolution (OpenCV)."""
        fovy_rad = np.deg2rad(self.model.cam_fovy[self.cam_id])
        fy = self.height / (2.0 * np.tan(fovy_rad / 2.0))
        fx = fy  # MuJoCo renders square pixels
        return np.array([[fx, 0.0, self.width / 2.0],
                         [0.0, fy, self.height / 2.0],
                         [0.0, 0.0, 1.0]])

    # -- extrinsics ----------------------------------------------------------
    def extrinsics(self) -> np.ndarray:
        """T_world_cam (4x4) of the OPENCV camera frame.
        Multiply camera-frame points/grasps by this to get world frame."""
        return mujoco_cam_to_cv_cam(self.data.cam_xpos[self.cam_id],
                                    self.data.cam_xmat[self.cam_id])

    # -- rendering -----------------------------------------------------------
    def render_rgb(self) -> np.ndarray:
        """(H,W,3) uint8 RGB."""
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.cam_id)
        return self.renderer.render().copy()

    def render_depth(self) -> np.ndarray:
        """(H,W) float32 metric depth in METERS (see module docstring).
        Far-plane / no-return pixels are set to 0 (RealSense convention)."""
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.cam_id)
        depth = self.renderer.render().astype(np.float32)
        self.renderer.disable_depth_rendering()
        depth[~np.isfinite(depth)] = 0.0
        depth[depth > self.depth_clip] = 0.0
        return depth

    def render_segmap(self, body_to_label: dict[int, int]) -> np.ndarray:
        """(H,W) float32 segmentation map: 0 = background, 1..N = objects.

        MuJoCo's segmentation renderer returns per-pixel (geom_id, obj_type).
        We map geom -> body -> our object label so the segmap matches what
        Contact-GraspNet expects (integer instance labels)."""
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.cam_id)
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()

        geom_ids = seg[:, :, 0].astype(np.int64)
        segmap = np.zeros(geom_ids.shape, dtype=np.float32)
        # geom -> body lookup table (geom_id -1 = background)
        geom_body = self.model.geom_bodyid
        valid = geom_ids >= 0
        bodies = np.zeros_like(geom_ids)
        bodies[valid] = geom_body[geom_ids[valid]]
        for body_id, label in body_to_label.items():
            segmap[valid & (bodies == body_id)] = float(label)
        return segmap

    def capture(self, body_to_label: dict[int, int] | None = None):
        """One synchronized observation: (rgb, depth, segmap, K, T_world_cam)."""
        rgb = self.render_rgb()
        depth = self.render_depth()
        segmap = self.render_segmap(body_to_label) if body_to_label else None
        return rgb, depth, segmap, self.intrinsics(), self.extrinsics()

    def close(self):
        self.renderer.close()
