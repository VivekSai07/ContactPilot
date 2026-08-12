"""sim_grasp — MuJoCo tabletop grasping simulation for Contact-GraspNet evaluation.

Modules
-------
frames           Coordinate frame conventions + transform helpers (READ THIS FIRST)
scene_generator  SceneGenerator: builds randomized MJCF tabletop scenes with a
                 Menagerie Franka Panda, settles physics
camera           CameraModule: RGB / metric depth / segmentation rendering,
                 intrinsics + extrinsics (OpenCV convention)
pointcloud       Depth -> point cloud reconstruction, frame transforms
grasp_predictor  GraspPredictor interface + ContactGraspNetPredictor implementation
feasibility      Table-collision grasp filter (no motion planning)
visualizer       Open3D grasp visualization (reuses Contact-GraspNet's drawing code)
"""

from sim_grasp.frames import mujoco_cam_to_cv_cam, transform_points, invert_se3
from sim_grasp.scene_generator import SceneConfig, SceneGenerator
from sim_grasp.camera import CameraModule
from sim_grasp.pointcloud import depth_to_pointcloud
from sim_grasp.grasp_predictor import GraspPredictor, GraspPrediction, ContactGraspNetPredictor
from sim_grasp.feasibility import GraspFeasibilityChecker
from sim_grasp.visualizer import Visualizer
