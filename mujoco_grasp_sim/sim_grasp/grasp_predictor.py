"""GraspPredictor interface + Contact-GraspNet implementation.

The interface is deliberately narrow so the backend can be swapped later
(AnyGrasp, GSNet/GraspNet-baseline, GIGA, FoundationPose+planner, ...)
without touching the simulation side:

    predictor = ContactGraspNetPredictor()
    pred = predictor.predict(depth, K, rgb=rgb, segmap=segmap)
    pred.grasps_cam[seg_id]  ->  (N,4,4) T_cam_grasp poses (OpenCV cam frame)

ContactGraspNetPredictor REUSES the existing contact_graspnet_pytorch
package (installed editable from the sibling repo): its GraspEstimator,
checkpoint loading and full grasp-generation pipeline. Nothing is
reimplemented.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CKPT_DIR = _REPO_ROOT / 'contact_graspnet_pytorch' / 'checkpoints' / 'contact_graspnet'


@dataclass
class GraspPrediction:
    """Backend-agnostic grasp prediction result.

    All poses are 4x4 homogeneous transforms T_cam_grasp in the OPENCV camera
    frame, grasp-frame convention: +Z approach, +X finger closing line
    (see frames.py).
    """
    grasps_cam: dict      # {seg_id: (N,4,4) float32}
    scores: dict          # {seg_id: (N,) float32}
    contact_pts: dict     # {seg_id: (N,3) float32}
    gripper_openings: dict = field(default_factory=dict)

    @property
    def num_grasps(self) -> int:
        return int(sum(len(g) for g in self.grasps_cam.values()))

    def best_grasp(self):
        """(seg_id, T_cam_grasp (4,4), score) of the overall best grasp, or None."""
        best = None
        for seg_id, s in self.scores.items():
            if len(s) == 0:
                continue
            i = int(np.argmax(s))
            if best is None or s[i] > best[2]:
                best = (seg_id, self.grasps_cam[seg_id][i], float(s[i]))
        return best


class GraspPredictor(ABC):
    """Implement this to plug in a different grasp detection backend."""

    @abstractmethod
    def predict(self, depth: np.ndarray, K: np.ndarray,
                rgb: np.ndarray | None = None,
                segmap: np.ndarray | None = None) -> GraspPrediction:
        """depth: (H,W) float32 meters; K: (3,3); rgb: (H,W,3) uint8;
        segmap: (H,W) integer instance labels, 0 = background."""


class ContactGraspNetPredictor(GraspPredictor):
    """Thin adapter around the existing contact_graspnet_pytorch pipeline."""

    def __init__(self, ckpt_dir: str | Path = DEFAULT_CKPT_DIR,
                 forward_passes: int = 1, z_range=(0.2, 1.8),
                 local_regions: bool = True, filter_grasps: bool = True,
                 arg_configs: list | None = None):
        # Imports deferred so the sim modules stay importable without the
        # CGN repo (e.g. when testing scene generation alone).
        from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
        from contact_graspnet_pytorch import config_utils
        from contact_graspnet_pytorch.checkpoints import CheckpointIO

        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f'Contact-GraspNet checkpoint dir not found: {ckpt_dir}')

        self.forward_passes = forward_passes
        self.z_range = list(z_range)
        self.local_regions = local_regions
        self.filter_grasps = filter_grasps

        global_config = config_utils.load_config(
            str(ckpt_dir), batch_size=forward_passes, arg_configs=arg_configs or [])
        self.estimator = GraspEstimator(global_config)
        checkpoint_io = CheckpointIO(checkpoint_dir=str(ckpt_dir / 'checkpoints'),
                                     model=self.estimator.model)
        checkpoint_io.load('model.pt')

    def predict(self, depth, K, rgb=None, segmap=None) -> GraspPrediction:
        use_segments = segmap is not None

        # Existing repo pipeline: depth -> full cloud + per-segment clouds
        pc_full, pc_segments, _pc_colors = self.estimator.extract_point_clouds(
            depth, K, segmap=segmap, rgb=rgb, z_range=self.z_range)

        return self.predict_clouds(pc_full, pc_segments if use_segments else None)

    def predict_clouds(self, pc_full: np.ndarray,
                       pc_segments: dict | None = None) -> GraspPrediction:
        """Predict directly from point clouds in an OpenCV CAMERA frame —
        the multi-camera fusion path (P2): fused world clouds are expressed
        in the primary camera's frame and passed here; grasps come back
        T_cam_grasp in that same frame, like the depth-image path."""
        use_segments = bool(pc_segments)
        pred_grasps_cam, scores, contact_pts, gripper_openings = \
            self.estimator.predict_scene_grasps(
                pc_full.astype(np.float32),
                pc_segments={k: np.asarray(v, dtype=np.float32)
                             for k, v in (pc_segments or {}).items()},
                local_regions=self.local_regions and use_segments,
                filter_grasps=self.filter_grasps and use_segments,
                forward_passes=self.forward_passes)

        return GraspPrediction(grasps_cam=pred_grasps_cam, scores=scores,
                               contact_pts=contact_pts,
                               gripper_openings=gripper_openings or {})
