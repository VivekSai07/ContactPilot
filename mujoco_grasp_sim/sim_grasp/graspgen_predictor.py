"""GraspGenPredictor: GraspPredictor backed by NVlabs/GraspGen.

Unlike ContactGraspNetPredictor, this ALWAYS runs via subprocess — GraspGen
lives in its own conda env (graspgen_torch) because its dependencies
conflict with cgn_torch's torch version, not merely for the memory-isolation
reason ContactGraspNetPredictor's subprocess path uses. There is no
in-process code path here.
"""

import os
from pathlib import Path

import numpy as np

from sim_grasp.grasp_predictor import GraspPredictor, GraspPrediction

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_DIR = _REPO_ROOT / 'graspgen_checkpoints'


def resolve_graspgen_python(override: str | None = None) -> Path:
    """Resolve the graspgen_torch interpreter: --graspgen-python CLI value,
    else GRASPGEN_PYTHON env var. Fails fast with a clear message — never
    falls back to sys.executable (that would run GraspGen under cgn_torch)."""
    candidate = override or os.environ.get('GRASPGEN_PYTHON')
    if not candidate:
        raise RuntimeError(
            'GraspGen backend requested but no interpreter configured. '
            'Set the GRASPGEN_PYTHON environment variable to the '
            'graspgen_torch env\'s interpreter, or pass --graspgen-python. '
            'See mujoco_grasp_sim/README.md "GraspGen backend setup".')
    path = Path(candidate)
    if not path.is_file():
        raise FileNotFoundError(f'GRASPGEN_PYTHON does not exist: {path}')
    return path


class GraspGenPredictor(GraspPredictor):
    def __init__(self, checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
                 graspgen_python: str | None = None,
                 num_grasps: int = 200, grasp_threshold: float = 0.8):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.gripper_config = self.checkpoint_dir / 'graspgen_franka_panda.yml'
        if not self.gripper_config.is_file():
            raise FileNotFoundError(
                f'GraspGen checkpoint not found: {self.gripper_config}. '
                'Run mujoco_grasp_sim/scripts/download_graspgen_checkpoint.py first.')
        self.python = resolve_graspgen_python(graspgen_python)
        self.num_grasps = num_grasps
        self.grasp_threshold = grasp_threshold

    def predict(self, depth, K, rgb=None, segmap=None) -> GraspPrediction:
        payload = dict(depth=depth, K=K)
        if rgb is not None:
            payload['rgb'] = rgb
        if segmap is not None:
            payload['segmap'] = segmap
        return self._run(payload)

    def predict_clouds(self, pc_full: np.ndarray,
                       pc_segments: dict | None = None) -> GraspPrediction:
        payload = {'pc_full': np.asarray(pc_full, dtype=np.float32)}
        for sid, pc in (pc_segments or {}).items():
            payload[f'pcseg_{float(sid):g}'] = np.asarray(pc, dtype=np.float32)
        return self._run(payload)

    def _run(self, payload: dict, work_dir: str | Path = '.') -> GraspPrediction:
        work_dir = Path(work_dir)
        obs_f = work_dir / '_graspgen_obs.npz'
        out_f = work_dir / '_graspgen_out.npz'
        np.savez(obs_f, **payload)
        worker = Path(__file__).parent / 'graspgen_worker.py'
        cmd = [str(self.python), str(worker), str(obs_f), str(out_f),
               '--gripper-config', str(self.gripper_config),
               '--num-grasps', str(self.num_grasps),
               '--grasp-threshold', str(self.grasp_threshold)]
        from sim_grasp.subprocess_utils import run_worker
        returncode = run_worker(cmd)
        if returncode != 0 or not out_f.exists():
            raise RuntimeError(f'GraspGen worker failed (exit code {returncode})')
        parts = {'grasps': {}, 'scores': {}, 'contacts': {}}
        with np.load(out_f) as z:
            for k in z.files:
                kind, sid = k.split('_', 1)
                parts[kind][float(sid)] = z[k]
        obs_f.unlink(missing_ok=True)
        out_f.unlink(missing_ok=True)
        return GraspPrediction(grasps_cam=parts['grasps'], scores=parts['scores'],
                               contact_pts=parts['contacts'], gripper_openings={})
