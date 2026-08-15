"""PromptSelector: resolves a text/click/box prompt to a target mask via
Meta SAM 3.

Always runs via subprocess — SAM 3 lives in its own conda env (sam3_torch,
Python 3.12) because it needs a newer Python than cgn_torch's 3.10. There is
no in-process code path here, matching GraspGenPredictor's isolation
pattern in sim_grasp/graspgen_predictor.py.
"""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SelectionResult:
    masks: np.ndarray    # (K,H,W) bool
    scores: np.ndarray   # (K,) float32
    boxes: np.ndarray    # (K,4) float32, pixel [x1,y1,x2,y2]

    @property
    def is_ambiguous(self) -> bool:
        return len(self.scores) > 1

    @property
    def is_empty(self) -> bool:
        return len(self.scores) == 0


def resolve_sam3_python(override: str | None = None) -> Path:
    """Resolve the sam3_torch interpreter: --sam3-python CLI value, else
    SAM3_PYTHON env var. Fails fast — never falls back to sys.executable
    (that would run SAM 3 under cgn_torch, which lacks Python 3.12)."""
    candidate = override or os.environ.get('SAM3_PYTHON')
    if not candidate:
        raise RuntimeError(
            'Promptable selection requested but no SAM 3 interpreter '
            'configured. Set the SAM3_PYTHON environment variable to the '
            'sam3_torch env\'s python, or pass --sam3-python. See '
            'mujoco_grasp_sim/README.md "Promptable selection setup".')
    path = Path(candidate)
    if not path.is_file():
        raise FileNotFoundError(f'SAM3_PYTHON does not exist: {path}')
    return path


class PromptSelector:
    def __init__(self, sam3_python: str | None = None, click_radius_px: int = 15):
        self.python = resolve_sam3_python(sam3_python)
        self.click_radius_px = click_radius_px

    def select(self, rgb: np.ndarray, prompt: str | None = None,
              click: tuple[float, float] | None = None,
              box: tuple[float, float, float, float] | None = None,
              work_dir: str | Path = '.') -> SelectionResult:
        modes = [m for m in (prompt, click, box) if m is not None]
        if len(modes) != 1:
            raise ValueError('Exactly one of prompt, click, box must be given')

        work_dir = Path(work_dir)
        rgb_f = work_dir / '_sam3_rgb.npy'
        out_f = work_dir / '_sam3_out.npz'
        np.save(rgb_f, np.asarray(rgb, dtype=np.uint8))

        worker = Path(__file__).parent / 'sam3_worker.py'
        cmd = [str(self.python), str(worker), str(rgb_f), str(out_f)]
        if prompt is not None:
            cmd += ['--prompt', prompt]
        elif click is not None:
            cmd += ['--click', f'{click[0]},{click[1]}']
        else:
            cmd += ['--box', f'{box[0]},{box[1]},{box[2]},{box[3]}']
        cmd += ['--click-radius-px', str(self.click_radius_px)]

        r = subprocess.run(cmd)
        if r.returncode != 0 or not out_f.exists():
            raise RuntimeError(f'SAM 3 worker failed (exit code {r.returncode})')

        with np.load(out_f) as z:
            result = SelectionResult(masks=z['masks'], scores=z['scores'], boxes=z['boxes'])
        rgb_f.unlink(missing_ok=True)
        out_f.unlink(missing_ok=True)
        return result
