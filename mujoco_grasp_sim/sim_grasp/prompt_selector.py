"""PromptSelector: resolves a text/click/box prompt to a target mask via
Meta SAM 3.

Always runs via subprocess — SAM 3 lives in its own conda env (sam3_torch,
Python 3.12) because it needs a newer Python than cgn_torch's 3.10. There is
no in-process code path here, matching GraspGenPredictor's isolation
pattern in sim_grasp/graspgen_predictor.py.
"""
import os
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


def resolve_real_label(gt_segmap: np.ndarray, mask: np.ndarray) -> int | None:
    """Real object label the resolved `mask` (H,W bool) actually overlaps
    most, per `gt_segmap` (H,W int, 0 = background). Returns None if the
    mask overlaps no real object (entirely over background).

    Sim-only bookkeeping: maps a SAM 3 mask (real perception, no ground
    truth involved in producing it) onto the ground-truth body-name/
    success-detection machinery the rest of the pipeline uses. A real
    camera deployment has no ground-truth segmap to compare against;
    success there would be graded some other way. Ground truth here is
    used ONLY to label an already-SAM3-selected mask, never to influence
    which mask/object gets selected in the first place."""
    overlap_labels = gt_segmap[mask]
    overlap_labels = overlap_labels[overlap_labels > 0]
    if len(overlap_labels) == 0:
        return None
    return int(np.bincount(overlap_labels.astype(int)).argmax())


def filter_selection_by_click(result: SelectionResult,
                              click: tuple[float, float]) -> SelectionResult:
    """Pure filter: keep only candidates in `result` whose mask contains
    the `click` pixel (x, y). No subprocess/model call here -- safe to
    unit-test directly with synthetic SelectionResult objects.

    Used to turn a category-wide detection pass (`select(rgb, prompt=...)`,
    which returns one mask per object instance in the scene) into a
    click-disambiguated selection: the click no longer needs to localize
    the object geometrically (SAM 3's click-as-box-exemplar mode only
    matches the locally clicked face's appearance -- measured IoU ~0.29
    against the true full-object mask, see prompt_selector click_to_select
    docstring), it only needs to land on the intended instance's mask."""
    if result.is_empty:
        return result
    x, y = int(click[0]), int(click[1])
    keep = np.where(result.masks[:, y, x])[0]
    if len(keep) == 0:
        return SelectionResult(
            masks=np.zeros((0,) + result.masks.shape[1:], dtype=bool),
            scores=np.zeros((0,), dtype=np.float32),
            boxes=np.zeros((0, 4), dtype=np.float32))
    return SelectionResult(masks=result.masks[keep], scores=result.scores[keep],
                           boxes=result.boxes[keep])


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

        from sim_grasp.subprocess_utils import run_worker
        returncode = run_worker(cmd)
        if returncode != 0 or not out_f.exists():
            raise RuntimeError(f'SAM 3 worker failed (exit code {returncode})')

        with np.load(out_f) as z:
            result = SelectionResult(masks=z['masks'], scores=z['scores'], boxes=z['boxes'])
        rgb_f.unlink(missing_ok=True)
        out_f.unlink(missing_ok=True)
        return result

    def click_to_select(self, rgb: np.ndarray, click: tuple[float, float],
                        category: str = 'a block',
                        work_dir: str | Path = '.') -> SelectionResult:
        """Click-based selection that returns a full-object mask, not just
        the locally-clicked face/color: a click-as-box-exemplar prompt
        (the old `select(rgb, click=...)` path) makes SAM 3 match the
        clicked region's *appearance*, which on a uniformly-lit cuboid
        face returns only that face (measured IoU ~0.29 against the true
        full-object mask on a real repro case, 2026-08-18). Instead, this
        runs a category-wide text-prompt detection pass -- genuine
        per-instance object detection, measured IoU 0.89-0.99 against the
        true full-object masks for every instance in the same scene --
        then keeps only the instance(s) whose mask contains the click
        pixel. `category` currently defaults to box/cuboid wording since
        this project's scenes only spawn box-shaped objects (see
        ROADMAP.md); pass a different category if that changes."""
        result = self.select(rgb, prompt=category, work_dir=work_dir)
        filtered = filter_selection_by_click(result, click)
        if filtered.is_empty and not result.is_empty:
            print(f"[prompt] category {category!r} matched {len(result.scores)} "
                  f"instance(s), but none contain click {click} -- try a "
                  "different --category or click location")
        return filtered
