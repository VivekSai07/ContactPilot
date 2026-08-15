"""Promptable-selection accuracy benchmark.

For each seed: generate a scene, capture RGB, read each object's REAL
spawned color from the compiled MuJoCo model (model.geom_rgba — not scene
metadata), build a genuine ground-truth prompt ("the {color} box") for one
target object, run PromptSelector, and check whether the resolved mask
actually overlaps the intended object's ground-truth segmap region (IoU).
Ground truth is used only for this grading step, never fed into the
selection pipeline itself.

Usage:
    python benchmark_prompt_selection.py --seeds 0-4 --tag baseline
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim_grasp import SceneConfig, SceneGenerator, CameraModule
from sim_grasp.color_utils import rgb_to_color_name
from sim_grasp.prompt_selector import PromptSelector

HERE = Path(__file__).resolve().parent


def parse_seeds(spec: str) -> list[int]:
    seeds = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def object_geom(model, body_name):
    """Mirrors run_sim_grasp_test.py's _object_geom() — (bid, gid, gtype)
    for a named object body."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    gid = model.body_geomadr[bid]
    return bid, gid, int(model.geom_type[gid])


def run_one(seed: int, selector: PromptSelector) -> dict:
    cfg = SceneConfig(seed=seed)
    gen = SceneGenerator(cfg)
    model, data = gen.generate()
    on_table = gen.objects_on_table()
    if not on_table:
        return {'seed': seed, 'skipped': 'no objects on table'}

    cam = CameraModule(model, data, cam_name=cfg.cam_name, width=640, height=480)
    rgb, depth, segmap, K, T_world_cam = cam.capture(gen.object_body_ids)
    cam.close()

    label_of = {name: i + 1 for i, name in enumerate(gen.object_names)}
    target_body = on_table[0]
    target_sid = label_of[target_body]
    _bid, gid, _gtype = object_geom(model, target_body)
    color_name = rgb_to_color_name(model.geom_rgba[gid][:3])
    prompt = f'the {color_name} box'

    result = selector.select(rgb, prompt=prompt)
    if result.is_empty:
        return {'seed': seed, 'prompt': prompt, 'matched': False, 'reason': 'no matches'}

    # best-scoring match (benchmark grades accuracy; it doesn't exercise
    # the CLI's disambiguation-required behavior)
    idx = int(np.argmax(result.scores))
    mask = result.masks[idx]
    gt_mask = segmap == target_sid
    intersection = np.logical_and(mask, gt_mask).sum()
    union = np.logical_or(mask, gt_mask).sum()
    iou = float(intersection) / float(union) if union > 0 else 0.0

    return {
        'seed': seed, 'prompt': prompt, 'target_object': target_sid,
        'num_matches': len(result.scores), 'matched': iou > 0.5, 'iou': round(iou, 3),
        'score': round(float(result.scores[idx]), 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', default='0-4', help='e.g. "0-4" or "1,3,7"')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--sam3-python', default=None)
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    tag = args.tag or time.strftime('%m%d_%H%M')
    bench_dir = HERE / 'output' / f'bench_prompt_{tag}'
    bench_dir.mkdir(parents=True, exist_ok=True)

    selector = PromptSelector(sam3_python=args.sam3_python)
    results = []
    for seed in seeds:
        print(f'[bench-prompt] seed {seed}...', flush=True)
        r = run_one(seed, selector)
        results.append(r)
        print(f'[bench-prompt]   {r}', flush=True)

    (bench_dir / 'summary.json').write_text(json.dumps(results, indent=2))
    graded = [r for r in results if 'matched' in r]
    n_correct = sum(1 for r in graded if r['matched'])
    print(f'\n[bench-prompt] ===== {n_correct}/{len(graded)} correct selections '
          f'({100 * n_correct / max(len(graded), 1):.0f}%) =====')
    iou_scored = [r for r in graded if 'iou' in r]
    if iou_scored:
        mean_iou = sum(r['iou'] for r in iou_scored) / len(iou_scored)
        print(f'[bench-prompt] mean IoU: {mean_iou:.3f}')
    print(f'[bench-prompt] full details: {bench_dir / "summary.json"}')


if __name__ == '__main__':
    main()
