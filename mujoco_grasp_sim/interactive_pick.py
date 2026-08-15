"""Interactive live pick: click an object in a live camera window, SAM 3
segments it, confirm the mask, then watch GraspGen/CGN pick it — live, in
the same window.

Usage:
    export MUJOCO_GL=osmesa
    export GRASPGEN_PYTHON=/path/to/graspgen_torch/bin/python
    export SAM3_PYTHON=/path/to/sam3_torch/bin/python
    python interactive_pick.py --seed 5 --backend graspgen

Controls: click an object in the window to select it. Once SAM 3 resolves
a mask, it's highlighted — press Enter/Space to confirm and execute the
pick, or Esc/'c' to try a different click. Close the window at any point
to cancel.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_sim_grasp_test import filter_feasible
from sim_grasp import (SceneConfig, SceneGenerator, CameraModule,
                       ContactGraspNetPredictor)
from sim_grasp.live_viewer import LiveViewer
from sim_grasp.prompt_selector import PromptSelector, resolve_real_label


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=None, help='randomization seed')
    ap.add_argument('--backend', choices=['cgn', 'graspgen'], default='cgn',
                    help='grasp-prediction backend')
    ap.add_argument('--graspgen-python', default=None,
                    help='path to graspgen_torch\'s python; overrides GRASPGEN_PYTHON')
    ap.add_argument('--sam3-python', default=None,
                    help='path to sam3_torch\'s python; overrides SAM3_PYTHON')
    ap.add_argument('--click-radius-px', type=int, default=15,
                    help='half-width of the box synthesized around your click for SAM 3')
    ap.add_argument('--save-dir', default=None, help='output dir (default: output/<timestamp>)')
    args = ap.parse_args()

    save_dir = Path(args.save_dir) if args.save_dir else \
        Path(__file__).parent / 'output' / time.strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f'[run] outputs -> {save_dir}')

    cfg = SceneConfig(seed=args.seed)
    default_cal = Path(__file__).parent / 'calibration_result.yaml'
    if default_cal.exists():
        cfg.calibration_file = str(default_cal)
        print(f'[camera] using real eye-to-hand calibration: {default_cal}')
    else:
        print(f'[camera] using generic look-at camera: pos={cfg.cam_pos}, target={cfg.cam_target}')
    gen = SceneGenerator(cfg)
    model, data = gen.generate()
    on_table = gen.objects_on_table()
    print(f'[scene] {len(gen.object_names)} objects spawned, {len(on_table)} on table')

    cam = CameraModule(model, data, cam_name=cfg.cam_name, width=640, height=480)
    rgb, depth, segmap, K, T_world_cam = cam.capture(gen.object_body_ids)

    viewer = LiveViewer()
    selector = PromptSelector(sam3_python=args.sam3_python, click_radius_px=args.click_radius_px)

    real_label = None
    mask = None
    print('[interactive] click an object in the window to select it '
          '(close the window to cancel)...')
    while real_label is None:
        xy = viewer.wait_for_click(rgb)
        if xy is None:
            print('[interactive] window closed — cancelled, nothing executed')
            viewer.close()
            cam.close()
            return
        print(f'[interactive] click at {xy} — running SAM 3...')
        result = selector.select(rgb, click=(float(xy[0]), float(xy[1])))
        if result.is_empty:
            print('[interactive] no object found at that point — click again')
            continue
        # SAM 3 doesn't guarantee its matches are score-sorted, and a click
        # can occasionally return more than one candidate (unlike the design
        # assumption) — always take the highest-scoring one, no CLI-index
        # disambiguation available in this interactive flow.
        best = int(np.argmax(result.scores))
        mask = result.masks[best]
        viewer.show_mask_overlay(rgb, mask)
        print(f'[interactive] SAM 3 match, score {float(result.scores[best]):.3f} — '
              'Enter/Space to confirm, Esc/c to retry')
        if not viewer.wait_for_confirm():
            if viewer.closed:
                print('[interactive] window closed — cancelled, nothing executed')
                viewer.close()
                cam.close()
                return
            print('[interactive] retry — click an object in the window again...')
            continue
        real_label = resolve_real_label(segmap, mask)
        if real_label is None:
            print('[interactive] resolved mask does not overlap any known '
                  'object — click again')
            continue

    new_segmap = np.zeros(rgb.shape[:2], dtype=segmap.dtype)
    new_segmap[mask] = real_label
    print(f'[interactive] confirmed: object {real_label}')

    if args.backend == 'graspgen':
        print('[graspgen] loading GraspGen...')
        from sim_grasp import GraspGenPredictor
        predictor = GraspGenPredictor(graspgen_python=args.graspgen_python)
    else:
        print('[cgn] loading Contact-GraspNet...')
        predictor = ContactGraspNetPredictor(forward_passes=3)
    t0 = time.time()
    pred = predictor.predict(depth, K, rgb=rgb, segmap=new_segmap)
    print(f'[{args.backend}] {pred.num_grasps} grasps in {time.time() - t0:.1f}s')

    grasps_cam, scores = pred.grasps_cam, pred.scores
    if pred.num_grasps > 0:
        grasps_cam, scores, feas_stats = filter_feasible(
            grasps_cam, scores, pred.gripper_openings, T_world_cam, cfg.table_height)
        print(f"[feasibility] kept {feas_stats['n_after']}/{feas_stats['n_before']} "
              f"({feas_stats['n_rejected']} table-colliding/underhand rejected)")

    s = np.asarray(scores.get(real_label, []))
    if len(s) == 0:
        print('[interactive] no feasible grasp found for the selected object — '
              'nothing to execute')
        viewer.close()
        cam.close()
        return

    i = int(np.argmax(s))
    T_cam_grasp = grasps_cam[real_label][i]
    score = float(s[i])
    T_world_grasp = T_world_cam @ np.asarray(T_cam_grasp)
    label_to_body = {lbl: gen.object_names[lbl - 1]
                     for lbl in gen.object_body_ids.values()}
    body = label_to_body[real_label]

    from sim_grasp.executor import GraspExecutor
    rec_cam = CameraModule(model, data, cam_name=cfg.record_cam_name, width=640, height=480)
    executor = GraspExecutor(model, data, camera_module=rec_cam, record_gif=True,
                             record_dir=save_dir / '_gif_frames',
                             on_frame=viewer.show_frame)
    print(f'[execute] object {real_label} ({body}), score {score:.3f} — watch the window...')
    res = executor.execute(T_world_grasp, target_body=body)
    res.update(object=real_label, score=score)
    print(f'[execute]   -> {res}')
    if res['success']:
        print(f"[execute] PICK SUCCESS (object raised {res['object_raised_m']} m)")
    else:
        print('[execute] pick failed')
    executor.save_gif(save_dir / 'execution.gif')
    (save_dir / 'metrics.json').write_text(json.dumps(
        {'seed': args.seed, 'backend': args.backend, 'object': real_label,
         'score': score, 'execution': res}, indent=2))
    print(f'[execute] video saved: {save_dir / "execution.gif"}')

    print('[interactive] done — close the window to exit')
    while not viewer.closed:
        viewer.show_frame(rec_cam.render_rgb())
    viewer.close()
    cam.close()
    rec_cam.close()


if __name__ == '__main__':
    main()
