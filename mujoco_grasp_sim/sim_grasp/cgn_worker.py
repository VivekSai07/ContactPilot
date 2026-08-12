"""Subprocess Contact-GraspNet worker.

Runs ONE grasp prediction in its own process and exits. The point: PyTorch +
CUDA hold gigabytes of Windows commit that are only returned to the OS when
the process dies. Multi-round --pick-all on 8 GB machines OOMs if the model
stays resident in the sim process (MuJoCo renders fail on ~1 MiB
allocations), so the run script shells out here once per round instead.

Usage:
    python sim_grasp/cgn_worker.py obs.npz out.npz \
        [--forward-passes N] [--arg-configs TEST.first_thres:0.14 ...]

obs.npz keys, depth mode:  depth (H,W) float32 m, K (3,3), rgb (H,W,3) uint8,
                           segmap (H,W)
            cloud mode:    pc_full (N,3) float32 in a camera frame, plus
                           pcseg_<sid> (Ni,3) per object (P2 fusion path)
out.npz keys: grasps_<sid>, scores_<sid>, contacts_<sid>, openings_<sid>.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('obs_npz')
    ap.add_argument('out_npz')
    ap.add_argument('--forward-passes', type=int, default=1)
    ap.add_argument('--arg-configs', nargs='*', default=[])
    args = ap.parse_args()

    from sim_grasp.grasp_predictor import ContactGraspNetPredictor

    obs = np.load(args.obs_npz)
    predictor = ContactGraspNetPredictor(forward_passes=args.forward_passes,
                                         arg_configs=list(args.arg_configs))
    if 'pc_full' in obs:           # cloud mode (multi-camera fusion)
        pc_segments = {float(k.split('_', 1)[1]): obs[k]
                       for k in obs.files if k.startswith('pcseg_')}
        pred = predictor.predict_clouds(obs['pc_full'], pc_segments or None)
    else:                          # depth-image mode
        pred = predictor.predict(obs['depth'], obs['K'],
                                 rgb=obs['rgb'] if 'rgb' in obs else None,
                                 segmap=obs['segmap'] if 'segmap' in obs else None)

    out = {}
    for sid in pred.grasps_cam:
        key = f'{float(sid):g}'
        out[f'grasps_{key}'] = np.asarray(pred.grasps_cam[sid])
        out[f'scores_{key}'] = np.asarray(pred.scores[sid])
        out[f'contacts_{key}'] = np.asarray(
            pred.contact_pts.get(sid, np.zeros((0, 3))))
        if sid in pred.gripper_openings:
            out[f'openings_{key}'] = np.asarray(pred.gripper_openings[sid])
    np.savez(args.out_npz, **out)
    print(f'[cgn-worker] {pred.num_grasps} grasps -> {args.out_npz}')


if __name__ == '__main__':
    main()
