"""Subprocess GraspGen worker — runs under the graspgen_torch interpreter.

Mirrors cgn_worker.py's CLI and output format exactly, so
run_sim_grasp_test.py's existing _subprocess_predict() result-parsing code
works unmodified regardless of which backend produced the npz.

Usage:
    python sim_grasp/graspgen_worker.py obs.npz out.npz \
        --gripper-config path/to/graspgen_franka_panda.yml \
        [--num-grasps 200] [--grasp-threshold 0.8]

obs.npz keys, depth mode:  depth (H,W) float32 m, K (3,3), rgb (H,W,3) uint8,
                           segmap (H,W)
            cloud mode:    pc_full (N,3) float32 in a camera frame, plus
                           pcseg_<sid> (Ni,3) per object (P2 fusion path)
out.npz keys: grasps_<sid>, scores_<sid>, contacts_<sid>.

Contact points and gripper openings are not produced by GraspGen (unlike
CGN) — contacts_<sid> is written as an empty (0,3) array and openings_<sid>
is omitted, matching GraspPrediction's documented defaults.
"""
import logging
import os
import warnings

if not os.environ.get('SIM_GRASP_VERBOSE'):
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    for _name in ('grasp_gen', 'OpenGL', 'torch', 'timm', 'spconv', 'sam3'):
        logging.getLogger(_name).setLevel(logging.ERROR)

import argparse
from pathlib import Path

import numpy as np
import torch


def object_point_clouds(obs) -> dict:
    """Return {seg_id: (Ni,3) float32 object point cloud in camera frame},
    from either depth+segmap or pre-extracted cloud-mode payloads."""
    if 'pc_full' in obs:
        return {float(k.split('_', 1)[1]): np.asarray(obs[k], dtype=np.float32)
                for k in obs.files if k.startswith('pcseg_')}

    from grasp_gen.utils.point_cloud_utils import depth_and_segmentation_to_point_clouds

    depth, K, segmap = obs['depth'], obs['K'], obs['segmap']
    rgb = obs['rgb'] if 'rgb' in obs else None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    out = {}
    for sid in sorted(s for s in np.unique(segmap) if s > 0):
        # depth_and_segmentation_to_point_clouds requires a single-object
        # mask (background=0 plus exactly one object id) — isolate this
        # object out of the full multi-object segmap before calling.
        single_object_mask = np.where(segmap == sid, sid, 0)
        _scene_pc, object_pc, _scene_c, _obj_c = depth_and_segmentation_to_point_clouds(
            depth_image=depth, segmentation_mask=single_object_mask,
            fx=fx, fy=fy, cx=cx, cy=cy, rgb_image=rgb,
            target_object_id=int(sid), remove_object_from_scene=True)
        out[float(sid)] = np.asarray(object_pc, dtype=np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('obs_npz')
    ap.add_argument('out_npz')
    ap.add_argument('--gripper-config', required=True)
    ap.add_argument('--num-grasps', type=int, default=200)
    ap.add_argument('--grasp-threshold', type=float, default=0.8)
    args = ap.parse_args()

    gripper_config = Path(args.gripper_config)
    if not gripper_config.is_file():
        raise FileNotFoundError(f'GraspGen gripper config not found: {gripper_config}')

    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
    from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal

    obs = np.load(args.obs_npz)
    per_object_pc = object_point_clouds(obs)

    grasp_cfg = load_grasp_cfg(str(gripper_config))
    sampler = GraspGenSampler(grasp_cfg)

    out = {}
    total = 0
    for sid, pc in per_object_pc.items():
        if pc.shape[0] < 30:
            continue
        pc_t = torch.from_numpy(pc)
        pc_filtered, _removed = point_cloud_outlier_removal(pc_t)
        pc_filtered = pc_filtered.numpy()
        grasps_t, conf_t = GraspGenSampler.run_inference(
            pc_filtered, sampler, grasp_threshold=args.grasp_threshold,
            num_grasps=args.num_grasps, topk_num_grasps=-1)
        grasps = grasps_t.cpu().numpy().astype(np.float32) if len(grasps_t) else \
            np.zeros((0, 4, 4), dtype=np.float32)
        scores = conf_t.cpu().numpy().astype(np.float32) if len(conf_t) else \
            np.zeros((0,), dtype=np.float32)
        if len(grasps):
            grasps[:, 3, 3] = 1.0
        key = f'{sid:g}'
        out[f'grasps_{key}'] = grasps
        out[f'scores_{key}'] = scores
        out[f'contacts_{key}'] = np.zeros((0, 3), dtype=np.float32)
        total += len(grasps)

    np.savez(args.out_npz, **out)
    print(f'[graspgen-worker] {total} grasps -> {args.out_npz}')


if __name__ == '__main__':
    main()
