"""Multi-camera point-cloud fusion (P2 of ROADMAP.md).

Fuses RGB-D observations from N calibrated cameras into one world-frame
cloud (per scene and per object), then expresses the result in a chosen
PRIMARY camera's frame so the existing Contact-GraspNet path — which expects
clouds in an OpenCV camera frame — runs unchanged. Grasps come back in the
primary camera frame, exactly like the single-camera pipeline.

Segmap reconciliation: in sim every camera shares ground-truth instance ids,
so per-object clouds fuse by id. On the real setup, per-camera segmentations
must be matched first (e.g. by cloud overlap); fuse_observations() takes the
ids as-is and is agnostic to where they came from.
"""

import numpy as np

from sim_grasp.frames import invert_se3, transform_points


def voxel_dedup(pc: np.ndarray, voxel: float = 0.003) -> np.ndarray:
    """Keep one point per `voxel`-sized cell — removes the double-density
    artifacts where camera frustums overlap (CGN's farthest-point sampling
    otherwise oversamples those regions)."""
    if len(pc) == 0:
        return pc
    keys = np.floor(pc / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pc[np.sort(idx)]


def fuse_observations(observations: list[dict], voxel: float = 0.003):
    """Fuse N single-camera observations into world-frame clouds.

    :param observations: dicts with keys depth (H,W) m, K (3,3),
        T_world_cam (4,4), and optionally segmap (H,W int, 0 = background)
    :param voxel: dedup cell size in meters
    :returns: (pc_world (M,3), seg_clouds_world {seg_id: (Mi,3)})
    """
    from sim_grasp.pointcloud import depth_to_pointcloud

    full, seg_clouds = [], {}
    for obs in observations:
        depth, K, T_wc = obs['depth'], obs['K'], obs['T_world_cam']
        pc = depth_to_pointcloud(depth, K)
        full.append(transform_points(T_wc, pc))
        segmap = obs.get('segmap')
        if segmap is None:
            continue
        for sid in np.unique(segmap):
            if sid <= 0:
                continue
            pc_s = depth_to_pointcloud(depth, K, mask=(segmap == sid))
            seg_clouds.setdefault(int(sid), []).append(
                transform_points(T_wc, pc_s))

    pc_world = voxel_dedup(np.vstack(full), voxel)
    seg_world = {sid: voxel_dedup(np.vstack(parts), voxel)
                 for sid, parts in seg_clouds.items()}
    return pc_world, seg_world


def clouds_to_camera(pc_world: np.ndarray, seg_world: dict,
                     T_world_cam: np.ndarray):
    """Express fused world clouds in one camera's OpenCV frame (the frame the
    grasp predictor works in; its grasps then come back T_cam_grasp as usual).
    """
    T_cw = invert_se3(T_world_cam)
    return (transform_points(T_cw, pc_world),
            {sid: transform_points(T_cw, pc) for sid, pc in seg_world.items()})
