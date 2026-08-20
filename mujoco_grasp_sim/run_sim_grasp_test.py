"""End-to-end MuJoCo tabletop grasping test with Contact-GraspNet.

Pipeline:
    MuJoCo scene (Menagerie Panda + table + random objects)
        -> settle physics
        -> RGB-D + segmentation capture (eye-to-hand camera)
        -> depth -> point cloud (OpenCV camera frame)
        -> Contact-GraspNet (existing repo pipeline)
        -> table-collision feasibility filter
        -> metrics + Open3D visualization

Usage:
    python run_sim_grasp_test.py                 # full run with visualization
    python run_sim_grasp_test.py --no-vis        # headless (CI / remote)
    python run_sim_grasp_test.py --seed 3 --n-objects 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim_grasp import (SceneConfig, SceneGenerator, CameraModule,
                       ContactGraspNetPredictor, GraspFeasibilityChecker,
                       Visualizer)
from sim_grasp.frames import transform_grasps, transform_points, invert_se3
from sim_grasp.grasp_predictor import GraspPrediction
from sim_grasp.perception import clean_depth, recenter_grasp
from sim_grasp.pointcloud import depth_to_pointcloud
from sim_grasp.placement_planner import (
    compute_object_footprint, build_bin_heightmap, OccupancyPlacementPlanner,
    compute_release_z)
from sim_grasp.executor import PLACE_RELEASE


def gpu_vram_gb() -> float:
    """Total VRAM via nvidia-smi — avoids importing torch (and committing its
    CUDA context) into this process."""
    import subprocess
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.total',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=15)
        return float(out.stdout.splitlines()[0]) * 1048576 / 1e9 \
            if out.returncode == 0 else 0.0
    except Exception:
        return 0.0


def predict_in_subprocess(depth, K, rgb, segmap, forward_passes, arg_configs,
                          work_dir, backend='cgn', graspgen_python=None) -> GraspPrediction:
    """Run one grasp prediction in a child process.

    PyTorch's multi-GB Windows commit is returned to the OS when the child
    exits, so the sim process keeps enough headroom to render — keeping the
    model resident here OOMs multi-round pick-and-place on 8 GB machines.
    (For backend='graspgen' this isolation is required regardless, since
    GraspGen needs its own conda env — see GraspGenPredictor.)"""
    if backend == 'graspgen':
        from sim_grasp import GraspGenPredictor
        return GraspGenPredictor(graspgen_python=graspgen_python).predict(
            depth, K, rgb=rgb, segmap=segmap)
    return _subprocess_predict(dict(depth=depth, K=K, rgb=rgb, segmap=segmap),
                               forward_passes, arg_configs, work_dir)


def predict_clouds_in_subprocess(pc_full_cam, pc_segments_cam, forward_passes,
                                 arg_configs, work_dir, backend='cgn',
                                 graspgen_python=None) -> GraspPrediction:
    """Cloud-mode subprocess prediction (P2 fusion: fused multi-camera cloud
    expressed in the primary camera frame)."""
    if backend == 'graspgen':
        from sim_grasp import GraspGenPredictor
        return GraspGenPredictor(graspgen_python=graspgen_python).predict_clouds(
            pc_full_cam, pc_segments_cam)
    payload = {'pc_full': np.asarray(pc_full_cam, dtype=np.float32)}
    for sid, pc in pc_segments_cam.items():
        payload[f'pcseg_{float(sid):g}'] = np.asarray(pc, dtype=np.float32)
    return _subprocess_predict(payload, forward_passes, arg_configs, work_dir)


def _subprocess_predict(payload, forward_passes, arg_configs,
                        work_dir) -> GraspPrediction:
    import subprocess
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    obs_f, out_f = work_dir / '_cgn_obs.npz', work_dir / '_cgn_out.npz'
    np.savez(obs_f, **payload)
    worker = Path(__file__).parent / 'sim_grasp' / 'cgn_worker.py'
    cmd = [sys.executable, str(worker), str(obs_f), str(out_f),
           '--forward-passes', str(forward_passes)]
    if arg_configs:
        cmd += ['--arg-configs', *arg_configs]
    r = subprocess.run(cmd)
    if r.returncode != 0 or not out_f.exists():
        raise RuntimeError(f'CGN worker failed (exit code {r.returncode})')
    parts = {'grasps': {}, 'scores': {}, 'contacts': {}, 'openings': {}}
    with np.load(out_f) as z:   # context manager: NpzFile keeps the file open
        for k in z.files:       # lazily; Windows can't unlink it otherwise
            kind, sid = k.split('_', 1)
            parts[kind][float(sid)] = z[k]
    obs_f.unlink(missing_ok=True)
    out_f.unlink(missing_ok=True)
    return GraspPrediction(grasps_cam=parts['grasps'], scores=parts['scores'],
                           contact_pts=parts['contacts'],
                           gripper_openings=parts['openings'])


def filter_feasible(grasps_cam, scores, openings, T_world_cam, table_height):
    """Table-collision filter (runs in world frame, returns camera frame)."""
    checker = GraspFeasibilityChecker(table_height=table_height)
    grasps_world = {k: transform_grasps(T_world_cam, np.asarray(G))
                    for k, G in grasps_cam.items()}
    kept_world, kept_scores, stats = checker.filter(grasps_world, scores, openings)
    T_cam_world = invert_se3(T_world_cam)
    kept_cam = {k: transform_grasps(T_cam_world, G) for k, G in kept_world.items()}
    return kept_cam, kept_scores, stats


# [P1] Shape-aware pick ordering: among objects still on the table in
# --pick-all, prefer parallel-sided shapes (cylinders/boxes — 9/14 and 9/16
# in the P1 verdict) over harder round ones (capsules 1/6, spheres 0/5) so
# clutter clears without bulldozing the objects most likely to escape anyway.
_GEOM_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
_SHAPE_PRIORITY = {
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): 0.06,
    int(mujoco.mjtGeom.mjGEOM_BOX): 0.04,
    int(mujoco.mjtGeom.mjGEOM_MESH): 0.0,
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): -0.03,
    int(mujoco.mjtGeom.mjGEOM_SPHERE): -0.06,
}


def _object_geom(model, body_name):
    """(body_id, first_geom_id, geom_type) for a named object body — objects
    have exactly one geom (primitive or mesh)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    gid = model.body_geomadr[bid]
    return bid, gid, int(model.geom_type[gid])


def _box_yaw_alignment_bonus(model, data, bid, gtype, closing_world, weight=0.15):
    """[P1] Bonus for grasps whose closing axis lines up with one of a box's
    horizontal edges, vs cutting across the diagonal (which tends to spin the
    box during closing). Neutral (0) for non-box shapes."""
    if gtype != _GEOM_BOX:
        return 0.0
    c_xy = closing_world[:2]
    n = np.linalg.norm(c_xy)
    if n < 1e-6:
        return 0.0
    c_xy = c_xy / n
    R_obj = data.xmat[bid].reshape(3, 3)
    best = 0.0
    for ax in range(3):
        a_xy = R_obj[:2, ax]
        an = np.linalg.norm(a_xy)
        if an < 0.3:   # this local axis is ~vertical (perpendicular to table)
            continue
        best = max(best, abs(float(np.dot(c_xy, a_xy / an))))
    return weight * (best - 0.85)


def rank_candidates(grasps_cam, scores, T_world_cam, model=None, data=None,
                    label_to_body=None):
    """All (rank_score, seg_id, idx) sorted best-first: CGN score plus
      - a bonus for downward approaches (near-horizontal ones bulldoze
        neighbors)
      - [P1] a shape-priority bonus (parallel-sided objects first)
      - [P1] a box-yaw alignment bonus (penalize diagonal closing across a
        box, which tends to spin it out)
    `model`/`data`/`label_to_body` are optional; pass all three to enable the
    P1 shape-aware terms (they contribute 0 if omitted)."""
    R_wc = T_world_cam[:3, :3]
    out = []
    for sid, s in scores.items():
        bid = gtype = None
        shape_bonus = 0.0
        if model is not None and label_to_body is not None:
            bid, _, gtype = _object_geom(model, label_to_body[int(sid)])
            shape_bonus = _SHAPE_PRIORITY.get(gtype, 0.0)
        for i in range(len(s)):
            T = np.asarray(grasps_cam[sid][i])
            approach_z = float((R_wc @ T[:3, 2])[2])
            rank = float(s[i]) + 0.25 * (-approach_z) + shape_bonus
            if data is not None and bid is not None:
                closing_world = R_wc @ T[:3, 0]
                rank += _box_yaw_alignment_bonus(model, data, bid, gtype,
                                                 closing_world)
            out.append((rank, sid, i))
    out.sort(key=lambda t: t[0], reverse=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=None, help='randomization seed')
    ap.add_argument('--n-objects', type=int, default=None,
                    help='fixed object count (default: random 1-10)')
    ap.add_argument('--forward-passes', type=int, default=1)
    ap.add_argument('--no-vis', action='store_true', help='skip Open3D window')
    ap.add_argument('--view-sim', action='store_true',
                    help='open the interactive MuJoCo viewer on the settled '
                         'scene before grasp prediction (close window to continue)')
    ap.add_argument('--no-feasibility', action='store_true',
                    help='skip table-collision grasp filtering')
    ap.add_argument('--save-dir', default=None,
                    help='output directory (default: output/<timestamp>)')
    ap.add_argument('--execute', action='store_true',
                    help='execute the best grasp(s) with the Panda (IK + '
                         'position control) and report pick success')
    ap.add_argument('--top-k', type=int, default=3,
                    help='with --execute: try up to K best grasps until one succeeds')
    ap.add_argument('--pick-all', action='store_true',
                    help='pick EVERY object and place it in the bin: re-observes '
                         'and re-runs Contact-GraspNet after each attempt (cleared '
                         'objects unblock occluded ones) until the table is empty, '
                         'each object failed 3x, or no grasps remain')
    ap.add_argument('--camera', choices=['calibrated', 'lookat', 'fused'],
                    default='calibrated',
                    help='observation camera setup: "calibrated" = real eye-to-hand '
                         'calibration (calibration_result.yaml); "lookat" = '
                         'generic angled look-at camera (the pre-calibration setup); '
                         '"fused" = BOTH (lookat primary + calibrated side cam, '
                         'point clouds fused in world frame — P2). '
                         'Use this to A/B compare Contact-GraspNet performance.')
    ap.add_argument('--backend', choices=['cgn', 'graspgen'], default='cgn',
                    help='grasp-prediction backend: "cgn" = Contact-GraspNet '
                         '(default), "graspgen" = NVlabs/GraspGen (needs the '
                         'graspgen_torch env — see README "GraspGen backend setup")')
    ap.add_argument('--graspgen-python', default=None,
                    help='path to the graspgen_torch env\'s interpreter; overrides '
                         'the GRASPGEN_PYTHON environment variable')
    ap.add_argument('--pick-object', type=int, default=None, metavar='SEG_ID',
                    help='only grasp THIS object (segmentation instance id, '
                         'see the printed per-object table / observation.png). '
                         'Works with --execute and --pick-all.')
    prompt_group = ap.add_mutually_exclusive_group()
    prompt_group.add_argument('--prompt', type=str, default=None,
                    help='select the target object by text description '
                         '(e.g. "the red box"), via SAM 3 on the rendered '
                         'RGB. Mutually exclusive with --click/--box/--pick-object.')
    prompt_group.add_argument('--click', type=str, default=None, metavar='X,Y',
                    help='select the target object by clicking a pixel '
                         '(observation.png coordinates), via SAM 3.')
    prompt_group.add_argument('--box', type=str, default=None, metavar='X1,Y1,X2,Y2',
                    help='select the target object by a pixel bounding box, via SAM 3.')
    ap.add_argument('--prompt-index', type=int, default=None, metavar='I',
                    help='with an ambiguous --prompt (multiple matches): pick '
                         'match #I from the printed ranked list')
    ap.add_argument('--category', type=str, default='a block',
                    help='text category for click-based whole-object detection '
                         "(e.g. 'a block', 'a cube'); only applies to --click, "
                         'not --prompt/--box (default: a block)')
    ap.add_argument('--sam3-python', default=None,
                    help='path to the sam3_torch env\'s python; overrides '
                         'the SAM3_PYTHON environment variable')
    ap.add_argument('--grasp-index', type=int, default=None, metavar='I',
                    help='with --execute: run candidate #I from the printed '
                         'ranked candidate list instead of auto-trying top-k')
    ap.add_argument('--clean-depth', action='store_true',
                    help='crop the depth map to the table workspace and remove '
                         'speckles before grasp prediction (sim_grasp/perception.py)')
    ap.add_argument('--recenter', action='store_true',
                    help='shift each executed grasp along its finger-closing '
                         'axis onto the target object cloud — counters the '
                         'dominant closed_on_air failure (lateral CGN offset)')
    ap.add_argument('--calibration', default='auto',
                    help='path to eye-to-hand calibration yaml; "auto" uses '
                         'calibration_result.yaml next to this script if present; '
                         '"none" uses the generic look-at camera (same as --camera lookat)')
    args = ap.parse_args()
    if args.pick_all:
        args.execute = True   # pick-all implies execution (thresholds, GPU prep)
    if args.pick_object is not None and (args.prompt or args.click or args.box):
        sys.exit('[prompt] --pick-object and --prompt/--click/--box are mutually '
                 'exclusive — both select a single target object, pick one mechanism.')
    if args.pick_all and (args.prompt or args.click or args.box):
        sys.exit('[prompt] --pick-all and --prompt/--click/--box are mutually '
                 'exclusive — promptable selection targets a single object; '
                 '"pick all objects matching X" is not supported.')

    save_dir = Path(args.save_dir) if args.save_dir else \
        Path(__file__).parent / 'output' / time.strftime('%Y%m%d_%H%M%S')
    vis = Visualizer(save_dir)
    print(f'[run] outputs -> {save_dir}')

    # ----------------------------------------------------------------- scene
    cfg = SceneConfig(seed=args.seed)
    if args.n_objects is not None:
        cfg.n_objects_range = (args.n_objects, args.n_objects)
    if args.camera == 'lookat':
        cfg.calibration_file = None
    elif args.camera == 'fused':
        cfg.calibration_file = None        # primary camera = lookat (camera A)
        side_cal = Path(__file__).parent / 'calibration_result.yaml' \
            if args.calibration in ('auto', 'none') else Path(args.calibration)
        if not side_cal.exists():
            sys.exit(f'[fusion] side-camera calibration yaml not found: {side_cal}')
        cfg.side_calibration_file = str(side_cal)
        print(f'[camera] FUSED: lookat primary + calibrated side cam ({side_cal})')
    elif args.calibration == 'auto':
        default_cal = Path(__file__).parent / 'calibration_result.yaml'
        cfg.calibration_file = str(default_cal) if default_cal.exists() else None
    elif args.calibration.lower() != 'none':
        cfg.calibration_file = args.calibration
    if cfg.calibration_file:
        print(f'[camera] using real eye-to-hand calibration: {cfg.calibration_file}')
    else:
        print(f'[camera] using generic look-at camera: pos={cfg.cam_pos}, '
              f'target={cfg.cam_target}')
    gen = SceneGenerator(cfg)

    t0 = time.time()
    model, data = gen.generate()
    on_table = gen.objects_on_table()
    print(f'[scene] {len(gen.object_names)} objects spawned, '
          f'{len(on_table)} on table after settling ({time.time() - t0:.1f}s)')

    # ------------------------------------------------- interactive sim viewer
    if args.view_sim:
        import mujoco.viewer
        print('[viewer] interactive MuJoCo window open — physics is live; '
              'close the window to continue the pipeline...')
        mujoco.viewer.launch(model, data)

    # --------------------------------------------------------------- capture
    cam = CameraModule(model, data, cam_name=cfg.cam_name, width=640, height=480)
    rgb, depth, segmap, K, T_world_cam = cam.capture(gen.object_body_ids)
    if args.clean_depth:
        n0 = int((depth > 0).sum())
        depth = clean_depth(depth, K, T_world_cam)
        print(f'[perception] workspace crop + speckle removal: '
              f'{n0 - int((depth > 0).sum())} px dropped')

    if args.prompt or args.click or args.box:
        from sim_grasp.prompt_selector import PromptSelector, resolve_real_label
        selector = PromptSelector(sam3_python=args.sam3_python)
        click_xy = tuple(float(v) for v in args.click.split(',')) if args.click else None
        box_xyxy = tuple(float(v) for v in args.box.split(',')) if args.box else None
        target_desc = f'click {click_xy}' if click_xy is not None else (args.prompt or args.box)
        if click_xy is not None:
            result = selector.click_to_select(rgb, click_xy, category=args.category)
        else:
            result = selector.select(rgb, prompt=args.prompt, box=box_xyxy)
        if result.is_empty:
            sys.exit(f'[prompt] no object matched: {target_desc!r}')
        if click_xy is not None:
            # a click already disambiguated a specific location -- unlike a
            # --prompt match spanning the whole image, there's no useful
            # index to ask the user for here. Match interactive_pick.py's
            # existing click-loop behavior: take the highest-scoring
            # candidate among whichever instance(s) contain the click.
            idx = int(np.argmax(result.scores))
        elif result.is_ambiguous and args.prompt_index is None:
            print(f'[prompt] {len(result.scores)} matches for '
                  f'{target_desc!r} — pass --prompt-index to disambiguate:')
            for i, (s, b) in enumerate(zip(result.scores, result.boxes)):
                print(f'  [{i}] score {float(s):.3f}  box {[round(float(v), 1) for v in b]}')
            sys.exit(1)
        else:
            idx = args.prompt_index if result.is_ambiguous else 0
        if not 0 <= idx < len(result.scores):
            sys.exit(f'[prompt] --prompt-index {idx} out of range (0..{len(result.scores) - 1})')
        mask = result.masks[idx]
        real_label = resolve_real_label(segmap, mask)
        if real_label is None:
            sys.exit('[prompt] resolved mask does not overlap any known object')
        new_segmap = np.zeros(rgb.shape[:2], dtype=segmap.dtype)
        new_segmap[mask] = real_label
        segmap = new_segmap
        print(f'[prompt] resolved to object {real_label}, score {float(result.scores[idx]):.3f}')

    visible_ids = sorted(int(s) for s in np.unique(segmap) if s > 0)
    print(f'[camera] K diag: fx={K[0,0]:.1f} fy={K[1,1]:.1f}; '
          f'visible object ids: {visible_ids}')
    vis.save_observation(rgb, depth, segmap)

    # ------------------------------------------- multi-camera fusion (P2)
    fused = args.camera == 'fused'
    side = None
    if fused:
        from sim_grasp.fusion import fuse_observations, clouds_to_camera
        side = CameraModule(model, data, cam_name=cfg.side_cam_name,
                            width=640, height=480)

        def capture_fused(primary_capture):
            """Fuse the primary capture with a fresh side-camera capture.
            Returns clouds in the PRIMARY camera frame. Depth is always
            workspace-cropped here: the fused cloud has no z_range filter
            downstream (the depth path applies one inside the predictor)."""
            rgb_p, d_p, seg_p, K_p, T_p = primary_capture
            rgb_s, d_s, seg_s, K_s, T_s = side.capture(gen.object_body_ids)
            obs_pair = [
                {'depth': clean_depth(d_p, K_p, T_p), 'K': K_p,
                 'T_world_cam': T_p, 'segmap': seg_p},
                {'depth': clean_depth(d_s, K_s, T_s), 'K': K_s,
                 'T_world_cam': T_s, 'segmap': seg_s}]
            pc_w, seg_w = fuse_observations(obs_pair)
            pc_c, seg_c = clouds_to_camera(pc_w, seg_w, T_p)
            print(f'[fusion] {len(pc_c)} pts from 2 cameras '
                  f'({len(seg_c)} objects with points)')
            return pc_c, seg_c

        pc_fused_cam, seg_fused_cam = capture_fused(
            (rgb, depth, segmap, K, T_world_cam))

    # ----------------------------------------------------- point cloud (info)
    pc_cam, pc_rgb = depth_to_pointcloud(depth, K, rgb=rgb)
    pc_world = transform_points(T_world_cam, pc_cam)
    print(f'[cloud] {pc_cam.shape[0]} points; world-z range '
          f'[{pc_world[:, 2].min():.3f}, {pc_world[:, 2].max():.3f}] m '
          f'(tabletop at {cfg.table_height} m)')

    # ------------------------------------------------------ Contact-GraspNet
    # --execute wants several candidates per object (top-k retry). On >=6GB
    # GPUs use more forward passes; on small GPUs (e.g. 4GB GTX 1650) that
    # OOMs, so lower the confidence thresholds instead.
    arg_configs = []
    if args.execute and args.forward_passes < 3:
        vram_gb = gpu_vram_gb()
        if vram_gb >= 6:
            print(f'[{args.backend}] --execute: raising forward_passes to 3 for denser candidates')
            args.forward_passes = 3
        else:
            print(f'[{args.backend}] --execute: small GPU ({vram_gb:.1f} GB) — lowering '
                  'confidence thresholds instead of extra forward passes')
            arg_configs = ['TEST.first_thres:0.14', 'TEST.second_thres:0.14']
    t0 = time.time()
    if args.pick_all:
        # pick-all re-runs the backend every round: keep torch OUT of this
        # process (see predict_in_subprocess) or MuJoCo rendering OOMs on
        # 8 GB RAM. For backend='graspgen' this isolation is required
        # regardless of --pick-all, since it needs its own conda env.
        print(f'[{args.backend}] pick-all: running in a subprocess per round...')
        predictor = None
        if fused:
            pred = predict_clouds_in_subprocess(pc_fused_cam, seg_fused_cam,
                                                args.forward_passes, arg_configs,
                                                save_dir, backend=args.backend,
                                                graspgen_python=args.graspgen_python)
        else:
            pred = predict_in_subprocess(depth, K, rgb, segmap,
                                         args.forward_passes, arg_configs, save_dir,
                                         backend=args.backend,
                                         graspgen_python=args.graspgen_python)
    elif args.backend == 'graspgen':
        print('[graspgen] loading GraspGen...')
        from sim_grasp import GraspGenPredictor
        predictor = GraspGenPredictor(graspgen_python=args.graspgen_python)
        pred = predictor.predict_clouds(pc_fused_cam, seg_fused_cam) if fused \
            else predictor.predict(depth, K, rgb=rgb, segmap=segmap)
    else:
        # in-process ContactGraspNetPredictor imports pyrender, which
        # creates its own OSMesa context via PyOpenGL -- if MuJoCo's
        # renderer (CameraModule, already constructed above) created ITS
        # OSMesa context first in this same process, PyOpenGL's context
        # bookkeeping breaks (`TypeError: unhashable type` deep in
        # OpenGL.contextdata, real repro: 2026-08-19). --pick-all already
        # avoids this by running CGN in a subprocess every round; do the
        # same here for the single-shot path instead of reordering
        # construction (which would still fail the moment MuJoCo needs to
        # render again, e.g. --view-sim or a second capture).
        print('[cgn] loading Contact-GraspNet (subprocess, avoids an OSMesa '
              "context conflict with MuJoCo's own renderer)...")
        predictor = None
        pred = predict_clouds_in_subprocess(
            pc_fused_cam, seg_fused_cam, args.forward_passes, arg_configs,
            save_dir, backend='cgn') if fused else predict_in_subprocess(
            depth, K, rgb, segmap, args.forward_passes, arg_configs,
            save_dir, backend='cgn')
    print(f'[{args.backend}] {pred.num_grasps} grasps in {time.time() - t0:.1f}s')

    # ------------------------------------------------- feasibility filtering
    grasps_cam, scores = pred.grasps_cam, pred.scores
    feas_stats = {'n_before': pred.num_grasps, 'n_after': pred.num_grasps, 'n_rejected': 0}
    if not args.no_feasibility and pred.num_grasps > 0:
        grasps_cam, scores, feas_stats = filter_feasible(
            grasps_cam, scores, pred.gripper_openings, T_world_cam, cfg.table_height)
        print(f"[feasibility] kept {feas_stats['n_after']}/{feas_stats['n_before']} "
              f"({feas_stats['n_rejected']} table-colliding/underhand rejected)")

    # ----------------------------------------------------------------- metrics
    per_object = {}
    for seg_id in sorted(grasps_cam.keys()):
        s = np.asarray(scores[seg_id])
        per_object[int(seg_id)] = {
            'num_grasps': int(len(s)),
            'best_score': float(s.max()) if len(s) else None,
            'mean_score': float(s.mean()) if len(s) else None,
        }
        print(f"  object {int(seg_id):2d}: {len(s):4d} grasps, "
              f"best {per_object[int(seg_id)]['best_score']}")

    best = None
    for seg_id, s in scores.items():
        if len(s) and (best is None or s.max() > best[2]):
            i = int(np.argmax(s))
            best = (int(seg_id), grasps_cam[seg_id][i], float(s[i]))

    metrics = {
        'backend': args.backend,
        'seed': args.seed,
        'objects_spawned': len(gen.object_names),
        'objects_on_table': len(on_table),
        'visible_objects': visible_ids,
        'num_grasps': int(sum(len(s) for s in scores.values())),
        'feasibility': feas_stats,
        'per_object': per_object,
        'T_world_cam': T_world_cam.tolist(),
        'camera_K': K.tolist(),
    }
    if best is not None:
        seg_id, T_cam_grasp, score = best
        T_world_grasp = T_world_cam @ T_cam_grasp
        metrics['best_grasp'] = {
            'object': seg_id, 'score': score,
            'T_cam_grasp': np.asarray(T_cam_grasp).tolist(),
            'T_world_grasp': T_world_grasp.tolist(),
        }
        print(f'[best] object {seg_id}, score {score:.3f}, world position '
              f'{np.round(T_world_grasp[:3, 3], 3).tolist()}')
    (save_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    # ------------------------------------------------------------ artifacts
    contact_pts = {k: pred.contact_pts.get(k, np.zeros((0, 3))) for k in grasps_cam}
    vis.save_predictions_npz(pc_cam, grasps_cam, scores, contact_pts, pc_rgb)
    print(f'[run] metrics + predictions saved to {save_dir}')

    # ------------------------------------------------------- grasp execution
    label_to_body = {lbl: gen.object_names[lbl - 1]
                     for lbl in gen.object_body_ids.values()}

    if args.pick_all:
        # Sequential pick-and-place of EVERY object: each round re-observes the
        # scene and re-runs CGN (the predictor stays loaded), so objects that
        # were occluded or scoreless in the first shot get fresh chances as the
        # table empties. GIF frames stream to disk to keep RAM flat.
        from sim_grasp.executor import GraspExecutor
        from sim_grasp.scene_generator import ARM_OBSERVE_QPOS

        rec_cam = CameraModule(model, data, cam_name=cfg.record_cam_name,
                               width=640, height=480)
        executor = GraspExecutor(model, data, camera_module=rec_cam,
                                 record_gif=True,
                                 record_dir=save_dir / '_gif_frames',
                                 gif_frame_interval=0.2)
        label_of = {name: i + 1 for i, name in enumerate(gen.object_names)}
        drop = gen.bin_drop_point()   # kept only as the legacy fallback target
        placement_planner = OccupancyPlacementPlanner(cfg.bin_center, cfg.bin_inner_half)
        n_total = len(on_table)
        fail_count: dict[str, int] = {}
        rounds_log = []
        max_rounds = n_total + 3   # each round costs a CGN subprocess (~30 s)
        cur = (grasps_cam, scores, T_world_cam)   # round 1 reuses initial CGN run
        obs = (depth, segmap, K)   # observation behind the current grasps
        low_thres = False          # last-resort CGN thresholds for hard objects
        # (this arg_configs-based escalation only affects --backend cgn; it's a
        # no-op for --backend graspgen, which has its own --grasp-threshold knob)

        for rnd in range(1, max_rounds + 1):
            if cur is not None:
                g_r, s_r, T_wc = cur
                cur = None
            else:
                executor.go_observe(ARM_OBSERVE_QPOS)
                rgb_r, depth_r, segmap_r, K_r, T_wc = cam.capture(gen.object_body_ids)
                if args.clean_depth:
                    depth_r = clean_depth(depth_r, K_r, T_wc)
                obs = (depth_r, segmap_r, K_r)
                cfgs_r = (['TEST.first_thres:0.08', 'TEST.second_thres:0.08']
                          if low_thres else arg_configs)
                if fused:
                    pc_f, seg_f = capture_fused(
                        (rgb_r, depth_r, segmap_r, K_r, T_wc))
                    pred_r = predict_clouds_in_subprocess(
                        pc_f, seg_f, args.forward_passes, cfgs_r, save_dir,
                        backend=args.backend, graspgen_python=args.graspgen_python)
                else:
                    pred_r = predict_in_subprocess(depth_r, K_r, rgb_r, segmap_r,
                                                   args.forward_passes, cfgs_r,
                                                   save_dir, backend=args.backend,
                                                   graspgen_python=args.graspgen_python)
                g_r, s_r = pred_r.grasps_cam, pred_r.scores
                if not args.no_feasibility and pred_r.num_grasps > 0:
                    g_r, s_r, _ = filter_feasible(g_r, s_r, pred_r.gripper_openings,
                                                  T_wc, cfg.table_height)

            in_bin_now = set(gen.objects_in_bin())
            remaining = [n for n in gen.objects_on_table()
                         if n not in in_bin_now and fail_count.get(n, 0) < 3]
            if args.pick_object is not None:        # P3: only the chosen one
                remaining = [n for n in remaining
                             if label_of[n] == args.pick_object]
            if not remaining:
                print(f'[pick-all] round {rnd}: nothing left to pick')
                break
            allowed = {label_of[n] for n in remaining}
            cand = [(rk, sid, i) for rk, sid, i in
                    rank_candidates(g_r, s_r, T_wc, model, data, label_to_body)
                    if int(sid) in allowed]
            if not cand:
                if not low_thres:
                    low_thres = True   # all later rounds keep the low thresholds
                    print(f'[pick-all] round {rnd}: no grasps for {remaining} — '
                          'retrying with lowered CGN thresholds (0.08)')
                    continue
                print(f'[pick-all] round {rnd}: no grasps for remaining objects '
                      f'{remaining} even at lowered thresholds — stopping')
                break

            _, sid, i = cand[0]
            body = label_to_body[int(sid)]
            score = float(s_r[sid][i])
            print(f'[pick-all] round {rnd}: picking object {int(sid)} ({body}), '
                  f'score {score:.3f} — {len(remaining)} object(s) remaining')
            T_world_grasp = T_wc @ np.asarray(g_r[sid][i])
            shift = 0.0
            if args.recenter:
                d_o, seg_o, K_o = obs
                pts_cam = depth_to_pointcloud(d_o, K_o, mask=(seg_o == int(sid)))
                T_world_grasp, shift = recenter_grasp(
                    T_world_grasp, transform_points(T_wc, pts_cam),
                    table_z=cfg.table_height)
                if shift:
                    print(f'[recenter] grasp shifted {shift * 1e3:+.1f} mm '
                          'along the closing axis')

            d_o, seg_o, K_o = obs
            footprint = compute_object_footprint(d_o, seg_o, int(sid), K_o, T_wc)
            place_pose = None
            if footprint is not None:
                heightmap = build_bin_heightmap(
                    d_o, seg_o, K_o, T_wc, cfg.bin_center, cfg.bin_inner_half,
                    exclude_seg_id=int(sid))
                place_pose = placement_planner.plan(footprint, heightmap)
            if place_pose is None:
                print('[placement] footprint/slot search failed for object '
                      f'{int(sid)} — falling back to the fixed bin drop point')
            # ground-truth grasp offset (sim-only diagnostic): where the
            # target body ACTUALLY is in the grasp frame at execution time —
            # x=closing, y=across fingers, z=approach (fingers sweep
            # z in [0.066, 0.112], so a well-centered grasp has z ~ 0.09)
            import mujoco as _mj
            bid_t = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, body)
            rel_t = data.xpos[bid_t] - T_world_grasp[:3, 3]
            gt_off = [round(float(rel_t @ T_world_grasp[:3, ax]), 4)
                      for ax in range(3)]
            res = executor.execute(T_world_grasp, target_body=body)
            entry = {'round': rnd, 'object': int(sid), 'body': body,
                     'score': score, 'recenter_shift_m': round(shift, 4),
                     'gt_offset_grasp_frame': gt_off, 'pick': res}
            if res['success']:
                if place_pose is not None:
                    release_z = compute_release_z(place_pose, T_world_grasp, footprint)
                    entry['place'] = executor.place(
                        place_pose.x, place_pose.y, release_z, place_pose.yaw)
                else:
                    entry['place'] = executor.place(
                        drop[0], drop[1], drop[2] + PLACE_RELEASE)
                entry['in_bin'] = body in gen.objects_in_bin()
                if entry['in_bin']:
                    print(f"[pick-all]   pick OK (raised {res['object_raised_m']} m)"
                          f' -> placed in bin')
                else:
                    fail_count[body] = fail_count.get(body, 0) + 1
                    print(f'[pick-all]   pick OK but object missed the bin '
                          f'(attempt {fail_count[body]}/3 for {body})')
            else:
                fail_count[body] = fail_count.get(body, 0) + 1
                print(f"[pick-all]   pick FAILED ({res.get('stage')}, raised "
                      f"{res.get('object_raised_m', 'n/a')}) — attempt "
                      f'{fail_count[body]}/3 for {body}')
            rounds_log.append(entry)

        in_bin = gen.objects_in_bin()
        left = [n for n in gen.objects_on_table() if n not in in_bin]
        fell = [n for n in gen.object_names if n not in in_bin and n not in left]
        print(f'[pick-all] DONE: {len(in_bin)}/{n_total} objects in the bin '
              f'{in_bin}; left on table: {left if left else "none"}; '
              f'knocked off table: {fell if fell else "none"}')
        executor.save_gif(save_dir / 'execution.gif')
        rec_cam.close()
        metrics['pick_all'] = {'objects_total': n_total, 'in_bin': in_bin,
                               'left_on_table': left, 'fell_off_table': fell,
                               'rounds': rounds_log}
        (save_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        print(f'[pick-all] video saved: {save_dir / "execution.gif"}')

    elif args.execute and any(len(s) for s in scores.values()):
        from sim_grasp.executor import GraspExecutor

        # Free Contact-GraspNet before execution: the model + CUDA allocator
        # caches hold gigabytes of Windows commit, and on 8 GB machines the
        # GIF frame buffer below then exhausts memory (numpy ArrayMemoryError).
        contact_pts = None
        del predictor, pred
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        cand = rank_candidates(grasps_cam, scores, T_world_cam, model, data,
                              label_to_body)
        if args.pick_object is not None:           # P3: user-selected target
            cand = [c for c in cand if int(c[1]) == args.pick_object]
            if not cand:
                sys.exit(f'[pick-object] no feasible grasps for object '
                         f'{args.pick_object} (available: '
                         f'{sorted(int(k) for k in grasps_cam)})')
        # P4: numbered candidate list — pick one with --grasp-index I
        print(f'[candidates] top {min(len(cand), 10)} of {len(cand)} '
              '(index | object | cgn score | world pos | approach):')
        for j, (rk, sid_, i_) in enumerate(cand[:10]):
            Tg = T_world_cam @ np.asarray(grasps_cam[sid_][i_])
            print(f'  [{j}] obj {int(sid_)}  score {float(scores[sid_][i_]):.3f}  '
                  f'pos {np.round(Tg[:3, 3], 3).tolist()}  '
                  f'approach {np.round(Tg[:3, 2], 2).tolist()}')
        if args.grasp_index is not None:
            if not 0 <= args.grasp_index < len(cand):
                sys.exit(f'[grasp-index] {args.grasp_index} out of range '
                         f'(0..{len(cand) - 1})')
            cand = [cand[args.grasp_index]]
        else:
            cand = cand[:args.top_k]
        ranked = [(int(sid), grasps_cam[sid][i], float(scores[sid][i]))
                  for _, sid, i in cand]

        # snapshot the settled state so each attempt starts identically
        qpos0, qvel0, ctrl0 = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
        # record the GIF from the close-up side camera, not the observation
        # camera — the top-down view shows nothing of the finger-object contact
        rec_cam = CameraModule(model, data, cam_name=cfg.record_cam_name,
                               width=640, height=480)
        executor = GraspExecutor(model, data, camera_module=rec_cam,
                                 record_gif=True,
                                 record_dir=save_dir / '_gif_frames')
        exec_results = []
        for attempt, (sid, T_cam_grasp, score) in enumerate(ranked, 1):
            data.qpos[:], data.qvel[:], data.ctrl[:] = qpos0, qvel0, ctrl0
            import mujoco as _mj; _mj.mj_forward(model, data)
            T_world_grasp = T_world_cam @ np.asarray(T_cam_grasp)
            shift = 0.0
            if args.recenter:
                pts_cam = depth_to_pointcloud(depth, K, mask=(segmap == int(sid)))
                T_world_grasp, shift = recenter_grasp(
                    T_world_grasp, transform_points(T_world_cam, pts_cam),
                    table_z=cfg.table_height)
                if shift:
                    print(f'[recenter] grasp shifted {shift * 1e3:+.1f} mm '
                          'along the closing axis')

            footprint = compute_object_footprint(depth, segmap, int(sid), K, T_world_cam)
            place_pose = None
            if footprint is not None:
                heightmap = build_bin_heightmap(
                    depth, segmap, K, T_world_cam, cfg.bin_center,
                    cfg.bin_inner_half, exclude_seg_id=int(sid))
                place_pose = OccupancyPlacementPlanner(
                    cfg.bin_center, cfg.bin_inner_half).plan(footprint, heightmap)
            body = label_to_body[int(sid)]
            print(f'[execute] attempt {attempt}/{len(ranked)}: object {int(sid)} '
                  f'({body}), score {score:.3f}')
            res = executor.execute(T_world_grasp, target_body=body)
            res.update(object=int(sid), score=score, recenter_shift_m=round(shift, 4))
            if res['success']:
                if place_pose is not None:
                    release_z = compute_release_z(place_pose, T_world_grasp, footprint)
                    res['place'] = executor.place(
                        place_pose.x, place_pose.y, release_z, place_pose.yaw)
                else:
                    drop = gen.bin_drop_point()
                    res['place'] = executor.place(
                        drop[0], drop[1], drop[2] + PLACE_RELEASE)
                res['in_bin'] = body in gen.objects_in_bin()
            exec_results.append(res)
            print(f'[execute]   -> {res}')
            if res['success']:
                print(f'[execute] PICK SUCCESS on attempt {attempt} '
                      f"(object raised {res['object_raised_m']} m)")
                if res['in_bin']:
                    print('[execute] placed in bin')
                else:
                    print('[execute] picked but missed the bin '
                          f"(place stage: {res['place']['stage']})")
                break
        executor.save_gif(save_dir / 'execution.gif')
        rec_cam.close()
        metrics['execution'] = exec_results
        (save_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        print(f'[execute] video saved: {save_dir / "execution.gif"}')

    # ------------------------------------------------------- 3D visualization
    if not args.no_vis:
        print('[vis] opening Open3D window (close it to exit)...')
        vis.show_grasps(pc_cam, grasps_cam, scores, pc_colors=pc_rgb)

    cam.close()


if __name__ == '__main__':
    main()
