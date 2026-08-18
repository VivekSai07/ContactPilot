# ContactPilot Grasping Pipeline — Roadmap

Guiding principle: **maximum practical grasp reliability** for a Franka Panda
on unseen tabletop objects with calibrated RGB-D — modular, reproducible,
deployable. No novelty for novelty's sake.

## P0 — Dockerized, reproducible pipeline  [IN PROGRESS]
- [x] `Dockerfile` + `docker-compose.yml` + `requirements-docker.txt` at repo root
- [x] Single-command build (`docker compose build`) and run (`docker compose up`)
- [x] GPU support: CUDA 12.8 image, torch cu128 wheels — one image covers
      GTX 1650 (sm_75) AND RTX 5090 (sm_120, Blackwell)
- [x] Docs: `DOCKER.md`
- [ ] Build verified on this laptop (first build downloads ~4 GB)
- [ ] Build verified on the RTX 5090 workstation

## P1 — Maximize grasp success rate  [IN PROGRESS]
- [x] Batch evaluation harness: `mujoco_grasp_sim/benchmark.py` (N seeds ×
      config → success-rate table; every change must move this number)
- [x] Failure taxonomy from batch runs: `mujoco_grasp_sim/analyze_failures.py`
      → taxonomy.json. **Baseline (seeds 0-4, pick-all, lookat, 2026-06-11):
      15/40 binned (38%)**; failures: closed_on_air 32 (78%!), missed_bin 4,
      knocked_off_table 4, ik 1. Initial-obs grasp coverage 17/40 objects.
- [x] Perception: workspace crop + speckle removal (`sim_grasp/perception.py`,
      `--clean-depth`); table segmentation deferred (sim segmaps are perfect —
      lab item for the RealSense)
- [x] Execution: 3-axis grasp re-centering onto the object cloud
      (`recenter_grasp()`, `--recenter`) + ground-truth offset logging
      (`gt_offset_grasp_frame` per attempt). Bin release height 0.17→0.15.
- **A/B verdict (2026-06-11, 2 iterations):** re-centering does NOT move the
  number (38% → 38% → 32%, within CGN stochastic noise). Ground truth shows
  why: position offset does not separate success from failure (|x| median
  4.9 vs 5.7 mm, |y| 10.1 vs 8.9 mm — both groups well-centered). Real
  failure structure is SHAPE-dependent dynamic escape during closing:
  cylinders 9 ok/5 fail, boxes 9/16, capsules 1/6, spheres 0/5. All grasps
  necessarily pinch the object's top sliver (z≈110 of the 66-112 mm sweep —
  the table blocks deeper approaches), so curved objects squirt out and boxes
  likely rotate out when the closing line crosses a diagonal. Keep
  `--recenter`/`--clean-depth` opt-in (harmless, help on real-sensor noise).
- [x] P1 levers implemented & validated (2026-06-14, seeds 0-4, pick-all,
      fused camera, 5-10 mixed-shape objects/scene): grasp-yaw-vs-object-face
      alignment bonus in ranking (`_box_yaw_alignment_bonus`), two-phase
      gentle closing in executor (fast approach to gripper ctrl=75, then slow
      squeeze to 0), fingertip-pad friction audit (torsional/rolling friction
      "1.0 0.01 0.004" patched onto the 5 `fingertip_pad_collision_*` geoms in
      panda.xml — **later found to be a no-op, see condim fix below**),
      shape-aware ranking bonus (`_SHAPE_PRIORITY`: cylinder >
      box > mesh > capsule > sphere). **Result: 21/40 binned (52%)**, up from
      17/40 (42%) fused baseline (+10pp). Knocked-off-table 10→7. Initial-obs
      grasp coverage 25/40 (vs 28/40 baseline — reordering changes WHICH
      grasps are attempted first, not raw coverage).
- [x] Box-only objects + fixed object count (2026-06-14): scenes now spawn
      exactly **3 box/cuboid objects** (`SceneConfig.n_objects_range=(3,3)`,
      `use_meshes=False`) at random reachable XY positions
      (`_sample_xy_positions`, unchanged). Cylinders/spheres/capsules/meshes
      removed — motivated by failures observed in lab testing concentrating
      on cylindrical objects during picking and the transit-to-bin transfer.
      5-seed fused pick-all (on top of the P1 levers above):
      **10/15 binned (67%)**, knocked-off-table **0** (was 7-10),
      initial-obs grasp coverage 14/15 objects, all 5 remaining failures at
      the `done` stage (closed-on-air / drop-in-transit class). `--n-objects
      N` or `SceneConfig(use_meshes=True, ...)` restore the old
      varied-object/varied-count behavior.
- [x] **Friction audit fix — condim=4 (2026-06-14):** user observed grasped
      cuboids slowly slipping out of the gripper during lift and during the
      transit-to-bin move. Root cause: the original friction-audit patch set
      `friction="1.0 0.01 0.004"` on the fingertip pads but never raised
      `condim` above MuJoCo's default of 3. With condim=3 only `friction[0]`
      (sliding) enters the contact's friction cone — and 1.0 is the
      unpatched global default too, so the patch changed *nothing*;
      torsional friction[1] was a dead value regardless of its size. Fixed
      by patching `friction="1.5 0.02 0.004" condim="4"` onto the 5
      `fingertip_pad_collision_*` geoms in panda.xml — this raises sliding
      friction above the rigid-body default AND activates torsional
      friction, which resists the off-center-grasp twisting torque that
      causes the slow walk-out under lift/transit acceleration. 5-seed fused
      pick-all (box-only/3-objects, on top of all prior P1 levers):
      **14/15 binned (93%)**, up from 10/15 (67%) — knocked-off-table stays
      at 0. The single remaining miss (seed 0) had only 2/3 objects with
      any feasible grasps to begin with (perception-limited, not grip-limited).
- [x] **GraspGen backend added + decisive comparison (2026-08-13, seeds 0-4,
      pick-all, fused camera, box-only/3-objects):** NVlabs/GraspGen wired in
      as a second grasp-prediction backend through the `GraspPredictor` ABC
      extension point (`sim_grasp/graspgen_predictor.py`, run as a subprocess
      via a separate `GRASPGEN_PYTHON` env — its own conda env, `graspgen_torch`
      — same off-process-VRAM pattern as the CGN worker), with `--backend
      {cgn,graspgen}` / `--graspgen-python` added to `run_sim_grasp_test.py`
      and `benchmark.py`. Run on the exact same seeds/camera/scene config as
      the recorded CGN baseline above (14/15 binned, 93%): **GraspGen scores
      15/15 binned (100%)**, knocked-off-table 0, initial-obs grasp coverage
      15/15 objects. Failure taxonomy (`analyze_failures.py`, 5 failure
      events across 20 pick attempts): `ik_unreachable` 4 (80%), `missed_bin`
      1 (20%) — zero `closed_on_air`, a full shift away from CGN's dominant
      failure mode (78% `closed_on_air` at the original 2026-06-11 baseline).
      Mean wall time 136s/run (5-seed batch, includes a fresh GraspGen
      subprocess inference per pick-all round). GraspGen is now the
      recommended backend for further P1 reliability work.
- [ ] Filtering: neighbor-object collision check (table collision exists),
      workspace-reachability pre-filter

## P2 — Multi-camera perception  [DONE 2026-06-11 except A/B verdict]
- New calibrated side camera available: `mujoco_grasp_sim/calibration_result.yaml`
  (T_cam_to_base, TSAI, 41 samples — beside the Franka at base-frame
  (0.025, 0.283, 0.647), looking down ~50°; replaces the old top-down pose)
- [x] Scene supports a second camera (`side_cam`, emitted when
      `SceneConfig.side_calibration_file` is set; placement from any yaml)
- [x] Point-cloud fusion in world frame: `sim_grasp/fusion.py`
      (voxel dedup, per-object id-based segmap fusion)
- [x] Fused cloud → CGN: `ContactGraspNetPredictor.predict_clouds()` +
      cloud-mode `cgn_worker` (npz keys pc_full / pcseg_<sid>); grasps return
      in the primary camera frame, downstream unchanged. `--camera fused`.
      Validated seed 5: 91k fused pts, 7/7 objects with points (vs 17/40
      initial-obs coverage single-cam in the baseline bench), first-try-class
      pick on a previously always-failing cylinder.
- [x] A/B evaluation single vs fused via the P1 harness (seeds 0-4,
      pick-all): **fused 17/40 binned (42%) vs single 15/40 (38%)** — and
      the real, low-noise win is PERCEPTION: initial-observation grasp
      coverage **28/40 vs 17/40 objects (+65%)**, pick success/attempt 23/52
      vs 19/52, zero IK failures. The binned rate barely moves because the
      bottleneck is execution dynamics (see P1 verdict), not perception;
      knocked_off rose 4→10 since fusion unlocks attempts on harder,
      previously-occluded objects. Fusion is the right default for the lab
      (occlusion robustness); pair it with the P1 execution levers.

## P3 — User-directed object selection ("pick THIS object")  [DONE 2026-06-11]
- [x] Select target by segmentation instance id: `--pick-object SEG_ID`
      (works in --execute and --pick-all; errors with the list of available
      ids if the target has no feasible grasps)
- [ ] Click-on-RGB selection (lab item — sim ids are printed/visualized)
- Keep interface compatible with future language-conditioned selection

## P4 — User-guided grasp selection  [DONE 2026-06-11]
- [x] Index-based CLI browse/select: --execute prints a ranked candidate
      table (object, score, world pos, approach); `--grasp-index I` executes
      exactly that candidate. Validated seed 5: --pick-object 6
      --grasp-index 0 → pick success.
- [ ] Open3D click-picking (optional polish)

## P5 — Promptable object selection (Meta SAM 3)  [BASELINE RECORDED 2026-08-15]
- [x] `PromptSelector` (`sim_grasp/prompt_selector.py`) wraps SAM 3 text/
      click/box prompting, run as a subprocess via a separate `SAM3_PYTHON`
      env (its own conda env, `sam3_torch`) — same off-process-VRAM pattern
      as the CGN/GraspGen workers. `--prompt`/`--click`/`--box` flags added
      to `run_sim_grasp_test.py`; `rgb_to_color_name` (`sim_grasp/color_utils.py`)
      derives ground-truth color names from `model.geom_rgba` for benchmark
      grading only (never fed into the selection pipeline).
- [x] **Decisive accuracy benchmark (2026-08-15, seeds 0-4,
      `benchmark_prompt_selection.py --seeds 0-4 --tag baseline`):** for each
      seed, one on-table object's real spawned color drives a genuine
      ground-truth prompt ("the {color} box"), SAM 3's best-scoring match is
      graded against that object's ground-truth segmap mask via IoU
      (`matched` = IoU > 0.5). Result: **3/5 correct selections (60%)**,
      mean IoU **0.733** over the 4 seeds SAM 3 returned any match for.
      Per-seed: seed 0 "brown" → IoU 0.983 (correct); seed 1 "yellow" → SAM 3
      returned 0 matches (ungraded, counts as incorrect); seed 2 "blue" → IoU
      0.975 (correct, despite 3 candidate matches); seed 3 "blue" → IoU 0.973
      (correct); seed 4 "pink" → IoU 0.0 (incorrect — best match locked onto
      the wrong object despite 2 candidates). Full detail in
      `mujoco_grasp_sim/output/bench_prompt_baseline/summary.json`. Takeaway:
      when SAM 3 finds the object, localization is excellent (mean IoU ~0.98
      on hits); the failure mode is prompt/color-name mismatches (missed
      "yellow" entirely, mis-selected on "pink") rather than segmentation
      quality — worth revisiting the color-name vocabulary and/or prompt
      phrasing before this is trusted as a primary selection path.
- [x] **Whole-object click selection fix (2026-08-18):** `--click`'s original
      implementation fed the click to SAM 3 as a small geometric-exemplar
      box, which only segments the locally-clicked face of an object —
      measured IoU **~0.29** against the true full-object mask (a real bug:
      a grasp predictor fed only a thin surface sliver produces
      push-not-grasp poses, not pick poses). Growing the box made it WORSE,
      not better. Fix: run a generic category text-detection pass (`"a
      block"`) for genuine whole-object instance masks (measured **IoU
      0.89–0.99** across every object in a real scene), then use the click
      only to pick which detected instance was meant — click as
      disambiguation, not a segmentation input
      (`PromptSelector.click_to_select`, `filter_selection_by_click`,
      `sim_grasp/prompt_selector.py`). This does **not** invalidate the
      decisive-accuracy-benchmark numbers above (60% / IoU 0.733) — that
      benchmark exercises the `--prompt` (text) path, not `--click`.

## P6 — Interactive live pick (click-to-select, live execution)  [DONE 2026-08-18]
- [x] `interactive_pick.py`: opens a live camera window, click an object to
      select it (via the P5 whole-object click-selection path above), SAM 3
      resolves and shows the mask for confirmation, then GraspGen/CGN
      predicts a grasp and the pick executes live in the same window
      (`sim_grasp/live_viewer.py`'s `LiveViewer`, `GraspExecutor`'s optional
      `on_frame` callback). Design:
      `docs/superpowers/plans/2026-08-15-interactive-pick.md`.
- [x] Retries the top-3 scoring grasps if the first IK-to-pregrasp attempt
      doesn't converge; pre-warms the CGN predictor in the background during
      the idle mask-review pause (GraspGen isn't pre-warmed — always a fresh
      subprocess per call, warming early would just double the cost).
- [x] **Real crash found & fixed (2026-08-16):** an unthrottled cv2 redraw
      loop (showing a "still working" status banner during SAM3/GraspGen's
      5-30s subprocess calls) segfaulted in Mesa's software rasterizer
      (`swrast_dri.so`) under this machine's WSL2 GPU passthrough, taking
      down the whole VM — confirmed via `journalctl -k` showing the segfault
      right after a burst of `dxg` ioctl failures, while a concurrent CUDA
      subprocess was also running. Fixed by throttling the redraw rate to
      ~10-30Hz (still smooth for a status banner, far less driver pressure)
      — see `/memories/repo/wsl2-gpu-display.md` for the general lesson.
- [x] **Place-in-bin gap found & fixed (2026-08-18):** both
      `interactive_pick.py` and `run_sim_grasp_test.py`'s single-object
      `--execute` mode stopped right after lifting the object —
      `--pick-all` was the only mode that carried it to the bin. Fixed by
      wiring both call sites to `GraspExecutor.place()` +
      `SceneGenerator.objects_in_bin()` (already proven by `--pick-all`),
      reported the same way (`res['place']`, `res['in_bin']`). Live-verified:
      click → pick → carry to bin → release, human-confirmed.

## Current state (2026-08-18)
Working end-to-end in MuJoCo sim, two grasp backends (CGN, GraspGen — P1) and
four object-selection paths (`--pick-object` ids, `--grasp-index` candidate
browsing, SAM 3 `--prompt`/`--click`/`--box`, P3-P5): scene gen → RGB-D +
segmap (single or fused, P2) → grasp prediction (subprocess, 8 GB RAM safe) →
feasibility filter → ranked execution (diff-IK) → pick → place-in-bin →
re-observe loop (`--pick-all`), or a single click-to-pick with live visual
feedback and the same pick → place-in-bin ending (`interactive_pick.py`, P6).
GraspGen is the recommended backend for further P1 reliability work (100%
binned vs CGN's 93% on the current box-only/3-object scene config, see P1
2026-08-13 entry). Camera A/B: top-down calibrated is hard mode for CGN
(sparse, low scores) vs inclined lookat/fused (dense, higher success) — lab
camera remounted inclined (P2).
