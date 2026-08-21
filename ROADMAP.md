# ContactPilot Grasping Pipeline — Roadmap

Guiding principle: **maximum practical grasp reliability** for a Franka Panda
on unseen tabletop objects with calibrated RGB-D — modular, reproducible,
deployable. No novelty for novelty's sake.

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
- [x] **Short-object finger-table collision fix (2026-08-21):**
      `GraspFeasibilityChecker.is_feasible()` validated the originally
      predicted grasp pose, but `GraspExecutor.execute()` advances the
      actually-executed pose `EXTRA_APPROACH` (12mm) further along the
      approach axis before closing, unvalidated — for short objects
      (already-tight table clearance), a plausible direct cause of
      fingers hitting the table. Found via the same live-testing round
      that led to the placement-pose-robustness fixes above; root-caused
      then deferred (see
      `docs/research/2026-08-21-short-object-finger-table-collision.md`),
      now fixed: `GraspFeasibilityChecker` gained an `extra_approach`
      constructor param (default 0.0, backward compatible),
      `run_sim_grasp_test.py` now constructs it with
      `extra_approach=EXTRA_APPROACH` so the feasibility check validates
      the pose actually closed on. Before/after on identical raw predicted
      grasps (5 seeds, box-only/3-objects, fused, GraspGen): **27/1687
      grasps (1.6%) newly rejected**, concentrated on 4 of 5 seeds.
      Regression check (`benchmark.py --seeds 0-4 --mode pick-all`, fix
      active): **15/15 binned (100%)**, 0 knocked off table — matches the
      last recorded GraspGen baseline, so the extra rejections cost no
      throughput. Sample size is small and not deliberately biased toward
      short objects, so 1.6% is a lower bound, not an isolated measurement
      of the short-object case specifically — the qualitative result (a
      real, previously-unvalidated 12mm blind spot is now checked) is the
      more meaningful takeaway.
- [x] **DiffIK elbow-flip fix (2026-08-21), found via live/manual testing
      feedback ("some poses are reached so weirdly"):**
      `DiffIK.solve()` tries 4 seeds per solve (the caller's actual current
      joint config, a canonical elbow-down pose, two random perturbations)
      and picked whichever *converged* result had the lowest position
      error — with no penalty for jumping to a very different
      elbow/shoulder posture. Since "converged" already means "good
      enough" (within 2x the tight tolerance), this silently discarded
      perfectly fine continuity with the arm's current pose whenever a
      differently-postured seed happened to be marginally more accurate,
      causing a visible elbow flip between waypoints (pre-grasp → grasp →
      lift → transit → release) within the same pick sequence. Fixed:
      extracted the seed-selection decision into
      `_pick_best_seed_result()` — if the current-pose-seeded (`q_init`)
      result converges at all, it wins unconditionally; only fall through
      to comparing the other seeds if `q_init` itself fails to converge —
      as a side effect, `solve()`'s restart loop now breaks as soon as ANY
      seed converges (previously only when the best-so-far result was
      *tight*, not merely converged), which is both simpler and cheaper
      (fewer wasted IK solves in the common case). Regression-tested with 5 synthetic scenarios in the new
      `test_executor_seed_selection.py` (tight convergence, loose
      convergence with a more-accurate competitor present, q_init failing
      entirely, nothing converging, single-result list), mirroring how
      `_ease()` was already pulled out of `_step_to` for unit-testability
      without a live MuJoCo model. **Validation (2026-08-21, same 10-seed
      GraspGen/fused config as the last recorded baseline): 30/30 binned
      (100%)**, 0 knocked off table — matches the prior
      placement-pose-robustness baseline exactly, so the fix costs no
      throughput. Visually confirmed the actual symptom is gone: sampled
      7 frames spanning a full 5-round `--pick-all` sequence (seed 2, the
      run with the most retries) and the elbow stayed in a consistent
      posture throughout, with no flip to a mirrored/different
      configuration at any waypoint transition.
- [x] **Smooth the retract-then-observe transition (2026-08-21), found via
      live/manual testing feedback ("it feels like a state
      transition/machine" going home between picks):** between
      `--pick-all` rounds, `place()`'s final open-handed retract move and
      `go_observe()`'s move to the fixed `ARM_OBSERVE_QPOS` were both
      linear (`_ease(smooth=False)`) — each has an instantaneous velocity
      jump at its start, so two consecutive full-speed-from-a-dead-stop
      motions in a row visually read as separate abrupt "snaps" rather
      than one continuous reach. Fixed: applied the same `smooth=True`
      easing already used for `place()`'s object-carrying motions to both
      of these — retract now decelerates to zero velocity exactly where
      `go_observe()`'s move picks up, so the pair reads as one fluid
      decelerate-then-reaccelerate. No hold durations were touched (those
      serve object-settling physics and capture-accuracy margin, out of
      scope for this fix — see the deferred idea below).
      **Validation (2026-08-21, same 10-seed GraspGen/fused config as the
      last recorded baseline): 30/30 binned (100%)**, 0 knocked off
      table — matches the DiffIK elbow-flip fix's own baseline exactly,
      confirming this change (which only alters the SHAPE of the joint
      trajectory between waypoints, never which waypoints are targeted)
      costs no throughput. Visual check: sampled densely-spaced frames
      around a round transition (seed 0, release → retract → go_observe →
      settle) and saw stable, consistent arm positioning with no visible
      discontinuity or odd artifact — though unlike the elbow-flip fix's
      more binary "did it flip" signature, confirming *smoothness* itself
      from low-fps stills alone has real limits; the benchmark's clean
      regression result is the stronger piece of evidence here.
- [ ] Skip the fixed `ARM_OBSERVE_QPOS` detour entirely when it isn't
      actually needed for camera-frustum clearance for the next target —
      a more thorough follow-up to the transition-smoothing fix above,
      considered during the same brainstorming session but deferred as
      more invasive (requires a real geometric check of what "in the
      frustum" means) and better proven-in-need after this smoothing fix
      is confirmed sufficient in practice. Note: `go_observe()` isn't only
      about frustum clearance — it also resets the arm to a known posture
      that becomes the IK seed (`q_now`) for the next pick's pre-grasp
      solve, which directly interacts with the DiffIK elbow-flip fix above
      (continuity-first seed selection). Skipping the detour would change
      what that seed is, so this follow-up needs to account for IK-seeding
      behavior, not just visibility.
- [ ] Audit the remaining linear (non-`smooth`) `_step_to()` call sites in
      `GraspExecutor.execute()` (the pre-grasp→grasp→lift sequence) for the
      same abrupt-junction pattern just fixed for the retract→observe
      transition — in particular the pre-grasp move directly into the
      grasp-approach move (no hold between them, both linear today) is the
      clearest remaining instance and may be more visually noticeable than
      the one just fixed, since both are large arm motions. Not fixed here
      because these motions have their own established anti-slip rationale
      (2026-06-14 P1 friction/gentle-closing work) that a smoothing change
      should be validated against separately, not bundled into this fix.
- [ ] Filtering: neighbor-object collision check (table collision exists),
      workspace-reachability pre-filter
- [ ] Nullspace redundancy resolution for DiffIK (continuous joint-space
      bias toward the current posture throughout DLS iteration, not just
      at seed selection) — considered as a follow-up to the elbow-flip fix
      above but deferred pending confirmation the seed-selection fix alone
      is sufficient in practice. The concrete residual case, if this turns
      out to be needed: the elbow-flip fix only guarantees continuity when
      `q_init` itself converges — if `q_init` fails (the baseline already
      shows occasional `ik_unreachable` pre-grasp retries) `solve()` falls
      through to the canonical elbow-down seed, and a posture flip is
      still possible there. Watch for this specifically in `place()`'s
      hover-target solve (`executor.py`'s `self.ik.solve(T_pre, q_now)`),
      where the target can be far enough from the current pose for `q_init`
      to fail outright.

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

## P7 — Intelligent bin placement (vision-only)  [DONE 2026-08-21]
- [x] Problem: `--pick-all`/`--execute`/`interactive_pick.py` all released
      every object at the same hardcoded world point
      (`SceneGenerator.bin_drop_point()`) at a fixed release height
      (`executor.PLACE_RELEASE`) — no XY offset between placements (2nd/3rd
      object risks landing on the 1st) and no per-object height accounting
      (a tall object could jam against the bin floor). Fix must be
      vision-only (depth/segmap/K/T_world_cam only, no MuJoCo internal-state
      queries) so it ports to a real RealSense camera later. Design:
      `docs/superpowers/specs/2026-08-20-intelligent-bin-placement-design.md`,
      plan: `docs/superpowers/plans/2026-08-20-intelligent-bin-placement.md`.
- [x] New `sim_grasp/placement_planner.py`: `compute_object_footprint()`
      (world XY size/yaw/height of the object about to be placed, from its
      own segmap mask) + `build_bin_heightmap()` (top-down occupancy grid of
      the bin's current contents) → `OccupancyPlacementPlanner.plan()`
      (free-space + 4-yaw-offset search over the heightmap) →
      `compute_release_z()` (release height from the *measured* grasp
      offset, not a fixed constant). Wired into all three entry points
      (`run_sim_grasp_test.py`'s `--pick-all` and single `--execute` paths,
      `interactive_pick.py`); `GraspExecutor.place()` signature changed from
      `place(drop_pos)` to `place(x, y, release_z, yaw=0.0)`.
- [x] **Real bug found & fixed during implementation:** the plan's own
      literal code for `build_bin_heightmap` used `np.maximum.at()` on a
      `NaN`-initialized grid — `np.maximum` propagates NaN (unlike
      `np.fmax`), so every touched cell degenerated to NaN and silently
      discarded every real height measurement. Fixed by initializing with
      `-inf` and checking `np.isfinite()` instead. Verified by reproducing
      both failure modes directly (the brief's original code fails its own
      test; the fix passes).
- [x] **Validation (2026-08-21, GraspGen backend, fused camera, seeds 0-9,
      box-only/3-object scenes — same config as the P1 100% baseline):**
      **29/30 binned (96.7%)**, zero knocked-off-table, zero stacking- or
      crushing-type failures across all 10 seeds. The 3 failures (all
      `missed_bin`, all in one seed) were confirmed non-systematic: (a) a
      same-day re-run of that exact seed with no code change succeeded 3/3,
      and (b) a same-day 10-seed run of the pre-change code (fixed drop
      point, isolated worktree at commit `529d673`) also hit one transient
      GraspGen-worker subprocess crash on an unrelated seed that likewise
      vanished on retry — both point to this sim's known CGN/GraspGen
      run-to-run stochasticity (see `CLAUDE.md`), not a placement-planner
      regression. Pre-change baseline over the same 10 seeds: 30/30 (100%,
      counting the retried crash). Net effect at this 3-object bin size:
      statistically indistinguishable aggregate success rate, with the
      original design-target failure modes (stacking, crushing) not
      observed in either condition — the 3-object/24cm-bin scene rarely
      exercises them either way; a larger `--n-objects` scene would be a
      sharper differentiator for future validation.
- [x] **Placement pose robustness fixes (2026-08-21), found via live/manual
      testing feedback (not caught by the benchmark's success/failure
      taxonomy, since neither symptom below always caused an outright
      `missed_bin`/failure — just a near-miss or visible instability)**:
      plan `docs/superpowers/plans/2026-08-21-placement-pose-robustness.md`.
  - **Bug 1 — corner-lock:** `OccupancyPlacementPlanner.plan()`'s
    clearance-scoring tie-break measured only distance to *other objects*,
    which is `inf` for every candidate when the bin is empty (true for the
    first object placed every round) — the tie-break silently defaulted to
    the first-scanned candidate, the near-corner of the search region,
    every single time. This is exactly why every run showed the first
    object landing in the same bin corner, touching the wall. Fixed by
    adding a wall-clearance term (`clearance = min(wall_clearance,
    occ_clearance)`), restoring what the design spec always said the
    objective should be. Regression-tested (`test_placement_planner.py`'s
    empty-bin case now asserts a near-center placement, not just "somewhere
    in the bin").
  - **Bug 2 — transit slip:** `GraspExecutor._step_to()` used pure linear
    interpolation for joint targets, which has an instantaneous velocity
    jump at the start of every motion — a classic cause of a held object
    slipping right as `place()`'s transit-to-hover move begins. Fixed by
    adding an optional smoothstep ease-in-ease-out profile (`_ease()`,
    zero velocity at both ends), applied only to `place()`'s two
    object-carrying motions (transit, lower); the already-tuned pick
    sequence in `execute()` and the open-handed release/retract steps are
    untouched.
  - **Deferred (separate future plan, not fixed here):** a third symptom
    (short objects' fingers hitting the table during closing) was
    root-caused to a *different* subsystem —
    `GraspFeasibilityChecker` validates the grasp pose before
    `execute()`'s `EXTRA_APPROACH` (12mm) deepens it further, unchecked.
    Logged at
    `docs/research/2026-08-21-short-object-finger-table-collision.md`.
  - **Re-validation (2026-08-21, same 10-seed GraspGen/fused config as
    above): 30/30 binned (100%)**, zero knocked-off-table, zero
    `missed_bin` failures (down from 3, all clustered in one seed, in the
    pre-fix run) — only 3 `ik_unreachable` pre-grasp retries (unrelated to
    placement, resolved on the next round each time). A clean improvement
    over the prior 29/30 (96.7%), though at this 3-object bin size the
    small failure count makes it hard to call the delta itself
    statistically decisive — the qualitative fix (no more corner-touching,
    smoother carries) is the more meaningful result here, confirmed via
    the live-testing feedback that motivated this fix in the first place.

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

## P8 — Natural-language task instructions (reasoning layer)  [Phase 1 IMPLEMENTED 2026-08-21]

Full exploration/rationale in `docs/research/2026-08-20-reasoning-layer-reflectvlm.md`.
Goal: instructions like "pick the blue cube first and put it on the left,
then the red one on the right" — controls both pick order and placement
destination. Explicitly **not** about physical-stability lookahead (that's
already solved deterministically by the intelligent bin-placement work,
expected to land as P7 — do not confuse the two).

Two-phase plan, strictly sequential — **do not start Phase 2 until Phase 1
is verified robust end-to-end**:

- **Phase 1 (start now, once P7 is verified robust end-to-end) — Option A:**
  a lightweight text-only LLM parses the instruction into an ordered step
  list (`{object_description, spatial_relation, reference}`).
  `object_description` resolves via the existing SAM 3 `PromptSelector`
  (unchanged, P5). `spatial_relation` becomes a directional bias fed into
  the existing `OccupancyPlacementPlanner` free-space search (P7) — still
  collision-safe, just region-preferring. Minimal VRAM footprint (small
  local model or a one-shot cloud API call), reuses everything already
  built.
- **Phase 2 (only after Phase 1 is proven robust) — Option C:** upgrade the
  text-only parser to a small combined vision+language model (e.g.
  Qwen2-VL 2B/7B) that sees the current camera image alongside the
  instruction — only pursue this if Phase 1's text-only grounding proves
  insufficient for attribute references like "the tallest one" that
  benefit from actually seeing the scene.

Explicitly rejected: adopting ReflectVLM as originally proposed (13B VLM +
diffusion dynamics model, ~26GB fp16 / would consume this machine's entire
8GB budget even 4-bit quantized; needs a goal *image* not text; trained on
a different task domain with no fine-tuning support yet) — see the research
note for the full reality-check.

- [x] **Phase 1 implemented (2026-08-21):** full architecture in
      `docs/superpowers/specs/2026-08-21-reasoning-layer-phase1-design.md`,
      built per `docs/superpowers/plans/2026-08-21-reasoning-layer-phase1.md`.
      Three new modules, none modifying already-shipped P5/P7 code:
      `sim_grasp/instruction_parser.py` (NIM `meta/llama-3.1-8b-instruct`
      call + JSON-schema validation into an ordered `Step` list),
      `sim_grasp/spatial_relation_resolver.py` (turns a step's
      `left_of`/`right_of`/`near`/`center` relation into a smaller,
      differently-positioned `OccupancyPlacementPlanner` instance, or
      `None` for no bias — `OccupancyPlacementPlanner` itself is untouched,
      since it only supports square regions, not true half-bin
      rectangles), and a new `run_sim_grasp_test.py --instruction TEXT`
      flag wired into the existing `--pick-all` round loop (resolves the
      active step's `pick_target` via SAM 3 each round, restricting
      `allowed` to that one object; biases placement via the resolver,
      falling back to the plain bin-wide planner). With no `--instruction`,
      every code path is byte-for-byte unchanged (confirmed by re-running
      the full existing `test_*.py` suite plus a baseline `--pick-all` run
      with identical 3/3-binned results to the pre-change baseline).
      Two real bugs were found and fixed via the required live smoke tests
      (not the unit tests, which don't exercise a multi-round pick-all
      sequence): (1) the active step's index never advanced after a
      *successful* pick, so a single-step instruction kept retrying the
      same already-satisfied step forever; (2) an ambiguous `pick_target`
      matching multiple visually-identical objects (e.g. three green
      cuboids in one seed) only checked SAM 3's single top-scoring match,
      which could be an object a prior step already placed — fixed by
      trying every match by descending score until one resolves to an
      object still on the table. A third bug, found by task review rather
      than live testing, was fixed the same way: the `near` relation's
      first-pass implementation dropped the design's location filter,
      taking SAM 3's top-scoring match unconditionally instead of first
      restricting to candidates whose centroid is actually inside the bin
      — fixed with a regression test covering the exact failure mode (see
      the design spec's Implementation notes for detail).
      **Live smoke test results:** a single-step instruction ("Pick up any
      cube and place it in the center of the bin.", seed 0, GraspGen/fused)
      parsed to exactly one step and picked exactly one object, then
      correctly stopped (`[pick-all] round 2: no instruction steps left to
      resolve`) rather than continuing to pick the remaining two — matching
      the instruction's stated scope. A two-step, relation-bearing
      instruction ("Pick a green cube and put it on the left, then pick a
      green cube and put it on the right.", seed 1, GraspGen/fused) parsed
      to two steps and executed both: 2/3 objects binned, zero crashes,
      zero knocked-off-table, landing in two distinct (non-stacked,
      non-overlapping) bin locations. The exact left/right sidedness (as a
      person looking at the camera image would call it) is guaranteed by
      `spatial_relation_resolver`'s own unit tests
      (`test_spatial_relation_resolver.py`), which directly verify
      `_camera_view_axis()`'s direction math against a known camera
      transform — the live run's role was end-to-end wiring validation,
      not re-deriving that geometry by eye from a 3rd-person GIF angle.
      Phase 2 (Option C, a vision+language model) remains un-started,
      gated on Phase 1 proving insufficient in practice, per the two-phase
      sequencing decision above.
  - **Fixed, distinguishable object colors (2026-08-21, found via the
    user manually running `--instruction` end to end):** scene objects
    previously got fully random RGBA (`scene_generator.py`'s old
    `_rand_rgba`), which frequently produced ambiguous or duplicate-ish
    colors (e.g. two shades of blue, no red in a given seed) — making
    instruction text like "the red cube" impossible to write correctly
    without first inspecting `observation.png`, and a live test with seed
    2 genuinely had no green object at all, so `"pick a green cube..."`
    correctly (but unhelpfully) matched nothing 6/6 attempts. Fixed:
    `color_utils.object_color(index)` assigns a fixed, deterministic
    (name, rgba) per spawn index — red/green/blue first, for the default
    3-object scene — using the same reference RGB `rgb_to_color_name`
    matches against, so it always round-trips correctly.
    `run_sim_grasp_test.py` now prints `[scene] object colors: obj_0=red,
    obj_1=green, obj_2=blue` right after scene generation, so an
    `--instruction` can be written correctly without opening
    `observation.png` first. Live-tested: seed 2 now deterministically
    spawns red/green/blue; `"Pick the red cube first and put it on the
    left, then pick the green cube and put it on the right."` correctly
    grounded and picked both (2/3 binned, 152-frame `execution.gif` —
    previously a 2-frame no-op when the instruction's color didn't exist
    in the scene).
  - **Quiet worker subprocess output (2026-08-21, found via the user
    manually running `--instruction` and reporting the terminal was
    "fully messy"):** `cgn_worker.py`/`graspgen_worker.py`/
    `sam3_worker.py` ran with no output capture and no suppression before
    importing their heavy third-party dependencies, so a normal
    successful run was flooded with `FutureWarning`/`UserWarning`
    deprecation noise (timm, spconv, `pkg_resources`, GraspGen's tensor
    construction) and INFO-level third-party logging (`grasp_gen.*`,
    `OpenGL.acceleratesupport`, `torch.fx._symbolic_trace`'s one-time
    warning), interleaved with this project's own `[bracket]` prints.
    Confirmed by direct source inspection that all of it is plain
    `warnings.warn()`/stdlib `logging` (no PyTorch C++/glog env var
    needed, despite the glog-style timestamp prefix on the `torch.fx`
    line). Fixed: a shared `sim_grasp/subprocess_utils.run_worker()`
    replaces the three near-identical `subprocess.run(cmd)` call sites,
    capturing output and only surfacing it on a nonzero exit code; each
    worker script gets a suppression guard (filters the known warning
    categories, raises the known noisy logger namespaces to `ERROR`)
    before any other import. A new `run_sim_grasp_test.py --verbose` flag
    sets `SIM_GRASP_VERBOSE`, restoring the original full output for
    debugging. Live-verified: the exact instruction that originally
    produced ~250 lines of noise now prints only this project's own
    `[bracket]` lines by default, and `--verbose` reproduces the original
    full output unchanged.

