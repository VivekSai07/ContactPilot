# Design: Interactive live pick (click-to-select, SAM 3, live execution)

Date: 2026-08-15

## Problem

`mujoco_grasp_sim` can select a target object by ground-truth id
(`--pick-object`) or by a pre-typed prompt/pixel coordinate
(`--prompt`/`--click`/`--box`, sub-project 2 of idea 2) — but every run is
headless: you type coordinates in advance from a saved `observation.png`,
the pipeline runs to completion invisibly, and you inspect the result
afterward via `execution.gif`. There's no way to *watch* a live camera feed,
click the object you want, see SAM 3's segmentation confirmed before
committing, and then watch the robot actually execute the pick — the
original motivation behind idea 2 from the start. This is sub-project 3 of
idea 2; sub-projects 1 (GraspGen backend) and 2 (promptable selection) are
merged and verified (GraspGen: 15/15 vs 14/15 binned; promptable selection:
3/5 correct, mean IoU 0.733).

## Goals

- A live 2D window showing the robot's actual camera feed.
- Click a pixel on that live feed to select a target object via SAM 3 (same
  `PromptSelector`/`sam3_worker.py` machinery as sub-project 2, unchanged).
- Visual confirmation before committing: the resolved mask is overlaid on
  the frame; the user confirms or clicks again to retry.
- Watch the actual pick-and-place execution live in the same window, not
  just as a GIF reviewed afterward (the GIF is still saved, unchanged).
- Support both grasp backends (`cgn`/`graspgen`) via the existing
  `--backend` flag, unchanged.

## Non-goals

- Live text-prompt entry in the window (typed `--prompt` still works from
  the CLI on a future extension if wanted, but this sub-project is
  click-only, per the approved design).
- Real-camera/RealSense integration — sim-only, same scope boundary as
  sub-project 2.
- Any change to `GraspPredictor`, `feasibility.py`, either backend
  predictor, or the diff-IK solver itself — this sub-project only adds a
  live-viewing/interaction layer around the existing pipeline.
- 60fps smooth video — `osmesa` is a CPU software rasterizer; "live" here
  means "you can watch it happen and it updates in near-real-time,"
  matching the GIF's existing `_frame_interval`, not a video-game frame
  rate.

## Architecture

```
interactive_pick.py --seed N [--backend cgn|graspgen] [--sam3-python PATH]
                    │
                    ▼
    SceneGenerator / CameraModule (existing, unchanged)
    MUJOCO_GL=osmesa (matches every other script in this project — the
    WSL2 fix for correct segmentation rendering; osmesa's output is a
    plain numpy RGB array like any other backend, so it displays in cv2
    exactly the same way — no backend-switching needed anywhere)
                    │  render_rgb() each loop iteration
                    ▼
    LiveViewer (NEW, sim_grasp/live_viewer.py)
       cv2 window: shows the live frame, captures mouse clicks
       (cv2.setMouseCallback), draws mask overlays, waits for confirm
                    │  user clicks (x, y)
                    ▼
    PromptSelector.select(rgb, click=(x, y))   ← existing, unchanged
                    │  SelectionResult (masks, scores, boxes)
                    ▼
    LiveViewer.show_mask_overlay(mask) → wait for confirm/retry keypress
                    │  confirmed
                    ▼
    Real-label lookup: mode of the (already-captured, osmesa-rendered)
    ground-truth segmap's values under the mask — same logic as
    promptable-selection's Task-6-fix (commit dbb50eb), factored into a
    small reusable function so both scripts share it (see Files below)
                    ▼
    GraspGenPredictor / ContactGraspNetPredictor (existing, unchanged)
                    ▼
    GraspFeasibilityChecker + ranking (existing, unchanged)
                    ▼
    GraspExecutor.execute(...) — diff-IK pick (existing, MODIFIED only to
    add an optional per-frame callback so interactive_pick.py can
    LiveViewer.show_frame() each step as it's captured — the existing
    GIF-buffering/streaming behavior is unchanged, the callback is
    additive)
                    ▼
    PICK SUCCESS/FAIL printed; execution.gif still saved (unchanged)
```

Everything below "SceneGenerator" is existing, working machinery reused
as-is. The new surface area is small and additive: one new file
(`live_viewer.py`), one new script (`interactive_pick.py`), one small,
additive modification to `executor.py` (an optional frame callback), and
one small refactor extracting the real-label-lookup logic (currently
inlined in `run_sim_grasp_test.py`'s prompt-resolution block) into a
shared, importable function so this script doesn't duplicate it.

## Files

- **Create:** `mujoco_grasp_sim/sim_grasp/live_viewer.py` — `LiveViewer`
  class: owns the `cv2` window, mouse-click capture, mask-overlay drawing,
  confirm/retry keypress handling, and a `show_frame(rgb)` method used
  both for the pre-selection live feed and for live execution playback.
- **Create:** `mujoco_grasp_sim/interactive_pick.py` — the new entry
  point; orchestrates the flow above by calling existing `sim_grasp`
  modules plus the new `LiveViewer`. CLI: `--seed`, `--backend`,
  `--graspgen-python`, `--sam3-python`, `--click-radius-px` (mirrors the
  relevant subset of `run_sim_grasp_test.py`'s existing flags — no new
  flag semantics, just fewer of them since this script is inherently
  single-object/single-run, not built for batch/CI use).
- **Modify:** `mujoco_grasp_sim/sim_grasp/executor.py` — add an optional
  `on_frame: Callable[[np.ndarray], None] | None = None` parameter to
  `GraspExecutor.__init__`, called (if provided) from inside the existing
  `_maybe_record()` alongside its current GIF-buffering logic. No change
  to any existing caller's behavior (parameter defaults to `None`).
- **Modify:** `mujoco_grasp_sim/run_sim_grasp_test.py` — extract the
  real-label-lookup logic (currently inlined: `overlap_labels = segmap[mask]; ...; real_label = int(np.bincount(...).argmax())`)
  into a new shared function `resolve_real_label(gt_segmap, mask) -> int`
  in `sim_grasp/prompt_selector.py` (alongside `PromptSelector`, since it's
  conceptually part of "resolving a SAM 3 selection to something the sim
  can use," not specific to the CLI script), and call it from both
  `run_sim_grasp_test.py` (replacing its current inline block, behavior
  unchanged) and the new `interactive_pick.py`.

## Error handling

- SAM 3 returns zero matches for a click (shouldn't normally happen for a
  click — clicks always yield exactly one target per sub-project 2's
  design — but if the resolved mask is empty): show a clear on-screen
  message ("no object at that point — click again") and return to the
  live-feed/click-capture state, not a crash.
- `resolve_real_label` finds no real object under the mask (mask entirely
  over background): same as above — clear message, click again.
- `SAM3_PYTHON`/`GRASPGEN_PYTHON` unset: fail fast at startup with the
  same clear error `resolve_sam3_python()`/`resolve_graspgen_python()`
  already raise — no new error-handling code needed, just don't swallow
  the existing exceptions.
- User closes the window before confirming: exit cleanly (no pick
  attempted), matching `--view-sim`'s existing "close the window to
  continue" convention but here closing means "cancel," not "continue."
- No feasible grasps survive filtering after a confirmed selection (can
  happen — observed in the user's own manual testing session): print the
  existing `[feasibility] kept 0/N ...` message and exit cleanly, same as
  today's headless behavior — no pick attempted, no crash.

## Testing / verification plan

This codebase has no automated test suite (documented convention); this
sub-project is inherently interactive/manual by nature (a human clicking a
window), so verification is necessarily manual smoke-testing, same
approach as sub-projects 1 and 2's manual verification steps, plus one
piece that *is* mechanically testable:

1. `resolve_real_label(gt_segmap, mask) -> int` — pure function, no GUI,
   no subprocess. Testable directly with synthetic numpy arrays (a
   plain script per this project's `test_color_utils.py` convention).
2. `LiveViewer`'s non-interactive pieces (mask-overlay compositing math)
   — testable directly with synthetic frames/masks, no window needed.
3. The full interactive flow itself — manual: run `interactive_pick.py`,
   click a real object, confirm the mask overlay looks correct, confirm
   the pick executes and is visible frame-by-frame in the window, confirm
   `execution.gif` still gets saved correctly afterward (regression check
   against sub-project 1/2's existing behavior).
4. Confirm `run_sim_grasp_test.py --prompt/--click/--box` still behaves
   identically after the `resolve_real_label` extraction (regression
   check — same real-label-lookup logic, just relocated).

## Risks

- `osmesa`'s CPU rendering speed determines how "live" this actually
  feels — if a single `render_rgb()` call takes too long, the execution
  playback will look like a slideshow, not live video. This was not an
  issue for GIF recording (which tolerates any latency, since it's saved
  after the fact), but is directly user-visible here. Mitigation: reuse
  the GIF's existing `_frame_interval`/`GIF_DOWNSAMPLE` cadence as the
  starting point for the live display rate too, rather than trying to
  push every physics step — if that's still too slow in practice, this is
  the first thing to tune.
- `cv2.imshow`'s window needs a display to attach to. This WSL2 machine
  has `DISPLAY=:0` (WSLg) and rendering to a real window was already
  confirmed working via `mujoco.viewer.launch()` (`--view-sim`) earlier in
  this project — `cv2.imshow` uses the same underlying X11/Wayland
  presentation path, so this is expected to work, but hasn't been
  directly confirmed for `cv2` specifically on this machine yet; worth an
  early smoke test in the first implementation task rather than assuming.
