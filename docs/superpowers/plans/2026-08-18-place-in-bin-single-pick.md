# Place-in-Bin for Single-Object Pick Flows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After a successful pick, both `interactive_pick.py` and `run_sim_grasp_test.py`'s single-object `--execute` mode should place the object in the bin (not just lift it), matching what `--pick-all` already does.

**Architecture:** `sim_grasp/executor.py`'s `GraspExecutor.place(drop_pos)` and `sim_grasp/scene_generator.py`'s `SceneGenerator.bin_drop_point()`/`objects_in_bin()` already exist and are already used correctly by `run_sim_grasp_test.py --pick-all`. The gap is purely that the two single-pick call sites (`interactive_pick.py`'s click loop, and `run_sim_grasp_test.py`'s non-`--pick-all` `--execute` branch) call `executor.execute(...)` and stop — they never call `.place(...)`. This plan wires both call sites to call `place()` after a successful pick and report bin status, with no changes to `executor.py` or `scene_generator.py` themselves.

**Tech Stack:** Python, MuJoCo, no new dependencies.

## Global Constraints

- Repo has no automated test suite. Verification for these two tasks is manual/live runs (this code path drives real MuJoCo physics + IK, not unit-testable in isolation) — matches how `execute()` itself and `--pick-all` were verified.
- Commit messages are plain text — never add a "Co-Authored-By: Claude" trailer or similar.
- Any live run needs `export MUJOCO_GL=osmesa` (and `GRASPGEN_PYTHON`/`SAM3_PYTHON` for the graspgen/SAM3 backends) — see `mujoco_grasp_sim/README.md`.
- Use `/home/vivek/miniconda3/envs/cgn_torch/bin/python` to run things (plain `python`/`python3` lacks numpy on this machine).
- On failure to place (bin unreachable, or object misses the bin), report the status and stop — no retry. This matches `--pick-all`'s existing per-attempt reporting style, not a new retry mechanism.
- All new/modified Python files must pass `get_errors` with no reported problems before moving to the next task.

---

### Task 1: Place in bin after a successful pick in `interactive_pick.py`

**Files:**
- Modify: `mujoco_grasp_sim/interactive_pick.py` (the pick-attempt loop, around lines 190-207)

**Interfaces:**
- Consumes: `GraspExecutor.place(drop_pos) -> dict` (`{'placed': bool, 'stage': str}`, existing, in `sim_grasp/executor.py`); `SceneGenerator.bin_drop_point() -> np.ndarray` and `SceneGenerator.objects_in_bin() -> list[str]` (existing, in `sim_grasp/scene_generator.py`); both already used identically by `run_sim_grasp_test.py --pick-all`.
- Produces: nothing consumed by Task 2 (both tasks are independent, mirroring the same pattern in two different files).

- [ ] **Step 1: Locate and replace the pick-attempt loop's success handling**

In `mujoco_grasp_sim/interactive_pick.py`, find this exact block:

```python
        res = executor.execute(T_world_grasp, target_body=body)
        res.update(object=real_label, score=score)
        print(f'[execute]   -> {res}')
        if res['success']:
            print(f"[execute] PICK SUCCESS on attempt {attempt} "
                  f"(object raised {res['object_raised_m']} m)")
            break
        print(f'[execute] attempt {attempt} failed at stage {res["stage"]}' +
              (' — trying next candidate' if attempt < len(order) else ' — all attempts failed'))
```

Replace it with:

```python
        res = executor.execute(T_world_grasp, target_body=body)
        res.update(object=real_label, score=score)
        if res['success']:
            res['place'] = executor.place(gen.bin_drop_point())
            res['in_bin'] = body in gen.objects_in_bin()
        print(f'[execute]   -> {res}')
        if res['success']:
            print(f"[execute] PICK SUCCESS on attempt {attempt} "
                  f"(object raised {res['object_raised_m']} m)")
            if res['in_bin']:
                print('[execute] placed in bin')
            else:
                print('[execute] picked but missed the bin '
                      f"(place stage: {res['place']['stage']})")
            break
        print(f'[execute] attempt {attempt} failed at stage {res["stage"]}' +
              (' — trying next candidate' if attempt < len(order) else ' — all attempts failed'))
```

(`gen` is the existing `SceneGenerator` instance already in scope earlier in `main()` — no new imports needed. `body` is already computed above this loop.)

- [ ] **Step 2: Check for errors**

Run `get_errors` on `interactive_pick.py`, confirm no problems.

- [ ] **Step 3: Manual live-GUI verification**

```bash
cd mujoco_grasp_sim
export MUJOCO_GL=osmesa SAM3_PYTHON=/home/vivek/miniconda3/envs/sam3_torch/bin/python GRASPGEN_PYTHON=/home/vivek/miniconda3/envs/graspgen_torch/bin/python DISPLAY=:0
/home/vivek/miniconda3/envs/cgn_torch/bin/python interactive_pick.py --seed 5 --backend graspgen
```

Click an object, confirm the mask, watch the window through the pick. Expected: after "PICK SUCCESS", the arm should visibly carry the object over to the bin and release it (not stop right after lifting), and the console should print either `[execute] placed in bin` or `[execute] picked but missed the bin (place stage: ...)`. Ask the human operator to confirm this visually — this repo's established pattern for GUI verification (no automated way to inspect a cv2/MuJoCo window's contents). After the run, check `output/<latest>/metrics.json` contains an `"in_bin"` key in its `"execution"` object.

- [ ] **Step 4: Commit**

```bash
git add mujoco_grasp_sim/interactive_pick.py
git commit -m "Place object in bin after a successful pick in interactive_pick.py"
```

---

### Task 2: Place in bin after a successful pick in `run_sim_grasp_test.py`'s single-object `--execute` mode

**Files:**
- Modify: `mujoco_grasp_sim/run_sim_grasp_test.py` (the single-object execute loop, around lines 723-732 — NOT the `--pick-all` branch above it, which already places correctly)

**Interfaces:**
- Consumes: same `GraspExecutor.place(...)` / `SceneGenerator.bin_drop_point()` / `SceneGenerator.objects_in_bin()` as Task 1.
- Produces: nothing consumed by Task 1 (independent).

- [ ] **Step 1: Locate and replace the single-object execute loop's success handling**

In `mujoco_grasp_sim/run_sim_grasp_test.py`, inside the `elif args.execute and any(len(s) for s in scores.values()):` branch (this is a **different** loop than `--pick-all`'s — do not touch the `--pick-all` branch, which already calls `.place(...)` correctly), find this exact block:

```python
            res = executor.execute(T_world_grasp, target_body=body)
            res.update(object=int(sid), score=score, recenter_shift_m=round(shift, 4))
            exec_results.append(res)
            print(f'[execute]   -> {res}')
            if res['success']:
                print(f'[execute] PICK SUCCESS on attempt {attempt} '
                      f"(object raised {res['object_raised_m']} m)")
                break
```

Replace it with:

```python
            res = executor.execute(T_world_grasp, target_body=body)
            res.update(object=int(sid), score=score, recenter_shift_m=round(shift, 4))
            if res['success']:
                res['place'] = executor.place(gen.bin_drop_point())
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
```

(`gen` is the existing `SceneGenerator` instance already in scope in `main()` — it's used a few lines above in the `--pick-all` branch and throughout the file. `body` is already computed above this loop via `label_to_body[int(sid)]`.)

- [ ] **Step 2: Check for errors**

Run `get_errors` on `run_sim_grasp_test.py`, confirm no problems.

- [ ] **Step 3: Regression test**

```bash
cd mujoco_grasp_sim
export MUJOCO_GL=osmesa SAM3_PYTHON=/home/vivek/miniconda3/envs/sam3_torch/bin/python GRASPGEN_PYTHON=/home/vivek/miniconda3/envs/graspgen_torch/bin/python
/home/vivek/miniconda3/envs/cgn_torch/bin/python run_sim_grasp_test.py --seed 5 --execute --backend graspgen --click 389,273 --no-vis --top-k 3
```

Expected: reaches `PICK SUCCESS`, then prints either `[execute] placed in bin` or `[execute] picked but missed the bin (place stage: ...)` — it must not stop silently right after `PICK SUCCESS` the way it did before this fix. Also confirm `--pick-all` still works unaffected (that branch wasn't touched):

```bash
/home/vivek/miniconda3/envs/cgn_torch/bin/python run_sim_grasp_test.py --seed 5 --pick-all --backend graspgen --no-vis
```

Expected: same `[pick-all] DONE: N/N objects in the bin ...` style output as before (this run's own behavior must be unchanged by this task's diff, since the diff is in a different branch of the `if`/`elif`).

- [ ] **Step 4: Commit**

```bash
git add mujoco_grasp_sim/run_sim_grasp_test.py
git commit -m "Place object in bin after a successful pick in single-object --execute mode"
```
