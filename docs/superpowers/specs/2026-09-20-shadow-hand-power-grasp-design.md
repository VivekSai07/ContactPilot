# Shadow Hand E3M5 — Mechanical Mount + Fixed Power-Grasp — Design

## Problem

ContactPilot picks with a 2-finger parallel-jaw gripper. The user wants to
explore dexterous (multi-fingered, humanoid-style) grasping, starting with
the **Shadow Hand E3M5** (`mujoco_menagerie/shadow_hand`) mounted on the
same Franka Panda arm. This is explicitly scoped small for v1: get the hand
mechanically mounted and moving with **one fixed power-grasp posture** —
not real dexterous grasp planning (that's the separate, larger
DexGraspNet2 integration, deliberately deferred until this proves out).

## Decisions (confirmed with the user)

- **Wrist positioning**: reuse the existing CGN/GraspGen predicted 6-DoF
  grasp pose unchanged — it places the *wrist*, same as it places the
  parallel gripper's TCP today. No new perception/grasp-quality work.
- **Closing posture source**: mine one validated, stable power grasp for
  the Shadow Hand from DexGraspNet's public dataset (real, simulator-
  validated joint angles — not hand-tuned guesswork), remapped from
  DexGraspNet's joint convention to Menagerie's `shadow_hand.xml` joint
  names. This is a one-time data-mining step, not a runtime dependency —
  DexGraspNet's code is never imported or run by ContactPilot.
- **Scope boundary, explicit**: exactly one fixed closing posture,
  triggered the same way `GRIPPER_CLOSED`/`GRIPPER_OPEN` already trigger
  the parallel gripper today. No per-object posture selection, no
  per-finger optimization, no orientation-aware posture variants. This is
  "treat the hand as a fancier parallel gripper" by design (per the
  user's own framing) — real dexterity is DexGraspNet2's job, later.

## Architecture

```
mujoco_grasp_sim/assets/panda_shadow_hand.xml (NEW, generated in-memory
by SceneGenerator, same pattern as the existing panda-only patched asset)
    │
    │  franka_emika_panda/panda.xml's link7, hand/left_finger/right_finger
    │  subtree + split tendon + its <equality> + actuator8 REMOVED
    │
    ▼
  shadow_hand/right_hand.xml's palm-and-fingers subtree GRAFTED at the
  exact mount frame the old "hand" body used to occupy
  (pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834" on link7) --
  shadow_hand's OWN forearm + rh_WRJ1/rh_WRJ2 wrist joints are dropped
  (the Panda arm already provides 7-DoF positioning; a second wrist adds
  uncontrolled DOF this plan doesn't drive) -- re-rooted starting at the
  palm body.
    │
    ▼
sim_grasp/end_effector.py (NEW): EndEffectorController ABC
  - ParallelGripperController: wraps today's exact data.ctrl[7] scalar
    behavior verbatim -- zero change for any Panda-gripper scene.
  - ShadowHandController: interpolates the hand's full actuator vector
    between an "open" posture and the DexGraspNet-derived power-grasp
    posture, driven by the SAME 0(open)-255(closed) scalar GraspExecutor
    already threads through _step_to/_hold/execute/place.
    │
    ▼
GraspExecutor (MODIFIED): constructed with an EndEffectorController
instance instead of hard-coded ctrl[7]/finger_joint1 references;
DiffIK's hand_bid updated to whatever body name the re-rooted Shadow Hand
subtree uses as its "wrist" (the Panda-side mount body, not shadow_hand's
own dropped rh_wrist).
```

## Components

**`sim_grasp/end_effector.py`** (new): `EndEffectorController` ABC with
one method, `ctrl_for(openness: float) -> np.ndarray` (or a small dict/
slice write), mapping the same 0–255 "openness" scalar to whatever the
concrete end effector's actuators need. `ParallelGripperController`
returns a length-1 array (today's `data.ctrl[7]` value, unchanged
behavior). `ShadowHandController` linearly interpolates between two
stored joint-angle vectors (open, power-grasp) sized to the hand's actual
actuator count, keyed by actuator name so the interpolation is robust to
any actuator-ordering assumptions.

**`sim_grasp/executor.py`** (modified): `GraspExecutor.__init__` gains an
`end_effector: EndEffectorController` parameter (default constructs a
`ParallelGripperController` for full backward compatibility with every
existing script — `run_sim_grasp_test.py`/`interactive_pick.py` need zero
changes). `_step_to`'s `gripper_ctrl` parameter is unchanged in name/type
(still the 0–255 scalar) but is now passed through
`self.end_effector.ctrl_for(gripper_ctrl)` before being written, instead
of directly to `data.ctrl[7]`. The pick success check
(`execute()`'s `finger_open = data.qpos[self.model.joint('finger_joint1').qposadr[0]]`)
becomes a method on `EndEffectorController` too
(`is_grasping(data) -> bool`), since "did we actually hold something" means
something different for a tendon gripper (finger separation) vs. a
multi-fingered hand (e.g. checking the lifted object's height alone,
already computed elsewhere in `execute()`, is sufficient and simpler than
re-deriving a hand-specific contact heuristic for v1).

**`scripts/extract_shadow_hand_posture.py`** (new, one-time/offline tool,
not part of the runtime pipeline): downloads or reads a small sample of
DexGraspNet's published ShadowHand grasp data, extracts one validated
power-grasp joint configuration, remaps DexGraspNet's joint names to
Menagerie's `shadow_hand.xml` joint names (a manually-verified mapping
table, since the two conventions differ), and writes the result as a
plain Python constant (`SHADOW_HAND_POWER_GRASP_POSTURE` dict) into
`sim_grasp/end_effector.py` — checked into the repo as a static value,
not fetched at runtime.

**`mujoco_grasp_sim/scene_generator.py`** (modified): gains an
`end_effector: str = 'parallel'` field on `SceneConfig` (`'parallel'` |
`'shadow_hand'`), branching which MJCF patch/asset gets generated and
which `EndEffectorController` `run_sim_grasp_test.py`/`interactive_pick.py`
construct. Default remains `'parallel'` — zero behavior change unless a
caller opts in.

## Error handling

- Same as today for IK failures (`ik_pregrasp`/`ik_grasp` stage failures)
  — the bigger hand's different collision envelope may fail IK more
  often against cluttered scenes, but that's an existing, already-handled
  failure mode, not new error-handling surface for this plan.
- If the DexGraspNet joint-name mapping is ever wrong (a joint silently
  maps to the wrong actuator), the effect is visually obvious (fingers
  curl in the wrong pattern) — caught by this plan's live visual smoke
  test, not by a runtime check (a static, one-time-verified mapping
  doesn't need a runtime guard).

## Testing

1. `end_effector.py`'s posture interpolation is a pure function (no
   MuJoCo model needed for the math itself, though constructing the
   controller may reference real actuator names) — standalone
   `test_end_effector.py`, matching this project's `test_*.py` convention.
2. Asset-loading sanity: a standalone script confirms
   `panda_shadow_hand.xml` loads into a real `mujoco.MjModel` with no
   XML errors, the expected actuator count, and the hand's body correctly
   parented under `link7` at the expected transform.
3. Live visual validation: run a manual pick attempt against the new
   scene config, save a GIF, visually confirm the Shadow Hand's fingers
   curl into the power-grasp posture over an object and the arm doesn't
   collide with itself — this is a genuinely new mechanical configuration,
   so a human/visual check matters here more than for a config change.
4. Regression: existing `run_sim_grasp_test.py --pick-all`/benchmark
   suite with the DEFAULT (`'parallel'`) config must show byte-identical
   behavior to before — confirms the new abstraction didn't disturb the
   parallel-gripper path.

## Deferred (explicit, not part of this plan)

- Real dexterous grasp synthesis (DexGraspNet2 integration) — separate,
  larger effort, revisited only once this fixed-posture milestone is
  proven out live.
- Per-object/per-orientation posture selection, wrist re-articulation
  (Shadow's own `rh_WRJ1`/`rh_WRJ2` are dropped in this plan, not driven).
- A hand-specific contact/force-based "did we actually grasp it" check —
  the existing lifted-height heuristic is reused as-is for v1.
