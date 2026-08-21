"""GraspExecutor — closed-loop pick execution in MuJoCo (no motion planner).

CONTROL ARCHITECTURE (read this to understand "how the Franka is controlled")
-----------------------------------------------------------------------------
The Menagerie Panda is actuated by JOINT-SPACE POSITION SERVOS: each arm
actuator is a position servo (gain 2000-4500) whose `data.ctrl[i]` value is a
JOINT ANGLE TARGET in radians. The gripper actuator drives a tendon coupling
both fingers; ctrl in [0, 255] maps closed -> 0.08 m opening (255 = open).

Cartesian grasp poses therefore go through INVERSE KINEMATICS:

    T_cam_grasp  --T_world_cam-->  T_world_grasp     (task space)
        -> differential IK (damped least squares on the 7 arm joints)
        -> q_target (joint space)
        -> linear joint-space interpolation fed to data.ctrl each step
        -> position servos track the reference while physics runs

This mirrors the real robot: MoveIt/franka_ros also plans in joint space and
streams joint position/velocity references to the arm's low-level controller.

GRASP -> HAND FRAME
-------------------
Contact-GraspNet grasp frame: +Z approach, +X finger-closing line, origin at
the gripper base. The Menagerie hand body frame has its origin at the same
physical point, +Z toward the fingers, but its fingers slide along +/-Y.
Mapping is therefore a pure +/-90 deg rotation about Z:

    T_world_hand = T_world_grasp @ Rz(+-pi/2)

Both signs are physically identical (symmetric fingers); we solve IK for both
and keep the better-converged / closer one.

PICK SEQUENCE
-------------
  1. pre-grasp : grasp pose retracted RETRACT_DIST along grasp -Z
  2. approach  : straight to the grasp pose (slow)
  3. close     : gripper ctrl 255 -> 0
  4. lift      : grasp pose raised LIFT_DIST along world +Z
  5. verdict   : success if the target object rose > SUCCESS_RAISE and the
                 fingers did not close completely (i.e. something is held)
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

ARM_JOINTS = [f'joint{i}' for i in range(1, 8)]
RETRACT_DIST = 0.10
EXTRA_APPROACH = 0.012   # advance a touch past the predicted pose before closing
                         # (deeper finger engagement; guards against rim/edge slip)
LIFT_DIST = 0.20
SUCCESS_RAISE = 0.08
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0
CLOSE_PHASE1_CTRL = 75.0  # [P1] end of the fast approach phase of closing;
                          # the remaining travel to GRIPPER_CLOSED happens
                          # slowly (gentle squeeze)
GIF_DOWNSAMPLE = 2     # record GIF at half resolution (1/4 the memory)
GIF_MAX_FRAMES = 1500  # hard cap on recorded frames

# Place sequence (heights of the HAND ORIGIN above the drop point;
# fingertips are ~0.10 m below the hand origin)
PLACE_HOVER = 0.24     # transit / retract height
PLACE_RELEASE = 0.15   # legacy fixed release height, kept only as a fallback
                       # constant for callers whose vision-based placement
                       # planner fails (see placement_planner.py) — fingertips
                       # are ~0.10 m below the hand origin, so the object
                       # drops ~5 cm; at 0.17 it dropped ~7 cm and bounced out
                       # of the 5 cm bin walls (taxonomy: missed_bin x4)
_HOVER_ABOVE_RELEASE = PLACE_HOVER - PLACE_RELEASE  # 0.09 m transit clearance,
                                                     # now anchored to whatever
                                                     # release_z callers pass in
TOPDOWN_HAND_R = np.array([[1.0, 0.0, 0.0],    # canonical hand-down pose,
                           [0.0, -1.0, 0.0],   # fallback orientation for the
                           [0.0, 0.0, -1.0]])  # place IK


def _ease(t: float, smooth: bool) -> float:
    """Interpolation parameter for _step_to: linear (smooth=False) or a
    smoothstep ease-in-ease-out curve (smooth=True, zero velocity at t=0
    and t=1) -- pulled out as its own function so the easing math is
    unit-testable without a live MuJoCo model."""
    return (3 * t ** 2 - 2 * t ** 3) if smooth else t


def _candidate_hand_orientations(R_cur: np.ndarray, yaw: float) -> tuple:
    """Ordered hand-orientation candidates for place(): (1) current hand
    orientation as-is, (2) canonical top-down rotated by `yaw` about world
    Z -- a top-down grasp preserves the object's on-table yaw through the
    pick, so this reliably rotates the placed object -- (3) canonical
    top-down with yaw=0, as a last resort matching the pre-existing
    unconditional fallback."""
    Rz_yaw = R.from_euler('z', yaw).as_matrix()
    return (R_cur, Rz_yaw @ TOPDOWN_HAND_R, TOPDOWN_HAND_R)

_RZ_P90 = np.eye(4); _RZ_P90[:3, :3] = R.from_euler('z', np.pi / 2).as_matrix()
_RZ_M90 = np.eye(4); _RZ_M90[:3, :3] = R.from_euler('z', -np.pi / 2).as_matrix()


@dataclass
class IKResult:
    qpos: np.ndarray
    pos_err: float
    ori_err: float
    converged: bool


def _pick_best_seed_result(results: list) -> 'IKResult':
    """Continuity-first seed selection: `results[0]` is always the
    q_init-seeded (current-pose) attempt. If it converged at all, it wins
    unconditionally -- 'converged' already means 'good enough' (see
    IKResult.converged's definition), so a numerically lower error from a
    differently-postured seed is not a reason to abandon joint-space
    continuity with the arm's current pose. Only when q_init itself fails
    to converge do we fall through to the best of the remaining attempts
    (preferring convergence, then lowest position error) -- unchanged from
    the prior behavior for that case."""
    q_init_result = results[0]
    if q_init_result.converged:
        return q_init_result
    best = q_init_result
    for res in results[1:]:
        if (res.converged and not best.converged) or \
                (res.converged == best.converged and res.pos_err < best.pos_err):
            best = res
    return best


class DiffIK:
    """Damped-least-squares IK for the 7 Panda arm joints, targeting the
    'hand' body frame. Runs on a scratch MjData so the live sim is untouched."""

    def __init__(self, model: mujoco.MjModel,
                 pos_tol=0.004, ori_tol=0.02, max_iters=200, damping=0.1):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
        self.dof_idx = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.qpos_idx = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.jnt_range = np.array([model.joint(j).range for j in ARM_JOINTS])
        self.pos_tol, self.ori_tol = pos_tol, ori_tol
        self.max_iters, self.damping = max_iters, damping

    def solve(self, T_world_hand: np.ndarray, q_init: np.ndarray) -> IKResult:
        """DLS with restarts: try the given seed (the caller's current joint
        configuration) first. If it converges, use it -- continuity with the
        arm's current pose beats a numerically lower error from a
        differently-postured seed (see _pick_best_seed_result). Only try the
        canonical elbow-down pose and two perturbed seeds if q_init itself
        fails to converge."""
        seeds = [q_init,
                 np.array([0.0, 0.35, 0.0, -1.8, 0.0, 2.2, -0.785]),
                 q_init + np.random.default_rng(0).uniform(-0.4, 0.4, 7),
                 q_init + np.random.default_rng(1).uniform(-0.7, 0.7, 7)]
        results = []
        for s in seeds:
            res = self._solve_single(T_world_hand, np.clip(
                s, self.jnt_range[:, 0], self.jnt_range[:, 1]))
            results.append(res)
            if res.converged:
                break   # q_init converged (or, having fallen through, this
                        # later seed did) -- _pick_best_seed_result will
                        # still apply the continuity-first rule below
        return _pick_best_seed_result(results)

    def _solve_single(self, T_world_hand: np.ndarray, q_init: np.ndarray) -> IKResult:
        d = self.scratch
        mujoco.mj_resetData(self.model, d)   # valid default qpos (incl. freejoint quats)
        d.qpos[self.qpos_idx] = q_init
        target_p = T_world_hand[:3, 3]
        target_R = T_world_hand[:3, :3]

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(self.max_iters):
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            cur_p = d.xpos[self.hand_bid]
            cur_R = d.xmat[self.hand_bid].reshape(3, 3)

            e_pos = target_p - cur_p
            e_ori = R.from_matrix(target_R @ cur_R.T).as_rotvec()
            if np.linalg.norm(e_pos) < self.pos_tol and np.linalg.norm(e_ori) < self.ori_tol:
                break

            mujoco.mj_jacBody(self.model, d, jacp, jacr, self.hand_bid)
            J = np.vstack([jacp[:, self.dof_idx], jacr[:, self.dof_idx]])  # 6x7
            e = np.concatenate([e_pos, e_ori])
            JJt = J @ J.T + (self.damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            dq = np.clip(dq, -0.2, 0.2)
            q = d.qpos[self.qpos_idx] + dq
            d.qpos[self.qpos_idx] = np.clip(q, self.jnt_range[:, 0], self.jnt_range[:, 1])

        mujoco.mj_kinematics(self.model, d)
        cur_p = d.xpos[self.hand_bid]
        cur_R = d.xmat[self.hand_bid].reshape(3, 3)
        pos_err = float(np.linalg.norm(target_p - cur_p))
        ori_err = float(np.linalg.norm(R.from_matrix(target_R @ cur_R.T).as_rotvec()))
        q_sol = d.qpos[self.qpos_idx].copy()
        return IKResult(q_sol, pos_err, ori_err,
                        pos_err < self.pos_tol * 2 and ori_err < self.ori_tol * 2)


class GraspExecutor:
    """Executes Contact-GraspNet grasps on the simulated Panda."""

    def __init__(self, model, data, camera_module=None, record_gif=False,
                 record_dir=None, gif_frame_interval=0.08,
                 on_frame: 'Callable[[np.ndarray], None] | None' = None):
        self.model, self.data = model, data
        self.ik = DiffIK(model)
        self.cam = camera_module       # reused for GIF recording (optional)
        self.record = record_gif
        # With record_dir, frames are streamed to disk as JPEGs and the GIF is
        # assembled at the end — peak RAM stays flat (essential when CGN stays
        # loaded during multi-round pick-and-place on 8 GB machines).
        self.record_dir = Path(record_dir) if (record_gif and record_dir) else None
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
        self.frames: list[np.ndarray] = []
        self._n_frames = 0
        self._frame_interval = gif_frame_interval  # sim seconds between frames
        self._last_frame_t = -1.0
        # Optional live-display hook (e.g. LiveViewer.show_frame) — called
        # with the same frame captured for the GIF, at the same cadence.
        # Requires record_gif=True (the cadence/frame-capture logic lives in
        # _maybe_record below); this does not change GIF-saving behavior.
        self.on_frame = on_frame

    # -- low-level motion helpers ---------------------------------------------
    def _step_to(self, q_target: np.ndarray, duration: float,
                 gripper_ctrl: float | None = None, smooth: bool = False):
        """Interpolate joint position references over `duration` seconds of
        sim time; the position servos do the tracking. `smooth=True` uses
        a smoothstep ease-in-ease-out profile (zero velocity at both ends)
        instead of linear interpolation -- linear interpolation has an
        instantaneous velocity jump at t=0, a real cause of a held object
        slipping right as a transit move begins."""
        model, data = self.model, self.data
        n = max(1, int(duration / model.opt.timestep))
        q_start = data.ctrl[:7].copy()
        for i in range(n):
            t = (i + 1) / n
            a = _ease(t, smooth)
            data.ctrl[:7] = (1 - a) * q_start + a * q_target
            if gripper_ctrl is not None:
                data.ctrl[7] = gripper_ctrl
            mujoco.mj_step(model, data)
            self._maybe_record()

    def _hold(self, duration: float):
        for _ in range(max(1, int(duration / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)
            self._maybe_record()

    def _maybe_record(self):
        if self.record and self.cam is not None and \
                self.data.time - self._last_frame_t >= self._frame_interval:
            if self._n_frames >= GIF_MAX_FRAMES:
                print(f'[executor] GIF frame cap ({GIF_MAX_FRAMES}) reached '
                      '— recording stopped')
                self.record = False
                return
            frame = np.ascontiguousarray(
                self.cam.render_rgb()[::GIF_DOWNSAMPLE, ::GIF_DOWNSAMPLE])
            if self.record_dir is not None:
                imageio.imwrite(self.record_dir / f'{self._n_frames:05d}.jpg', frame)
            else:
                self.frames.append(frame)
            self._n_frames += 1
            self._last_frame_t = self.data.time
            if self.on_frame is not None:
                self.on_frame(frame)

    # -- grasp execution --------------------------------------------------------
    def _hand_targets(self, T_world_grasp: np.ndarray, q_seed: np.ndarray):
        """IK for both +/-90deg hand yaws; returns best (IKResult, T_world_hand)."""
        best = None
        for Rz in (_RZ_P90, _RZ_M90):
            T_hand = T_world_grasp @ Rz
            res = self.ik.solve(T_hand, q_seed)
            score = res.pos_err + 0.1 * res.ori_err
            if best is None or (res.converged and not best[0].converged) or \
                    (res.converged == best[0].converged and score < best[2]):
                best = (res, T_hand, score)
        return best[0], best[1]

    def execute(self, T_world_grasp: np.ndarray, target_body: str) -> dict:
        """Run the full pick sequence for one grasp. Returns a result dict."""
        model, data = self.model, self.data
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body)
        obj_z0 = float(data.xpos[bid][2])
        q_now = data.qpos[self.ik.qpos_idx].copy()

        # pre-grasp: retract along grasp approach axis (-Z)
        T_pre = T_world_grasp.copy()
        T_pre[:3, 3] -= RETRACT_DIST * T_world_grasp[:3, 2]

        ik_pre, T_hand_pre = self._hand_targets(T_pre, q_now)
        if not ik_pre.converged:
            return {'success': False, 'stage': 'ik_pregrasp',
                    'pos_err': ik_pre.pos_err, 'ori_err': ik_pre.ori_err}
        # final grasp pose: same hand yaw choice, seeded from pre-grasp solution;
        # advanced EXTRA_APPROACH along the approach axis for deeper engagement
        T_hand_grasp = T_hand_pre.copy()
        T_hand_grasp[:3, 3] = (T_world_grasp[:3, 3]
                               + EXTRA_APPROACH * T_world_grasp[:3, 2])
        ik_grasp = self.ik.solve(T_hand_grasp, ik_pre.qpos)
        if not ik_grasp.converged:
            return {'success': False, 'stage': 'ik_grasp',
                    'pos_err': ik_grasp.pos_err, 'ori_err': ik_grasp.ori_err}
        # lift pose: straight up in world
        T_hand_lift = T_hand_grasp.copy()
        T_hand_lift[2, 3] += LIFT_DIST
        ik_lift = self.ik.solve(T_hand_lift, ik_grasp.qpos)

        # ---- run the sequence (position servos track the references) -------
        self._step_to(ik_pre.qpos, 2.0, gripper_ctrl=GRIPPER_OPEN)
        self._step_to(ik_grasp.qpos, 1.2, gripper_ctrl=GRIPPER_OPEN)
        self._hold(0.2)
        # [P1] two-phase close: fast approach to near-contact, then a slow
        # gentle squeeze. A single fast close "punches" curved objects
        # (spheres/capsules) out of the gripper before friction can grip;
        # slowing the final travel gives contact forces time to build up.
        self._step_to(ik_grasp.qpos, 0.4, gripper_ctrl=CLOSE_PHASE1_CTRL)  # fast approach
        self._step_to(ik_grasp.qpos, 1.2, gripper_ctrl=GRIPPER_CLOSED)     # gentle squeeze
        self._hold(0.3)
        self._step_to(ik_lift.qpos, 1.5, gripper_ctrl=GRIPPER_CLOSED)   # lift
        self._hold(0.5)

        # ---- verdict --------------------------------------------------------
        obj_z1 = float(data.xpos[bid][2])
        finger_open = float(data.qpos[self.model.joint('finger_joint1').qposadr[0]])
        raised = obj_z1 - obj_z0
        success = raised > SUCCESS_RAISE and finger_open > 0.001
        return {'success': bool(success), 'stage': 'done',
                'object_raised_m': round(raised, 4),
                'finger_opening_m': round(2 * finger_open, 4),
                'ik_errors_mm': [round(ik_pre.pos_err * 1e3, 1),
                                 round(ik_grasp.pos_err * 1e3, 1),
                                 round(ik_lift.pos_err * 1e3, 1)]}

    # -- place & housekeeping ---------------------------------------------------
    def place(self, x: float, y: float, release_z: float, yaw: float = 0.0) -> dict:
        """Carry the held object above (x, y), lower until the hand origin
        reaches `release_z`, open the fingers, retract. Tries, in order:
        (1) the current hand orientation as-is, (2) a canonical top-down
        orientation rotated by `yaw`, (3) a canonical top-down orientation
        with no yaw (last resort) -- see _candidate_hand_orientations."""
        data = self.data
        q_now = data.qpos[self.ik.qpos_idx].copy()
        R_cur = data.xmat[self.ik.hand_bid].reshape(3, 3).copy()

        plan = None
        for R_hand in _candidate_hand_orientations(R_cur, yaw):
            T_pre = np.eye(4)
            T_pre[:3, :3] = R_hand
            T_pre[:3, 3] = [x, y, release_z + _HOVER_ABOVE_RELEASE]
            ik_pre = self.ik.solve(T_pre, q_now)
            if not ik_pre.converged:
                continue
            T_rel = T_pre.copy()
            T_rel[2, 3] = release_z
            ik_rel = self.ik.solve(T_rel, ik_pre.qpos)
            if ik_rel.converged:
                plan = (ik_pre, ik_rel)
                break
        if plan is None:
            # bin unreachable from this grasp pose: release in place so the
            # object drops back on the table and can be retried next round
            self._step_to(data.ctrl[:7].copy(), 0.5, gripper_ctrl=GRIPPER_OPEN)
            self._hold(0.3)
            return {'placed': False, 'stage': 'ik_place'}

        ik_pre, ik_rel = plan
        # smooth=True only for the two motions performed while still
        # holding the object -- linear interpolation's instant velocity
        # jump at the start of a move is a real cause of slip; release and
        # retract happen open-handed, so they're left as linear.
        self._step_to(ik_pre.qpos, 2.2, gripper_ctrl=GRIPPER_CLOSED, smooth=True)   # transit
        self._step_to(ik_rel.qpos, 1.0, gripper_ctrl=GRIPPER_CLOSED, smooth=True)   # lower
        self._hold(0.2)
        self._step_to(ik_rel.qpos, 0.6, gripper_ctrl=GRIPPER_OPEN)     # release
        self._hold(0.4)
        self._step_to(ik_pre.qpos, 0.8, gripper_ctrl=GRIPPER_OPEN)     # retract
        return {'placed': True, 'stage': 'place_done'}

    def go_observe(self, q_observe, duration=2.5):
        """Joint move back to the observation pose (gripper open) so the next
        capture sees the table without the arm in the frustum."""
        self._step_to(np.asarray(q_observe, dtype=float), duration,
                      gripper_ctrl=GRIPPER_OPEN)
        self._hold(0.3)

    def save_gif(self, path, fps=None):
        """Assemble the GIF; in record_dir mode frames are streamed from disk
        one at a time so peak memory stays flat."""
        fps = fps or max(1, round(1.0 / self._frame_interval))
        if self.record_dir is not None:
            files = sorted(self.record_dir.glob('*.jpg'))
            if not files:
                return
            with imageio.get_writer(path, mode='I', fps=fps, loop=0) as w:
                for f in files:
                    w.append_data(imageio.imread(f))
            shutil.rmtree(self.record_dir, ignore_errors=True)
        elif self.frames:
            imageio.mimsave(path, self.frames, fps=fps, loop=0)
