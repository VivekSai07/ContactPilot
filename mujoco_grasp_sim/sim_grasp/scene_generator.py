"""SceneGenerator — randomized MuJoCo tabletop scenes with a Menagerie Franka Panda.

Responsibilities
----------------
1. Patch the Menagerie panda.xml (strip keyframe — its qpos size would clash
   with our added free joints — and lift link0 to tabletop height).
2. Generate an MJCF scene: pedestal + table + eye-to-hand camera + N random
   objects (primitives and, if available, meshes from assets/objects/).
3. Spawn objects without catastrophic penetration (rejection-sampled XY,
   staggered drop heights) and settle physics before observation capture.

The patched panda.xml and the generated scene XML are written into
mujoco_grasp_sim/assets/ (NOT into the vendored mujoco_menagerie/ tree —
that directory is a git submodule and must stay pristine so it never shows
as locally modified). Because MJCF resolves meshdir/<include> paths
relative to the top-level XML file's own directory, the patch step rewrites
panda.xml's <compiler meshdir="assets"> to an absolute path pointing back
at mujoco_menagerie/franka_emika_panda/assets, so mesh loading still works
regardless of where the generated files live.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from sim_grasp.frames import look_at_xyaxes

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent                              # mujoco_grasp_sim/
REPO_ROOT = PROJECT_ROOT.parent                              # repo root
MENAGERIE_PANDA_DIR = REPO_ROOT / 'mujoco_menagerie' / 'franka_emika_panda'
MESH_OBJECT_DIR = PROJECT_ROOT / 'assets' / 'objects'        # drop YCB .obj/.stl here
GENERATED_DIR = PROJECT_ROOT / 'assets'                       # patched/generated MJCF lands here

# Arm joint targets while observing (folded back so the arm stays out of the
# camera frustum). Position-servo actuators hold these via data.ctrl.
ARM_OBSERVE_QPOS = np.array([0.0, -1.4, 0.0, -2.7, 0.0, 1.45, -0.7853])
GRIPPER_OPEN_CTRL = 255.0


@dataclass
class SceneConfig:
    """All randomization & geometry knobs in one place."""
    # Table (matches a typical lab workstation; tabletop = z TABLE_HEIGHT)
    table_height: float = 0.75
    table_size_x: float = 0.40          # half-size: table spans 0.8 m in x
    table_size_y: float = 0.45          # half-size: 0.9 m in y
    table_center_x: float = 0.60        # table center in front of the robot
    table_thickness: float = 0.04

    # Object spawn region ON the table (kept inside camera view + arm reach;
    # also fully covered by the calibrated top-down camera at (0.49, 0.27))
    spawn_x: tuple = (0.38, 0.68)
    spawn_y: tuple = (-0.12, 0.32)
    min_object_spacing: float = 0.09    # min center distance at spawn time

    # Objects
    # [box-only] Fixed at 3 box/cuboid objects per scene, randomly placed
    # within the reachable spawn region (see _sample_xy_positions). Cylinders,
    # spheres, capsules and YCB meshes were dropped: curved/non-box shapes
    # dominated pick failures (rolling/slipping during closing and during the
    # transit-to-bin transfer).
    n_objects_range: tuple = (3, 3)
    use_meshes: bool = False            # disabled: YCB meshes aren't box-shaped
    mesh_probability: float = 0.4       # unused while use_meshes=False

    # Eye-to-hand camera. Two placement modes:
    #  1. calibration_file set -> camera placed EXACTLY like the real lab
    #     setup: T_world_cam = Trans(0,0,table_height) @ T_base_cam (the
    #     robot base frame sits at tabletop height, see frames.py).
    #  2. otherwise -> generic look-at placement below.
    cam_name: str = 'ext_cam'
    calibration_file: str | None = None  # path to calibration_result.yaml
    cam_pos: tuple = (1.35, 0.0, 1.40)
    cam_target: tuple = (0.55, 0.0, 0.75)
    cam_fovy_deg: float = 58.0          # D455 vertical FOV ~ 58-65 deg

    # Optional SECOND observation camera for multi-camera fusion (P2):
    # placed from its own calibration yaml (the inclined side mount beside
    # the Franka), independent of the main camera above. Only emitted into
    # the scene when side_calibration_file is set.
    side_cam_name: str = 'side_cam'
    side_calibration_file: str | None = None
    side_cam_fovy_deg: float = 58.0

    # Recording camera for execution GIFs: a side view past the -y table edge,
    # pulled back far enough that the whole pick -> transit -> place-in-bin
    # sequence stays in frame (the top-down observation camera shows almost
    # nothing of the grasp itself).
    record_cam_name: str = 'record_cam'
    record_cam_pos: tuple = (0.62, -0.92, 1.18)
    record_cam_target: tuple = (0.50, -0.05, 0.80)  # between spawn region & bin
    record_cam_fovy_deg: float = 52.0

    # Place bin: a static open-top tray on the table, outside the object spawn
    # region, within arm reach (0.54 m from the base). Pick-and-place targets
    # its center.
    bin_center: tuple = (0.45, -0.30)   # XY on the tabletop
    bin_inner_half: float = 0.12        # inner half-extent (square)
    bin_wall_height: float = 0.05
    bin_wall_half_thickness: float = 0.006

    # Physics settling
    settle_time: float = 3.0            # seconds of free simulation
    max_extra_settle: float = 4.0       # extra time if objects still moving
    settle_qvel_thresh: float = 0.02    # rad/s or m/s — "at rest" threshold

    seed: int | None = None


# ---------------------------------------------------------------------------
# Random object descriptions
# ---------------------------------------------------------------------------
@dataclass
class ObjectSpec:
    name: str
    xml: str            # the <body>...</body> snippet
    spawn_half_height: float


def _rand_rgba(rng):
    c = rng.uniform(0.15, 0.95, size=3)
    return f'{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1'


def _make_primitive(rng, name: str) -> ObjectSpec:
    """Random graspable box/cuboid. Sizes chosen to fit the Panda gripper
    (max opening 0.08 m) along at least one dimension."""
    rgba = _rand_rgba(rng)
    yaw = rng.uniform(0, 2 * np.pi)
    quat = f'{np.cos(yaw / 2):.5f} 0 0 {np.sin(yaw / 2):.5f}'

    # Panda max opening is 0.08 m -> keep at least one graspable dimension
    # comfortably below ~0.06 m (fingers need clearance to wrap, not pinch rims)
    sx, sy, sz = rng.uniform(0.012, 0.028), rng.uniform(0.012, 0.028), rng.uniform(0.02, 0.055)
    geom = f'<geom type="box" size="{sx:.4f} {sy:.4f} {sz:.4f}" rgba="{rgba}"/>'
    half_h = float(np.sqrt(sx**2 + sy**2 + sz**2))

    body = (f'<body name="{name}" pos="0 0 0" quat="{quat}">'
            f'<freejoint name="{name}_joint"/>'
            f'{geom}'
            f'</body>')
    return ObjectSpec(name=name, xml=body, spawn_half_height=half_h)


def _list_mesh_files():
    if not MESH_OBJECT_DIR.is_dir():
        return []
    return sorted([p for p in MESH_OBJECT_DIR.iterdir()
                   if p.suffix.lower() in ('.stl', '.obj')])


def _make_mesh_object(rng, name: str, mesh_path: Path) -> ObjectSpec | None:
    """Load a mesh object (e.g. a YCB model dropped into assets/objects/).

    The mesh is auto-scaled so its largest bounding-box dimension is <= 12 cm
    (graspable). Convex-hull collision is MuJoCo's default for meshes.
    """
    try:
        import trimesh
        m = trimesh.load(str(mesh_path), force='mesh')
        ext = m.bounding_box.extents
    except Exception:
        return None
    max_ext = float(max(ext))
    if max_ext <= 0:
        return None
    scale = min(1.0, 0.12 / max_ext)
    half_h = 0.5 * max_ext * scale * 1.74  # diag/2 upper bound

    yaw = rng.uniform(0, 2 * np.pi)
    quat = f'{np.cos(yaw / 2):.5f} 0 0 {np.sin(yaw / 2):.5f}'
    rgba = _rand_rgba(rng)
    mesh_asset_name = f'{name}_mesh'
    # NOTE: absolute path so meshdir (which belongs to panda.xml) is bypassed.
    asset = f'<mesh name="{mesh_asset_name}" file="{mesh_path.resolve()}" scale="{scale:.5f} {scale:.5f} {scale:.5f}"/>'
    body = (f'<body name="{name}" pos="0 0 0" quat="{quat}">'
            f'<freejoint name="{name}_joint"/>'
            f'<geom type="mesh" mesh="{mesh_asset_name}" rgba="{rgba}" density="400"/>'
            f'</body>')
    spec = ObjectSpec(name=name, xml=body, spawn_half_height=half_h)
    spec.extra_asset = asset  # type: ignore[attr-defined]
    return spec


# ---------------------------------------------------------------------------
# SceneGenerator
# ---------------------------------------------------------------------------
class SceneGenerator:
    """Builds and settles one randomized tabletop scene.

    Usage:
        gen = SceneGenerator(SceneConfig(seed=0))
        model, data = gen.generate()
        seg_id_of_body = gen.object_body_ids   # {body_id: seg_label}
    """

    def __init__(self, config: SceneConfig | None = None):
        self.cfg = config or SceneConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.object_names: list[str] = []
        self.object_body_ids: dict[int, int] = {}   # mj body id -> seg label (1..N)
        self.scene_xml_path: Path | None = None

        if not MENAGERIE_PANDA_DIR.is_dir():
            raise FileNotFoundError(
                f'Menagerie Panda not found at {MENAGERIE_PANDA_DIR}. Clone with:\n'
                'git clone --depth 1 --filter=blob:none --sparse '
                '--config core.autocrlf=false '
                'https://github.com/google-deepmind/mujoco_menagerie.git\n'
                'git -C mujoco_menagerie sparse-checkout set franka_emika_panda')

    # -- panda.xml patching --------------------------------------------------
    def _patched_panda_xml(self) -> str:
        """Strip the keyframe (qpos-size clash with added free joints), lift
        link0 onto the pedestal at table height, and rewrite meshdir to an
        absolute path (the output file no longer lives next to panda.xml,
        so the original relative meshdir="assets" would otherwise break).
        Returns the patched filename (written into GENERATED_DIR)."""
        src = (MENAGERIE_PANDA_DIR / 'panda.xml').read_text(encoding='utf-8')
        patched = re.sub(r'<keyframe>.*?</keyframe>', '', src, flags=re.S)
        patched = patched.replace(
            '<body name="link0" childclass="panda">',
            f'<body name="link0" childclass="panda" pos="0 0 {self.cfg.table_height}">',
            1)
        if f'pos="0 0 {self.cfg.table_height}"' not in patched:
            raise RuntimeError('Failed to patch link0 mount height in panda.xml '
                               '(upstream file structure changed?)')

        # meshdir="assets" is relative to panda.xml's own directory; since
        # the patched file is now written into GENERATED_DIR (not next to
        # panda.xml), rewrite it to an absolute path so mesh loading still
        # resolves correctly regardless of where the top-level scene XML
        # that includes this file actually lives.
        abs_meshdir = str((MENAGERIE_PANDA_DIR / 'assets').resolve())
        patched, n_meshdir = re.subn(
            r'meshdir="assets"',
            f'meshdir="{abs_meshdir}"',
            patched)
        if n_meshdir != 1:
            raise RuntimeError(
                f'Expected to patch exactly one meshdir="assets" in panda.xml, '
                f'patched {n_meshdir} (upstream file structure changed?)')

        # [P1 friction audit] Compliant rubber fingertip pads grip harder
        # against rotation/slip than MuJoCo's rigid-body default
        # (1 0.005 0.0001, condim=3) — bump sliding+torsional friction on the
        # 5 fingertip pad collision boxes, the actual contact surfaces during
        # a grasp, to resist the slow sliding/torsional "walk-out" of
        # off-center grasps during lift/transit-to-bin.
        # condim="4" is REQUIRED here: MuJoCo's default condim=3 only puts
        # friction[0] (sliding) into the contact friction cone, so the
        # torsional term friction[1] is a dead value no matter how high it's
        # set without it (the original 1.0/0.01/0.004 patch was a no-op,
        # since 1.0 sliding == the unpatched default).
        patched, n_friction = re.subn(
            r'(<default class="fingertip_pad_collision_\d">\s*'
            r'<geom type="box" size="[^"]*" pos="[^"]*")/>',
            r'\1 friction="1.5 0.02 0.004" condim="4"/>',
            patched)
        if n_friction != 5:
            raise RuntimeError(
                f'Expected to patch friction on 5 fingertip pad collision '
                f'geoms in panda.xml, patched {n_friction} '
                '(upstream file structure changed?)')
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        out = GENERATED_DIR / '_panda_sim_patched.xml'
        out.write_text(patched, encoding='utf-8')
        return out.name

    # -- object sampling -----------------------------------------------------
    def _sample_objects(self) -> tuple[list[ObjectSpec], list[str]]:
        n = int(self.rng.integers(self.cfg.n_objects_range[0],
                                  self.cfg.n_objects_range[1] + 1))
        mesh_files = _list_mesh_files() if self.cfg.use_meshes else []
        specs, extra_assets = [], []
        for i in range(n):
            name = f'obj_{i}'
            spec = None
            if mesh_files and self.rng.random() < self.cfg.mesh_probability:
                spec = _make_mesh_object(self.rng, name, Path(self.rng.choice(mesh_files)))
                if spec is not None and hasattr(spec, 'extra_asset'):
                    extra_assets.append(spec.extra_asset)  # type: ignore[attr-defined]
            if spec is None:
                spec = _make_primitive(self.rng, name)
            specs.append(spec)
        return specs, extra_assets

    def _sample_xy_positions(self, n: int) -> np.ndarray:
        """Rejection-sample XY spawn positions keeping min spacing (avoids
        catastrophic initial penetration between objects)."""
        cfg = self.cfg
        positions = []
        for _ in range(n):
            for _attempt in range(300):
                xy = np.array([self.rng.uniform(*cfg.spawn_x),
                               self.rng.uniform(*cfg.spawn_y)])
                if all(np.linalg.norm(xy - p) >= cfg.min_object_spacing for p in positions):
                    positions.append(xy)
                    break
            else:
                # workspace saturated — accept closest-effort placement
                positions.append(xy)
        return np.array(positions)

    # -- scene XML -----------------------------------------------------------
    def _camera_placement(self) -> tuple[np.ndarray, str]:
        """Camera world pos + MJCF xyaxes string.

        With a calibration file, mirrors the real lab mount:
            T_world_camCV = Trans(0,0,table_h) @ T_base_cam
        then converts the OpenCV camera rotation to MuJoCo convention
        (R_mj = R_cv @ diag(1,-1,-1), see frames.py) and emits its x/y axes.
        """
        cfg = self.cfg
        if cfg.calibration_file:
            return self._calibrated_placement(cfg.calibration_file)
        _, _, xyaxes = look_at_xyaxes(cfg.cam_pos, cfg.cam_target)
        return np.asarray(cfg.cam_pos, dtype=np.float64), xyaxes

    def _calibrated_placement(self, calibration_file: str):
        """World pos + MJCF xyaxes for a camera mounted per a lab calibration
        yaml (T_base_cam with the robot base at tabletop height)."""
        from sim_grasp.calibration import load_T_base_cam
        T_base_cam = load_T_base_cam(calibration_file)
        T_world_base = np.eye(4)
        T_world_base[2, 3] = self.cfg.table_height
        T_world_cam = T_world_base @ T_base_cam
        R_mj = T_world_cam[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        xyaxes = ' '.join(f'{v:.6f}' for v in (*R_mj[:, 0], *R_mj[:, 1]))
        return T_world_cam[:3, 3], xyaxes

    def _build_scene_xml(self, specs: list[ObjectSpec], extra_assets: list[str]) -> Path:
        cfg = self.cfg
        cam_pos, cam_xyaxes = self._camera_placement()
        _, _, rec_xyaxes = look_at_xyaxes(cfg.record_cam_pos, cfg.record_cam_target)
        side_cam_xml = ''
        if cfg.side_calibration_file:
            sp, sx = self._calibrated_placement(cfg.side_calibration_file)
            side_cam_xml = (
                f'<!-- Second observation camera (P2 fusion) -->\n    '
                f'<camera name="{cfg.side_cam_name}" '
                f'pos="{sp[0]:.6f} {sp[1]:.6f} {sp[2]:.6f}" '
                f'xyaxes="{sx}" fovy="{cfg.side_cam_fovy_deg}"/>')
        panda_file = self._patched_panda_xml()

        table_z = cfg.table_height - cfg.table_thickness / 2
        leg_h = (cfg.table_height - cfg.table_thickness) / 2
        leg_xy = [(cfg.table_center_x - cfg.table_size_x + 0.05,  cfg.table_size_y - 0.05),
                  (cfg.table_center_x - cfg.table_size_x + 0.05, -cfg.table_size_y + 0.05),
                  (cfg.table_center_x + cfg.table_size_x - 0.05,  cfg.table_size_y - 0.05),
                  (cfg.table_center_x + cfg.table_size_x - 0.05, -cfg.table_size_y + 0.05)]
        legs = '\n    '.join(
            f'<geom type="cylinder" size="0.025 {leg_h:.4f}" pos="{x:.3f} {y:.3f} {leg_h:.4f}" '
            f'rgba="0.45 0.33 0.22 1" contype="0" conaffinity="0"/>'
            for x, y in leg_xy)

        objects_xml = '\n    '.join(s.xml for s in specs)
        assets_xml = '\n    '.join(extra_assets)

        # Place bin: floor slab + 4 walls, static, sitting on the tabletop
        bx, by = cfg.bin_center
        bi, wt = cfg.bin_inner_half, cfg.bin_wall_half_thickness
        wh = cfg.bin_wall_height / 2
        bo = bi + 2 * wt                       # outer half-extent
        bz = cfg.table_height
        bin_rgba = '0.50 0.55 0.62 1'
        bin_xml = '\n    '.join([
            f'<geom name="bin_floor" type="box" size="{bo:.4f} {bo:.4f} 0.004" '
            f'pos="{bx} {by} {bz + 0.004:.4f}" rgba="{bin_rgba}"/>',
            f'<geom name="bin_wall_xp" type="box" size="{wt:.4f} {bo:.4f} {wh:.4f}" '
            f'pos="{bx + bi + wt:.4f} {by} {bz + 0.008 + wh:.4f}" rgba="{bin_rgba}"/>',
            f'<geom name="bin_wall_xm" type="box" size="{wt:.4f} {bo:.4f} {wh:.4f}" '
            f'pos="{bx - bi - wt:.4f} {by} {bz + 0.008 + wh:.4f}" rgba="{bin_rgba}"/>',
            f'<geom name="bin_wall_yp" type="box" size="{bo:.4f} {wt:.4f} {wh:.4f}" '
            f'pos="{bx} {by + bi + wt:.4f} {bz + 0.008 + wh:.4f}" rgba="{bin_rgba}"/>',
            f'<geom name="bin_wall_ym" type="box" size="{bo:.4f} {wt:.4f} {wh:.4f}" '
            f'pos="{bx} {by - bi - wt:.4f} {bz + 0.008 + wh:.4f}" rgba="{bin_rgba}"/>'])

        xml = f"""<mujoco model="panda_tabletop_grasping">
  <include file="{panda_file}"/>

  <statistic center="0.55 0 0.9" extent="1.2"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global offwidth="1280" offheight="960" azimuth="120" elevation="-20"/>
    <map znear="0.005"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.5 0.6" rgb2="0.1 0.1 0.15" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.25 0.3 0.35" rgb2="0.15 0.2 0.25"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.1"/>
    <texture type="2d" name="tabletex" builtin="flat" rgb1="0.82 0.71 0.55" width="32" height="32"/>
    <material name="table_mat" texture="tabletex" reflectance="0.05"/>
    {assets_xml}
  </asset>

  <worldbody>
    <light pos="0.6 0 2.2" dir="0 0 -1" directional="true" diffuse="0.7 0.7 0.7"/>
    <light pos="1.4 0.8 1.8" dir="-0.4 -0.4 -1" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- Robot pedestal: lifts link0 (robot base frame) to tabletop height.
         T_world_base = Trans(0, 0, {cfg.table_height}) -->
    <geom name="pedestal" type="cylinder" size="0.10 {cfg.table_height / 2:.4f}"
          pos="0 0 {cfg.table_height / 2:.4f}" rgba="0.3 0.3 0.32 1"/>

    <!-- Table: top surface is the z = {cfg.table_height} plane -->
    <geom name="table_top" type="box"
          size="{cfg.table_size_x} {cfg.table_size_y} {cfg.table_thickness / 2:.4f}"
          pos="{cfg.table_center_x} 0 {table_z:.4f}" material="table_mat"
          friction="0.9 0.005 0.0001"/>
    {legs}

    <!-- Place bin (open-top tray, pick-and-place target) -->
    {bin_xml}

    <!-- Eye-to-hand RGB-D camera, fixed in the WORLD frame (like a tripod-
         mounted RealSense D455 across the table). MuJoCo convention here;
         conversion to OpenCV happens in CameraModule. -->
    <camera name="{cfg.cam_name}" pos="{cam_pos[0]:.6f} {cam_pos[1]:.6f} {cam_pos[2]:.6f}"
            xyaxes="{cam_xyaxes}" fovy="{cfg.cam_fovy_deg}"/>

    <!-- Close-up side camera: only used to record execution GIFs -->
    <camera name="{cfg.record_cam_name}"
            pos="{cfg.record_cam_pos[0]:.6f} {cfg.record_cam_pos[1]:.6f} {cfg.record_cam_pos[2]:.6f}"
            xyaxes="{rec_xyaxes}" fovy="{cfg.record_cam_fovy_deg}"/>
    {side_cam_xml}

    {objects_xml}
  </worldbody>
</mujoco>
"""
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        out = GENERATED_DIR / '_generated_scene.xml'
        out.write_text(xml, encoding='utf-8')
        return out

    # -- public API ------------------------------------------------------------
    def generate(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Build the scene, spawn objects, settle physics. Returns (model, data)
        ready for observation capture."""
        cfg = self.cfg
        specs, extra_assets = self._sample_objects()
        self.scene_xml_path = self._build_scene_xml(specs, extra_assets)

        model = mujoco.MjModel.from_xml_path(str(self.scene_xml_path))
        data = mujoco.MjData(model)
        self.model, self.data = model, data
        self.object_names = [s.name for s in specs]

        # Map MuJoCo body ids -> segmentation labels 1..N (0 = background)
        self.object_body_ids = {}
        for label, s in enumerate(specs, start=1):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s.name)
            self.object_body_ids[bid] = label

        # --- initial robot configuration (held by position actuators) -------
        for j, q in enumerate(ARM_OBSERVE_QPOS):
            data.qpos[j] = q
            data.ctrl[j] = q
        data.ctrl[7] = GRIPPER_OPEN_CTRL          # gripper open
        data.qpos[7] = 0.04                       # finger joints open
        data.qpos[8] = 0.04

        # --- place objects: random XY + staggered drop heights ---------------
        xy = self._sample_xy_positions(len(specs))
        for i, s in enumerate(specs):
            jadr = model.joint(f'{s.name}_joint').qposadr[0]
            drop_z = cfg.table_height + s.spawn_half_height + 0.015 + 0.04 * i
            data.qpos[jadr:jadr + 3] = [xy[i, 0], xy[i, 1], drop_z]
            # keep the randomized yaw quaternion already baked into the body;
            # freejoint quat starts as identity -> re-randomize here instead:
            yaw = self.rng.uniform(0, 2 * np.pi)
            data.qpos[jadr + 3:jadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

        mujoco.mj_forward(model, data)
        self._settle()
        return model, data

    def _settle(self):
        """Step physics until objects rest (cfg.settle_time guaranteed, then up
        to cfg.max_extra_settle until max object speed < threshold)."""
        cfg, model, data = self.cfg, self.model, self.data
        n_steps = int(cfg.settle_time / model.opt.timestep)
        for _ in range(n_steps):
            mujoco.mj_step(model, data)

        extra_steps = int(cfg.max_extra_settle / model.opt.timestep)
        check_every = max(1, int(0.1 / model.opt.timestep))
        for i in range(extra_steps):
            mujoco.mj_step(model, data)
            if i % check_every == 0 and self._max_object_speed() < cfg.settle_qvel_thresh:
                break

    def _max_object_speed(self) -> float:
        speeds = [0.0]
        for name in self.object_names:
            jadr = self.model.joint(f'{name}_joint').dofadr[0]
            speeds.append(float(np.abs(self.data.qvel[jadr:jadr + 6]).max()))
        return max(speeds)

    def bin_drop_point(self) -> np.ndarray:
        """World point above which the executor releases objects."""
        bx, by = self.cfg.bin_center
        return np.array([bx, by, self.cfg.table_height + 0.02])

    def objects_in_bin(self) -> list[str]:
        """Names of objects currently inside the place bin."""
        cfg, out = self.cfg, []
        bx, by = cfg.bin_center
        tol = cfg.bin_inner_half + 0.02
        for name in self.object_names:
            jadr = self.model.joint(f'{name}_joint').qposadr[0]
            x, y, z = self.data.qpos[jadr:jadr + 3]
            if (abs(x - bx) < tol and abs(y - by) < tol
                    and cfg.table_height - 0.01 < z < cfg.table_height + 0.20):
                out.append(name)
        return out

    def objects_on_table(self) -> list[str]:
        """Names of objects that are still on the tabletop after settling."""
        cfg, out = self.cfg, []
        for name in self.object_names:
            jadr = self.model.joint(f'{name}_joint').qposadr[0]
            x, y, z = self.data.qpos[jadr:jadr + 3]
            if (z > cfg.table_height - 0.05
                    and abs(x - cfg.table_center_x) < cfg.table_size_x
                    and abs(y) < cfg.table_size_y):
                out.append(name)
        return out
