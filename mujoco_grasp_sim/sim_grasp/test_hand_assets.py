"""Standalone check for the merged Panda+Shadow-Hand MJCF asset -- run
directly, no pytest (this codebase has no automated test suite)."""
import mujoco

from sim_grasp.hand_assets import build_panda_shadow_hand_xml
from sim_grasp.scene_generator import GENERATED_DIR

xml_name = build_panda_shadow_hand_xml()
xml_path = GENERATED_DIR / xml_name
assert xml_path.exists(), f'{xml_path} was not written'

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)   # confirms the model is physically valid, not just parseable

# 7 arm actuators + 18 Shadow Hand actuators, in that order.
assert model.nu == 25, f'expected 25 actuators (7 arm + 18 hand), got {model.nu}'
expected_hand_actuators = [
    'rh_A_THJ5', 'rh_A_THJ4', 'rh_A_THJ3', 'rh_A_THJ2', 'rh_A_THJ1',
    'rh_A_FFJ4', 'rh_A_FFJ3', 'rh_A_FFJ0',
    'rh_A_MFJ4', 'rh_A_MFJ3', 'rh_A_MFJ0',
    'rh_A_RFJ4', 'rh_A_RFJ3', 'rh_A_RFJ0',
    'rh_A_LFJ5', 'rh_A_LFJ4', 'rh_A_LFJ3', 'rh_A_LFJ0',
]
assert len(expected_hand_actuators) == 18
for i, name in enumerate(expected_hand_actuators):
    actual = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 7 + i)
    assert actual == name, f'actuator {7+i}: expected {name!r}, got {actual!r}'

# rh_palm must be a direct child of link7 (the old "hand" body's former
# parent), fixed (no free/hinge joint of its own -- the wrist was dropped).
palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'rh_palm')
assert palm_bid != -1, 'rh_palm body not found'
link7_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link7')
assert model.body_parentid[palm_bid] == link7_bid, \
    'rh_palm must be a direct child of link7'
assert model.body_jntnum[palm_bid] == 0, \
    'rh_palm must have no joint of its own (wrist dropped, rigidly fixed)'

# The old Panda gripper bodies/joints must be gone.
assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand') == -1
assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint1') == -1
assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'rh_WRJ1') == -1
assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'rh_WRJ2') == -1

print('All hand_assets checks passed.')
