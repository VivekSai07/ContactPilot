"""One-time, offline tool: extracts one validated ShadowHand power grasp
from a downloaded DexGraspNet .npy grasp file and writes
sim_grasp/shadow_hand_posture.py. Not part of the runtime pipeline --
DexGraspNet's own code is never imported; only its published data format
(a dict keyed by joint-name strings, per their own quick_example.ipynb)
is read here.

Usage: python scripts/extract_shadow_hand_posture.py <path/to/grasp.npy> [--index N]
"""
import argparse
import numpy as np

# Menagerie actuator name -> DexGraspNet joint-name key, per the verified
# rule: rh_{finger}J{n} = DexGraspNet robot0:{finger}J{n-1}. For the 3
# fingers whose middle+distal are tendon-coupled on the real hardware (and
# in this Menagerie model) into one actuator, the DISTAL joint's angle is
# used for that shared actuator target -- an accepted v1 approximation
# (see this plan's Global Constraints), not an exact reproduction.
ACTUATOR_TO_DEXGRASPNET_JOINT = {
    'rh_A_THJ5': 'robot0:THJ4', 'rh_A_THJ4': 'robot0:THJ3',
    'rh_A_THJ3': 'robot0:THJ2', 'rh_A_THJ2': 'robot0:THJ1',
    # NOTE: rh_A_THJ1 ('thdistal' class, joint 'rh_THJ1') is intentionally
    # NOT mapped here. Verified directly against both MJCFs: DexGraspNet's
    # 'robot0:THJ0' has axis="0 1 0", range="-1.571 0", while Menagerie's
    # rh_THJ1 has axis="1 0 0" (inherited from the top-level 'right_hand'
    # class -- 'thdistal' only overrides range), range="-0.261799 1.5708".
    # These are different rotational axes of the same body, not a sign-flip
    # of the same axis (unlike thmiddle/THJ1, which IS a clean sign-flip and
    # is mapped as-is above) -- there is no valid linear transform (copy,
    # negate, scale) from DexGraspNet's THJ0 onto this DOF. A fixed manual
    # value is substituted for it below instead of a DexGraspNet-sourced one.
    'rh_A_FFJ4': 'robot0:FFJ3', 'rh_A_FFJ3': 'robot0:FFJ2',
    'rh_A_FFJ0': 'robot0:FFJ0',    # coupled FFJ2+FFJ1 -> use distal (FFJ0)
    'rh_A_MFJ4': 'robot0:MFJ3', 'rh_A_MFJ3': 'robot0:MFJ2',
    'rh_A_MFJ0': 'robot0:MFJ0',
    'rh_A_RFJ4': 'robot0:RFJ3', 'rh_A_RFJ3': 'robot0:RFJ2',
    'rh_A_RFJ0': 'robot0:RFJ0',
    'rh_A_LFJ5': 'robot0:LFJ4', 'rh_A_LFJ4': 'robot0:LFJ3',
    'rh_A_LFJ3': 'robot0:LFJ2', 'rh_A_LFJ0': 'robot0:LFJ0',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('npy_path')
    ap.add_argument('--index', type=int, default=0,
                    help='which grasp entry in the file to use (default: first)')
    args = ap.parse_args()

    grasp_data = np.load(args.npy_path, allow_pickle=True)
    qpos = grasp_data[args.index]['qpos']

    power_grasp = {act: float(qpos[joint])
                   for act, joint in ACTUATOR_TO_DEXGRASPNET_JOINT.items()}
    # rh_A_THJ1 has no valid DexGraspNet-sourced equivalent (see the comment
    # by the mapping table above -- different rotational axis, not merely a
    # differently-signed or out-of-range value). Use a fixed, safely-in-range
    # (thdistal class: [-0.261799, 1.5708]) modest thumb-tip curl instead, the
    # same "known v1 approximation" pattern already used for the
    # tendon-coupled middle/distal joints elsewhere in this mapping.
    power_grasp['rh_A_THJ1'] = 0.5
    # Open posture: all zeros is the Shadow Hand's natural relaxed-open
    # pose for every joint in this model (each joint's <default> range
    # straddles or starts at 0 -- confirmed against right_hand.xml's
    # <default> ranges).
    open_posture = {act: 0.0 for act in ACTUATOR_TO_DEXGRASPNET_JOINT}
    open_posture['rh_A_THJ1'] = 0.0

    out_path = 'sim_grasp/shadow_hand_posture.py'
    with open(out_path, 'w') as f:
        f.write('"""Fixed Shadow Hand postures -- see extract_shadow_hand_posture.py '
               'for provenance (mined from DexGraspNet, CC BY-NC 4.0, grasp '
               f'{args.npy_path!r} index {args.index}). '
               'rh_A_THJ1 is NOT DexGraspNet-sourced: it has no valid mapping '
               '(different rotational axis from any DexGraspNet thumb joint) '
               'and is instead a fixed, manually-chosen, safely-in-range '
               'value (0.5 rad)."""\n\n')
        f.write(f'SHADOW_HAND_OPEN_POSTURE = {open_posture!r}\n\n')
        f.write(f'SHADOW_HAND_POWER_GRASP_POSTURE = {power_grasp!r}\n')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
