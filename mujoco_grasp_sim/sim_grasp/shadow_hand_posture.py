"""Fixed Shadow Hand postures -- see extract_shadow_hand_posture.py for
provenance. Mined from DexGraspNet (CC BY-NC 4.0), object
'ddg-gd_box_poisson_019' (a box -- matching this project's own box-shaped
scene objects), grasp index 263, chosen as the entry with the fullest,
most consistent 4-finger curl in that file (mean |joint angle| 0.466 rad,
vs. 0.175-0.35 for most other entries in the same file) -- a genuine
enclosing power grasp, not an edge/precision pinch."""

SHADOW_HAND_OPEN_POSTURE = {'rh_A_THJ5': 0.0, 'rh_A_THJ4': 0.0, 'rh_A_THJ3': 0.0, 'rh_A_THJ2': 0.0, 'rh_A_THJ1': 0.0, 'rh_A_FFJ4': 0.0, 'rh_A_FFJ3': 0.0, 'rh_A_FFJ0': 0.0, 'rh_A_MFJ4': 0.0, 'rh_A_MFJ3': 0.0, 'rh_A_MFJ0': 0.0, 'rh_A_RFJ4': 0.0, 'rh_A_RFJ3': 0.0, 'rh_A_RFJ0': 0.0, 'rh_A_LFJ5': 0.0, 'rh_A_LFJ4': 0.0, 'rh_A_LFJ3': 0.0, 'rh_A_LFJ0': 0.0}

SHADOW_HAND_POWER_GRASP_POSTURE = {'rh_A_THJ5': 0.3315754532814026, 'rh_A_THJ4': 1.1239268779754639, 'rh_A_THJ3': 0.00886788684874773, 'rh_A_THJ2': -0.5216392278671265, 'rh_A_THJ1': -1.2037444114685059, 'rh_A_FFJ4': 0.25657305121421814, 'rh_A_FFJ3': 0.2358376532793045, 'rh_A_FFJ0': 0.5914084315299988, 'rh_A_MFJ4': -0.0019584910478442907, 'rh_A_MFJ3': 0.22507211565971375, 'rh_A_MFJ0': 0.5931413173675537, 'rh_A_RFJ4': -0.11233185231685638, 'rh_A_RFJ3': 0.14670445024967194, 'rh_A_RFJ0': 0.5680378079414368, 'rh_A_LFJ5': 0.22676876187324524, 'rh_A_LFJ4': -0.3344805836677551, 'rh_A_LFJ3': 0.25026214122772217, 'rh_A_LFJ0': 0.594383180141449}
