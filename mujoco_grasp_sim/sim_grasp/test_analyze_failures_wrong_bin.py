"""analyze_failures must classify a props-scene object that was binned
into the WRONG bin as 'wrong_bin' (P10 SP1), and must not flag a
correctly-sorted or a legacy boxes-mode round. Pure dicts, no MuJoCo.
Run directly from mujoco_grasp_sim/ with PYTHONPATH=., no pytest."""
from analyze_failures import analyze_run

ok_pick = {'success': True}
m = {'pick_all': {'rounds': [
    {'round': 1, 'body': 'obj_0', 'score': 0.9, 'pick': ok_pick,
     'place': {'stage': 'placed'}, 'in_bin': True,
     'category': 'food', 'target_bin': 'A', 'landed_bin': 'B'},
    {'round': 2, 'body': 'obj_1', 'score': 0.8, 'pick': ok_pick,
     'place': {'stage': 'placed'}, 'in_bin': True,
     'category': 'non_food', 'target_bin': 'B', 'landed_bin': 'B'},
    {'round': 3, 'body': 'obj_2', 'score': 0.7, 'pick': ok_pick,
     'place': {'stage': 'placed'}, 'in_bin': True},        # boxes-mode round
], 'fell_off_table': []}}

ev = analyze_run(m, 'seed_0')
assert [e['category'] for e in ev] == ['wrong_bin'], ev
assert ev[0]['body'] == 'obj_0' and 'B' in ev[0]['detail'] and 'A' in ev[0]['detail']
print('analyze_failures wrong_bin checks passed.')
