"""Failure taxonomy from batch runs (P1 of ROADMAP.md).

Reads metrics.json files produced by run_sim_grasp_test.py (directly, or the
seed_*/ subdirs of a benchmark.py output dir) and classifies every failed
attempt into the taxonomy:

    no_grasp_prediction  CGN returned zero grasps for an object at the initial
                         observation (coverage failure / occlusion)
    ik_unreachable       IK did not converge for pre-grasp or grasp pose
    closed_on_air        fingers closed fully -> grasp was laterally offset
                         from the object (prediction offset)
    object_displaced     object ended LOWER than it started -> bulldozed /
                         pushed during approach (collision)
    unstable_slip        fingers held something but the object did not rise
                         enough -> bad contact, slipped out during lift
    place_unreachable    pick succeeded but the bin was unreachable by IK
    missed_bin           placed, but the object bounced/landed outside the bin
    knocked_off_table    object left the table entirely during the run

Usage:
    python analyze_failures.py output/bench_baseline_pickall
    python analyze_failures.py output/run_20260611_xyz        # single run
Writes <dir>/taxonomy.json and prints a summary table.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

CLOSED_ON_AIR_M = 0.003   # finger opening below this = nothing between fingers
DISPLACED_M = -0.02       # object ended this much lower = pushed off/over


def classify_pick(res: dict) -> tuple[str, str]:
    """Classify one failed executor.execute() result -> (category, detail)."""
    stage = res.get('stage')
    if stage in ('ik_pregrasp', 'ik_grasp'):
        return 'ik_unreachable', (f"{stage}, pos_err {res.get('pos_err', 0) * 1e3:.0f} mm")
    raised = res.get('object_raised_m')
    opening = res.get('finger_opening_m')
    if opening is not None and opening <= CLOSED_ON_AIR_M:
        return 'closed_on_air', f'opening {opening * 1e3:.1f} mm, raised {raised} m'
    if raised is not None and raised < DISPLACED_M:
        return 'object_displaced', f'raised {raised} m (pushed down/away)'
    return 'unstable_slip', f'opening {opening} m, raised {raised} m'


def analyze_run(m: dict, seed) -> list[dict]:
    """All failure events from one run's metrics dict."""
    events = []

    def add(cat, detail, **kw):
        events.append({'seed': seed, 'category': cat, 'detail': detail, **kw})

    for sid, info in m.get('per_object', {}).items():
        if not info.get('num_grasps'):
            add('no_grasp_prediction', f"object {sid} at initial observation",
                object=int(sid))

    for k, res in enumerate(m.get('execution', []), 1):
        if not res.get('success'):
            cat, detail = classify_pick(res)
            add(cat, detail, attempt=k, object=res.get('object'),
                score=res.get('score'))

    pa = m.get('pick_all', {})
    for r in pa.get('rounds', []):
        pick = r.get('pick', {})
        where = {'round': r.get('round'), 'body': r.get('body'),
                 'score': r.get('score')}
        if not pick.get('success'):
            cat, detail = classify_pick(pick)
            add(cat, detail, **where)
        elif r.get('place', {}).get('stage') == 'ik_place':
            add('place_unreachable', 'released in place, retried later', **where)
        elif not r.get('in_bin', True):
            add('missed_bin', 'placed but landed outside the bin', **where)
    for body in pa.get('fell_off_table', []):
        add('knocked_off_table', 'off the table by end of run', body=body)
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dir', help='benchmark dir (with seed_*/) or single run dir')
    args = ap.parse_args()
    root = Path(args.run_dir)

    runs = []   # (label, metrics dict)
    if (root / 'metrics.json').exists():
        runs.append((root.name, json.loads((root / 'metrics.json').read_text())))
    for d in sorted(root.glob('seed_*')):
        f = d / 'metrics.json'
        if f.exists():
            runs.append((d.name, json.loads(f.read_text())))
    if not runs:
        sys.exit(f'no metrics.json found under {root}')

    events, totals = [], Counter()
    attempts = succ = binned = objects = 0
    for label, m in runs:
        ev = analyze_run(m, label)
        events.extend(ev)
        totals.update(e['category'] for e in ev)
        objects += m.get('objects_on_table') or 0
        pa = m.get('pick_all', {})
        for r in pa.get('rounds', []):
            attempts += 1
            succ += bool(r.get('pick', {}).get('success'))
        binned += len(pa.get('in_bin', []))
        for res in m.get('execution', []):
            attempts += 1
            succ += bool(res.get('success'))

    print(f'[taxonomy] {len(runs)} runs, {objects} objects, '
          f'{attempts} pick attempts ({succ} ok), {binned} binned')
    print(f'[taxonomy] {len(events)} failure events:')
    for cat, n in totals.most_common():
        print(f'  {cat:<22} {n}')
    print('[taxonomy] details:')
    for e in events:
        loc = e.get('body') or e.get('object')
        rnd = f" r{e['round']}" if e.get('round') is not None else ''
        print(f"  {e['seed']}{rnd}  {e['category']:<22} {loc}: {e['detail']}")

    out = root / 'taxonomy.json'
    out.write_text(json.dumps({'totals': dict(totals), 'events': events}, indent=2))
    print(f'[taxonomy] written: {out}')


if __name__ == '__main__':
    main()
