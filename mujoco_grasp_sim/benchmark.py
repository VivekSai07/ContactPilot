"""Batch evaluation harness for the grasping pipeline (P1 of ROADMAP.md).

Runs run_sim_grasp_test.py headless over many seeds, parses each run's
metrics.json, and aggregates success rates + failure statistics. Every
reliability change should be judged by this number, not by single runs
(Contact-GraspNet inference is stochastic).

Usage:
    python benchmark.py --seeds 0-9 --camera lookat --mode execute --top-k 5
    python benchmark.py --seeds 0-4 --camera lookat --mode pick-all
    python benchmark.py --seeds 1,3,7 --mode predict        # prediction only

Outputs land in output/bench_<tag>/: one subdir per seed plus summary.json,
and a table is printed at the end.
"""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_seeds(spec: str) -> list[int]:
    seeds = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def summarize_sorting(results: list[dict]) -> dict:
    """Aggregate truth-scored sorting outcomes over completed seeds only."""
    ok = [r for r in results if not r['crashed']]
    return {'completed': len(ok), 'crashed': len(results) - len(ok),
            'total': sum(r.get('total') or 0 for r in ok),
            'correct': sum(r.get('in_correct_bin', 0) for r in ok),
            'wrong': sum(r.get('in_wrong_bin', 0) for r in ok),
            'binned': sum(r.get('in_bin', 0) for r in ok),
            'fell_off': sum(r.get('fell_off', 0) for r in ok)}


def format_location_agreement(values: list[float]) -> str:
    """Show mean graph location agreement without hiding small mismatches."""
    # Why: whole-percent rounding made a 99.58% cohort appear perfectly 100%.
    return f'{100 * sum(values) / max(len(values), 1):.1f}%'


def effective_identity(scene_graph: bool, identity: str | None) -> str | None:
    """Return the actual NIM identity mode that the run script will use."""
    # Why: the run script defaults graph mode to perceived; summaries must agree.
    return identity or ('perceived' if scene_graph else None)


def run_one(seed: int, args, run_dir: Path) -> dict:
    cmd = [sys.executable, str(HERE / 'run_sim_grasp_test.py'),
           '--seed', str(seed), '--no-vis', '--camera', args.camera,
           '--save-dir', str(run_dir), '--backend', args.backend]
    if args.backend == 'graspgen' and args.graspgen_python:
        cmd += ['--graspgen-python', args.graspgen_python]
    if args.n_objects:
        cmd += ['--n-objects', str(args.n_objects)]
    if args.recenter:
        cmd += ['--recenter']
    if args.clean_depth:
        cmd += ['--clean-depth']
    if args.filter_neighbors:
        cmd += ['--filter-neighbors']
    if args.scene != 'boxes':          # keep the boxes command line byte-identical
        cmd += ['--scene', args.scene]
    if args.scene_graph:
        cmd += ['--scene-graph']
        if args.identity:
            cmd += ['--identity', args.identity]
    if args.routing != 'oracle':
        cmd += ['--routing', args.routing]
    if args.mode == 'execute':
        cmd += ['--execute', '--top-k', str(args.top_k)]
    elif args.mode == 'pick-all':
        cmd += ['--pick-all']
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0

    out: dict = {'seed': seed, 'wall_s': round(wall, 1),
                 'crashed': proc.returncode != 0}
    if out['crashed']:
        out['error_tail'] = (proc.stderr or proc.stdout or '')[-800:]
        return out
    try:
        m = json.loads((run_dir / 'metrics.json').read_text())
    except Exception as e:  # run finished but metrics unreadable
        out['crashed'], out['error_tail'] = True, f'metrics.json: {e}'
        return out

    out['objects'] = m.get('objects_on_table')
    out['num_grasps'] = m.get('num_grasps')
    out['objects_with_grasps'] = sum(
        1 for v in m.get('per_object', {}).values() if v.get('num_grasps'))
    if args.mode == 'execute':
        attempts = m.get('execution', [])
        out['attempts'] = len(attempts)
        out['pick_success'] = any(a.get('success') for a in attempts)
        out['fail_stages'] = [a.get('stage') for a in attempts
                              if not a.get('success')]
    elif args.mode == 'pick-all':
        pa = m.get('pick_all', {})
        out['in_bin'] = len(pa.get('in_bin', []))
        out['total'] = pa.get('objects_total')
        if 'in_correct_bin' in pa:
            out['in_correct_bin'] = len(pa['in_correct_bin'])
            out['in_wrong_bin'] = len(pa.get('in_wrong_bin', []))
        out['fell_off'] = len(pa.get('fell_off_table', []))
        out['rounds'] = len(pa.get('rounds', []))
        out['fail_stages'] = [r['pick'].get('stage')
                              for r in pa.get('rounds', [])
                              if not r['pick'].get('success')]
        sgm = m.get('scene_graph')
        if sgm:
            objs = sgm.get('objects', {})
            out['cat_correct'] = sum(1 for o in objs.values() if o.get('correct'))
            out['cat_total'] = len(objs)
            out['loc_agreement'] = sgm.get('location_agreement')
            out['nim_calls'] = sgm.get('nim_calls')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', default='0-9', help='e.g. "0-9" or "1,3,7"')
    ap.add_argument('--camera', choices=['calibrated', 'lookat', 'fused'],
                    default='lookat')
    ap.add_argument('--mode', choices=['predict', 'execute', 'pick-all'],
                    default='execute')
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--n-objects', type=int, default=None)
    ap.add_argument('--recenter', action='store_true',
                    help='forward --recenter to the run script')
    ap.add_argument('--clean-depth', action='store_true',
                    help='forward --clean-depth to the run script')
    ap.add_argument('--filter-neighbors', action='store_true',
                    help='forward --filter-neighbors to the run script')
    ap.add_argument('--scene', choices=['boxes', 'props'], default='boxes',
                    help='passed through to run_sim_grasp_test.py (P10)')
    ap.add_argument('--scene-graph', action='store_true',
                    help='forward --scene-graph (props pick-all only)')
    ap.add_argument('--identity', choices=['perceived', 'oracle'], default=None,
                    help='forward --identity (needs --scene-graph)')
    ap.add_argument('--routing', choices=['oracle', 'scene-graph'], default='oracle',
                    help='forward --routing to the run script (P10 SP3)')
    ap.add_argument('--backend', choices=['cgn', 'graspgen'], default='cgn',
                    help='forward --backend to the run script')
    ap.add_argument('--graspgen-python', default=None,
                    help='forward --graspgen-python to the run script (or rely '
                         'on the GRASPGEN_PYTHON env var, same as the run script)')
    ap.add_argument('--tag', default=None, help='output/bench_<tag>/')
    args = ap.parse_args()
    # why: same guard as run_sim_grasp_test.py; fail before any run starts
    if args.identity is not None and not args.scene_graph:
        ap.error('--identity requires --scene-graph')
    if args.routing == 'scene-graph' and not (args.scene == 'props' and
                                              args.mode == 'pick-all' and args.scene_graph):
        ap.error('--routing scene-graph requires --scene props --mode pick-all --scene-graph')
    args.identity = effective_identity(args.scene_graph, args.identity)

    seeds = parse_seeds(args.seeds)
    tag = args.tag or f'{args.mode}_{args.camera}_{time.strftime("%m%d_%H%M")}'
    bench_dir = HERE / 'output' / f'bench_{tag}'
    bench_dir.mkdir(parents=True, exist_ok=True)
    print(f'[bench] {len(seeds)} seeds, mode={args.mode}, camera={args.camera} '
          f'-> {bench_dir}')

    results = []
    for k, seed in enumerate(seeds, 1):
        print(f'[bench] run {k}/{len(seeds)} (seed {seed})...', flush=True)
        r = run_one(seed, args, bench_dir / f'seed_{seed}')
        results.append(r)
        print(f'[bench]   {r}', flush=True)
        (bench_dir / 'summary.json').write_text(json.dumps(
            {'args': vars(args), 'results': results}, indent=2))

    # ---- aggregate ----------------------------------------------------------
    ok = [r for r in results if not r['crashed']]
    print(f'\n[bench] ===== {len(ok)}/{len(results)} runs completed '
          f'(crashed: {len(results) - len(ok)}) =====')
    if args.scene == 'props':
        print(f'[bench] routing={args.routing}, identity='
              f'{args.identity if args.scene_graph else "none"}; '
              'correct-bin outcomes use simulator truth for offline scoring')
    if args.mode == 'execute' and ok:
        n_succ = sum(r['pick_success'] for r in ok)
        print(f'[bench] pick success: {n_succ}/{len(ok)} scenes '
              f'({100 * n_succ / len(ok):.0f}%)')
    elif args.mode == 'pick-all' and ok:
        binned = sum(r['in_bin'] for r in ok)
        total = sum(r['total'] or 0 for r in ok)
        fell = sum(r['fell_off'] for r in ok)
        print(f'[bench] objects binned: {binned}/{total} '
              f'({100 * binned / max(total, 1):.0f}%), knocked off table: {fell}')
        if any('in_correct_bin' in r for r in ok):
            sorting = summarize_sorting(results)
            corr = sorting['correct']
            print(f'[bench] objects in correct bin: {corr}/{total} '
                  f'({100 * corr / max(total, 1):.0f}%)')
            print(f'[bench] objects in wrong bin: {sorting["wrong"]}/{total}')
        if any('cat_total' in r for r in ok):
            cc = sum(r.get('cat_correct', 0) for r in ok)
            ct = sum(r.get('cat_total', 0) for r in ok)
            la = [r['loc_agreement'] for r in ok if r.get('loc_agreement') is not None]
            # Why: oracle identity tests knowledge from true names, not visual identification.
            identity_label = ('oracle-identity' if args.identity == 'oracle'
                              else 'perceived')
            print(f'[bench] {identity_label} category accuracy: {cc}/{ct} '
                  f'({100 * cc / max(ct, 1):.0f}%), mean location agreement: '
                  f'{format_location_agreement(la)}')
    if ok:
        cov_have = sum(r.get('objects_with_grasps') or 0 for r in ok)
        cov_all = sum(r.get('objects') or 0 for r in ok)
        print(f'[bench] initial-observation grasp coverage: '
              f'{cov_have}/{cov_all} objects')
        stages = Counter(s for r in ok for s in r.get('fail_stages', []))
        if stages:
            print(f'[bench] failure stages: {dict(stages)}')
        mean_t = sum(r['wall_s'] for r in ok) / len(ok)
        print(f'[bench] mean wall time per run: {mean_t:.0f}s')
    print(f'[bench] full per-run details: {bench_dir / "summary.json"}')


if __name__ == '__main__':
    main()
