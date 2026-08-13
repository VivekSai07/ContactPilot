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
        out['fell_off'] = len(pa.get('fell_off_table', []))
        out['rounds'] = len(pa.get('rounds', []))
        out['fail_stages'] = [r['pick'].get('stage')
                              for r in pa.get('rounds', [])
                              if not r['pick'].get('success')]
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
    ap.add_argument('--backend', choices=['cgn', 'graspgen'], default='cgn',
                    help='forward --backend to the run script')
    ap.add_argument('--graspgen-python', default=None,
                    help='forward --graspgen-python to the run script (or rely '
                         'on the GRASPGEN_PYTHON env var, same as the run script)')
    ap.add_argument('--tag', default=None, help='output/bench_<tag>/')
    args = ap.parse_args()

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
