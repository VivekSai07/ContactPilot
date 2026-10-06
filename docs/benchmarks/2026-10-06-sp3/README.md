# SP3 paired sorting benchmark — 2026-10-06

These are the three `benchmark.py` `summary.json` outputs from the same
GraspGen/fused props seeds 0–9. Only a final newline was added when copying
the files here; their parsed JSON values match the local run outputs.
Per-seed renders, GIFs, and full `metrics.json` files remain in ignored local
`mujoco_grasp_sim/output/` directories. No credential is included here.

Run from `mujoco_grasp_sim/` in the `cgn_torch` environment, with downloaded
props/checkpoints, a working `GRASPGEN_PYTHON`, and `NVIDIA_API_KEY` for
graph modes:

```bash
MUJOCO_GL=osmesa python benchmark.py --seeds 0-9 --mode pick-all --scene props --camera fused --backend graspgen --routing oracle --tag sp3_oracle_20261006
MUJOCO_GL=osmesa python benchmark.py --seeds 0-9 --mode pick-all --scene props --camera fused --backend graspgen --scene-graph --routing scene-graph --identity perceived --tag sp3_graph_perceived_20261006
MUJOCO_GL=osmesa python benchmark.py --seeds 0-9 --mode pick-all --scene props --camera fused --backend graspgen --scene-graph --routing scene-graph --identity oracle --tag sp3_graph_oracle_identity_20261006
```

The three tracked JSON files preserve every seed's completion flag and
correct-bin, wrong-bin, category, and location results. `ROADMAP.md` P10
contains the interpreted comparison and known limitations. The benchmarks
are stochastic and the NIM service is external, so reruns need not reproduce
the same exact counts.
