# Semantic props (ROADMAP P10)

`manifest.json` lists the 12 box-shaped household props used by
`run_sim_grasp_test.py --scene props` (6 `food`, 6 `non_food`), with a
per-prop mass. The meshes/textures themselves are **not committed** —
fetch them once (idempotent):

    cd mujoco_grasp_sim
    python scripts/download_props.py

They land in `gso/<model_id>/` (gitignored).

## Attribution

3D models from **Google Scanned Objects** (Downs et al., 2022,
"Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household
Items"), licensed **CC-BY 4.0**, via the MuJoCo conversion
[kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects)
(MJCF files MIT), pinned at commit `6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c`.
Props are uniformly scaled at load time (see `sim_grasp/props.py`).
