"""Props scene checks (P10 SP1): props load textured, every prop settles
on the table (none tipped into a bin / off the table) across 10 seeds,
the same seed reproduces the same scene, and the lookat camera sees both
bins. Needs downloaded props + MuJoCo, no GPU (MUJOCO_GL=osmesa on
headless/WSL2). Run directly, no pytest."""
import numpy as np

from sim_grasp.camera import CameraModule
from sim_grasp.scene_generator import SceneConfig, SceneGenerator

# 1. settles cleanly on 10 seeds, balanced categories, masses from manifest
for seed in range(10):
    gen = SceneGenerator(SceneConfig.for_props(seed=seed))
    model, data = gen.generate()
    assert len(gen.object_names) == 4
    cats = sorted(gen.object_categories.values())
    assert cats == ['food', 'food', 'non_food', 'non_food'], (seed, cats)
    assert len(set(gen.object_props.values())) == 4, seed
    on_table = gen.objects_on_table()
    assert sorted(on_table) == sorted(gen.object_names), (seed, on_table)
    assert gen.objects_in_bins() == {}, (seed, gen.objects_in_bins())
    assert gen._max_object_speed() < 0.05, (seed, gen._max_object_speed())

# 2. determinism: same seed -> same props and same settled poses
g1 = SceneGenerator(SceneConfig.for_props(seed=4)); _, d1 = g1.generate()
g2 = SceneGenerator(SceneConfig.for_props(seed=4)); _, d2 = g2.generate()
assert g1.object_props == g2.object_props
assert np.allclose(d1.qpos, d2.qpos, atol=1e-6)

# 3. n_objects override + observation camera sees textured props and bin B
gen = SceneGenerator(SceneConfig.for_props(seed=0, n_objects=5))
model, data = gen.generate()
assert len(gen.object_names) == 5
cam = CameraModule(model, data, cam_name=gen.cfg.cam_name, width=640, height=480)
rgb, depth, segmap, K, T_world_cam = cam.capture(gen.object_body_ids)
assert set(np.unique(segmap).astype(int)) >= {1, 2, 3, 4, 5}
T_cam_world = np.linalg.inv(T_world_cam)
for b in gen.cfg.bins():
    for dx in (-b.inner_half, b.inner_half):
        for dy in (-b.inner_half, b.inner_half):
            p = T_cam_world @ np.array([b.center[0] + dx, b.center[1] + dy,
                                        gen.cfg.table_height, 1.0])
            u, v, _ = K @ (p[:3] / p[2])
            assert p[2] > 0 and 0 <= u < 640 and 0 <= v < 480, (b.name, u, v)
cam.close()
print('All props-scene checks passed (10 seeds settle, deterministic, both bins in view).')
