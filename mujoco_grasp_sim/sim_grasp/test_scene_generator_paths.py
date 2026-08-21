"""Standalone check that generated/patched MJCF lands in GENERATED_DIR (not
inside the vendored mujoco_menagerie/ tree) and that the model still loads
correctly (i.e. the absolute-meshdir rewrite didn't break mesh resolution).
Run directly, no pytest (this codebase has no automated test suite)."""
from sim_grasp.scene_generator import (
    GENERATED_DIR, MENAGERIE_PANDA_DIR, SceneConfig, SceneGenerator,
)

gen = SceneGenerator(SceneConfig(seed=0))
model, data = gen.generate()

# Generated files must live in GENERATED_DIR, never inside the vendored dir.
assert gen.scene_xml_path.parent == GENERATED_DIR, (
    f'generated scene XML written to {gen.scene_xml_path.parent}, '
    f'expected {GENERATED_DIR}')
patched_panda = GENERATED_DIR / '_panda_sim_patched.xml'
assert patched_panda.is_file(), f'{patched_panda} was not written'
assert not (MENAGERIE_PANDA_DIR / '_panda_sim_patched.xml').exists(), (
    'stale patched panda.xml still present inside the vendored menagerie dir')
assert not (MENAGERIE_PANDA_DIR / '_generated_scene.xml').exists(), (
    'stale generated scene XML still present inside the vendored menagerie dir')

# Model must have actually loaded meshes (not silently fallen back to zero
# geoms) -- the panda arm alone has dozens of mesh geoms.
n_mesh_geoms = sum(1 for i in range(model.ngeom)
                    if model.geom_type[i] == 7)  # mjGEOM_MESH == 7
assert n_mesh_geoms > 10, (
    f'expected >10 mesh geoms from the Panda arm, got {n_mesh_geoms} '
    '(meshdir rewrite likely broke mesh resolution)')

print(f'All scene_generator path checks passed ({n_mesh_geoms} mesh geoms loaded).')
