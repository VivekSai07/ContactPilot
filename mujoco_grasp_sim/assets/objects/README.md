# Mesh object library

Drop `.stl` / `.obj` mesh files here (e.g. YCB models) and the scene
generator will randomly mix them with primitive shapes
(`SceneConfig.mesh_probability`, default 0.4).

Meshes are auto-scaled so their largest dimension is <= 12 cm (graspable by
the Panda gripper). Collision uses MuJoCo's default convex hull.

YCB models: https://www.ycbbenchmarks.com/object-models/
(use the `google_16k` .obj versions; textures are optional, geometry is enough)
