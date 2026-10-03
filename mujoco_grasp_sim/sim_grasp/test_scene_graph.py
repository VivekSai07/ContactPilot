"""Standalone checks for scene_graph.build (no MuJoCo/network)."""
import json
import numpy as np

from sim_grasp.scene_generator import BinSpec
from sim_grasp.scene_graph import build, MIN_POINTS, NEAR_DIST

H, W = 120, 160
K = np.array([[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]])
# camera 1 m above the table looking straight down; image-right = world +x
T = np.eye(4); T[:3, :3] = np.diag([1.0, -1.0, -1.0]); T[:3, 3] = [0.5, 0.0, 1.75]
TABLE = 0.75
bins = [BinSpec('A', (0.80, 0.0), 0.08, 'food'), BinSpec('B', (0.20, 0.0), 0.08, 'non_food')]


def pix(x, y):          # world (x, y) on the table -> pixel (u, v)
    return int(round(K[0, 0] * (x - 0.5) / 1.0 + W / 2)), int(round(K[1, 1] * -(y - 0.0) / 1.0 + H / 2))


depth = np.full((H, W), 1.0, np.float32)        # tabletop everywhere
segmap = np.zeros((H, W), np.float32)
rgb = np.full((H, W, 3), 200, np.uint8)

def blob(label, x, y, half_px, z_top, colour):
    u, v = pix(x, y)
    segmap[v - half_px:v + half_px, u - half_px:u + half_px] = label
    depth[v - half_px:v + half_px, u - half_px:u + half_px] = 1.75 - z_top
    rgb[v - half_px:v + half_px, u - half_px:u + half_px] = colour

blob(1, 0.45, 0.0, 4, TABLE + 0.02, (255, 0, 0))        # table, left
blob(2, 0.53, 0.0, 4, TABLE + 0.02, (0, 0, 255))        # table, right, near #1
blob(3, 0.80, 0.0, 4, TABLE + 0.14, (0, 255, 0))        # tall prop standing in bin A: top face above the 5 cm wall
u, v = pix(0.30, 0.10); segmap[v, u:u + 3] = 4             # 3-pixel sliver
# label 5 deliberately absent (occluded object)

g = build(depth, segmap, K, T, rgb, bins, TABLE)
assert set(g.nodes) == {1, 2, 3, 4}, g.nodes.keys()          # no node for an absent label
assert g.nodes[1].location == 'table' and g.nodes[2].location == 'table'
assert g.nodes[3].location == 'A'                            # bin footprint, regardless of height
assert g.nodes[4].location == 'unknown'                      # < MIN_POINTS
assert g.nodes[1].colour_name == 'red' and g.nodes[2].colour_name == 'blue'
assert np.allclose(g.nodes[1].centroid_world[:2], [0.45, 0.0], atol=0.01)
x0, y0, x1, y1 = g.nodes[1].pixel_bbox
assert x0 < pix(0.45, 0)[0] < x1 and y0 < pix(0.45, 0)[1] < y1
rels = {(e.src, e.dst, e.relation) for e in g.edges}
assert (1, 2, 'left_of') in rels and (2, 1, 'right_of') in rels
assert (1, 2, 'near') in rels and (2, 1, 'near') in rels       # 8 cm < NEAR_DIST
assert not any(3 in (e.src, e.dst) or 4 in (e.src, e.dst) for e in g.edges)  # edges: table nodes only
assert g.table_nodes() == [1, 2]
js = json.loads(json.dumps(g.to_json()))
assert js['nodes']['1']['location'] == 'table' and js['nodes']['1']['identity'] is None
assert {'src': 1, 'dst': 2, 'relation': 'left_of'} in js['edges']
assert MIN_POINTS == 30 and NEAR_DIST == 0.12
print('All scene_graph checks passed.')
