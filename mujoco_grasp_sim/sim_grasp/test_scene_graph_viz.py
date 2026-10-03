"""Standalone checks for scene_graph_viz (pure OpenCV)."""
import numpy as np

from sim_grasp.scene_graph import Edge, Node, SceneGraph
from sim_grasp.scene_graph_viz import LOCATION_COLOURS, draw_scene_graph

rgb = np.full((120, 160, 3), 128, np.uint8)
g = SceneGraph(nodes={
    1: Node(1, np.zeros(3), np.zeros(3), 'red', 'table', (10, 10, 40, 40), 'box of cookies', 'food'),
    2: Node(2, np.zeros(3), np.zeros(3), 'blue', 'A', (90, 60, 130, 100), None, None)},
    edges=[Edge(1, 2, 'left_of'), Edge(1, 2, 'near')])
out = draw_scene_graph(rgb, g, {'A': 'food', 'B': 'non_food'}, gt_categories={1: 'non_food'})
assert out.dtype == np.uint8 and out.shape == (120 + 28, 160, 3)
assert (rgb == 128).all()                                  # input not modified
assert tuple(out[25, 10]) == LOCATION_COLOURS['table']     # left box edge (labels sit above the top edge)
assert tuple(out[80, 90]) == LOCATION_COLOURS['A']
assert (out[120:] != 128).any()                            # legend strip drawn
print('All scene_graph_viz checks passed.')
