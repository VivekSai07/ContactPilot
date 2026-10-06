"""Behavior checks for SP3 bin decisions; run with PYTHONPATH=."""
import numpy as np

from sim_grasp.scene_generator import BinSpec
from sim_grasp.scene_graph import Node, SceneGraph
from sim_grasp.sorting_policy import choose_bin, target_bin, next_failure_count


bins = [BinSpec('A', (0.45, -0.30), 0.1, 'food'),
        BinSpec('B', (0.45, 0.30), 0.1, 'non_food')]


def raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('expected ValueError')


assert target_bin('food', bins).name == 'A'
assert target_bin('non_food', bins).name == 'B'
raises_value_error(lambda: target_bin(None, bins))
raises_value_error(lambda: target_bin('unknown', bins))
raises_value_error(lambda: target_bin('food', [bins[0], bins[0]]))

node = Node(7, np.zeros(3), np.ones(3), 'red', 'table', (0, 0, 20, 20),
            identity='ink cartridge', category='non_food')
graph = SceneGraph(nodes={7: node})


def forbidden_oracle():
    raise AssertionError('graph routing read simulator category')


selected, category = choose_bin('scene-graph', 7, graph, bins, forbidden_oracle)
assert (selected.name, category) == ('B', 'non_food')
assert choose_bin('oracle', 7, graph, bins, lambda: 'food')[0].name == 'A'
raises_value_error(lambda: choose_bin('scene-graph', 99, graph, bins, forbidden_oracle))
node.location = 'A'
raises_value_error(lambda: choose_bin('scene-graph', 7, graph, bins, forbidden_oracle))
node.location = 'table'
node.category = None
raises_value_error(lambda: choose_bin('scene-graph', 7, graph, bins, forbidden_oracle))

# A successfully lifted object can land back on the table. Its retry budget
# must advance independently of the simulator's post-place bin judgement.
assert next_failure_count('scene-graph', pick_succeeded=True,
                          landed_bin='A', old_count=0) == 1
assert next_failure_count('scene-graph', pick_succeeded=True,
                          landed_bin=None, old_count=1) == 2
assert next_failure_count('oracle', pick_succeeded=True,
                          landed_bin='A', old_count=0) == 0
assert next_failure_count('oracle', pick_succeeded=True,
                          landed_bin=None, old_count=0) == 1
assert next_failure_count('scene-graph', pick_succeeded=False,
                          landed_bin=None, old_count=1) == 2

print('SP3 sorting policy checks passed.')
