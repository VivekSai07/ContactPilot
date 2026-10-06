"""Pure P10 bin-routing and retry decisions; no MuJoCo or model imports."""
from typing import Callable


def target_bin(category: str, bins: list):
    """Return the unique bin configured for category, or fail closed."""
    matches = [b for b in bins if b.category == category]
    if category is None or len(matches) != 1:
        # Why: a missing/ambiguous category must not silently choose a bin.
        raise ValueError(f'no unique destination bin for category {category!r}')
    return matches[0]


def choose_bin(routing: str, seg_id: int, graph, bins: list,
               oracle_category: Callable[[], str]):
    """Return (BinSpec, category) for one candidate, preserving source split."""
    if routing == 'scene-graph':
        node = None if graph is None else graph.nodes.get(seg_id)
        if node is None or node.location != 'table':
            # Why: only a currently visible table node may drive a graph route.
            raise ValueError(f'seg_id {seg_id} has no current table graph node')
        category = node.category
    elif routing == 'oracle':
        category = oracle_category()
    else:
        raise ValueError(f'unknown routing mode {routing!r}')
    return target_bin(category, bins), category


def next_failure_count(routing: str, pick_succeeded: bool,
                       landed_bin: str | None, old_count: int) -> int:
    """Account for one attempt without truth-derived retry state in graph mode."""
    if routing == 'scene-graph':
        # Why: the next graph observation, not the sim bin oracle, decides
        # whether this object is still pickable after a successful lift.
        return old_count + 1
    if routing == 'oracle':
        return old_count if pick_succeeded and landed_bin is not None else old_count + 1
    raise ValueError(f'unknown routing mode {routing!r}')


def evaluate_placement(gt_category: str, landed_bin: str | None,
                       bin_categories: dict[str, str]) -> dict:
    """Offline truth-labelled outcome; never feed this into a route decision."""
    # Why: a failed/unbinned pick has no bin result to judge as right or wrong.
    correct = None if landed_bin is None else bin_categories[landed_bin] == gt_category
    return {'gt_category': gt_category, 'landed_bin': landed_bin,
            'correct_bin': correct}
