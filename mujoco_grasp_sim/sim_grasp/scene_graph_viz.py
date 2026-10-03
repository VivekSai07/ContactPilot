"""[P10 SP2] Annotated scene-graph image: a box per node coloured by
location, '#id identity - category' labels (red when the category
disagrees with sim ground truth -- a sim-only debugging aid), left_of
arrows and near lines between table objects, and a bin->category legend."""
import cv2
import numpy as np

LOCATION_COLOURS = {'table': (255, 200, 0), 'A': (0, 200, 0), 'B': (0, 120, 255),
                    'unknown': (160, 160, 160)}
_WRONG = (255, 0, 0)
_LEGEND_H = 28


def _centre(n):
    x0, y0, x1, y1 = n.pixel_bbox
    return (x0 + x1) // 2, (y0 + y1) // 2


def draw_scene_graph(rgb, graph, bin_categories, gt_categories=None) -> np.ndarray:
    img = np.ascontiguousarray(rgb[..., :3]).copy()
    for e in graph.edges:
        if e.src not in graph.nodes or e.dst not in graph.nodes:
            continue
        p, q = _centre(graph.nodes[e.src]), _centre(graph.nodes[e.dst])
        if e.relation == 'left_of':
            cv2.arrowedLine(img, p, q, (255, 255, 255), 1, tipLength=0.15)
        elif e.relation == 'near' and e.src < e.dst:      # draw each near pair once
            cv2.line(img, p, q, (255, 255, 0), 1, lineType=cv2.LINE_AA)
    for sid, n in graph.nodes.items():
        colour = LOCATION_COLOURS.get(n.location, LOCATION_COLOURS['unknown'])
        x0, y0, x1, y1 = n.pixel_bbox
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), colour, 1)
        label = f'#{sid} {n.identity or "?"} - {n.category or "?"}'
        wrong = gt_categories is not None and n.category is not None \
            and gt_categories.get(sid) not in (None, n.category)
        cv2.putText(img, label, (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    _WRONG if wrong else colour, 1, cv2.LINE_AA)
    legend = np.full((_LEGEND_H, img.shape[1], 3), 32, np.uint8)
    text = '  '.join(f'bin {b}: {c}' for b, c in sorted(bin_categories.items()))
    cv2.putText(legend, text + '   (red label = wrong vs sim truth)', (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([img, legend])
