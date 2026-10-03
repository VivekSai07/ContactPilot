"""[P10 SP2] Vision-only scene graph over one capture: a node per segmented
object (world centroid/extents, colour, location table|bin|unknown) and
left_of/right_of/near edges between table objects. Pure -- no MuJoCo, no
network -- so it runs unchanged on a real camera. Identity/category are
filled in later by object_knowledge."""
from dataclasses import dataclass, field

import numpy as np

from sim_grasp.color_utils import rgb_to_color_name
from sim_grasp.pointcloud import depth_to_pointcloud
from sim_grasp.spatial_relation_resolver import _camera_view_axis

MIN_POINTS = 30          # fewer points -> location 'unknown' (sliver/occluded)
NEAR_DIST = 0.12         # m, centroid distance for a 'near' edge


@dataclass
class Node:
    seg_id: int
    centroid_world: np.ndarray
    extents: np.ndarray
    colour_name: str
    location: str
    pixel_bbox: tuple
    identity: 'str | None' = None
    category: 'str | None' = None


@dataclass
class Edge:
    src: int
    dst: int
    relation: str


@dataclass
class SceneGraph:
    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)

    def table_nodes(self) -> list:
        return sorted(s for s, n in self.nodes.items() if n.location == 'table')

    def to_json(self) -> dict:
        return {'nodes': {str(s): {'seg_id': n.seg_id,
                                   'centroid_world': [round(float(v), 4) for v in n.centroid_world],
                                   'extents': [round(float(v), 4) for v in n.extents],
                                   'colour_name': n.colour_name, 'location': n.location,
                                   'pixel_bbox': [int(v) for v in n.pixel_bbox],
                                   'identity': n.identity, 'category': n.category}
                          for s, n in sorted(self.nodes.items())},
                'edges': [{'src': e.src, 'dst': e.dst, 'relation': e.relation}
                          for e in self.edges]}


def _location(pts_world, bins, table_height) -> str:
    # bin = XY footprint only: a tall prop's top face sits above the bin wall
    in_any = np.zeros(len(pts_world), bool)
    votes = {}
    for b in bins:
        inside = ((np.abs(pts_world[:, 0] - b.center[0]) <= b.inner_half)
                  & (np.abs(pts_world[:, 1] - b.center[1]) <= b.inner_half))
        votes[b.name] = int(inside.sum())
        in_any |= inside
    z = pts_world[:, 2]
    votes['table'] = int((~in_any & (z >= table_height - 0.01) & (z <= table_height + 0.30)).sum())
    best = max(votes.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else 'unknown'


def build(depth, segmap, K, T_world_cam, rgb, bins, table_height) -> SceneGraph:
    seg = np.asarray(segmap).reshape(depth.shape)
    g = SceneGraph()
    for sid in sorted(int(s) for s in np.unique(seg) if s > 0):
        mask = seg == sid
        vs, us = np.nonzero(mask)
        bbox = (int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1)
        pc, cols = depth_to_pointcloud(depth, K, rgb=rgb, mask=mask)
        if len(pc) < MIN_POINTS:
            g.nodes[sid] = Node(sid, np.zeros(3), np.zeros(3), 'unknown', 'unknown', bbox)
            continue
        pw = pc @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
        g.nodes[sid] = Node(sid, pw.mean(axis=0), pw.max(axis=0) - pw.min(axis=0),
                            rgb_to_color_name(cols.mean(axis=0)),
                            _location(pw, bins, table_height), bbox)
    # edges between table objects only; bin membership is a node attribute
    table = g.table_nodes()
    axis, sign = _camera_view_axis(T_world_cam)
    k = 0 if axis == 'x' else 1
    order = sorted(table, key=lambda s: sign * g.nodes[s].centroid_world[k])
    for a, b in zip(order, order[1:]):          # immediate neighbours keep the image readable
        g.edges += [Edge(a, b, 'left_of'), Edge(b, a, 'right_of')]
    for i, a in enumerate(table):
        for b in table[i + 1:]:
            if np.linalg.norm(g.nodes[a].centroid_world[:2] - g.nodes[b].centroid_world[:2]) < NEAR_DIST:
                g.edges += [Edge(a, b, 'near'), Edge(b, a, 'near')]
    return g
