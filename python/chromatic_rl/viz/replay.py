"""Replay engine graph operations over a full trace.

Each node's graph derives deterministically from its parent's via the recorded op,
mirroring the C++ primitives exactly:

  delete/insert(u,v)  Graph::deleteEdge / insertEdge - same vertex set.
  contract(u,v)       Graph::contractEdge - v'=max(u,v) merges into u'=min(u,v),
                      labels > v' shift down by one, parallels/self-loops dropped.
  piece(verts)        inducedSubgraph - verts[i] -> i in the given order.
  ie_piece            synthetic pieceGraph (complementCliqueDecompose): NOT replayable;
                      the engine records its edges, and root-identity tracking stops
                      (verts=None) for it and all its descendants.

Identity: verts[i] is the frozenset of ROOT vertices that local vertex i represents.
"""
from __future__ import annotations

from dataclasses import dataclass


class ReplayError(RuntimeError):
    """Replayed n/m disagrees with the engine-recorded checksum."""


@dataclass
class ReplayedNode:
    edges: list          # sorted list of (u, v), u < v
    verts: list | None   # list of frozenset(root ids), or None below ie_piece


def _norm(edges):
    return sorted((u, v) if u < v else (v, u) for (u, v) in edges)


def _contract(edges, n, a, b):
    u, v = min(a, b), max(a, b)

    def remap(x):
        y = u if x == v else x
        return y if y < v else y - 1

    out = set()
    for (x, y) in edges:
        rx, ry = remap(x), remap(y)
        if rx != ry:
            out.add((min(rx, ry), max(rx, ry)))
    return sorted(out), n - 1


def replay(trace, root_n, root_edges):
    """trace: list of FullTraceRecord (or dicts with the same attrs). Returns
    {id: ReplayedNode}. Records must be topologically ordered (parents first) -
    the recorder guarantees this."""
    def attr(r, name):
        return r[name] if isinstance(r, dict) else getattr(r, name)

    by_id = {attr(r, "id"): r for r in trace}
    out = {}
    for r in trace:
        rid, parent, via = attr(r, "id"), attr(r, "parent_id"), attr(r, "via")
        if via == "root":
            edges = _norm(root_edges)
            verts = [frozenset([i]) for i in range(root_n)]
        else:
            p = out[parent]
            pn = len(p.verts) if p.verts is not None else attr(by_id[parent], "n")
            if via == "delete":
                u, v = attr(r, "via_u"), attr(r, "via_v")
                e = (min(u, v), max(u, v))
                edges = [x for x in p.edges if x != e]
                verts = p.verts
            elif via == "insert":
                u, v = attr(r, "via_u"), attr(r, "via_v")
                edges = _norm(set(p.edges) | {(min(u, v), max(u, v))})
                verts = p.verts
            elif via == "contract":
                u, v = attr(r, "via_u"), attr(r, "via_v")
                edges, _ = _contract(p.edges, pn, u, v)
                if p.verts is None:
                    verts = None
                else:
                    lo, hi = min(u, v), max(u, v)
                    verts = [
                        (p.verts[lo] | p.verts[hi]) if x == lo else p.verts[x]
                        for x in range(pn) if x != hi
                    ]
            elif via == "piece":
                vv = list(attr(r, "via_verts"))
                idx = {pv: i for i, pv in enumerate(vv)}
                edges = sorted(
                    (min(idx[x], idx[y]), max(idx[x], idx[y]))
                    for (x, y) in p.edges if x in idx and y in idx
                )
                verts = None if p.verts is None else [p.verts[pv] for pv in vv]
            elif via == "ie_piece":
                edges = _norm(attr(r, "edges"))
                verts = None
            else:
                raise ReplayError(f"unknown via {via!r} at node {rid}")
        node = ReplayedNode(edges=edges, verts=verts)
        en, em = attr(r, "n"), attr(r, "m")
        nn = len(node.verts) if node.verts is not None else en
        if nn != en or len(node.edges) != em:
            raise ReplayError(
                f"replay mismatch at node {rid} (via={via}): "
                f"got n={nn} m={len(node.edges)}, engine n={en} m={em}"
            )
        out[rid] = node
    return out
