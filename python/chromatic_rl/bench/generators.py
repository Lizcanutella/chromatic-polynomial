"""Graph generators for the benchmark harness.

Each generator returns (Graph, meta) where meta = {family, n, params, seed}.
Thin wrappers over networkx, piped through chromatic_rl.from_networkx so we
benchmark the engine, not a re-implementation of graph generation.
"""
from __future__ import annotations

import random as _random
from typing import Callable

import networkx as nx

from chromatic_rl import from_networkx


def _wrap(G, family, n, params, seed):
    return from_networkx(G), {
        "family": family, "n": int(n), "params": params, "seed": seed,
    }


def erdos_renyi(n, p, seed):
    return _wrap(nx.gnp_random_graph(n, p, seed=seed), "erdos_renyi", n, {"p": p}, seed)


def random_regular(n, d, seed):
    return _wrap(nx.random_regular_graph(d, n, seed=seed), "random_regular", n, {"d": d}, seed)


def grid(rows, cols):
    G = nx.grid_2d_graph(rows, cols)
    return _wrap(G, "grid", rows * cols, {"rows": rows, "cols": cols}, None)


def path(n):
    return _wrap(nx.path_graph(n), "path", n, {}, None)


def cycle(n):
    return _wrap(nx.cycle_graph(n), "cycle", n, {}, None)


def complete(n):
    return _wrap(nx.complete_graph(n), "complete", n, {}, None)


def cocktail_party(r, s):
    """Complete r-partite graph with r parts of size s (K_{s,s,...,s}).

    Its complement is r disjoint cliques (disconnected) -> the join reduction
    fully decomposes it, so it is a sanity family for the join toggle.
    """
    G = nx.complete_multipartite_graph(*([s] * r))
    return _wrap(G, "cocktail_party", r * s, {"r": r, "s": s}, None)


def random_tree(n, seed):
    try:
        G = nx.random_labeled_tree(n, seed=seed)
    except AttributeError:  # older networkx
        G = nx.random_tree(n, seed=seed)
    return _wrap(G, "random_tree", n, {}, seed)


def _ez(a, b):
    return frozenset((a, b))


def _planar_add_face(faces, edge_faces, tri):
    fid = len(faces)
    faces.append(tri)
    a, b, c = tuple(tri)
    for e in (_ez(a, b), _ez(b, c), _ez(a, c)):
        edge_faces.setdefault(e, set()).add(fid)


def _planar_remove_face(faces, edge_faces, fid):
    a, b, c = tuple(faces[fid])
    for e in (_ez(a, b), _ez(b, c), _ez(a, c)):
        edge_faces[e].discard(fid)
    last = len(faces) - 1
    if fid != last:
        ma, mb, mc = tuple(faces[last])
        for e in (_ez(ma, mb), _ez(mb, mc), _ez(ma, mc)):
            edge_faces[e].discard(last)
            edge_faces[e].add(fid)
        faces[fid] = faces[last]
    faces.pop()


def _random_maximal_planar_edges(n, seed):
    """Edge set (frozenset pairs) of a random maximal-planar graph on n vertices,
    via edge-flip random triangulation. Shared by random_planar (maximal) and
    random_planar_sparse (thinned). Raises ValueError for n < 3.
    """
    if n < 3:
        raise ValueError(f"random planar generators require n >= 3, got {n}")
    rng = _random.Random(seed)
    edges = {_ez(0, 1), _ez(0, 2), _ez(1, 2)}
    faces, edge_faces = [], {}
    # base triangle bounds two faces on the sphere (the two sides)
    _planar_add_face(faces, edge_faces, frozenset((0, 1, 2)))
    _planar_add_face(faces, edge_faces, frozenset((0, 1, 2)))
    for v in range(3, n):
        fid = rng.randrange(len(faces))
        x, y, z = tuple(faces[fid])
        _planar_remove_face(faces, edge_faces, fid)
        _planar_add_face(faces, edge_faces, frozenset((x, y, v)))
        _planar_add_face(faces, edge_faces, frozenset((y, z, v)))
        _planar_add_face(faces, edge_faces, frozenset((x, z, v)))
        edges.update({_ez(x, v), _ez(y, v), _ez(z, v)})
    edge_list = list(edges)
    for _ in range(8 * (3 * n - 6)):          # ~8 sweeps; internal mixing budget
        uv = rng.choice(edge_list)
        fset = edge_faces.get(uv)
        if not fset or len(fset) != 2:
            continue
        f1, f2 = tuple(fset)
        u, v = tuple(uv)
        a = next(iter(faces[f1] - uv))
        b = next(iter(faces[f2] - uv))
        if a == b or _ez(a, b) in edges:
            continue
        for fid in sorted((f1, f2), reverse=True):   # high id first: swap-pop stable
            _planar_remove_face(faces, edge_faces, fid)
        edges.discard(uv)
        edges.add(_ez(a, b))
        _planar_add_face(faces, edge_faces, frozenset((u, a, b)))
        _planar_add_face(faces, edge_faces, frozenset((v, a, b)))
        edge_list = list(edges)
    return edges


def random_planar(n, seed):
    """Random maximal-planar graph (|E| = 3n-6) via edge-flip random triangulation."""
    edges = _random_maximal_planar_edges(n, seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(tuple(e) for e in edges)
    return _wrap(G, "random_planar", n, {}, seed)


def random_planar_sparse(n, seed):
    """Sparse random-planar graph: a random maximal-planar triangulation thinned
    to ~65% of its edges, removing edges while preserving connectivity (a spanning
    tree is always kept). Result is simple, connected, planar, NON-maximal
    (|E| < 3n-6). At a given n it is a stronger selector benchmark than the maximal
    family, which CRT-1 decomposes too easily.
    """
    _THIN = 0.35                                    # fraction of edges removed (internal)
    edges = [tuple(sorted(tuple(e))) for e in _random_maximal_planar_edges(n, seed)]
    rng = _random.Random(seed * 1000 + 7)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    tree = {tuple(sorted(e)) for e in nx.minimum_spanning_tree(G).edges()}
    rest = [e for e in edges if e not in tree]
    rng.shuffle(rest)
    target = max(n - 1, round((1 - _THIN) * len(edges)))
    keep = set(tree) | set(rest[: max(0, target - len(tree))])
    H = nx.Graph()
    H.add_nodes_from(range(n))
    H.add_edges_from(keep)
    return _wrap(H, "random_planar_sparse", n, {}, seed)


GENERATORS: dict[str, Callable] = {
    "erdos_renyi": erdos_renyi,
    "random_regular": random_regular,
    "grid": grid,
    "path": path,
    "cycle": cycle,
    "complete": complete,
    "cocktail_party": cocktail_party,
    "random_tree": random_tree,
    "random_planar": random_planar,
    "random_planar_sparse": random_planar_sparse,
}
