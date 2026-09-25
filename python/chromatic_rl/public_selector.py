"""A small linear + residual-MLP edge-selection scorer.

Computes a fixed 9-feature vector per candidate edge and applies the shipped
weights (``weights/learned_weights.json`` / ``weights/mlp_weights.json``) to
pick the argmax edge. Usable directly as ``selector=`` in ``crl.compute(...)``.
"""
from __future__ import annotations

import json
import os

import numpy as np

import chromatic_rl as crl

_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
NUM_FEATURES = 9


def edge_features(graph):
    """(features, edges): float64 (num_edges, 9), rows aligned to edges in
    canonical u<v order."""
    edges = graph.edges()
    n = graph.num_vertices()
    m = len(edges)
    denom = n * (n - 1) / 2.0
    density = (m / denom) if denom > 0 else 0.0
    nm1 = float(n - 1) if n > 1 else 1.0
    inv_n = (1.0 / n) if n > 0 else 0.0
    deg = [graph.degree(v) for v in range(n)]
    nbr = [set(graph.neighbours(v)) for v in range(n)]

    feats = np.empty((m, NUM_FEATURES), dtype=np.float64)
    for i, (u, v) in enumerate(edges):
        du, dv = deg[u], deg[v]
        common = len(nbr[u] & nbr[v])
        feats[i, 0] = (du + dv) / (2.0 * nm1)
        feats[i, 1] = (du * dv) / (nm1 * nm1)
        feats[i, 2] = min(du, dv) / nm1
        feats[i, 3] = max(du, dv) / nm1
        feats[i, 4] = (common / (n - 2)) if n > 2 else 0.0
        feats[i, 5] = inv_n
        feats[i, 6] = density
        l_gpe = len(crl.clique_separator(graph.contract_edge(u, v)))
        l_gme = len(crl.clique_separator(graph.delete_edge(u, v)))
        feats[i, 7] = (1.0 / l_gpe) if l_gpe > 0 else 0.0
        feats[i, 8] = (1.0 / l_gme) if l_gme > 0 else 0.0
    return feats, edges


def _argmax_edge(edges, scores):
    return edges[int(np.argmax(scores))]


def learned_selector(weights_path=None):
    """Linear policy: score(e) = w . phi(e)."""
    with open(weights_path or os.path.join(_WEIGHTS_DIR, "learned_weights.json")) as f:
        w = np.asarray(json.load(f)["weights"], dtype=np.float64)

    def selector(graph):
        feats, edges = edge_features(graph)
        return _argmax_edge(edges, feats @ w)
    selector.__name__ = "learned"
    return selector


def mlp_selector(weights_path=None):
    """Residual policy: score(e) = w . phi(e) + W2 . relu(W1 . phi(e) + b1) + b2."""
    with open(weights_path or os.path.join(_WEIGHTS_DIR, "mlp_weights.json")) as f:
        d = json.load(f)
    lin = np.asarray(d["linear"], dtype=np.float64)
    w1 = np.asarray(d["mlp"]["w1"], dtype=np.float64)
    b1 = np.asarray(d["mlp"]["b1"], dtype=np.float64)
    w2 = np.asarray(d["mlp"]["w2"], dtype=np.float64)
    b2 = np.asarray(d["mlp"]["b2"], dtype=np.float64)

    def selector(graph):
        feats, edges = edge_features(graph)
        h = np.maximum(feats @ w1.T + b1, 0.0)
        scores = feats @ lin + h @ w2.reshape(-1) + float(b2.reshape(-1)[0])
        return _argmax_edge(edges, scores)
    selector.__name__ = "mlp"
    return selector
