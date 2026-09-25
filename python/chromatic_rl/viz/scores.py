"""Per-edge policy scores for branch-node inspection ("why did it pick that edge").

Reuses the feature extractor and deployed weight files from public_selector.py:
  learned:  score = w . phi(e)
  mlp:      score = w . phi(e) + W2 relu(W1 phi(e) + b1) + b2
"""
from __future__ import annotations

import json
import os

import numpy as np

_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")


def load_scorer(selector_name):
    """Return scorer(graph) -> (edges, list_of_float_scores), or None if the
    selector has no scoreable weights (built-in heuristics, callables)."""
    from chromatic_rl.public_selector import edge_features

    if selector_name == "learned":
        with open(os.path.join(_WEIGHTS_DIR, "learned_weights.json")) as f:
            w = np.asarray(json.load(f)["weights"], dtype=np.float64)

        def scorer(graph):
            feats, edges = edge_features(graph)
            return edges, (feats.astype(np.float64) @ w).tolist()
        return scorer

    if selector_name == "mlp":
        with open(os.path.join(_WEIGHTS_DIR, "mlp_weights.json")) as f:
            d = json.load(f)
        lin = np.asarray(d["linear"], dtype=np.float64)
        w1 = np.asarray(d["mlp"]["w1"], dtype=np.float64)
        b1 = np.asarray(d["mlp"]["b1"], dtype=np.float64)
        w2 = np.asarray(d["mlp"]["w2"], dtype=np.float64)
        b2 = np.asarray(d["mlp"]["b2"], dtype=np.float64)

        def scorer(graph):
            feats, edges = edge_features(graph)
            x = feats.astype(np.float64)
            h = np.maximum(x @ w1.T + b1, 0.0)
            s = x @ lin + h @ w2.reshape(-1) + float(b2.reshape(-1)[0])
            return edges, s.tolist()
        return scorer

    return None
