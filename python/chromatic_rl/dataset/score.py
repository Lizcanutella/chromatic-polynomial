"""Selector-name resolution for the visualizer's /compute endpoint.

Covers exactly the selectors the Build UI offers: none, min_degree_sum,
max_degree_sum, learned, mlp.
"""
from __future__ import annotations

import chromatic_rl as crl
from chromatic_rl.public_selector import learned_selector, mlp_selector

_NAMED = {
    "none": None,
    "min_degree_sum": crl.MIN_DEGREE_SUM,
    "max_degree_sum": crl.MAX_DEGREE_SUM,
}


def resolve_selector(name: str):
    if name in _NAMED:
        return _NAMED[name]
    if name == "learned":
        return learned_selector()
    if name == "mlp":
        return mlp_selector()
    raise ValueError(
        f"unknown selector {name!r}; known: {sorted(list(_NAMED) + ['learned', 'mlp'])}"
    )
