"""Run benchmark cells and collect metrics.

A *spec* is one fully-expanded run:
    {family, gen_kwargs, selector, use_cache, repeat}
`selector` is a name in SELECTOR_NAMES or a Python callable (Graph) -> (u, v).
"""
from __future__ import annotations

import time
from typing import Iterable, Iterator

import chromatic_rl as crl

from .generators import GENERATORS

_SELECTORS = {
    "none": None,
    "min_degree_sum": crl.MIN_DEGREE_SUM,
    "max_degree_sum": crl.MAX_DEGREE_SUM,
    "learned": "learned",
    "mlp": "mlp",
}
SELECTOR_NAMES = frozenset(_SELECTORS)


def _resolve_selector(selector):
    if callable(selector):
        return selector
    if selector == "learned":
        from chromatic_rl.public_selector import learned_selector
        return learned_selector()
    if selector == "mlp":
        from chromatic_rl.public_selector import mlp_selector
        return mlp_selector()
    return _SELECTORS[selector]


def run_one(spec) -> dict:
    family = spec["family"]
    gen_kwargs = spec["gen_kwargs"]
    selector = spec["selector"]
    use_cache = spec["use_cache"]
    repeat = spec["repeat"]
    sel_name = selector if isinstance(selector, str) else getattr(selector, "__name__", "callable")

    row = {
        "family": family,
        "n": gen_kwargs.get("n"),
        "params": {k: v for k, v in gen_kwargs.items() if k not in ("n", "seed")},
        "seed": gen_kwargs.get("seed"),
        "selector": sel_name,
        "use_cache": use_cache,
        "repeat": repeat,
    }
    try:
        graph, meta = GENERATORS[family](**gen_kwargs)
        sel = _resolve_selector(selector)
        best = None
        stats = None
        for _ in range(repeat):
            t0 = time.perf_counter()
            res = crl.compute(graph, selector=sel, use_cache=use_cache)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
            stats = res.stats
        row.update(
            n=meta["n"], params=meta["params"], seed=meta["seed"],
            nodes=stats.nodes, branches=stats.branches, hits=stats.hits,
            misses=stats.misses, hit_rate=stats.hit_rate, wall_s=best,
        )
    except Exception as e:  # noqa: BLE001 - one bad cell must not abort a sweep
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def run_sweep(specs: Iterable[dict]) -> Iterator[dict]:
    for spec in specs:
        yield run_one(spec)
