"""Run the cache-ON engine across multiple processes.

The nauty isomorphism cache uses nauty's process-global state, so a single process can only
run one cached ``compute()`` at a time (``canon.hpp`` enforces this with a loud guard: a
concurrent call raises ``RuntimeError`` rather than silently corrupting canonical keys).
Parallelism therefore needs *separate processes*, each with its own private nauty globals.

This module provides exactly that: a process pool where each worker runs an independent
cache-ON engine. It is the supported way to parallelise RL rollout environments (or any batch
of instances) without giving up the cache.

Usage::

    from chromatic_rl.parallel import parallel_compute

    graphs = [(4, [(0,1),(1,2),(2,3),(3,0)]), my_crl_graph, ...]
    results = parallel_compute(graphs, selector="min_degree_sum", workers=4)
    results[0].branches, results[0].coeffs

Selector specs must be **picklable** so workers can rebuild them (each worker builds the
selector once, in an initializer):

* ``None`` / ``"min_degree_sum"`` / ``"max_degree_sum"`` are built-in selectors;
* ``"learned"`` / ``"mlp"`` are the deployed policies (chromatic_rl.public_selector);
* any picklable callable ``(Graph) -> (u, v)``.

A *live, unpicklable* Python policy cannot cross the process boundary. Distil it to weights
(the ``("learned"/"mlp", ...)`` spec) or run single-process with ``crl.compute`` directly.
"""
from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

GraphSpec = Union[Tuple[int, Sequence[Tuple[int, int]]], Any]  # (n, edges) or a crl.Graph
SelectorSpec = Union[None, str, Tuple[str, Any], Any]


@dataclass
class ParallelResult:
    """Picklable result of one engine run (the fields that survive a process boundary)."""
    coeffs: List[int]            # chromatic polynomial, power-basis integer coefficients
    branches: int               # the RL cost signal
    nodes: int
    hits: int
    misses: int
    trace: Optional[List[dict]] = None  # per-decision records when record=True, else None


def _as_graphspec(g: GraphSpec) -> Tuple[int, List[Tuple[int, int]]]:
    """Normalise a graph argument to a picklable ``(n, edges)`` tuple."""
    if isinstance(g, tuple) and len(g) == 2 and isinstance(g[0], int):
        n, edges = g
        return n, [tuple(e) for e in edges]
    # Assume a crl.Graph (or anything with num_vertices()/edges()).
    return g.num_vertices(), [tuple(e) for e in g.edges()]


def _resolve_selector(spec: SelectorSpec):
    """Rebuild a selector from a picklable spec, inside a worker process."""
    if spec == "learned":
        from chromatic_rl.public_selector import learned_selector
        return learned_selector()
    if spec == "mlp":
        from chromatic_rl.public_selector import mlp_selector
        return mlp_selector()
    if spec is None or isinstance(spec, str):
        return spec  # None and built-in names are accepted by compute() directly
    if callable(spec):
        return spec
    raise ValueError(f"unsupported selector spec: {spec!r}")


# Per-worker state, populated once by the pool initializer (avoids rebuilding the selector
# and re-importing per task).
_WORKER: dict = {}


def _init_worker(selector_spec: SelectorSpec, use_cache: bool, record: bool) -> None:
    import chromatic_rl  # noqa: F401  (warm the import in the worker)
    _WORKER["selector"] = _resolve_selector(selector_spec)
    _WORKER["use_cache"] = use_cache
    _WORKER["record"] = record


def _trace_to_dicts(trace) -> List[dict]:
    return [
        {
            "id": r.id, "parent_id": r.parent_id, "n": r.n,
            "edges": [tuple(e) for e in r.edges], "chosen": tuple(r.chosen),
            "subtree_branches": r.subtree_branches,
        }
        for r in trace
    ]


def _run_one(task: Tuple[int, List[Tuple[int, int]]]) -> ParallelResult:
    import chromatic_rl as crl

    n, edges = task
    g = crl.Graph(n, edges)
    res = crl.compute(g, selector=_WORKER["selector"],
                      use_cache=_WORKER["use_cache"], record=_WORKER["record"])
    s = res.stats
    trace = _trace_to_dicts(res.trace) if _WORKER["record"] and res.trace is not None else None
    return ParallelResult(
        coeffs=list(res.poly.coeffs()),
        branches=s.branches, nodes=s.nodes, hits=s.hits, misses=s.misses,
        trace=trace,
    )


def parallel_compute(
    graphs: Sequence[GraphSpec],
    selector: SelectorSpec = None,
    use_cache: bool = True,
    record: bool = False,
    workers: Optional[int] = None,
) -> List[ParallelResult]:
    """Compute chromatic polynomials for ``graphs`` across a process pool.

    Each worker is a fresh process (``spawn`` context) with its own nauty globals, so the
    cache is safe to use in parallel. Results are returned in input order.

    Args:
        graphs: iterable of ``(n, edges)`` tuples or ``crl.Graph`` objects.
        selector: a picklable selector spec (see module docstring).
        use_cache: run the engine with the isomorphism cache (the point of this helper).
        record: also return per-decision traces (as picklable dicts).
        workers: process count; default ``min(len(graphs), os.cpu_count())``.

    Returns:
        list of :class:`ParallelResult`, aligned with ``graphs``.
    """
    tasks = [_as_graphspec(g) for g in graphs]
    if not tasks:
        return []

    if workers is None:
        workers = min(len(tasks), os.cpu_count() or 1)
    workers = max(1, int(workers))

    if workers == 1:
        # Single-process fast path: no pool overhead, identical results. The in-process
        # engine is still single-threaded, so the nauty guard is never tripped.
        _init_worker(selector, use_cache, record)
        try:
            return [_run_one(t) for t in tasks]
        finally:
            _WORKER.clear()

    ctx = multiprocessing.get_context("spawn")  # clean, independent nauty globals per worker
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(selector, use_cache, record),
    ) as ex:
        return list(ex.map(_run_one, tasks))
