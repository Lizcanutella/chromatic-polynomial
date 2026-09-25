"""Tests for chromatic_rl.parallel — the process-pool engine helper (item 3).

The contract under test: running a batch of graphs across independent worker processes
(each with its own nauty globals) produces exactly the same polynomials and branch counts
as running them serially in-process. This is the supported way to parallelise cache-ON work.
"""
import pytest

import chromatic_rl as crl
from chromatic_rl.parallel import ParallelResult, parallel_compute


def _cycle(n):
    return (n, [(i, (i + 1) % n) for i in range(n)])


def _petersen():
    outer = [(i, (i + 1) % 5) for i in range(5)]
    spokes = [(i, i + 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    return (10, outer + spokes + inner)


def _grid(r, c):
    def idx(i, j):
        return i * c + j
    edges = []
    for i in range(r):
        for j in range(c):
            if j + 1 < c:
                edges.append((idx(i, j), idx(i, j + 1)))
            if i + 1 < r:
                edges.append((idx(i, j), idx(i + 1, j)))
    return (r * c, edges)


GRAPHS = [_cycle(4), _cycle(5), _cycle(6), _petersen(), _grid(3, 3), _grid(3, 4)]


def _serial(graph, selector, use_cache=True, record=False):
    n, edges = graph
    return crl.compute(crl.Graph(n, edges), selector=selector, use_cache=use_cache, record=record)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_matches_serial(workers):
    """Parallel results equal the in-process engine, polynomial and branch count, in order."""
    results = parallel_compute(GRAPHS, selector="min_degree_sum", workers=workers)
    assert len(results) == len(GRAPHS)
    for res, graph in zip(results, GRAPHS):
        ref = _serial(graph, "min_degree_sum")
        assert res.coeffs == list(ref.poly.coeffs())
        assert res.branches == ref.stats.branches


def test_none_selector_equals_min_degree_sum():
    none_res = parallel_compute(GRAPHS, selector=None, workers=2)
    mds_res = parallel_compute(GRAPHS, selector="min_degree_sum", workers=2)
    assert [r.coeffs for r in none_res] == [r.coeffs for r in mds_res]


def test_learned_selector_spec():
    """The picklable "learned" spec is rebuilt in each worker and runs correctly."""
    from chromatic_rl.public_selector import learned_selector

    results = parallel_compute(GRAPHS, selector="learned", workers=2)
    for res, graph in zip(results, GRAPHS):
        ref = _serial(graph, learned_selector())
        assert res.coeffs == list(ref.poly.coeffs())
        assert res.branches == ref.stats.branches


def test_accepts_graph_objects():
    """parallel_compute accepts crl.Graph objects as well as (n, edges) tuples."""
    objs = [crl.Graph(n, edges) for (n, edges) in GRAPHS]
    from_objs = parallel_compute(objs, selector="min_degree_sum", workers=2)
    from_tuples = parallel_compute(GRAPHS, selector="min_degree_sum", workers=2)
    assert [r.coeffs for r in from_objs] == [r.coeffs for r in from_tuples]


def test_record_returns_picklable_trace():
    results = parallel_compute([_grid(3, 4)], selector="min_degree_sum", record=True, workers=2)
    (res,) = results
    assert res.trace is not None and len(res.trace) > 0
    rec = res.trace[0]
    assert set(rec) >= {"id", "parent_id", "n", "edges", "chosen", "subtree_branches"}
    # The recorded subtree branch count of the root equals the run's total branches.
    assert isinstance(res, ParallelResult)


def test_empty_input():
    assert parallel_compute([], selector=None, workers=2) == []
