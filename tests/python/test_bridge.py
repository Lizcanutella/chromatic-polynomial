import chromatic_rl


def test_module_imports_with_version():
    assert chromatic_rl.__version__ == "0.1.0"


import numpy as np
from chromatic_rl import Graph


def test_graph_construction_and_accessors():
    # triangle plus a pendant: 0-1-2-0, 2-3
    g = Graph(4, [(0, 1), (1, 2), (0, 2), (2, 3)])
    assert g.num_vertices() == 4
    assert g.num_edges() == 4
    assert g.degree(2) == 3
    assert sorted(g.neighbours(2)) == [0, 1, 3]
    assert (0, 1) in g.edges()
    assert all(u < v for (u, v) in g.edges())


def test_graph_to_arrays():
    g = Graph(3, [(0, 1), (1, 2)])
    n, edge_index = g.to_arrays()
    assert n == 3
    assert isinstance(edge_index, np.ndarray)
    assert edge_index.shape == (2, 2)
    assert edge_index.dtype == np.int64
    cols = {tuple(edge_index[:, i]) for i in range(edge_index.shape[1])}
    assert cols == {(0, 1), (1, 2)}


import itertools
import random
import networkx as nx
from chromatic_rl import compute


def _brute_force_proper_colorings(edges, n, k):
    """Count proper k-colorings by brute force (independent oracle)."""
    count = 0
    for coloring in itertools.product(range(k), repeat=n):
        if all(coloring[u] != coloring[v] for (u, v) in edges):
            count += 1
    return count


def test_compute_matches_closed_forms():
    # Cycle C_4: chromatic poly (k-1)^4 + (k-1) -> at k=3 equals 18.
    c4 = Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert compute(c4).poly.evaluate(3) == 18
    # Complete K_4: 4! = 24 at k=4.
    k4 = Graph(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    assert compute(k4).poly.evaluate(4) == 24


def test_compute_matches_brute_force_oracle():
    rng = random.Random(0)
    for _ in range(15):
        n = rng.randint(4, 7)
        G = nx.gnp_random_graph(n, 0.5, seed=rng.randint(0, 1_000_000))
        edges = [(min(u, v), max(u, v)) for u, v in G.edges()]
        g = Graph(n, edges)
        poly = compute(g).poly
        for k in range(n + 1):
            assert poly.evaluate(k) == _brute_force_proper_colorings(edges, n, k)


def test_cache_reduces_branches():
    G = nx.gnp_random_graph(13, 0.5, seed=42)
    edges = [(min(u, v), max(u, v)) for u, v in G.edges()]
    g = Graph(13, edges)
    cached = compute(g, use_cache=True).stats
    uncached = compute(g, use_cache=False).stats
    assert cached.branches < uncached.branches
    assert cached.hits > 0
    assert 0.0 <= cached.hit_rate <= 1.0


import pytest
from chromatic_rl import MIN_DEGREE_SUM


def test_python_selector_parity_with_builtin():
    G = nx.gnp_random_graph(8, 0.5, seed=7)
    edges = [(min(u, v), max(u, v)) for u, v in G.edges()]
    g = Graph(8, edges)

    def py_min_degree_sum(graph):
        return min(graph.edges(),
                   key=lambda e: graph.degree(e[0]) + graph.degree(e[1]))

    via_callback = compute(g, selector=py_min_degree_sum).poly
    via_builtin = compute(g, selector=MIN_DEGREE_SUM).poly
    assert via_callback.coeffs() == via_builtin.coeffs()


def test_selector_invariance_across_boundary():
    G = nx.gnp_random_graph(8, 0.5, seed=11)
    edges = [(min(u, v), max(u, v)) for u, v in G.edges()]
    g = Graph(8, edges)

    def pick_first(graph):
        return graph.edges()[0]

    def pick_last(graph):
        return graph.edges()[-1]

    assert compute(g, selector=pick_first).poly.coeffs() \
        == compute(g, selector=pick_last).poly.coeffs()


def test_selector_returning_non_edge_raises():
    # complement(C_6): edges are all non-adjacent pairs in C_6 = 0-1-2-3-4-5-0.
    # This graph is connected, not a base case (9 edges on 6 vertices), has no CRT-1
    # clique separator, and its complement (= C_6, a chordless cycle) is connected and
    # has no clique separator of any size -> neither the join nor the complement
    # clique-separator reduction fires, so the engine must invoke the selector at the root.
    # (C_6+chords {0,3},{1,4} can no longer be used: its complement has a cut vertex at 2,
    # so the complement clique-sep reduction decomposes it without calling the selector.
    # K_{2,3} was already retired when the join reduction was added.)
    g = Graph(6, [(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 5)])

    def bad(graph):
        return (0, 1)   # not an edge in this graph (that's a C_6 edge = non-edge of complement(C_6))

    with pytest.raises(ValueError):
        compute(g, selector=bad)


def test_selector_exception_propagates():
    # Same complement(C_6) graph so the selector is actually invoked (see note above).
    g = Graph(6, [(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 5)])

    def boom(graph):
        raise RuntimeError("policy failed")

    with pytest.raises(RuntimeError):
        compute(g, selector=boom)


def test_trace_disabled_by_default():
    g = Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert compute(g).trace is None


def test_trace_integrity():
    G = nx.gnp_random_graph(10, 0.5, seed=3)
    edges = [(min(u, v), max(u, v)) for u, v in G.edges()]
    g = Graph(10, edges)
    res = compute(g, record=True)
    trace = res.trace
    assert trace is not None and len(trace) == res.stats.branches

    by_id = {r.id: r for r in trace}
    roots = [r for r in trace if r.parent_id == -1]
    # decision roots partition the branch budget (robust to disconnected inputs)
    assert sum(r.subtree_branches for r in roots) == res.stats.branches

    for r in trace:
        assert tuple(r.chosen) in {tuple(e) for e in r.edges}   # chosen is a candidate
        if r.parent_id != -1:
            assert r.parent_id in by_id                          # tree reconstructs


from chromatic_rl import from_networkx, to_networkx


def test_from_networkx_roundtrip_and_compute():
    G = nx.gnp_random_graph(7, 0.5, seed=21)
    g = from_networkx(G)
    assert g.num_vertices() == G.number_of_nodes()
    assert g.num_edges() == G.number_of_edges()
    # compute via the converted graph agrees with the brute-force oracle
    edges = [(min(u, v), max(u, v)) for u, v in G.edges()]
    poly = compute(g).poly
    for k in range(G.number_of_nodes() + 1):
        assert poly.evaluate(k) == _brute_force_proper_colorings(edges, G.number_of_nodes(), k)


def test_to_networkx_roundtrip():
    g = Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    G = to_networkx(g)
    assert G.number_of_nodes() == 4
    assert {tuple(sorted(e)) for e in G.edges()} == {(0, 1), (1, 2), (2, 3), (0, 3)}


def test_branch_budget_aborts_on_hard_graph():
    import chromatic_rl as crl
    import networkx as nx
    # a dense graph that branches well past a tiny budget cache-OFF
    g = crl.from_networkx(nx.gnp_random_graph(12, 0.6, seed=3))
    with pytest.raises(crl.BudgetExceeded):
        crl.compute(g, use_cache=False, branch_budget=5)


def test_branch_budget_none_is_unlimited_and_exact():
    import chromatic_rl as crl
    # K4: chi(k) = k(k-1)(k-2)(k-3); budget=None must not alter the result
    g = crl.Graph(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    ref = crl.compute(g, use_cache=False).poly.coeffs()
    assert crl.compute(g, use_cache=False, branch_budget=None).poly.coeffs() == ref


def test_branch_budget_under_cap_returns_correct_poly():
    import chromatic_rl as crl
    # path P4 solves in 0 branches (tree reduction); a generous budget is a no-op
    g = crl.Graph(4, [(0, 1), (1, 2), (2, 3)])
    ref = crl.compute(g, use_cache=False).poly.coeffs()
    assert crl.compute(g, use_cache=False, branch_budget=1000).poly.coeffs() == ref


def test_reductions_none_matches_default():
    import chromatic_rl as crl
    g = crl.Graph(7, [(0,1),(0,2),(1,2),(2,3),(3,4),(3,5),(4,5),(5,6),(4,6),(2,4)])
    a = crl.compute(g, use_cache=True)
    b = crl.compute(g, use_cache=True, reductions=None)
    assert a.poly.coeffs() == b.poly.coeffs()
    assert a.stats.branches == b.stats.branches

def test_disabling_reduction_keeps_poly_changes_cost():
    import chromatic_rl as crl
    g = crl.Graph(7, [(0,1),(0,2),(1,2),(2,3),(3,4),(3,5),(4,5),(5,6),(4,6),(2,4)])
    on = crl.compute(g, use_cache=True)
    off = crl.compute(g, use_cache=True,
                      reductions={"clique_sep": {"enabled": False, "p": 1.0}})
    assert on.poly.coeffs() == off.poly.coeffs()          # answer unchanged
    assert off.stats.branches >= on.stats.branches        # cost may rise

def test_defer_seed_deterministic():
    import chromatic_rl as crl
    g = crl.Graph(7, [(0,1),(0,2),(1,2),(2,3),(3,4),(3,5),(4,5),(5,6),(4,6),(2,4)])
    red = {"clique_sep": {"enabled": True, "p": 0.5}}
    r1 = crl.compute(g, use_cache=True, reductions=red, defer_seed=7)
    r2 = crl.compute(g, use_cache=True, reductions=red, defer_seed=7)
    assert r1.stats.branches == r2.stats.branches
    assert r1.poly.coeffs() == r2.poly.coeffs()

def test_memory_proxy_fields_present():
    import chromatic_rl as crl
    # complement(C_6): 9 edges, no CRT-1 clique sep, no join. forces at least one
    # branch so cache.store() is called and cache_words > 0.
    # (The brief's 7-vertex graph resolves entirely via reductions — 0 branches — so
    # cache_words would be 0 there; this graph is a verified substitute.)
    g = crl.Graph(6, [(0,2),(0,3),(0,4),(1,3),(1,4),(1,5),(2,4),(2,5),(3,5)])
    res = crl.compute(g, use_cache=True)
    assert res.stats.peak_live_words > 0
    assert res.stats.cache_words > 0
