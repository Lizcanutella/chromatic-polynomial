"""chromatic_rl: Python bridge to the verified C++ chromatic-polynomial engine."""
from ._core import (  # noqa: F401
    __version__, Graph, Poly, Stats, Result, TraceRecord, FullTraceRecord, compute,
    random_selector,
    clique_separator, clique_separator_candidates, clique_separator_pieces,
    builtin_selector, BudgetExceeded,
)

MIN_DEGREE_SUM = "min_degree_sum"
MAX_DEGREE_SUM = "max_degree_sum"

__all__ = [
    "__version__", "Graph", "Poly", "Stats", "Result", "TraceRecord", "FullTraceRecord",
    "compute",
    "random_selector",
    "clique_separator", "clique_separator_candidates", "clique_separator_pieces",
    "builtin_selector", "BudgetExceeded",
    "MIN_DEGREE_SUM", "MAX_DEGREE_SUM",
    "from_networkx", "to_networkx",
]


def from_networkx(G):
    """Build an engine Graph from a networkx.Graph.

    Nodes are relabelled to a contiguous 0..n-1 range in sorted node order.
    """
    nodes = sorted(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    edges = [(index[u], index[v]) for u, v in G.edges()]
    return Graph(len(nodes), edges)


def to_networkx(g):
    """Build a networkx.Graph from an engine Graph."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(g.num_vertices()))
    G.add_edges_from(g.edges())
    return G
