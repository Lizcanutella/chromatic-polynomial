"""Build the trace-doc JSON consumed by the viewer template."""
from __future__ import annotations

import chromatic_rl as crl

from .replay import replay
from .scores import load_scorer

_MINUS = "-"  # typographic minus


def poly_html(coeffs):
    """Pretty-print ascending power-basis coeffs as HTML (k<sup>i</sup> terms)."""
    terms = []
    for i in range(len(coeffs) - 1, -1, -1):
        c = int(coeffs[i])
        if c == 0:
            continue
        mag = abs(c)
        if i == 0:
            body = str(mag)
        else:
            body = ("" if mag == 1 else str(mag)) + "k" + (
                f"<sup>{i}</sup>" if i > 1 else "")
        terms.append(("-" if c < 0 else "+", body))
    if not terms:
        return "0"
    out = (_MINUS if terms[0][0] == "-" else "") + terms[0][1]
    for sign, body in terms[1:]:
        out += f" {_MINUS if sign == '-' else '+'} {body}"
    return out


def _spring_layout(n, edges):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=7)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span_x = (max(xs) - min(xs)) or 1.0
    span_y = (max(ys) - min(ys)) or 1.0
    return [
        [0.05 + 0.9 * (pos[i][0] - min(xs)) / span_x,
         0.05 + 0.9 * (pos[i][1] - min(ys)) / span_y]
        for i in range(n)
    ]


def build_doc(result, root_n, root_edges, *, selector_name, use_cache,
              mode="export", positions=None, want_scores=False, input_desc=""):
    """result: Result from crl.compute(..., record='full')."""
    trace = result.trace
    if not trace or not hasattr(trace[0], "via"):
        raise ValueError("build_doc requires compute(..., record='full')")
    scorer = None
    if want_scores:
        scorer = load_scorer(selector_name)
        if scorer is None:
            raise ValueError(
                f"selector {selector_name!r} has no scoreable weights; "
                "scores are available for 'learned' and 'mlp' only")

    root_edges = [tuple(e) for e in root_edges]
    nodes = replay(trace, root_n, root_edges)

    # depth (forward pass) + subtree branch counts (reverse pass; ids are DFS-ordered)
    depth = {}
    subtree = {r.id: (1 if r.kind == "branch" else 0) for r in trace}
    for r in trace:
        depth[r.id] = 0 if r.parent_id == -1 else depth[r.parent_id] + 1
    for r in reversed(trace):
        if r.parent_id != -1:
            subtree[r.parent_id] += subtree[r.id]

    out_nodes = []
    for r in trace:
        node = nodes[r.id]
        scores = None
        if scorer is not None and r.kind == "branch":
            edges, sc = scorer(crl.Graph(r.n, node.edges))
            order = {tuple(e): i for i, e in enumerate(edges)}
            scores = [sc[order[e]] for e in node.edges]
        out_nodes.append({
            "id": r.id, "parent": r.parent_id, "via": r.via,
            "via_edge": ([r.via_u, r.via_v]
                         if r.via in ("delete", "contract", "insert") else None),
            "via_verts": list(r.via_verts) if r.via == "piece" else None,
            "kind": r.kind, "n": r.n, "m": r.m,
            "edges": [list(e) for e in node.edges],
            "verts": (None if node.verts is None
                      else [sorted(s) for s in node.verts]),
            "chosen": list(r.chosen) if r.kind == "branch" else None,
            "scores": scores,
            "sep": list(r.sep) if r.kind in ("clique_sep", "complement_sep") else None,
            "pieces": ([list(p) for p in r.pieces]
                       if r.kind == "complement_sep" else None),
            "depth": depth[r.id],
            "subtree_branches": subtree[r.id],
        })

    st = result.stats
    return {
        "meta": {
            "mode": mode, "selector": selector_name, "use_cache": use_cache,
            "scores": want_scores,
            "input": {"desc": input_desc, "n": root_n,
                      "edges": [list(e) for e in root_edges]},
            "stats": {"branches": st.branches, "nodes": st.nodes,
                      "hits": st.hits, "misses": st.misses,
                      "hit_rate": st.hit_rate},
            "poly": {"coeffs": [str(c) for c in result.poly.coeffs()],
                     "html": poly_html(list(result.poly.coeffs()))},
        },
        "layout": (positions if positions is not None
                   else _spring_layout(root_n, root_edges)),
        "nodes": out_nodes,
    }
