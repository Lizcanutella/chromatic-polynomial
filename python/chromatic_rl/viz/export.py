"""Render a trace doc into the self-contained viewer HTML."""
from __future__ import annotations

import json
import os

import chromatic_rl as crl

from .doc import build_doc

_STATIC = os.path.join(os.path.dirname(__file__), "static")


def render(doc):
    with open(os.path.join(_STATIC, "viewer.html")) as f:
        html = f.read()
    with open(os.path.join(_STATIC, "d3.v7.min.js")) as f:
        d3 = f.read()
    payload = json.dumps(doc).replace("</", "<\\/")   # never close the script tag
    html = html.replace("/*__D3__*/", d3, 1)
    html = html.replace("/*__VIZ_DOC__*/null", payload, 1)
    return html


def export_html(graph, *, selector="none", use_cache=True, scores=False,
                out="run.html", input_desc="", positions=None):
    from chromatic_rl.dataset.score import resolve_selector
    sel = resolve_selector(selector)
    res = crl.compute(graph, selector=sel, use_cache=use_cache, record="full")
    doc = build_doc(res, graph.num_vertices(), graph.edges(),
                    selector_name=selector, use_cache=use_cache, mode="export",
                    positions=positions, want_scores=scores, input_desc=input_desc)
    with open(out, "w") as f:
        f.write(render(doc))
    return out
