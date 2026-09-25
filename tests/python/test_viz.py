"""Tests for the reduction visualizer: enriched trace, replay, doc builder, export."""
import pytest

import chromatic_rl as crl


def petersen():
    import networkx as nx
    return crl.from_networkx(nx.petersen_graph())


def square_plus_tail():
    # 4-cycle 0-1-2-3 plus a pendant path 3-4-5: exercises branch + reductions.
    return crl.Graph(6, [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5)])


class TestFullTrace:
    def test_record_false_and_true_unchanged(self):
        g = square_plus_tail()
        assert crl.compute(g, use_cache=False).trace is None
        legacy = crl.compute(g, use_cache=False, record=True).trace
        assert all(isinstance(r, crl.TraceRecord) for r in legacy)

    def test_full_trace_structure(self):
        g = square_plus_tail()
        res = crl.compute(g, use_cache=False, record="full")
        tr = res.trace
        assert len(tr) >= 1
        root = tr[0]
        assert (root.id, root.parent_id, root.via) == (0, -1, "root")
        assert root.n == 6 and root.m == 6
        kinds = {r.kind for r in tr}
        assert "unset" not in kinds
        for r in tr:
            assert r.parent_id < r.id
        n_branch = sum(1 for r in tr if r.kind == "branch")
        assert n_branch == res.stats.branches

    def test_full_matches_legacy_branch_view(self):
        g = petersen()
        full = crl.compute(g, use_cache=False, record="full").trace
        legacy = crl.compute(g, use_cache=False, record=True).trace
        fb = [r for r in full if r.kind == "branch"]
        assert len(fb) == len(legacy)
        for f, l in zip(fb, legacy):
            assert f.n == l.n
            assert tuple(f.chosen) == tuple(l.chosen)
            assert f.subtree_branches == l.subtree_branches

    def test_reduction_kinds_visible(self):
        # C4 + pendant path: the engine finds a cut vertex (clique_sep l=1)
        res = crl.compute(square_plus_tail(), use_cache=False, record="full")
        kinds = {r.kind for r in res.trace}
        assert "clique_sep" in kinds
        sepnode = next(r for r in res.trace if r.kind == "clique_sep")
        assert len(sepnode.sep) >= 1
        children = [r for r in res.trace if r.parent_id == sepnode.id]
        assert children and all(c.via == "piece" for c in children)
        assert all(len(c.via_verts) >= 1 for c in children)

    def test_zero_branch_run_octahedron(self):
        # K_{2,2,2}: fully decomposed by join/complement machinery -> 0 branches
        import networkx as nx
        g = crl.from_networkx(nx.complete_multipartite_graph(2, 2, 2))
        res = crl.compute(g, use_cache=False, record="full")
        assert res.stats.branches == 0
        assert len(res.trace) >= 1          # the reduction tree is still visible
        assert all(r.kind != "branch" for r in res.trace)

    def test_cache_hit_recorded(self):
        res = crl.compute(petersen(), use_cache=True, record="full")
        assert res.stats.hits == sum(1 for r in res.trace if r.kind == "cache_hit")

    def test_record_bad_value_raises(self):
        with pytest.raises((ValueError, TypeError)):
            crl.compute(square_plus_tail(), record="fuller")


class TestReplay:
    def _replayed(self, g, **kw):
        from chromatic_rl.viz.replay import replay
        res = crl.compute(g, use_cache=False, record="full", **kw)
        return res.trace, replay(res.trace, g.num_vertices(), g.edges())

    def test_checksums_and_kinds(self):
        tr, nodes = self._replayed(petersen())
        assert set(nodes) == {r.id for r in tr}
        for r in tr:
            assert len(nodes[r.id].edges) == r.m       # replay reproduces engine m
            if nodes[r.id].verts is not None:
                assert len(nodes[r.id].verts) == r.n

    def test_contract_child_matches_engine_op(self):
        # For every contract child, replayed edges == crl.Graph.contract_edge oracle.
        tr, nodes = self._replayed(petersen())
        by_id = {r.id: r for r in tr}
        checked = 0
        for r in tr:
            if r.via != "contract":
                continue
            parent = nodes[r.parent_id]
            pg = crl.Graph(len(parent.verts), parent.edges)
            expect = sorted(tuple(e) for e in pg.contract_edge(r.via_u, r.via_v).edges())
            assert nodes[r.id].edges == expect
            checked += 1
        assert checked > 0

    def test_identity_partition(self):
        # Root-identity sets at any node are disjoint and subsets of root vertices.
        g = petersen()
        tr, nodes = self._replayed(g)
        for r in tr:
            v = nodes[r.id].verts
            if v is None:
                continue
            allv = [x for s in v for x in s]
            assert len(allv) == len(set(allv))
            assert set(allv) <= set(range(g.num_vertices()))

    def test_contract_merges_identity(self):
        # n=4 with a single non-edge (as in the task brief) has a disconnected
        # complement, so the join reduction fully decomposes it with zero
        # contractions. Use a slightly larger graph (networkx gnp seed=10, n=7)
        # that survives join/clique_sep/complement_sep and actually branches.
        g = crl.Graph(7, [(0, 2), (0, 4), (1, 3), (1, 5), (1, 6),
                           (2, 5), (3, 5), (3, 6), (4, 6)])
        tr, nodes = self._replayed(g)
        merged = [
            n for r in tr
            if r.via == "contract" and (n := nodes[r.id]).verts is not None
            for s in n.verts if len(s) > 1
        ]
        assert merged   # at least one merged vertex carries a multi-root identity

    def test_ie_piece_stops_identity(self):
        # Dense-ish graph with an independent-set separator in the complement.
        import networkx as nx
        g = crl.from_networkx(nx.complement(nx.gnp_random_graph(9, 0.75, seed=5)))
        res = crl.compute(g, use_cache=False, record="full")
        tr = res.trace
        if not any(r.kind == "complement_sep" for r in tr):
            pytest.skip("no complement_sep fired on this instance")
        from chromatic_rl.viz.replay import replay
        nodes = replay(tr, g.num_vertices(), g.edges())
        ies = [r for r in tr if r.via == "ie_piece"]
        assert ies
        for r in ies:
            assert nodes[r.id].verts is None
            assert len(nodes[r.id].edges) == r.m       # engine-recorded edges used


class TestScores:
    def test_learned_scorer_argmax_matches_engine(self):
        from chromatic_rl.viz.scores import load_scorer
        from chromatic_rl.dataset.score import resolve_selector
        g = petersen()
        sel = resolve_selector("learned")
        res = crl.compute(g, use_cache=False, record="full", selector=sel)
        scorer = load_scorer("learned")
        from chromatic_rl.viz.replay import replay
        nodes = replay(res.trace, g.num_vertices(), g.edges())
        checked = 0
        for r in res.trace:
            if r.kind != "branch":
                continue
            node = nodes[r.id]
            edges, sc = scorer(crl.Graph(r.n, node.edges))
            best = max(sc)
            chosen_score = sc[edges.index(tuple(r.chosen))]
            assert chosen_score >= best - 1e-9      # chosen edge is (tied-)argmax
            checked += 1
        assert checked > 0

    def test_mlp_scorer_argmax_matches_engine(self):
        from chromatic_rl.viz.scores import load_scorer
        from chromatic_rl.dataset.score import resolve_selector
        g = square_plus_tail()
        sel = resolve_selector("mlp")
        res = crl.compute(g, use_cache=False, record="full", selector=sel)
        scorer = load_scorer("mlp")
        from chromatic_rl.viz.replay import replay
        nodes = replay(res.trace, g.num_vertices(), g.edges())
        for r in res.trace:
            if r.kind != "branch":
                continue
            edges, sc = scorer(crl.Graph(r.n, nodes[r.id].edges))
            assert sc[edges.index(tuple(r.chosen))] >= max(sc) - 1e-9

    def test_unknown_selector_has_no_scorer(self):
        from chromatic_rl.viz.scores import load_scorer
        assert load_scorer("min_degree_sum") is None


class TestDoc:
    def _doc(self, g, **kw):
        from chromatic_rl.viz.doc import build_doc
        res = crl.compute(g, use_cache=kw.pop("use_cache", False), record="full",
                          selector=kw.pop("selector", None))
        return res, build_doc(res, g.num_vertices(), g.edges(),
                              selector_name="min_degree_sum",
                              use_cache=False, **kw)

    def test_schema_and_json_serializable(self):
        import json
        res, doc = self._doc(petersen(), input_desc="petersen")
        json.dumps(doc)                              # must be pure-JSON types
        assert doc["meta"]["input"]["n"] == 10
        assert doc["meta"]["poly"]["coeffs"] == [str(c) for c in res.poly.coeffs()]
        assert len(doc["layout"]) == 10
        assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in doc["layout"])
        ids = [nd["id"] for nd in doc["nodes"]]
        assert ids == sorted(ids)

    def test_subtree_branches_matches_engine(self):
        res, doc = self._doc(petersen())
        eng = {r.id: r.subtree_branches for r in res.trace if r.kind == "branch"}
        for nd in doc["nodes"]:
            if nd["kind"] == "branch":
                assert nd["subtree_branches"] == eng[nd["id"]]
        assert doc["nodes"][0]["subtree_branches"] == res.stats.branches

    def test_depth_and_parenting(self):
        _, doc = self._doc(square_plus_tail())
        by_id = {nd["id"]: nd for nd in doc["nodes"]}
        for nd in doc["nodes"]:
            if nd["parent"] == -1:
                assert nd["depth"] == 0
            else:
                assert nd["depth"] == by_id[nd["parent"]]["depth"] + 1

    def test_scores_attached_only_when_asked(self):
        from chromatic_rl.viz.doc import build_doc
        from chromatic_rl.dataset.score import resolve_selector
        g = petersen()
        res = crl.compute(g, use_cache=False, record="full",
                          selector=resolve_selector("learned"))
        doc = build_doc(res, 10, g.edges(), selector_name="learned",
                        use_cache=False, want_scores=True)
        br = [nd for nd in doc["nodes"] if nd["kind"] == "branch"]
        assert br and all(
            nd["scores"] is not None and len(nd["scores"]) == len(nd["edges"])
            for nd in br)
        doc2 = build_doc(res, 10, g.edges(), selector_name="learned",
                         use_cache=False, want_scores=False)
        assert all(nd["scores"] is None for nd in doc2["nodes"])

    def test_scores_refused_for_unscorable_selector(self):
        from chromatic_rl.viz.doc import build_doc
        g = petersen()
        res = crl.compute(g, use_cache=False, record="full")
        with pytest.raises(ValueError, match="scores"):
            build_doc(res, 10, g.edges(), selector_name="min_degree_sum",
                      use_cache=False, want_scores=True)

    def test_positions_override_layout(self):
        from chromatic_rl.viz.doc import build_doc
        g = square_plus_tail()
        res = crl.compute(g, use_cache=False, record="full")
        pos = [[i / 10, 1 - i / 10] for i in range(6)]
        doc = build_doc(res, 6, g.edges(), selector_name="none", use_cache=False,
                        positions=pos)
        assert doc["layout"] == pos

    def test_poly_html(self):
        from chromatic_rl.viz.doc import poly_html
        # chi(K3) = k^3 - 3k^2 + 2k  (ascending coeffs [0, 2, -3, 1])
        assert poly_html([0, 2, -3, 1]) == "k<sup>3</sup> − 3k<sup>2</sup> + 2k"
        assert poly_html([0]) == "0"
        assert poly_html([5]) == "5"

    def test_rejects_legacy_trace_even_when_empty(self):
        from chromatic_rl.viz.doc import build_doc
        g = crl.Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])   # 4-cycle: 0 branches
        for record in (True, False):
            res = crl.compute(g, use_cache=False, record=record)
            with pytest.raises(ValueError, match="record='full'"):
                build_doc(res, 4, g.edges(), selector_name="none", use_cache=False)


class TestExport:
    def test_render_injects_doc_and_d3(self, tmp_path):
        from chromatic_rl.viz.export import export_html
        import json, re
        g = square_plus_tail()
        out = export_html(g, selector="min_degree_sum", use_cache=False,
                          out=str(tmp_path / "run.html"), input_desc="test graph")
        html = open(out).read()
        assert "/*__VIZ_DOC__*/" not in html and "/*__D3__*/" not in html
        # JSON string content escapes "</" as "<\/", so the first "</script>"
        # after the assignment closes this tag — non-greedy match is safe.
        m = re.search(r"window\.VIZ_DOC = (\{.*?\});</script>", html, re.S)
        doc = json.loads(m.group(1))   # json accepts the \/ escape natively
        assert doc["meta"]["input"]["n"] == 6
        assert doc["nodes"]
        assert "d3" in html.lower()

    def test_cli_export_family(self, tmp_path):
        import subprocess, sys, os
        out = tmp_path / "er.html"
        r = subprocess.run(
            [sys.executable, "-m", "chromatic_rl.viz", "export",
             "--family", "erdos_renyi", "--n", "8", "--p", "0.3", "--seed", "1",
             "--selector", "min_degree_sum", "--no-cache", "--out", str(out)],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert out.exists() and out.stat().st_size > 100_000   # d3 inlined

    def test_cli_export_dataset_instance(self, tmp_path):
        import subprocess, sys, json
        ds = "datasets/maps"
        # smallest instance (n=2), so the export stays a fast unit test
        insts = [json.loads(l) for l in open(f"{ds}/instances.jsonl")]
        iid = min(insts, key=lambda r: r["n"])["id"]
        out = tmp_path / "inst.html"
        r = subprocess.run(
            [sys.executable, "-m", "chromatic_rl.viz", "export",
             "--dataset", ds, "--id", iid,
             "--selector", "none", "--out", str(out)],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert out.exists()


import os
import re
import shutil
import subprocess
import tempfile

import chromatic_rl as crl
from chromatic_rl.viz.doc import build_doc


def _viewer_path():
    from chromatic_rl.viz import export as _ex
    return os.path.join(os.path.dirname(_ex.__file__), "static", "viewer.html")


def extract_viewer_script(html):
    """Return the app's inline JS (the last <script> block), with the D3 and
    VIZ_DOC injection placeholders neutralized so `node --check` sees pure JS."""
    blocks = re.findall(r"<script>(.*?)</script>", html, flags=re.DOTALL)
    # blocks[0] = D3 placeholder, blocks[1] = VIZ_DOC line, blocks[-1] = the app
    return blocks[-1]


class TestViewerContract:
    def test_viewer_js_syntax(self):
        node = shutil.which("node")
        if node is None:
            import pytest
            pytest.skip("node not available")
        with open(_viewer_path()) as f:
            html = f.read()
        js = extract_viewer_script(html)
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tf:
            tf.write("const window={};const document={};const d3={};\n" + js)
            path = tf.name
        try:
            r = subprocess.run([node, "--check", path],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
        finally:
            os.unlink(path)

    def test_triptych_data_contract(self):
        # square_plus_tail triggers a branch AND a clique_sep decomposition.
        g = crl.Graph(6, [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5)])
        res = crl.compute(g, use_cache=False, record="full")
        doc = build_doc(res, 6, g.edges(), selector_name="none", use_cache=False)
        nodes = {nd["id"]: nd for nd in doc["nodes"]}
        children = {i: [] for i in nodes}
        for nd in doc["nodes"]:
            if nd["parent"] != -1:
                children[nd["parent"]].append(nd["id"])
        # Every branch node has exactly 2 derived graphs (delete + contract),
        # each drawable (non-empty edges OR n<=1) with a via we can label.
        for nd in doc["nodes"]:
            kids = [nodes[c] for c in children[nd["id"]]]
            if nd["kind"] == "branch":
                assert len(kids) == 2, (nd["id"], [k["via"] for k in kids])
                assert {k["via"] for k in kids} == {"delete", "contract"}
            for k in kids:
                assert k["via"] in ("delete", "contract", "insert",
                                    "piece", "ie_piece")
                assert isinstance(k["edges"], list)
                assert k["n"] >= 1
        # At least one decomposition node with >= 2 piece children exists.
        decomp = [nd for nd in doc["nodes"]
                  if nd["kind"] in ("clique_sep", "complement_sep",
                                    "disconnected", "join")]
        assert decomp, "expected a decomposition node in this graph"
        assert any(len(children[d["id"]]) >= 2 for d in decomp)
