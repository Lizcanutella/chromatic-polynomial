from chromatic_rl import compute
from chromatic_rl.bench import generators as gen


def test_path_generator_shape_and_meta():
    g, meta = gen.path(5)
    assert g.num_vertices() == 5
    assert g.num_edges() == 4
    assert meta == {"family": "path", "n": 5, "params": {}, "seed": None}


def test_closed_form_chromatic_polynomials():
    # P_4: k(k-1)^3 -> at k=3 == 24
    assert compute(gen.path(4)[0]).poly.evaluate(3) == 24
    # C_4: (k-1)^4 + (k-1) -> at k=3 == 18
    assert compute(gen.cycle(4)[0]).poly.evaluate(3) == 18
    # K_4: k(k-1)(k-2)(k-3) -> at k=4 == 24
    assert compute(gen.complete(4)[0]).poly.evaluate(4) == 24
    # random tree on n nodes: k(k-1)^(n-1) -> at k=3, n=5 == 3*2^4 == 48
    assert compute(gen.random_tree(5, seed=0)[0]).poly.evaluate(3) == 48


def test_erdos_renyi_is_seed_deterministic():
    g1, _ = gen.erdos_renyi(10, 0.5, seed=7)
    g2, _ = gen.erdos_renyi(10, 0.5, seed=7)
    assert sorted(g1.edges()) == sorted(g2.edges())


def test_grid_and_regular_shapes():
    g, meta = gen.grid(3, 4)
    assert g.num_vertices() == 12
    assert meta["params"] == {"rows": 3, "cols": 4}
    r, rmeta = gen.random_regular(6, 3, seed=1)
    assert all(r.degree(v) == 3 for v in range(6))
    assert rmeta["family"] == "random_regular"


def test_registry_lists_all_families():
    assert set(gen.GENERATORS) == {
        "erdos_renyi", "random_regular", "grid",
        "path", "cycle", "complete", "cocktail_party", "random_tree", "random_planar",
        "random_planar_sparse",
    }


from chromatic_rl.bench import runner


def _spec(family, gen_kwargs, selector="none", use_cache=True, repeat=1):
    return {"family": family, "gen_kwargs": gen_kwargs, "selector": selector,
            "use_cache": use_cache, "repeat": repeat}


def test_run_one_returns_full_schema():
    row = runner.run_one(_spec("cycle", {"n": 5}))
    for key in ("family", "n", "params", "seed", "selector", "use_cache",
                "nodes", "branches", "hits", "misses", "hit_rate", "wall_s", "repeat"):
        assert key in row, key
    assert row["family"] == "cycle"
    assert row["n"] == 5
    assert row["selector"] == "none"
    assert isinstance(row["branches"], int)
    assert row["wall_s"] >= 0.0
    assert "error" not in row


def test_run_one_branches_are_deterministic():
    a = runner.run_one(_spec("erdos_renyi", {"n": 9, "p": 0.5, "seed": 3}))
    b = runner.run_one(_spec("erdos_renyi", {"n": 9, "p": 0.5, "seed": 3}))
    assert a["branches"] == b["branches"]


def test_run_one_records_error_without_raising():
    # random_regular requires n*d even and n > d; (n=5, d=3) is invalid in networkx
    row = runner.run_one(_spec("random_regular", {"n": 5, "d": 3, "seed": 0}))
    assert "error" in row
    assert "branches" not in row
    assert row["family"] == "random_regular"


def test_run_one_accepts_callable_selector():
    def first_edge(g):
        return g.edges()[0]
    row = runner.run_one(_spec("path", {"n": 6}, selector=first_edge))
    assert row["selector"] == "first_edge"
    assert isinstance(row["branches"], int)


def test_run_sweep_yields_one_row_per_spec():
    specs = [_spec("path", {"n": 4}), _spec("cycle", {"n": 4})]
    rows = list(runner.run_sweep(specs))
    assert [r["family"] for r in rows] == ["path", "cycle"]


import pytest
from chromatic_rl.bench import config


def test_parse_expands_cartesian_grid():
    raw = {
        "out": "x.jsonl",
        "selectors": ["none", "min_degree_sum"],
        "runs": [{"family": "erdos_renyi", "n": [8, 10], "p": [0.5], "seeds": [0, 1]}],
    }
    out, specs = config.parse(raw)
    assert out == "x.jsonl"
    # 2 sizes * 1 p * 2 seeds * 2 selectors = 8
    assert len(specs) == 8
    one = specs[0]
    assert one["family"] == "erdos_renyi"
    assert set(one["gen_kwargs"]) == {"n", "p", "seed"}
    assert one["selector"] in {"none", "min_degree_sum"}


def test_deterministic_family_needs_no_seed():
    raw = {"runs": [{"family": "path", "n": [4, 5]}]}
    _, specs = config.parse(raw)
    assert len(specs) == 2  # 2 sizes * 1 default selector (none)
    assert "seed" not in specs[0]["gen_kwargs"]


def test_unknown_family_raises():
    with pytest.raises(config.ConfigError):
        config.parse({"runs": [{"family": "nope", "n": [4]}]})


def test_missing_required_param_raises():
    with pytest.raises(config.ConfigError):
        config.parse({"runs": [{"family": "erdos_renyi", "n": [8]}]})  # no p


def test_seeds_on_deterministic_family_raises():
    with pytest.raises(config.ConfigError):
        config.parse({"runs": [{"family": "path", "n": [4], "seeds": [0]}]})


def test_unknown_selector_raises():
    with pytest.raises(config.ConfigError):
        config.parse({"selectors": ["bogus"], "runs": [{"family": "path", "n": [4]}]})


@pytest.mark.parametrize("bad", [0, -1, 2.0, "3", True])
def test_nonpositive_or_nonint_repeat_raises(bad):
    # repeat<1 (or a non-int) must fail fast at parse time, not slip through to an error row.
    with pytest.raises(config.ConfigError):
        config.parse({"runs": [{"family": "path", "n": [4], "repeat": bad}]})


def test_top_level_repeat_default_applies_and_is_validated():
    _, specs = config.parse({"repeat": 3, "runs": [{"family": "path", "n": [4]}]})
    assert specs[0]["repeat"] == 3
    with pytest.raises(config.ConfigError):
        config.parse({"repeat": 0, "runs": [{"family": "path", "n": [4]}]})


import json
from chromatic_rl.bench import report


def test_append_writes_one_json_line_per_row(tmp_path):
    path = tmp_path / "out.jsonl"
    report.append({"family": "path", "branches": 1}, path)
    report.append({"family": "cycle", "branches": 2}, path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["family"] == "path"


def test_summarize_groups_by_family_and_selector():
    rows = [
        {"family": "path", "selector": "none", "branches": 10, "wall_s": 0.1},
        {"family": "path", "selector": "none", "branches": 20, "wall_s": 0.3},
        {"family": "path", "selector": "none", "error": "boom"},
    ]
    summary = report.summarize(rows)
    assert len(summary) == 1
    s = summary[0]
    assert s["runs"] == 3
    assert s["errors"] == 1
    assert s["mean_branches"] == 15
    assert s["median_branches"] == 15


def test_format_table_is_printable_string():
    summary = report.summarize([
        {"family": "path", "selector": "none", "branches": 5, "wall_s": 0.2},
    ])
    text = report.format_table(summary)
    assert "family" in text
    assert "path" in text


from chromatic_rl.bench.__main__ import main as bench_main


def test_cli_end_to_end_writes_jsonl_and_returns_zero(tmp_path, capsys):
    cfg = tmp_path / "sweep.yaml"
    cfg.write_text(
        "selectors: [none, min_degree_sum]\n"
        "runs:\n"
        "  - family: cycle\n"
        "    n: [4, 5]\n"
    )
    out = tmp_path / "bench.jsonl"
    rc = bench_main(["--config", str(cfg), "--out", str(out)])
    assert rc == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 4  # 2 sizes * 2 selectors
    printed = capsys.readouterr().out
    assert "cycle" in printed


def test_package_exports_public_api():
    import chromatic_rl.bench as b
    assert hasattr(b, "run_sweep")
    assert hasattr(b, "GENERATORS")
    assert hasattr(b, "load")


import networkx as nx
from chromatic_rl.bench.generators import GENERATORS, random_planar, random_planar_sparse
from chromatic_rl import to_networkx


def _planar_nx(n, seed):
    g, meta = random_planar(n, seed)
    assert meta == {"family": "random_planar", "n": n, "params": {}, "seed": seed}
    return to_networkx(g)


@pytest.mark.parametrize("n", [4, 8, 12, 16])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_planar_is_maximal_planar(n, seed):
    G = _planar_nx(n, seed)
    assert G.number_of_nodes() == n
    assert nx.check_planarity(G)[0] is True          # planar
    assert G.number_of_edges() == 3 * n - 6          # maximal planar
    assert nx.is_connected(G)                        # connected
    assert nx.number_of_selfloops(G) == 0            # simple


def test_random_planar_registered():
    assert GENERATORS["random_planar"] is random_planar


def test_random_planar_seed_determinism():
    a = sorted(map(tuple, _planar_nx(12, 5).edges()))
    b = sorted(map(tuple, _planar_nx(12, 5).edges()))
    c = sorted(map(tuple, _planar_nx(12, 6).edges()))
    assert a == b          # same seed -> identical graph
    assert a != c          # different seed -> generally different


def test_random_planar_rejects_tiny_n():
    with pytest.raises(ValueError):
        random_planar(2, 0)


@pytest.mark.parametrize("n", [10, 16, 24])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_planar_sparse_is_sparse_planar(n, seed):
    g, meta = random_planar_sparse(n, seed)
    assert meta == {"family": "random_planar_sparse", "n": n, "params": {}, "seed": seed}
    G = to_networkx(g)
    assert G.number_of_nodes() == n
    assert nx.check_planarity(G)[0] is True       # planar (subgraph of planar)
    assert nx.is_connected(G)                     # connected (keeps a spanning tree)
    assert nx.number_of_selfloops(G) == 0         # simple
    assert G.number_of_edges() < 3 * n - 6        # NON-maximal (thinned)
    assert G.number_of_edges() >= n - 1           # at least a spanning tree


def test_random_planar_sparse_registered():
    assert GENERATORS["random_planar_sparse"] is random_planar_sparse


def test_random_planar_sparse_determinism():
    a = sorted(map(tuple, to_networkx(random_planar_sparse(20, 5)[0]).edges()))
    b = sorted(map(tuple, to_networkx(random_planar_sparse(20, 5)[0]).edges()))
    c = sorted(map(tuple, to_networkx(random_planar_sparse(20, 6)[0]).edges()))
    assert a == b          # same seed -> identical graph
    assert a != c          # different seed -> generally different


def test_random_planar_sparse_rejects_tiny_n():
    with pytest.raises(ValueError):
        random_planar_sparse(2, 0)


def test_cocktail_party_generator():
    from chromatic_rl.bench.generators import GENERATORS
    graph, meta = GENERATORS["cocktail_party"](r=3, s=2)   # K_{2,2,2}, octahedron
    assert meta["family"] == "cocktail_party"
    assert meta["n"] == 6
    assert graph.num_edges() == 12                          # 3 parts of 2: complete 3-partite
