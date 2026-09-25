"""CLI: python -m chromatic_rl.viz {export|serve} ..."""
from __future__ import annotations

import argparse


def make_graph_from_args(a):
    """Returns (crl.Graph, desc). Family route uses chromatic_rl.bench generators."""
    import chromatic_rl as crl
    if a.dataset and a.id:
        from chromatic_rl.dataset import load
        _manifest, instances = load(a.dataset)
        inst = next((i for i in instances if i.id == a.id), None)
        if inst is None:
            raise SystemExit(f"instance {a.id!r} not in {a.dataset}")
        return inst.graph, f"{a.dataset}:{a.id}"
    if not a.family:
        raise SystemExit("need --family … or --dataset … --id …")
    from chromatic_rl.bench import GENERATORS
    gen = GENERATORS[a.family]
    kwargs = {}
    if a.family == "erdos_renyi":
        kwargs = dict(n=a.n, p=a.p, seed=a.seed)
    elif a.family == "random_regular":
        kwargs = dict(n=a.n, d=a.d, seed=a.seed)
    elif a.family == "grid":
        kwargs = dict(rows=a.rows, cols=a.cols)
    elif a.family == "random_tree":
        kwargs = dict(n=a.n, seed=a.seed)
    else:                                   # path / cycle / complete
        kwargs = dict(n=a.n)
    g, meta = gen(**kwargs)
    desc = f"{a.family} " + " ".join(f"{k}={v}" for k, v in kwargs.items())
    return g, desc


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m chromatic_rl.viz")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("export", help="run the engine and write a viewer HTML")
    ex.add_argument("--family")
    ex.add_argument("--n", type=int, default=10)
    ex.add_argument("--p", type=float, default=0.3)
    ex.add_argument("--d", type=int, default=3)
    ex.add_argument("--rows", type=int, default=3)
    ex.add_argument("--cols", type=int, default=4)
    ex.add_argument("--seed", type=int, default=1)
    ex.add_argument("--dataset")
    ex.add_argument("--id")
    ex.add_argument("--selector", default="none")
    ex.add_argument("--no-cache", action="store_true")
    ex.add_argument("--scores", action="store_true")
    ex.add_argument("--out", default="run.html")

    sv = sub.add_parser("serve", help="interactive editor + viewer on localhost")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--dataset", help="dataset dir for quick-start instances")
    sv.add_argument("--maps-dataset", default=None, dest="maps_dataset",
                    help="maps dataset dir for the Gallery tab (default: datasets/maps if present)")
    sv.add_argument("--geodata", default=None,
                    help="Natural Earth shapefile dir for the Gallery 'see original' "
                         "outline overlay (default: datasets/naturalearth if present; "
                         "needs the [geo] extra)")
    sv.add_argument("--budget", type=int, default=50000)  # match maps dataset branch_cap
    sv.add_argument("--timeout", type=float, default=60.0)
    sv.add_argument("--no-browser", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "export":
        from .export import export_html
        g, desc = make_graph_from_args(a)
        out = export_html(g, selector=a.selector, use_cache=not a.no_cache,
                          scores=a.scores, out=a.out, input_desc=desc)
        print(f"wrote {out}")
    else:
        import os
        from .serve import serve
        maps_dir = a.maps_dataset if a.maps_dataset else "datasets/maps"
        maps_dataset = maps_dir if os.path.isdir(maps_dir) else None
        geo_dir = a.geodata if a.geodata else "datasets/naturalearth"
        geodata = geo_dir if os.path.isdir(geo_dir) else None
        serve(port=a.port, dataset=a.dataset, maps_dataset=maps_dataset,
              geodata=geodata, budget=a.budget, timeout=a.timeout,
              open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
