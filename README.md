***This is a subset of a larger private research project.** It includes the full computation engine and the deployed policies, but not the RL training code, feature-engineering rationale or experiment history. You may refer to `results/RESULTS.md` for what's disclosed about performance.*

*This repository will be periodically updated with the full code.*

# Chromatic Polynomial Engine + RL Edge Selector

An exact C++ engine for computing chromatic polynomials by deletion-contraction (with structural reductions for speed optimization), and a **learned edge-selection policy**: at each branching step the engine picks an edge to delete/contract, and which edge you pick changes your future steps; affecting how much work the computation will take in the long-run. This repo ships two small trained policies (a linear model and a residual MLP) that pick better edges than the obvious hand-written heuristics, distilled down to fast, dependency-free scorers. See [`results/RESULTS.md`](results/RESULTS.md) for the numbers.

It also ships an interactive reduction visualizer: draw or generate a graph, pick a selector, and watch the search tree branch live. It includes a **Gallery of real-world political maps** (216 countries/subdivisions, from [Natural Earth](https://www.naturalearth.com/) data) rendered as their adjacency graphs, each with its certified chromatic polynomial. 

*Note: The real-world political maps have not all been verified, and are known to contain errors (missing or extra edges; extra nodes). Feel free to reach out with any corrections.*


## Quickstart

### Build the C++ core

```bash
sudo apt-get install -y g++ libgmp-dev   # Ubuntu/Debian; libgmp-dev provides gmpxx.h
./build.sh                               # builds and runs the validation suite
```

`validate` cross-checks the engine against closed-form polynomials (paths, cycles, complete graphs) *and* an independent brute-force proper-colouring oracle on random graphs which make up our correctness gate.

### Python bridge + visualizer

The engine's isomorphism cache needs [nauty](https://pallini.di.uniroma1.it/)
(its license asks you to fetch it yourself, so it is not bundled here). Download a release tarball (e.g. `nauty2_9_3.tar.gz`), drop it at the repo root, then:

```bash
./build_nauty.sh                         # builds nauty portable + -fPIC, once
uv pip install -e .                      # or: pip install -e .
python -m chromatic_rl.viz serve         # opens the visualizer at localhost
```

Or use it as a library:

```python
import chromatic_rl as crl

g = crl.Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
res = crl.compute(g, selector=crl.MIN_DEGREE_SUM)
print(res.poly.coeffs(), res.stats.branches)

from chromatic_rl.public_selector import learned_selector
res = crl.compute(g, selector=learned_selector())
```

`selector=` accepts `None`, `"min_degree_sum"`/`"max_degree_sum"` (the two built-in C++ heuristics), or any Python callable `(Graph) -> (u, v)` which is how the shipped `learned`/`mlp` policies plug in, and how you'd plug in your own.

### Benchmarking

```bash
uv run python -m chromatic_rl.bench --config bench/sweep.example.yaml --out bench.jsonl
```

Runs a configurable sweep of graph families against a set of selectors and records branch counts, cache hit rate, and wall-clock. See [`results/RESULTS.md`](results/RESULTS.md) for a sample sweep's output.

### Tests

```bash
uv run pytest tests/python/
ctest --test-dir build   # C++ engine tests (needs cmake -B build && cmake --build build)
```

## Layout

```
include/             the engine: graph rep, exact GMP polynomials, deletion-contraction +
                     structural reductions, nauty-backed isomorphism cache
bindings/            pybind11 module exposing the engine to Python
python/chromatic_rl/ Python package: bench sweeps, dataset loading, the visualizer,
                     map-image-to-graph upload, and the deployed edge-selection policies
                     (public_selector.py)
datasets/maps/       216 certified real-world map instances (countries + subdivisions)
                     backing the visualizer's Gallery tab
results/             benchmark numbers for the deployed policies (RESULTS.md + raw data)
tests/               C++ (tests/) and Python (tests/python/) test suites
```

## The maps Gallery

The Gallery loads standalone off the committed `datasets/maps/`. No extra setup needed. One toggle, "see original," overlays real region borders reconstructed from raw Natural Earth shapefiles; that needs the optional `[geo]` extra (`uv pip install geopandas shapely`) plus your own download of the Natural Earth admin-0/admin-1 datasets, neither of which ships here. Without
it, the Gallery still works, but you won't get the border overlay.

## License

MIT: See [LICENSE](LICENSE).
