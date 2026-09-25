# Results

Headline numbers from the deployed edge-selection policies, measured against the engine's two hand-written baseline heuristics (`min_degree_sum`, `max_degree_sum`). "Branches" is the search-tree cost the policies are trained to minimize: fewer branches means less work to compute the chromatic polynomial.

Raw data behind these numbers: [`bench.jsonl`](bench.jsonl) has one row per (graph family, size, selector), generated with `python -m chromatic_rl.bench` over the families shipped with this repo (`erdos_renyi`, `random_regular`, `grid`, `path`, `cycle`, `complete`, `random_tree`). Every row's polynomial is exact and cache-verified by the engine itself.

## Branch-count reduction vs. `min_degree_sum`

Pooled over the non-trivial instances (graphs where the baseline actually branches; i.e. excluding paths/cycles/complete graphs/trees, which the engine solves in closed form with zero branches regardless of selector):

| Selector | Branches vs. `min_degree_sum` |
|---|---|
| `max_degree_sum` (the *other* hand-written baseline) | **6.16×** (*more*) |
| `learned` (linear policy) | **0.73×** (27% fewer) |
| `mlp` (residual policy) | **0.60×** (40% fewer) |

## By graph family

| Family | `min_degree_sum` | `learned` | `mlp` |
|---|---|---|---|
| grid | 404.0 | 300.3 (0.74×) | 242.7 (0.60×) |
| erdos_renyi | 10.0 | 3.6 (0.36×) | 3.4 (0.34×) |
| random_regular | 10.0 | 9.0 (0.90×) | 10.7 (1.07×) |

mean branch count per family; small sample sizes (this is an illustrative benchmark)

The `mlp` policy doesn't uniformly beat `min_degree_sum`: on `random_regular` it's slightly *worse* (1.07×). This pattern is seen throughout our work: there is no single policy which wins on every graph family, and any published number should be read as "on these families, at these sizes," not as a universal claim.

## Reproducing this

```
python -m chromatic_rl.bench --config bench/sweep.example.yaml --out bench.jsonl
```

or drive it interactively through the visualizer (`python -m chromatic_rl.viz serve`), which lets you draw or generate a graph, pick a selector, and watch it branch live.
