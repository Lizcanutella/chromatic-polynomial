"""Read a materialized dataset directory back into engine-ready records."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import chromatic_rl as crl

from .schema import Instance, Manifest, coeffs_from_str


@dataclass
class LoadedInstance:
    id: str
    family: str
    n: int
    params: dict
    seed: int | None
    tier: str
    graph: object              # crl.Graph
    reference_poly: list | None  # list[int]; None for uncertified maps (Unit 6)
    build_stats: dict
    verified: dict
    positions: list            # list[[x, y]] normalized; [] for non-map instances
    labels: list               # list[str] region names; [] when absent
    excluded_reason: str | None = None  # why an uncertified instance ships without a poly


def _to_loaded(inst: Instance) -> LoadedInstance:
    edges = [(int(u), int(v)) for (u, v) in inst.edges]
    graph = crl.Graph(inst.n, edges)
    return LoadedInstance(
        id=inst.id, family=inst.family, n=inst.n, params=inst.params,
        seed=inst.seed, tier=inst.tier, graph=graph,
        reference_poly=(coeffs_from_str(inst.reference_poly)
                        if inst.reference_poly is not None else None),
        build_stats=inst.build_stats, verified=inst.verified,
        positions=inst.positions,
        labels=inst.labels,
        excluded_reason=inst.excluded_reason,
    )


def load(dataset_dir: str) -> tuple:
    with open(os.path.join(dataset_dir, "manifest.json")) as f:
        manifest = Manifest.from_json(json.load(f))
    loaded = []
    with open(os.path.join(dataset_dir, "instances.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            loaded.append(_to_loaded(Instance.from_json(json.loads(line))))
    return manifest, loaded
