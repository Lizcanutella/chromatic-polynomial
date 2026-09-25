"""Parse + validate a YAML sweep file into fully-expanded run specs.

Top-level keys (out, selectors, use_cache, repeat) set defaults; each entry in
`runs` is a block whose list-valued generator params are expanded as a cartesian
product (x selectors x seeds). A block may override the top-level defaults.
"""
from __future__ import annotations

import itertools

import yaml

from .generators import GENERATORS
from .runner import SELECTOR_NAMES

# Required generator params per family (excluding seed).
_REQUIRED = {
    "erdos_renyi": ["n", "p"],
    "random_regular": ["n", "d"],
    "grid": ["rows", "cols"],
    "path": ["n"],
    "cycle": ["n"],
    "complete": ["n"],
    "cocktail_party": ["r", "s"],
    "random_tree": ["n"],
    "random_planar": ["n"],
    "random_planar_sparse": ["n"],
}
_RANDOMIZED = {"erdos_renyi", "random_regular", "random_tree", "random_planar", "random_planar_sparse"}


class ConfigError(ValueError):
    """Raised for malformed sweep configs (author error - fail fast)."""


def load(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return parse(raw)


def parse(raw):
    if not isinstance(raw, dict) or not raw.get("runs"):
        raise ConfigError("config must be a mapping with a non-empty 'runs' list")
    defaults = {
        "selectors": raw.get("selectors", ["none"]),
        "use_cache": raw.get("use_cache", True),
        "repeat": raw.get("repeat", 1),
    }
    specs = []
    for block in raw["runs"]:
        specs.extend(_expand_block(block, defaults))
    return raw.get("out", "bench.jsonl"), specs


def _as_list(v):
    return v if isinstance(v, list) else [v]


def _expand_block(block, defaults):
    family = block.get("family")
    if family not in GENERATORS:
        raise ConfigError(f"unknown family {family!r}; known: {sorted(GENERATORS)}")

    selectors = _as_list(block.get("selectors", defaults["selectors"]))
    for sel in selectors:
        if sel not in SELECTOR_NAMES:
            raise ConfigError(f"unknown selector {sel!r}; known: {sorted(SELECTOR_NAMES)}")
    use_cache = block.get("use_cache", defaults["use_cache"])
    repeat = block.get("repeat", defaults["repeat"])
    # Fail fast on a nonsensical repeat count here, rather than letting repeat<1
    # slip through and surface downstream as an opaque per-run `error` row.
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ConfigError(f"repeat must be an integer >= 1, got {repeat!r}")

    param_names = _REQUIRED[family]
    grids = []
    for name in param_names:
        if name not in block:
            raise ConfigError(f"family {family!r} requires param {name!r}")
        grids.append(_as_list(block[name]))

    if family in _RANDOMIZED:
        seeds = _as_list(block.get("seeds", [0]))
    else:
        if block.get("seeds") is not None:
            raise ConfigError(f"family {family!r} is deterministic; remove 'seeds'")
        seeds = [None]

    specs = []
    for combo in itertools.product(*grids):
        base = dict(zip(param_names, combo))
        for seed in seeds:
            gen_kwargs = dict(base)
            if family in _RANDOMIZED:
                gen_kwargs["seed"] = seed
            for sel in selectors:
                specs.append({
                    "family": family,
                    "gen_kwargs": gen_kwargs,
                    "selector": sel,
                    "use_cache": use_cache,
                    "repeat": repeat,
                })
    return specs
