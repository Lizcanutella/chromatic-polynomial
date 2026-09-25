"""Dataset record + manifest schema and (de)serialization.

Pure data, no engine import. Chromatic-polynomial coefficients are GMP
integers; they are stored as decimal strings in JSON (B-ready format) and
parsed back to Python ints on load.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


def coeffs_to_str(coeffs: list[int]) -> list[str]:
    return [str(c) for c in coeffs]


def coeffs_from_str(coeffs: list[str]) -> list[int]:
    return [int(c) for c in coeffs]


@dataclass(frozen=True)
class Instance:
    id: str
    family: str
    n: int
    params: dict
    seed: int | None
    tier: str
    edges: list                  # list[list[int]]
    reference_poly: list | None  # list[str] decimal coeffs; None for uncertified maps
    build_stats: dict            # {branches, nodes, hits, misses, hit_rate, wall_s}; {} when uncertified
    verified: dict               # {engine: bool, oracle: bool, oracle_k_checked: list[int]}
    positions: list = field(default_factory=list)  # list[[x, y]] normalized; [] for non-map instances
    labels: list = field(default_factory=list)     # list[str] region names; [] when absent
    excluded_reason: str | None = None  # why an uncertified instance ships without a poly

    def to_json(self) -> dict:
        d = asdict(self)
        if not self.positions:
            d.pop("positions")          # keep byte-identity with position-free datasets
        if not self.labels:
            d.pop("labels")             # keep byte-identity with label-free datasets
        if self.excluded_reason is None:
            d.pop("excluded_reason")    # keep byte-identity with certified datasets
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Instance":
        return cls(**d)


@dataclass
class Manifest:
    version: str
    budget: dict
    oracle_n_max: int
    engine_version: str
    engine_commit: str
    generator: str
    n_instances: int
    counts: dict          # family -> count
    tier_counts: dict     # tier -> count
    extra: dict = field(default_factory=dict)  # geo convention (eps, data source, exclusions)

    def to_json(self) -> dict:
        d = asdict(self)
        if not self.extra:
            d.pop("extra")
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Manifest":
        return cls(**d)
