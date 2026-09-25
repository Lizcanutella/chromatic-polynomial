"""Geodata pipeline (Unit C of the map-aware visualizer).

Turns Natural Earth administrative polygons into the project-standard
(n, edges, positions) graph and feeds them through chromatic_rl.dataset's
verify/write path. Depends on the optional ``[geo]`` extra (geopandas +
shapely); import-guarded so the core package and /compute keep working
without it. Viewing a committed ``maps`` dataset needs no geo deps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

try:
    import geopandas  # noqa: F401
    import shapely  # noqa: F401
    HAVE_GEO = True
except Exception:  # noqa: BLE001 - any import failure means the extra is absent
    HAVE_GEO = False


class MapsGeoUnavailable(RuntimeError):
    """Raised when the [geo] extra is not installed."""


@dataclass
class GeoGraph:
    n: int
    edges: list = field(default_factory=list)      # list[tuple[int, int]], u < v, sorted
    positions: list = field(default_factory=list)  # list[[x, y]] normalized 0..1, y flipped (north up)
    labels: list = field(default_factory=list)     # list[str], index == vertex id
    granularity: str | None = None                 # admin level ("states"/"provinces"/...) or None


def require_geo() -> None:
    if not HAVE_GEO:
        raise MapsGeoUnavailable(
            "the 'geo' extra is required for the geodata pipeline - "
            "install it with:  uv pip install geopandas shapely")


__all__ = ["GeoGraph", "MapsGeoUnavailable", "HAVE_GEO", "require_geo"]
