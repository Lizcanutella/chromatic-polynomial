"""chromatic_rl.dataset: load and inspect the benchmark dataset format
(instances.jsonl + manifest.json), and resolve the visualizer's selector names."""
from .schema import Instance, Manifest, coeffs_to_str, coeffs_from_str
from .loader import load, LoadedInstance
from .score import resolve_selector

__all__ = [
    "Instance", "Manifest", "coeffs_to_str", "coeffs_from_str",
    "load", "LoadedInstance",
    "resolve_selector",
]
