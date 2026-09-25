"""Persist benchmark rows as JSONL and print a grouped summary table."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict


def append(row, path):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def summarize(rows):
    """Group rows by (family, selector) -> mean/median branches and mean wall_s."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["family"], r["selector"])].append(r)

    summary = []
    for (family, selector), rs in sorted(groups.items()):
        ok = [r for r in rs if "error" not in r]
        branches = [r["branches"] for r in ok]
        walls = [r["wall_s"] for r in ok]
        summary.append({
            "family": family,
            "selector": selector,
            "runs": len(rs),
            "errors": len(rs) - len(ok),
            "mean_branches": statistics.mean(branches) if branches else None,
            "median_branches": statistics.median(branches) if branches else None,
            "mean_wall_s": statistics.mean(walls) if walls else None,
        })
    return summary


def _fmt(x, places=1):
    return "-" if x is None else f"{x:.{places}f}"


def format_table(summary):
    header = (f"{'family':<16}{'selector':<16}{'runs':>5}{'errs':>5}"
              f"{'mean_br':>12}{'med_br':>12}{'mean_s':>10}")
    lines = [header, "-" * len(header)]
    for s in summary:
        lines.append(
            f"{s['family']:<16}{s['selector']:<16}{s['runs']:>5}{s['errors']:>5}"
            f"{_fmt(s['mean_branches']):>12}{_fmt(s['median_branches']):>12}"
            f"{_fmt(s['mean_wall_s'], 4):>10}"
        )
    return "\n".join(lines)
