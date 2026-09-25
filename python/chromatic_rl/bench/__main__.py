"""CLI: python -m chromatic_rl.bench --config sweep.yaml [--out bench.jsonl]"""
from __future__ import annotations

import argparse

from . import config, report
from .runner import run_sweep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="chromatic_rl.bench")
    ap.add_argument("--config", required=True, help="path to a YAML sweep config")
    ap.add_argument("--out", default=None, help="JSONL output path (overrides config 'out')")
    args = ap.parse_args(argv)

    out_path, specs = config.load(args.config)
    if args.out:
        out_path = args.out

    rows = []
    for row in run_sweep(specs):
        report.append(row, out_path)
        rows.append(row)

    print(report.format_table(report.summarize(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
