"""Compare two eval_report.json files (Route A vs Route B).

Usage:
    python scripts/compare_reports.py runs/cx_b0/eval_report.json runs/baseline_a0/eval_report.json \
        --labels "Route B (cx)" "Route A (LM-head)"
"""

from __future__ import annotations

import argparse
import json
import sys


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def row(report: dict) -> dict:
    out = {}
    for key, res in report.items():
        name = key.split("/")[-1].replace("stage1_", "").replace(".jsonl", "")
        for tag in ("T=1.0", "fitted"):
            m = res.get(tag)
            if not m:
                continue
            label = f"{name}/{tag}"
            out[label] = {
                "acc": m.get("acc_all", 0.0),
                "nll": m.get("choice/nll", 0.0),
                "ece": m.get("choice/ece", 0.0),
                "acc@80": m.get("choice/acc@80", 0.0),
                "T": res.get("fitted_T", 1.0) if tag == "fitted" else 1.0,
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--labels", nargs="+", default=None)
    args = ap.parse_args()
    labels = args.labels or [f"report{i}" for i in range(len(args.reports))]
    tables = [row(load(p)) for p in args.reports]

    keys = sorted(set().union(*[t.keys() for t in tables]))
    header = f"{'metric':28s}" + "".join(f"{l:>22s}" for l in labels)
    print(header)
    print("-" * len(header))
    for k in keys:
        line = f"{k:28s}"
        for t in tables:
            v = t.get(k)
            if v is None:
                line += f"{'-':>22s}"
            else:
                line += f"{v['acc']:>7.3f}/{v['nll']:>5.2f}/{v['ece']:>5.3f}"
                line += "  "
        print(line)
    print("\ncolumns: acc / nll / ece")


if __name__ == "__main__":
    main()
