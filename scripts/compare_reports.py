"""Compare eval_report.json files (Route A vs Route B, 4 tasks vs 24 tasks).

Usage:
    python scripts/compare_reports.py runs/a.json runs/b.json \
        --labels "A" "B" [--per-task]
"""

from __future__ import annotations

import argparse
import json


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def row(report: dict) -> dict:
    out = {}
    for key, res in report.items():
        name = key.split("/")[-1].replace("stage1_", "").replace("stage2_", "").replace(".jsonl", "")
        for tag in ("T=1.0", "fitted"):
            m = res.get(tag)
            if not m:
                continue
            out[f"{name}/{tag}"] = {
                "acc": m.get("acc_all", 0.0),
                "nll": m.get("choice/nll", 0.0),
                "ece": m.get("choice/ece", 0.0),
                "acc@80": m.get("choice/acc@80", 0.0),
            }
    return out


def macro_table(reports: list[dict], labels: list[str]) -> None:
    tables = [row(r) for r in reports]
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
                line += f"{v['acc']:>7.3f}/{v['nll']:>5.2f}/{v['ece']:>5.3f}  "
        print(line)
    print("\ncolumns: acc / nll / ece")


def per_task_table(reports: list[dict], labels: list[str]) -> None:
    tasks = sorted(
        {t for r in reports for res in r.values() for t in res.get("per_task", {})}
    )
    header = f"{'task':34s}" + "".join(f"{l:>24s}" for l in labels)
    print(header)
    print("-" * len(header))
    for t in tasks:
        line = f"{t:34s}"
        for r in reports:
            cell = "-"
            for res in r.values():
                m = res.get("per_task", {}).get(t)
                if m is not None:
                    cell = (
                        f"{m.get('acc_all',0):.3f}/"
                        f"{m.get('choice/nll',0):.2f}/"
                        f"{m.get('choice/ece',0):.3f}"
                    )
                    break
            line += f"{cell:>24s}"
        print(line)
    print("\ncolumns: acc / nll / ece")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--per-task", action="store_true")
    args = ap.parse_args()
    labels = args.labels or [f"report{i}" for i in range(len(args.reports))]
    reports = [load(p) for p in args.reports]
    macro_table(reports, labels)
    if args.per_task:
        print()
        per_task_table(reports, labels)


if __name__ == "__main__":
    main()
