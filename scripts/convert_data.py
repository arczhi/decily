"""Convert a registered dataset into unified decision-tuple JSONL.

Usage:
    python scripts/convert_data.py --dataset sst2 --split train --limit 5000 \
        --out data/raw/sst2_train.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decision_model.data import converters  # noqa: E402,F401
from decision_model.data.converters.hf_tasks import REGISTRY  # noqa: E402
from decision_model.data.schema import validate, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    it = REGISTRY[args.dataset](split=args.split, limit=args.limit)
    examples = []
    n_err = 0
    for ex in it:
        errors = validate(ex)
        if errors:
            n_err += 1
            if n_err <= 5:
                print(f"[warn] {ex.task}: {errors}", file=sys.stderr)
            continue
        examples.append(ex)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = write_jsonl(args.out, iter(examples))
    print(f"[convert] {args.dataset}/{args.split}: wrote {n} examples -> {args.out} (skipped {n_err})")


if __name__ == "__main__":
    main()
