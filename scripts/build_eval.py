"""Build a mixed evaluation JSONL from converted shards.

Usage:
    python scripts/build_eval.py --spec sst2=200,mnli=200,arc_easy=200 \
        --data-dir data/raw --out data/eval/val_mixed.jsonl
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decision_model.data.schema import read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="task=count[,task=count...]")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--split", default="val", help="shard suffix to read")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = []
    for part in args.spec.split(","):
        task, count = part.split("=")
        path = os.path.join(args.data_dir, f"{task}_{args.split}.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"missing shard: {path}")
        rows = list(read_jsonl(path))
        rng.shuffle(rows)
        out.extend(rows[: int(count)])
        print(f"[eval] {task}: {min(int(count), len(rows))}/{len(rows)}")

    rng.shuffle(out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = write_jsonl(args.out, iter(out))
    print(f"[eval] wrote {n} -> {args.out}")


if __name__ == "__main__":
    main()
