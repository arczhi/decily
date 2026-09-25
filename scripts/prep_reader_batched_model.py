"""Prepare a standard-format tower dir (mlx-lm compatible) from the v5 MLX bundle.

Writes (under the reader project, the original model is untouched):
    models/v5_batched/config.json            standard Qwen3 config
    models/v5_batched/model.safetensors      backbone, keys "model.*"
    models/v5_batched/decision_head.safetensors  pool.*/head.*
    models/v5_batched/tokenizer*             copied

The dir can be quantized with `mlx_lm.convert -q` afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import mlx.core as mx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/coding/decision-model/models_mlx/v5_mlx"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "models", "v5_batched"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    w = dict(mx.load(os.path.join(args.src, "model.safetensors")))
    tower, head = {}, {}
    for k, v in w.items():
        if k.startswith("encoder.model."):
            tower["model." + k[len("encoder.model."):]] = v
        elif k.startswith("encoder."):
            tower["model." + k[len("encoder."):]] = v
        else:
            head[k] = v
    print(f"[prep] tower {len(tower)} tensors, head {len(head)} tensors")

    mx.save_safetensors(os.path.join(args.out, "model.safetensors"), tower)
    mx.save_safetensors(os.path.join(args.out, "decision_head.safetensors"), head)

    cfg = json.load(open(os.path.join(args.src, "config.json")))
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
        src = os.path.join(args.src, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, name))
    print("[prep] wrote", args.out)


if __name__ == "__main__":
    main()
