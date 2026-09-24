"""Export a decision-model checkpoint to a portable safetensors bundle (for MLX).

Writes: model.safetensors (backbone + decision head, same key names as the
training checkpoint), config.json (backbone arch + decision-head spec), and the
tokenizer files. No torch needed on the consumer side.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["state"] if isinstance(ckpt, dict) and "state" in ckpt else ckpt
    state = {k: v.contiguous() for k, v in state.items()}

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, "model.safetensors")
    save_file(state, out_file, metadata={"format": "pt", "source": os.path.basename(args.ckpt)})
    print(f"[export] wrote {out_file} ({len(state)} tensors)")

    with open(os.path.join(args.backbone_dir, "config.json")) as f:
        arch = json.load(f)
    arch["decision_head"] = {
        "pool": "attention_full_sequence",
        "head_hidden_mult": 4,
        "roles": ["state", "question", "candidate"],
        "concat_order": ["state", "question", "candidate"],
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(arch, f, indent=2)
    print("[export] wrote config.json")

    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
        src = os.path.join(args.backbone_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out_dir, name))
    print("[export] copied tokenizer files")
    print("EXPORT_DONE")


if __name__ == "__main__":
    main()
