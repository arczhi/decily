"""Extract the text tower from Qwen3.5-2B-Base (VL) into a text-only model dir.

Produces: model.safetensors (keys "model.*"), config.json (text-only
Qwen3_5ForCausalLM, mirroring the public decider-2b text config for the same
base), tokenizer files. This lets the standard AutoModel path be used as the
decision-model encoder.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/models/Qwen3.5-2B-Base")
    ap.add_argument("--template-config", default="/root/models/decider-2b/config.json")
    ap.add_argument("--out", default="/root/models/Qwen3.5-2B-Base-Text")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    state = load_file(os.path.join(args.src, "model.safetensors"))
    text = {}
    skipped = 0
    for k, v in state.items():
        if k.startswith("model.language_model."):
            text["model." + k[len("model.language_model."):]] = v.contiguous()
        else:
            skipped += 1
    print(f"[extract] text tensors: {len(text)}, skipped: {skipped}")

    out_file = os.path.join(args.out, "model.safetensors")
    save_file(text, out_file, metadata={"format": "pt", "source": "Qwen3.5-2B-Base"})
    print(f"[extract] wrote {out_file}")

    with open(args.template_config) as f:
        cfg = json.load(f)
    cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("[extract] wrote config.json (text-only template)")

    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
        src = os.path.join(args.src, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, name))
    print("[extract] copied tokenizer files")
    print("EXTRACT_DONE")


if __name__ == "__main__":
    main()
