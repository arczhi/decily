"""Export the Stage-1 Qwen3.5 decision model to an MLX-ready bundle.

Steps:
  1. build our model per config, load the LoRA checkpoint, merge the adapter;
  2. map merged backbone weights back to the ORIGINAL VL key names
     ("model.language_model.*") so mlx-lm's sanitize() handles them;
  3. append the decision head (pool.*/head.*);
  4. write model.safetensors + config.json (VL config) + tokenizer files.

Verification: the mapped backbone key set must equal the base model's text keys.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import yaml  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from decision_model.models.cross_encoder import (  # noqa: E402
    CrossEncoderConfig,
    CrossEncoderDecisionModel,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base-dir", default="/root/models/Qwen3.5-2B-Base")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mc = {k: v for k, v in cfg["model"].items() if k not in ("route", "init_ckpt")}
    model = CrossEncoderDecisionModel(CrossEncoderConfig(**mc))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["state"] if isinstance(ckpt, dict) and "state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    if getattr(model.encoder.model, "merge_and_unload", None) is not None:
        model.encoder.model = model.encoder.model.merge_and_unload()
        print("[export] LoRA merged")
    merged = {k: v.detach().to(torch.bfloat16) for k, v in model.state_dict().items()}
    print("[export] merged tensors:", len(merged))

    base = load_file(os.path.join(args.base_dir, "model.safetensors"))
    base_text_keys = {k for k in base if k.startswith("model.language_model.")}
    print("[export] base text keys:", len(base_text_keys))

    out: dict = {}
    head: dict = {}
    for k, v in merged.items():
        if k.startswith("encoder."):
            stem = k[len("encoder."):]
            if stem.startswith("model."):
                stem = stem[len("model."):]
            orig = "model.language_model." + stem
            if orig not in base_text_keys:
                raise SystemExit(f"unmapped key: {k} -> {orig}")
            out[orig] = v.contiguous()
        else:
            head[k] = v.contiguous()

    missing = base_text_keys - set(out)
    if missing:
        raise SystemExit(f"missing {len(missing)} base keys, e.g. {sorted(missing)[:3]}")
    print("[export] backbone mapped OK, head tensors:", len(head))

    os.makedirs(args.out_dir, exist_ok=True)
    save_file({**out, **head}, os.path.join(args.out_dir, "model.safetensors"),
              metadata={"format": "pt", "source": os.path.basename(args.ckpt)})
    shutil.copy(os.path.join(args.base_dir, "config.json"),
                os.path.join(args.out_dir, "config.json"))
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
        src = os.path.join(args.base_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out_dir, name))
    print("[export] wrote", args.out_dir)
    print("EXPORT_MLX_DONE")


if __name__ == "__main__":
    main()
