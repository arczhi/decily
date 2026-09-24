"""Merge an SFT LoRA checkpoint into a full model checkpoint (for full FT warm start).

Usage:
    python scripts/merge_sft_adapter.py --config configs/cx_qwen17b_bigk.yaml \
        --ckpt runs/cx_qwen17b_bigk/ckpt_last.pt --out runs/merged_sft/model.pt
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

from decision_model.models.cross_encoder import (  # noqa: E402
    CrossEncoderConfig,
    CrossEncoderDecisionModel,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "route"}
    model_cfg["dtype"] = "float32"  # merge in fp32 for accuracy
    model = CrossEncoderDecisionModel(CrossEncoderConfig(**model_cfg))

    state = torch.load(args.ckpt, map_location="cpu")
    state = state["state"] if isinstance(state, dict) and "state" in state else state
    incompatible = model.load_state_dict(state, strict=False)
    print("missing:", len(incompatible.missing_keys), "unexpected:", len(incompatible.unexpected_keys))

    if getattr(model.encoder.model, "merge_and_unload", None) is not None:
        model.encoder.model = model.encoder.model.merge_and_unload()
        print("merged LoRA into base")
    else:
        print("WARNING: model has no peft wrapper; nothing to merge")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}
    torch.save({"state": save}, args.out)
    print("wrote", args.out, f"({sum(v.numel() for v in save.values())/1e9:.2f}B params, bf16)")


if __name__ == "__main__":
    main()
