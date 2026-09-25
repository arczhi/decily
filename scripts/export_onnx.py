"""Export a decision model to ONNX (cross-platform CPU / CUDA / DirectML).

The exported graph takes a batched joint sequence [state; question; candidate]
and returns one scalar logit per row:

    input_ids      [B, L] int64
    attention_mask [B, L] int64
    -> logits      [B]     float32

Candidate probabilities are a softmax over the rows of one question done by the
caller. Optionally quantizes to dynamic int8.

Usage:
    python scripts/export_onnx.py --config configs/stage1_qwen35.yaml \
        --ckpt runs/stage1_qwen35/ckpt_last.pt --out models_onnx/stage1 \
        [--quantize] [--samples 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import yaml  # noqa: E402
from torch import nn  # noqa: E402

from decision_model.data.collate import RouteACollator, RouteBCollator  # noqa: E402
from decision_model.models.cross_encoder import (  # noqa: E402
    CrossEncoderConfig,
    CrossEncoderDecisionModel,
)


class OnnxDecisionModule(nn.Module):
    """enc(input_ids, mask) -> pooled -> head -> scalar logit (per row)."""

    def __init__(self, model: CrossEncoderDecisionModel) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.pool = model.pool
        self.head = model.head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids, attention_mask)
        pooled = self.pool(h, attention_mask)
        return self.head(pooled).squeeze(-1)


def build_model(cfg: dict, ckpt: str):
    mc = {k: v for k, v in cfg["model"].items() if k not in ("route", "init_ckpt")}
    model = CrossEncoderDecisionModel(CrossEncoderConfig(**mc))
    state = torch.load(ckpt, map_location="cpu")
    state = state["state"] if isinstance(state, dict) and "state" in state else state
    model.load_state_dict(state, strict=False)
    return model.float().eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--dynamo", action="store_true", help="use the dynamo exporter (default: legacy)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(args.out, exist_ok=True)

    model = build_model(cfg, args.ckpt)
    wrapper = OnnxDecisionModule(model).eval()
    d = model.encoder.d_model

    onnx_path = os.path.join(args.out, "model.onnx")
    dummy_ids = torch.ones(2, 16, dtype=torch.long)
    dummy_mask = torch.ones(2, 16, dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=args.dynamo,
    )
    print(f"[onnx] exported {onnx_path} ({os.path.getsize(onnx_path)/1e6:.0f} MB)")

    # numeric validation torch vs onnxruntime
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    import random

    rng = random.Random(0)
    max_diff = 0.0
    with torch.no_grad():
        for _ in range(args.samples):
            b = rng.randint(1, 3)
            length = rng.randint(12, 48)
            ids = torch.randint(0, 1000, (b, length), dtype=torch.long)
            mask = torch.ones(b, length, dtype=torch.long)
            mask[-1, -3:] = 0  # exercise padding
            ref = wrapper(ids, mask).numpy()
            got = sess.run(["logits"], {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
            max_diff = max(max_diff, float(np.abs(ref - got).max()))
    print(f"[onnx] numeric check: max |torch - onnx| = {max_diff:.5f}")

    meta = {
        "source_config": args.config,
        "source_ckpt": os.path.basename(args.ckpt),
        "d_model": d,
        "max_diff_torch_onnx": max_diff,
        "quantized": bool(args.quantize),
    }

    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q_path = os.path.join(args.out, "model.int8.onnx")
        quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
        print(f"[onnx] quantized {q_path} ({os.path.getsize(q_path)/1e6:.0f} MB)")
        meta["int8_path"] = os.path.basename(q_path)

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("EXPORT_ONNX_DONE")


if __name__ == "__main__":
    main()
