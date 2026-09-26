#!/usr/bin/env python3
"""Export a cross-encoder decision checkpoint to ONNX (+ dynamic int8).

The exported graph contains the scoring path only:

    input_ids      [B, T]  int64
    attention_mask [B, T]  int64
    -> logits      [B]     float32

where each row is one candidate's `[state | question | candidate]` token
sequence (the caller builds the rows and softmaxes the logits).

No base-model download is needed: the backbone is instantiated from the
config and every parameter is then loaded from the bundle (strict), so the
pretrained base weights are never fetched.

Usage:
    python scripts/export_decision_onnx.py --bundle models_hf/Decily-1.7B \
        --out models_hf/Decily-ONNX-int8 --parity data/fair_suite/fair_suite.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402


def patch_auto_model_config_only() -> None:
    """Make AutoModel.from_pretrained build from config without downloading."""

    def from_pretrained(cls, name, *args, **kwargs):
        config = kwargs.pop("config", None) or AutoConfig.from_pretrained(name)
        attn = kwargs.get("attn_implementation")
        if attn:
            config._attn_implementation = attn
        model = AutoModel.from_config(config)
        dtype = kwargs.get("dtype")
        return model.to(dtype) if dtype is not None else model

    AutoModel.from_pretrained = classmethod(from_pretrained)


class OnnxLogits(nn.Module):
    """[B, T] ids + mask -> [B] logits (state+question+candidate rows)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.model.encoder(input_ids, attention_mask)
        pooled = self.model.pool(h, attention_mask)
        return self.model.head(pooled).squeeze(-1)


def build_model(bundle: str):
    from decision_model.models.cross_encoder import CrossEncoderConfig, CrossEncoderDecisionModel

    cfg = CrossEncoderConfig(
        backbone=bundle,
        pool="attention",
        head_hidden_mult=4,
        dropout=0.0,
        dtype="float32",
        amp_dtype="float32",
        lora=False,
        freeze_backbone=False,
        attn_implementation="eager",
    )
    model = CrossEncoderDecisionModel(cfg)
    from safetensors.torch import load_file

    state = load_file(os.path.join(bundle, "model.safetensors"))
    incompatible = model.load_state_dict(state, strict=False)
    missing = [k for k in incompatible.missing_keys]
    unexpected = [k for k in incompatible.unexpected_keys]
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval()
    print(f"[export] model built from config + bundle ({len(state)} tensors loaded)", flush=True)
    return model


def build_parity_rows(args, tok):
    """Real examples -> (ids, mask, meta) with one row per candidate."""
    from decision_model.data.collate import RouteBCollator
    from decision_model.data.schema import read_jsonl

    coll = RouteBCollator(
        tok,
        max_state_tokens=args.max_state_tokens,
        max_question_tokens=args.max_question_tokens,
        max_option_tokens=args.max_option_tokens,
    )
    examples = []
    for row in read_jsonl(args.parity):
        if len(row.questions[0].options) > args.max_options:
            continue
        examples.append(row)
        if len(examples) >= args.n_parity:
            break
    batch = coll(examples)
    return examples, batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="dir with model.safetensors + config.json + tokenizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--parity", default=None, help="JSONL of decision examples for parity checks")
    ap.add_argument("--n-parity", type=int, default=8)
    ap.add_argument("--max-state-tokens", type=int, default=256)
    ap.add_argument("--max-question-tokens", type=int, default=96)
    ap.add_argument("--max-option-tokens", type=int, default=64)
    ap.add_argument("--max-options", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    patch_auto_model_config_only()
    model = build_model(args.bundle)
    wrapper = OnnxLogits(model).eval()
    tok = AutoTokenizer.from_pretrained(args.bundle)

    fp32_path = os.path.join(args.out, "model.fp32.onnx")
    dummy = (
        torch.randint(0, 1000, (2, 64), dtype=torch.long),
        torch.ones((2, 64), dtype=torch.long),
    )
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        dummy,
        fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "tokens"},
            "attention_mask": {0: "batch", 1: "tokens"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[export] fp32 ONNX written in {time.time()-t0:.0f}s", flush=True)

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_fp32 = ort.InferenceSession(fp32_path, sess_options=so, providers=["CPUExecutionProvider"])

    # parity on real examples
    if args.parity:
        examples, batch = build_parity_rows(args, tok)
        with torch.no_grad():
            torch_logits = model(batch)["logits"].float().numpy()  # [b, q, k]
        ids, mask, _ = model._joint_inputs(batch)
        ids, mask = ids.numpy(), mask.numpy()
        b, q, k = torch_logits.shape
        ort_logits = sess_fp32.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
        ort_logits = ort_logits.reshape(b, q, k)
        diff = np.abs(torch_logits - ort_logits).max()
        print(f"[parity] fp32 max|diff| = {diff:.2e} over {b}x{q}x{k} rows", flush=True)
        if diff > 1e-3:
            raise SystemExit("fp32 parity check failed")

    if not args.no_int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = os.path.join(args.out, "model.int8.onnx")
        t0 = time.time()
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
        print(f"[export] int8 written in {time.time()-t0:.0f}s "
              f"({os.path.getsize(int8_path)/1e6:.1f} MB)", flush=True)
        sess_int8 = ort.InferenceSession(int8_path, sess_options=so, providers=["CPUExecutionProvider"])
        if args.parity:
            i8 = sess_int8.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0].reshape(b, q, k)
            diff = np.abs(torch_logits - i8).max()
            # top-1 per (b, q) over candidates k
            agree = (torch_logits.argmax(-1) == i8.argmax(-1)).mean()
            print(f"[parity] int8 max|diff| = {diff:.2e} | top-1 agreement = {agree:.3f}", flush=True)

        # latency: one decision with 4 candidates at the training limits
        T = args.max_state_tokens + args.max_question_tokens + args.max_option_tokens
        for B in (4, 16):
            x = np.random.randint(1, 1000, (B, T), dtype=np.int64)
            m = np.ones((B, T), dtype=np.int64)
            sess_int8.run(["logits"], {"input_ids": x, "attention_mask": m})
            t0 = time.perf_counter()
            sess_int8.run(["logits"], {"input_ids": x, "attention_mask": m})
            dt = time.perf_counter() - t0
            print(f"[bench] int8 CPU B={B:2d} T={T}: {dt*1000:7.0f} ms "
                  f"({dt/B*1000:.0f} ms/row)", flush=True)

    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
        src = os.path.join(args.bundle, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, name))
    print("EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
