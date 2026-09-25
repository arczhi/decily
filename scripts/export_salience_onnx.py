#!/usr/bin/env python3
"""Export a trained whole-document salience student to ONNX (+ int8).

The exported graph takes the packed-segment batch directly:
    input_ids      [B, T]     int64
    attention_mask [B, T]     int64
    span_mask      [B, S, T]  float32   (1.0 on a sentence's tokens)
and returns
    logits         [B, S]     float32   (per-sentence salience logit)

Pooling (span mean) and the MLP head are inside the graph, so the reader only
needs onnxruntime + a tokenizer.

Usage:
    python scripts/export_salience_onnx.py --ckpt runs/sal_doc_minilm/ckpt_best.pt \
        --out models_onnx/salience_minilm
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from train_salience_doc import SalienceDocModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone = ck["backbone"]
    model = SalienceDocModel(backbone)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.ckpt} backbone={backbone}", flush=True)

    tok = AutoTokenizer.from_pretrained(backbone)
    tok.save_pretrained(args.out)

    dummy = (
        torch.randint(0, 1000, (2, 128), dtype=torch.long),
        torch.ones((2, 128), dtype=torch.long),
        torch.zeros((2, 8, 128), dtype=torch.float32),
    )
    dummy[2][:, :, :16] = 1.0
    dummy[2][0, 4:, :] = 0.0
    torch.onnx.export(
        model,
        dummy,
        os.path.join(args.out, "model.fp32.onnx"),
        input_names=["input_ids", "attention_mask", "span_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "tokens"},
            "attention_mask": {0: "batch", 1: "tokens"},
            "span_mask": {0: "batch", 1: "sentences", 2: "tokens"},
            "logits": {0: "batch", 1: "sentences"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print("exported model.fp32.onnx", flush=True)

    import onnx

    m = onnx.load(os.path.join(args.out, "model.fp32.onnx"))
    print("graph inputs:", [(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
    print("graph outputs:", [(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])

    # numerical check (fp32)
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(os.path.join(args.out, "model.fp32.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    feed = {
        "input_ids": dummy[0].numpy(),
        "attention_mask": dummy[1].numpy(),
        "span_mask": dummy[2].numpy(),
    }
    with torch.no_grad():
        ref = model(*dummy).numpy()
    got = sess.run(["logits"], feed)[0]
    print(f"fp32 max|diff| = {np.abs(ref - got).max():.2e}", flush=True)

    if not args.no_int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = os.path.join(args.out, "model.int8.onnx")
        quantize_dynamic(os.path.join(args.out, "model.fp32.onnx"), int8_path, weight_type=QuantType.QInt8)
        sess8 = ort.InferenceSession(int8_path, sess_options=so, providers=["CPUExecutionProvider"])
        got8 = sess8.run(["logits"], feed)[0]
        print(f"int8 max|diff| vs torch = {np.abs(ref - got8).max():.2e}", flush=True)
        print("int8 size:", os.path.getsize(int8_path) / 1e6, "MB", flush=True)
    print("EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
