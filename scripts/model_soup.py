"""Greedy Model Soup over decision-model checkpoints (Wortsman et al., ICML 2022).

Protocol:
  - candidates are materialized as full fp32 weights (LoRA adapters merged);
  - the best single model on the SELECTION split seeds the soup;
  - remaining candidates are added one by one if they improve selection accuracy;
  - the final soup is evaluated on a disjoint TEST split (+ per-task), with
    temperature fitting, and saved in bf16.

Usage:
    python scripts/model_soup.py --select data/eval/stage1_heldout_big.jsonl \
        --test data/eval/stage1_heldout.jsonl \
        --labels SFT v1 v2 fullFT \
        --configs configs/cx_qwen17b_bigk.yaml configs/cx_qwen17b_bigk.yaml \
                  configs/cx_qwen17b_bigk.yaml configs/fullft_from_sft.yaml \
        --ckpts runs/cx_qwen17b_bigk/ckpt_last.pt runs/rlcd_17b/ckpt_last.pt \
                runs/rlcd_v2_17b/ckpt_last.pt runs/fullft_sft17b/ckpt_step1500.pt \
        --out runs/soup/model.pt --report runs/soup/report.json
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from decision_model.data.collate import RouteBCollator  # noqa: E402
from decision_model.data.schema import read_jsonl  # noqa: E402
from decision_model.eval.harness import (  # noqa: E402
    collect_logits,
    fit_temperature_rows,
    metrics_by_task,
    metrics_from_rows,
)
from decision_model.models.cross_encoder import (  # noqa: E402
    CrossEncoderConfig,
    CrossEncoderDecisionModel,
)


def _model_cfg(cfg: dict, dtype: str) -> dict:
    mc = {k: v for k, v in cfg["model"].items() if k not in ("route", "init_ckpt")}
    mc["dtype"] = dtype
    return mc


def materialize(cfg_path: str, ckpt_path: str) -> dict:
    """Full fp32 state dict with LoRA merged (if present)."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = CrossEncoderDecisionModel(CrossEncoderConfig(**_model_cfg(cfg, "float32")))
    state = torch.load(ckpt_path, map_location="cpu")
    state = state["state"] if isinstance(state, dict) and "state" in state else state
    model.load_state_dict(state, strict=False)
    if getattr(model.encoder.model, "merge_and_unload", None) is not None:
        model.encoder.model = model.encoder.model.merge_and_unload()
    out = {k: v.detach().float() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    return out


@torch.no_grad()
def evaluate_state(
    state: dict,
    cfg_model: dict,
    collator,
    examples: list,
    device: str = "cuda",
    batch_size: int = 8,
) -> tuple[dict, float]:
    # states are always materialized (LoRA merged) full weights -> plain model
    cfg_model = dict(cfg_model)
    cfg_model["lora"] = False
    model = CrossEncoderDecisionModel(CrossEncoderConfig(**cfg_model))
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    rows = collect_logits(model, examples, collator, batch_size=batch_size, device=device)
    t_fit, _ = fit_temperature_rows(rows)
    metrics = metrics_from_rows(rows, t_fit)
    del model
    torch.cuda.empty_cache()
    return metrics, t_fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--select-n", type=int, default=3000)
    ap.add_argument("--eval-batch", type=int, default=8)
    args = ap.parse_args()

    with open(args.configs[0]) as f:
        ref_cfg = yaml.safe_load(f)
    tok = AutoTokenizer.from_pretrained(ref_cfg["model"]["backbone"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    coll = RouteBCollator(tok, **ref_cfg["data"]["collator"])

    select_examples = list(read_jsonl(args.select))[: args.select_n]
    test_examples = list(read_jsonl(args.test))
    print(f"[soup] select={len(select_examples)} test={len(test_examples)}", flush=True)

    report: dict = {"singles": {}, "greedy": []}
    states: dict[str, dict] = {}
    for label, cfgp, ckpt in zip(args.labels, args.configs, args.ckpts):
        print(f"[soup] materialize {label} ...", flush=True)
        states[label] = materialize(cfgp, ckpt)
        model_cfg = _model_cfg(yaml.safe_load(open(cfgp)), "bfloat16")
        metrics, t = evaluate_state(
            states[label], model_cfg, coll, select_examples,
            batch_size=args.eval_batch,
        )
        report["singles"][label] = {"select": metrics, "fitted_T": t}
        print(
            f"[soup] single {label}: select acc={metrics['acc_all']:.4f} "
            f"ece={metrics['choice/ece']:.4f} T={t:.2f}",
            flush=True,
        )

    # seed = best single on selection accuracy
    best_label = max(report["singles"], key=lambda k: report["singles"][k]["select"]["acc_all"])
    print(f"[soup] seed = {best_label}", flush=True)
    soup_sum = {k: v.clone() for k, v in states[best_label].items()}
    n = 1

    order = sorted(
        [l for l in args.labels if l != best_label],
        key=lambda k: -report["singles"][k]["select"]["acc_all"],
    )
    for label in order:
        cand = {k: (soup_sum[k] + states[label][k]) / (n + 1) for k in soup_sum}
        model_cfg = _model_cfg(yaml.safe_load(open(args.configs[args.labels.index(label)])), "bfloat16")
        metrics, t = evaluate_state(cand, model_cfg, coll, select_examples, batch_size=args.eval_batch)
        entry = {
            "added": label,
            "n_models": n + 1,
            "select": metrics,
            "fitted_T": t,
        }
        baseline = report["greedy"][-1]["select"]["acc_all"] if report["greedy"] else report["singles"][best_label]["select"]["acc_all"]
        improve = metrics["acc_all"] > baseline
        entry["accepted"] = bool(improve)
        report["greedy"].append(entry)
        print(
            f"[soup] try +{label}: acc={metrics['acc_all']:.4f} (base {baseline:.4f}) "
            f"{'ACCEPT' if improve else 'reject'}",
            flush=True,
        )
        if improve:
            soup_sum = cand
            n += 1

    soup = {k: (v / n) for k, v in soup_sum.items()}

    # final test evaluation (bf16 model on the disjoint test split)
    final_metrics, t_final = evaluate_state(
        soup, _model_cfg(ref_cfg, "bfloat16"), coll, test_examples, batch_size=args.eval_batch
    )
    rows = None
    model = CrossEncoderDecisionModel(CrossEncoderConfig(**_model_cfg(ref_cfg, "bfloat16")))
    model.load_state_dict(soup, strict=False)
    model.to("cuda").eval()
    rows = collect_logits(model, test_examples, coll, batch_size=args.eval_batch, device="cuda")
    report["final"] = {
        "n_models": n,
        "models": [best_label] + [e["added"] for e in report["greedy"] if e["accepted"]],
        "test": final_metrics,
        "fitted_T": t_final,
        "test_per_task": metrics_by_task(rows, temperature=1.0),
    }
    print(f"[soup] FINAL test acc={final_metrics['acc_all']:.4f} ece={final_metrics['choice/ece']:.4f}", flush=True)
    print(json.dumps(report["final"]["test_per_task"], indent=1)[:1200], flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save = {k: v.to(torch.bfloat16).cpu() for k, v in soup.items()}
    torch.save({"state": save}, args.out)
    print(f"[soup] wrote {args.out}", flush=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[soup] wrote {args.report}", flush=True)
    print("SOUP_DONE", flush=True)


if __name__ == "__main__":
    main()
