"""Stage 1: train Route B-Mini (or Route A baseline) from a YAML config.

Usage:
    python -m decision_model.train.stage1 --config configs/mini_b0.yaml
"""

from __future__ import annotations

import argparse

import yaml
from transformers import AutoTokenizer

from ..data.collate import RouteACollator, RouteBCollator
from ..data.mixture import MixtureItem, MixtureSampler
from ..data.schema import read_jsonl
from ..models.decision_model import RouteBConfig, RouteBDecisionModel
from ..models.route_a import RouteAConfig, RouteADecisionModel
from .common import TrainConfig, Trainer


def build(cfg: dict):
    model_cfg = dict(cfg["model"])
    route = model_cfg.pop("route", "b")
    if route == "b":
        model = RouteBDecisionModel(RouteBConfig(**model_cfg))
        collator_cls = RouteBCollator
    elif route == "cx":
        from ..models.cross_encoder import CrossEncoderConfig, CrossEncoderDecisionModel

        model = CrossEncoderDecisionModel(CrossEncoderConfig(**model_cfg))
        collator_cls = RouteBCollator
    else:
        model = RouteADecisionModel(RouteAConfig(**model_cfg))
        collator_cls = RouteACollator
    tok = AutoTokenizer.from_pretrained(model_cfg["backbone"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    data_cfg = cfg["data"]
    collator = collator_cls(tok, **data_cfg.get("collator", {}))
    items = [MixtureItem(**it) for it in data_cfg["manifest"]]
    sampler = MixtureSampler(
        items,
        seed=cfg.get("seed", 0),
        min_options=data_cfg.get("min_options", 2),
        max_options=data_cfg.get("max_options", 10),
    )
    eval_examples = list(read_jsonl(data_cfg["eval_path"]))
    probe_examples = None
    if data_cfg.get("probe_path"):
        probe_examples = list(read_jsonl(data_cfg["probe_path"]))[:16]
    return model, collator, sampler, eval_examples, probe_examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model, collator, sampler, eval_examples, probe_examples = build(cfg)
    trainer = Trainer(
        model,
        collator,
        sampler,
        eval_examples,
        eval_collator=collator,
        cfg=TrainConfig(**cfg.get("train", {})),
        probe_examples=probe_examples,
    )
    trainer.run()


if __name__ == "__main__":
    main()
