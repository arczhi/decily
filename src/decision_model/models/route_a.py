"""Route A baseline: LM-head letter logits at the answer slot (decider-style)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM

from ..data.collate import RouteABatch


@dataclass
class RouteAConfig:
    backbone: str = "Qwen/Qwen3-0.6B"
    freeze_backbone: bool = True
    lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    temperature: float = 1.0
    label_smoothing: float = 0.05
    attn_implementation: str = "sdpa"


class RouteADecisionModel(nn.Module):
    def __init__(self, cfg: RouteAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.backbone, attn_implementation=cfg.attn_implementation
        )
        self.model.config.use_cache = False
        if cfg.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False
        if cfg.lora:
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)

    def forward(self, batch: RouteABatch) -> dict[str, Tensor]:
        out = self.model(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        )
        b = batch.input_ids.shape[0]
        q = batch.option_token_ids.shape[1]
        slot_logits = out.logits[torch.arange(b, device=out.logits.device), batch.slot_positions]
        slot_logits = slot_logits.unsqueeze(1).expand(b, q, -1)
        logits = torch.gather(slot_logits, 2, batch.option_token_ids)
        return {"logits": logits}

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_backbone:
            self.model.eval()
        return self
