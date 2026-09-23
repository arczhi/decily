"""Cross-encoder scorer (monoBERT / RankT5 style).

The candidate is scored jointly with state and question through the encoder:

    [state tokens] [question tokens] [candidate tokens] -> pool -> scalar

This is the mature reranking architecture (Nogueira et al. 2019 monoBERT;
Zhuang et al. 2022 RankT5): the query/candidate interaction happens inside
the encoder, and the score head is trained with a listwise softmax loss.
Compared with the late-interaction scorer, question conditioning is direct,
which fixes the multi-task interference observed with frozen features.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..data.collate import RouteBBatch
from .encoders import DTYPES, HFEncoder
from .scorer import AttentionPooling


@dataclass
class CrossEncoderConfig:
    backbone: str = "answerdotai/ModernBERT-base"
    freeze_backbone: bool = True
    dtype: str = "float32"
    lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    pool: str = "attention"  # attention | cls | mean
    dropout: float = 0.0
    head_hidden_mult: int = 4
    label_smoothing: float = 0.05
    n_task_types: int = 3
    attn_implementation: str = "sdpa"
    grad_checkpointing: bool = False
    rejector: bool = False


class CrossEncoderDecisionModel(nn.Module):
    def __init__(self, cfg: CrossEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = HFEncoder(
            cfg.backbone,
            add_role_embeddings=False,
            freeze=cfg.freeze_backbone,
            attn_implementation=cfg.attn_implementation,
            dtype=cfg.dtype,
        )
        if cfg.lora:
            from peft import LoraConfig, get_peft_model

            self.encoder.model = get_peft_model(
                self.encoder.model,
                LoraConfig(
                    r=cfg.lora_r,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules="all-linear",
                ),
            )
        if cfg.grad_checkpointing:
            self.encoder.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.encoder.model, "enable_input_require_grads"):
                self.encoder.model.enable_input_require_grads()
        d = self.encoder.d_model
        self.pool = AttentionPooling(d)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.head_hidden_mult * d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_mult * d, 1),
        )
        self.reject_pool = AttentionPooling(d) if cfg.rejector else None
        self.reject_head = (
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, cfg.head_hidden_mult * d),
                nn.GELU(),
                nn.Linear(cfg.head_hidden_mult * d, 1),
            )
            if cfg.rejector
            else None
        )
        # Trainable modules stay in fp32 (LoRA adapters + head); the frozen
        # encoder holds the low-precision weights. Forward runs under autocast
        # so matmuls use bf16 while optimizer states stay fp32 (AMP/QLoRA style).
        self._compute_dtype = DTYPES.get(cfg.dtype, torch.float32)

    def _joint_inputs(
        self, batch: RouteBBatch
    ) -> tuple[Tensor, Tensor, tuple[int, int, int, int]]:
        state = batch.state_input_ids
        question = batch.question_input_ids
        option = batch.option_input_ids
        b, q, k = option.shape[:3]
        ls, lq, lc = state.size(1), question.size(2), option.size(3)

        state_e = state[:, None, None, :].expand(b, q, k, ls).reshape(b * q * k, ls)
        state_m = batch.state_mask[:, None, None, :].expand(b, q, k, ls).reshape(b * q * k, ls)
        question_e = (
            question.reshape(b, q, 1, lq).expand(b, q, k, lq).reshape(b * q * k, lq)
        )
        question_m = (
            batch.question_mask.reshape(b, q, 1, lq).expand(b, q, k, lq).reshape(b * q * k, lq)
        )
        option_e = option.reshape(b * q * k, lc)
        option_m = batch.option_mask.reshape(b * q * k, lc)

        ids = torch.cat([state_e, question_e, option_e], dim=1)
        mask = torch.cat([state_m, question_m, option_m], dim=1)
        return ids, mask, (b, q, k, ls + lq + lc)

    def forward(self, batch: RouteBBatch) -> dict[str, Tensor]:
        use_amp = self._compute_dtype != torch.float32 and torch.cuda.is_available()
        with torch.autocast(
            device_type="cuda", dtype=self._compute_dtype, enabled=use_amp
        ):
            ids, mask, (b, q, k, _) = self._joint_inputs(batch)
            h = self.encoder(ids, mask)
            if self.cfg.pool == "attention":
                pooled = self.pool(h, mask)
            elif self.cfg.pool == "cls":
                pooled = h[:, 0]
            else:
                pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(
                    1, keepdim=True
                ).clamp(min=1)
            logits = self.head(pooled).view(b, q, k)
            reject_logit = None
            if self.reject_head is not None:
                cand_mask = batch.option_mask[:, :, :, 0]
                set_repr = self.reject_pool(
                    pooled.view(b * q, k, -1), cand_mask.reshape(b * q, k)
                ).view(b, q, -1)
                reject_logit = self.reject_head(set_repr).squeeze(-1)
        out = {"logits": logits.float(), "pooled": pooled.view(b, q, k, -1), "batch": batch}
        if reject_logit is not None:
            out["reject_logit"] = reject_logit.float()
        return out
