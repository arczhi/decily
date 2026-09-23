"""Route B: explicit candidate scorer decision model."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.collate import RouteBBatch
from .encoders import HFEncoder
from .scorer import CrossAttentionScorer

STATE_ROLE, QUESTION_ROLE, CANDIDATE_ROLE = 0, 1, 2


@dataclass
class RouteBConfig:
    backbone: str = "answerdotai/ModernBERT-base"
    freeze_backbone: bool = True
    share_encoder: bool = True
    add_role_embeddings: bool = True
    scorer_layers: int = 3
    scorer_heads: int = 8
    scorer_dropout: float = 0.0
    dot_feature: bool = True
    l2_normalize: bool = False
    center_pooled: bool = False
    attn_implementation: str = "sdpa"
    label_smoothing: float = 0.05
    temperature: float = 1.0
    n_task_types: int = 3
    extra: dict = field(default_factory=dict)


class RouteBDecisionModel(nn.Module):
    def __init__(self, cfg: RouteBConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = HFEncoder(
            cfg.backbone,
            add_role_embeddings=cfg.add_role_embeddings,
            freeze=cfg.freeze_backbone,
            attn_implementation=cfg.attn_implementation,
        )
        self.scorer = CrossAttentionScorer(
            self.encoder.d_model,
            n_layers=cfg.scorer_layers,
            n_heads=cfg.scorer_heads,
            dropout=cfg.scorer_dropout,
            dot_feature=cfg.dot_feature,
            l2_normalize=cfg.l2_normalize,
            center_pooled=cfg.center_pooled,
        )
        self.task_type_emb = nn.Embedding(cfg.n_task_types, self.encoder.d_model)
        self.task_proj = nn.Linear(self.encoder.d_model, self.encoder.d_model)

    def encode(self, batch: RouteBBatch) -> tuple[Tensor, Tensor, Tensor]:
        b, q, k = batch.option_input_ids.shape[:3]
        lc = batch.option_input_ids.shape[-1]
        d = self.encoder.d_model

        h_state = self.encoder(batch.state_input_ids, batch.state_mask, STATE_ROLE)
        h_q = self.encoder(
            batch.question_input_ids.reshape(b * q, -1),
            batch.question_mask.reshape(b * q, -1),
            QUESTION_ROLE,
        ).view(b, q, -1, d)
        h_c = self.encoder(
            batch.option_input_ids.reshape(b * q * k, lc),
            batch.option_mask.reshape(b * q * k, lc),
            CANDIDATE_ROLE,
        ).view(b, q, k, lc, d)
        return h_state, h_q, h_c

    def forward(self, batch: RouteBBatch) -> dict[str, Tensor]:
        b, q, k = batch.option_input_ids.shape[:3]
        h_state, h_q, h_c = self.encode(batch)
        d = self.encoder.d_model

        t_emb = self.task_proj(self.task_type_emb(batch.task_type)).view(b, 1, 1, d)
        h_q = h_q + t_emb
        cond = torch.cat([h_state.unsqueeze(1).expand(b, q, -1, d), h_q], dim=2)
        cond_mask = torch.cat(
            [batch.state_mask.unsqueeze(1).expand(b, q, -1), batch.question_mask], dim=2
        )
        center_mask = (batch.task_type != 2).unsqueeze(1).expand(b, q)
        logits, pooled = self.scorer(
            h_c, batch.option_mask, cond, cond_mask, center_mask
        )
        return {"logits": logits, "pooled": pooled, "batch": batch}


def masked_choice_logits(logits: Tensor, valid: Tensor) -> Tensor:
    return logits.masked_fill(~valid, torch.finfo(logits.dtype).min)


def decision_loss(
    logits: Tensor,
    batch: RouteBBatch,
    label_smoothing: float = 0.0,
    normalize_by_k: bool = False,
) -> dict[str, Tensor]:
    """CE for choice/score rows, BCE for noul/score independent heads.

    score rows participate in both: CE over levels and BCE per level are
    averaged, matching the "every level judged alone" recipe.
    """
    logits = logits.float()
    if logits.dim() == 3:
        if logits.size(1) != 1:
            raise ValueError(f"decision_loss expects Q=1, got Q={logits.size(1)}")
        logits = logits[:, 0]
    valid = batch.option_valid[:, 0]
    losses: dict[str, Tensor] = {}

    choice_rows = batch.task_type == 0
    sigmoid_rows = (batch.task_type == 1) | (batch.task_type == 2)

    if choice_rows.any():
        lg = logits[choice_rows].masked_fill(~valid[choice_rows], -1e9)
        n_cls = lg.size(-1)
        raw = batch.answer_index[choice_rows, 0]
        tgt = raw.clamp(min=0)
        smooth = torch.full_like(lg, label_smoothing / max(n_cls - 1, 1))
        smooth.scatter_(1, tgt.unsqueeze(1), 1.0 - label_smoothing)
        smooth = smooth.masked_fill(~valid[choice_rows], 0.0)
        smooth = torch.where(raw.unsqueeze(1) >= 0, smooth, torch.zeros_like(smooth))
        smooth = smooth / smooth.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        per_row = -(smooth * F.log_softmax(lg, dim=-1)).sum(dim=-1)
        if normalize_by_k:
            n_opts = valid[choice_rows].sum(dim=-1).clamp(min=2).float()
            per_row = per_row / n_opts.log()
        losses["choice_ce"] = per_row.mean()

    if sigmoid_rows.any():
        lg = logits[sigmoid_rows][valid[sigmoid_rows]]
        y = batch.answer_mask[sigmoid_rows, 0][valid[sigmoid_rows]]
        losses["sigmoid_bce"] = F.binary_cross_entropy_with_logits(lg, y)

    if not losses:
        raise ValueError("batch has no trainable rows")
    total = sum(losses.values())
    losses["total"] = total
    return losses
