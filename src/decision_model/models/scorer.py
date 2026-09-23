"""Interaction modules: candidate tokens score against state+question latents."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CrossBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: Tensor, kv: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        q = self.norm_q(x)
        k = v = self.norm_kv(kv)
        attn_out = self.attn(
            q, k, v, key_padding_mask=key_padding_mask, need_weights=False
        )[0]
        x = x + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        scores = self.proj(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        w = torch.softmax(scores, dim=-1)
        if mask is not None:
            w = torch.nan_to_num(w, nan=0.0)
        out = torch.einsum("bl,bld->bd", w, x)
        return out


class CrossAttentionScorer(nn.Module):
    """L2 interaction: candidate tokens (query) attend to [state; question] (kv)."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.0,
        dot_feature: bool = True,
        l2_normalize: bool = False,
        center_pooled: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.pool = AttentionPooling(d_model)
        self.cond_pool = AttentionPooling(d_model)
        self.dot_feature = dot_feature
        self.l2_normalize = l2_normalize
        self.center_pooled = center_pooled
        head_in = d_model + (1 if dot_feature else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, 1),
        )

    def forward(
        self,
        cand: Tensor,
        cand_mask: Tensor,
        cond: Tensor,
        cond_mask: Tensor,
        center_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """cand [B, Q, K, Lc, d]; cond [B, Q, L, d] -> logits [B, Q, K], pooled [B, Q, K, d].

        center_mask [B, Q] bool: subtract the per-question candidate mean for
        those rows (choice/score). Relative scores remove the huge shared
        component that otherwise dominates gradients and collapses training.
        """
        b, q, k, lc, d = cand.shape
        cond_len = cond.shape[2]

        x = cand.reshape(b * q * k, lc, d)
        c_mask = cand_mask.reshape(b * q * k, lc) if cand_mask is not None else None
        kv = (
            cond.unsqueeze(2)
            .expand(b, q, k, cond_len, d)
            .reshape(b * q * k, cond_len, d)
        )
        kv_mask = (
            cond_mask.unsqueeze(2)
            .expand(b, q, k, cond_len)
            .reshape(b * q * k, cond_len)
            if cond_mask is not None
            else None
        )

        key_padding = (kv_mask == 0) if kv_mask is not None else None
        for layer in self.layers:
            x = layer(x, kv, key_padding)

        pooled = self.pool(x, c_mask)
        cond_pooled = self.cond_pool(kv, kv_mask)
        pooled = pooled.view(b, q, k, d)
        cond_pooled = cond_pooled.view(b, q, k, d)

        if self.l2_normalize:
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        if self.center_pooled:
            valid = c_mask.view(b, q, k, -1).any(dim=-1)
            if center_mask is not None:
                valid = valid & center_mask.unsqueeze(-1)
            n_valid = valid.sum(dim=2, keepdim=True).clamp(min=1)
            shared = (pooled * valid.unsqueeze(-1)).sum(dim=2, keepdim=True) / n_valid.unsqueeze(-1)
            pooled = pooled - shared

        feats = [pooled]
        if self.dot_feature:
            cos = torch.nn.functional.cosine_similarity(pooled, cond_pooled, dim=-1)
            feats.append(cos.unsqueeze(-1))
        logits = self.head(torch.cat(feats, dim=-1)).squeeze(-1)
        return logits, pooled
