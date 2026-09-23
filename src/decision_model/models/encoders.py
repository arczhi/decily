"""Encoder wrapper with optional role embeddings (state / question / candidate)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel


DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class HFEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        add_role_embeddings: bool = True,
        n_roles: int = 3,
        freeze: bool = True,
        attn_implementation: str | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        kwargs = {"dtype": DTYPES.get(dtype, torch.float32)}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        config = AutoConfig.from_pretrained(model_name)
        config.use_cache = False
        self.model = AutoModel.from_pretrained(model_name, config=config, **kwargs)
        self.model = self.model.to(kwargs["dtype"])
        d_model = config.hidden_size
        self.role_emb = nn.Embedding(n_roles, d_model) if add_role_embeddings else None
        if self.role_emb is not None:
            nn.init.normal_(self.role_emb.weight, std=0.02)
            self.role_emb.to(kwargs["dtype"])
        self.freeze = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, role_id: int | Tensor | None = None
    ) -> Tensor:
        emb = self.model.get_input_embeddings()(input_ids)
        if self.role_emb is not None and role_id is not None:
            role = torch.as_tensor(role_id, device=input_ids.device)
            if role.dim() == 0:
                role = role.expand(input_ids.shape[0])
            emb = emb + self.role_emb(role).unsqueeze(1)
        out = self.model(inputs_embeds=emb, attention_mask=attention_mask)
        return out.last_hidden_state

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze and mode:
            has_trainable = any(p.requires_grad for p in self.model.parameters())
            if not has_trainable:
                self.model.eval()
        return self
