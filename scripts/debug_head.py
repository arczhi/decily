"""Train the exact Route B head on frozen pooled features (encoder out of the loop)."""

from __future__ import annotations

import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, "/root/decision-model/src")

import yaml
from transformers import AutoTokenizer

from decision_model.data.collate import RouteBCollator
from decision_model.data.schema import read_jsonl
from decision_model.models.decision_model import RouteBConfig, RouteBDecisionModel


def to_device(batch, device="cuda"):
    for f in batch.__dataclass_fields__:
        v = getattr(batch, f)
        if isinstance(v, torch.Tensor):
            setattr(batch, f, v.to(device))
    return batch


CFG = yaml.safe_load(open("/root/decision-model/configs/diag_b1.yaml"))
torch.manual_seed(0)
model_cfg = {k: v for k, v in CFG["model"].items() if k != "route"}
model = RouteBDecisionModel(RouteBConfig(**model_cfg)).cuda().eval()
tok = AutoTokenizer.from_pretrained(CFG["model"]["backbone"])
coll = RouteBCollator(tok, **CFG["data"]["collator"])
examples = list(read_jsonl("/root/decision-model/data/raw/sst2_train.jsonl"))


def collect(exs, shard=64):
    pooled_all, gold_all = [], []
    with torch.no_grad():
        for i in range(0, len(exs), shard):
            batch = to_device(coll(exs[i : i + shard]))
            out = model(batch)
            pooled_all.append(out["pooled"][:, 0].cpu())  # [b, 2, d]
            gold_all.append(batch.answer_index[:, 0].cpu())
    return torch.cat(pooled_all), torch.cat(gold_all)


P_tr, y_tr = collect(examples[:2048])
P_te, y_te = collect(examples[2048:3072])
P_tr, y_tr = P_tr.cuda(), y_tr.cuda()
P_te, y_te = P_te.cuda(), y_te.cuda()
d = P_tr.shape[-1]
print("pooled", P_tr.shape, "cuda", torch.cuda.memory_allocated() // 2**20, "MB")


def cos(a, b, dim=-1):
    return F.cosine_similarity(a, b, dim=dim)


def head_features(P, dot=True):
    # mimic scorer: pooled candidate + cosine(pooled_cand, pooled_cond=mean of the two)
    cond = P.mean(dim=1, keepdim=True)
    feats = [P]
    if dot:
        feats.append(cos(P, cond).unsqueeze(-1))
    return torch.cat(feats, dim=-1)  # [N, 2, d(+1)]


@torch.no_grad()
def eval_head(fn, head, X, y):
    logits = head(X)
    acc = (logits.argmax(-1) == y).float().mean()
    return acc.item()


def train_head(name, head, X_tr, X_te, lr=1e-3, steps=800, wd=0.01):
    head = head.cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    ls = 0.05
    for step in range(1, steps + 1):
        logits = head(X_tr)
        smooth = torch.full_like(logits, ls / (logits.shape[-1] - 1))
        smooth.scatter_(1, y_tr.unsqueeze(1), 1 - ls)
        loss = -(smooth * F.log_softmax(logits, -1)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                acc_tr = eval_head(None, head, X_tr, y_tr)
                acc_te = eval_head(None, head, X_te, y_te)
            print(f"[{name}] step {step} loss={loss.item():.4f} acc_tr={acc_tr:.3f} acc_te={acc_te:.3f}", flush=True)
    return head


class ExactHead(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 4 * d_hidden),
            nn.GELU(),
            nn.Linear(4 * d_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


X_tr = head_features(P_tr)
X_te = head_features(P_te)

# T1: exact head shape, lr 1e-3
train_head("exact_lr1e-3", ExactHead(X_tr.shape[-1], d), X_tr, X_te, lr=1e-3)
# T2: exact head shape, lr 1e-4
train_head("exact_lr1e-4", ExactHead(X_tr.shape[-1], d), X_tr, X_te, lr=1e-4)


# T3: dot-product score on the difference of the two candidates
class DotHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, x):
        a, b = x[:, 0, :d], x[:, 1, :d]
        s_a = (self.w.unsqueeze(0) @ a.T).squeeze(0)
        s_b = (self.w.unsqueeze(0) @ b.T).squeeze(0)
        return torch.stack([s_a, s_b], dim=-1)


train_head("dot", DotHead(d), X_tr, X_te, lr=1e-3)

# T4: linear classifier on diff feature, exact same loss
class DiffHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, x):
        a, b = x[:, 0, :d], x[:, 1, :d]
        s = self.lin(a - b).squeeze(-1)
        return torch.stack([s, -s], dim=-1)


train_head("diff_linear", DiffHead(d), X_tr, X_te, lr=1e-3)
print("DONE")
