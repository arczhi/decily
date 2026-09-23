"""Are the frozen-backbone candidate features linearly separable?

Fresh init model -> pooled candidate features -> logistic regression probe.
"""

from __future__ import annotations

import sys

import torch

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
train_ex = examples[:1024]
test_ex = examples[1024:2048]


def features(exs):
    feats, labels = [], []
    with torch.no_grad():
        for i in range(0, len(exs), 64):
            batch = to_device(coll(exs[i : i + 64]))
            out = model(batch)
            pooled = out["pooled"][:, 0]  # [B, K, d]
            b, k, d = pooled.shape
            for j in range(b):
                diff = pooled[j, 0] - pooled[j, 1]  # candidate 0 minus 1
                cos = torch.nn.functional.cosine_similarity(
                    pooled[j, 0], pooled[j, 1], dim=-1
                )
                feats.append(torch.cat([diff, cos.view(1)]))
                labels.append(1.0 if batch.answer_index[j, 0].item() == 0 else 0.0)
    X = torch.stack(feats)
    y = torch.tensor(labels, device=X.device)
    return X, y


Xtr, ytr = features(train_ex)
Xte, yte = features(test_ex)
print(f"features: dim={Xtr.shape[1]}, train={Xtr.shape[0]}, test={Xte.shape[0]}")
print(f"label balance train: {ytr.mean().item():.3f}")

# cosine similarity between the two candidate pooled vectors
cos_tr = Xtr[:, -1]
print(f"cos(cand0, cand1): mean={cos_tr.mean():.4f} std={cos_tr.std():.4f}")
print(f"||diff||: mean={Xtr[:,:-1].norm(dim=-1).mean():.4f}")

# logistic regression probe on frozen features
W = torch.zeros(Xtr.shape[1], device="cuda", requires_grad=True)
b = torch.zeros(1, device="cuda", requires_grad=True)
opt = torch.optim.Adam([W, b], lr=1e-2)
for step in range(500):
    logits = Xtr @ W + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ytr)
    opt.zero_grad()
    loss.backward()
    opt.step()
with torch.no_grad():
    acc_tr = ((Xtr @ W + b > 0) == ytr.bool()).float().mean()
    acc_te = ((Xte @ W + b > 0) == yte.bool()).float().mean()
print(f"probe loss={loss.item():.4f} acc_train={acc_tr.item():.4f} acc_test={acc_te.item():.4f}")

# two-layer probe (nonlinear head on frozen features)
h = 256
W1 = (torch.randn(Xtr.shape[1], h, device="cuda") * 0.02).requires_grad_()
b1 = torch.zeros(h, device="cuda", requires_grad=True)
W2 = (torch.randn(h, 1, device="cuda") * 0.02).requires_grad_()
b2 = torch.zeros(1, device="cuda", requires_grad=True)
opt = torch.optim.Adam([W1, b1, W2, b2], lr=1e-3)
for step in range(1000):
    hd = torch.relu(Xtr @ W1 + b1)
    logits = (hd @ W2 + b2).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ytr)
    opt.zero_grad()
    loss.backward()
    opt.step()
with torch.no_grad():
    hd_te = torch.relu(Xte @ W1 + b1)
    acc_te2 = (((hd_te @ W2 + b2).squeeze(-1) > 0) == yte.bool()).float().mean()
print(f"mlp probe loss={loss.item():.4f} acc_test={acc_te2.item():.4f}")
