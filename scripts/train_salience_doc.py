#!/usr/bin/env python3
"""Whole-document salience student: segment encoder + per-sentence span pooling.

Design (local CPU benchmarks, 2026-09-25):
- One 8192-token pass is 75s on CPU: attention is quadratic and ModernBERT's
  long-context regime is memory-bound. 8x512 batched runs at ~1.8k tok/s,
  ~40x faster per token than the 8192 single pass.
- So the document is packed into <= max-seg-tokens segments (greedy sentence
  packing), segments are batched, and each sentence's score comes from mean
  pooling its token hidden states through a small MLP head.

Training signal (mixed per step):
- KD: teacher probability distributions over candidate sentences
  (data/salience/salience_train.jsonl, from stage1 with the fixed question
  "Which sentence is most important for understanding this text?").
  Loss = KL(teacher || softmax(student logits over the same candidates), T).
- BCE: CNN/DailyMail sentence labels (ROUGE-1 F >= 0.5 against highlights).

Usage:
    python scripts/train_salience_doc.py \
        --backbone sentence-transformers/all-MiniLM-L6-v2 \
        --teacher data/salience/salience_train.jsonl \
        --bce data/doc_salience/train.jsonl \
        --val-teacher data/salience/salience_eval.jsonl \
        --val-bce data/doc_salience/val.jsonl \
        --out runs/salience_doc_minilm --steps 20000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from salience_pack import pack_segments, split_builder, tokenize_sentences  # noqa: E402


def stream_jsonl(path: str):
    """Yield parsed rows; skip empty/corrupt lines (remote file has one truncation)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


class JsonlIndex:
    """Random access to a JSONL file by line byte offsets (for the 3.7GB file)."""

    def __init__(self, path: str, limit: int | None = None):
        self.path = path
        self.offsets: list[int] = []
        with open(path, "rb") as f:
            if limit is None:
                for line in f:
                    if line.strip():
                        self.offsets.append(f.tell() - len(line))
            else:
                rng = random.Random(1234)
                n_seen = 0
                for line in f:
                    if line.strip():
                        n_seen += 1
                        if len(self.offsets) < limit:
                            self.offsets.append(f.tell() - len(line))
                        else:
                            j = rng.randrange(n_seen)
                            if j < limit:
                                self.offsets[j] = f.tell() - len(line)
        self._fh = None

    def __len__(self) -> int:
        return len(self.offsets)

    def get(self, i: int) -> dict | None:
        if self._fh is None:
            self._fh = open(self.path, "rb")
        self._fh.seek(self.offsets[i])
        line = self._fh.readline()
        try:
            return json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None


class SalienceDocModel(nn.Module):
    """Encoder over packed segments + per-sentence span pooling + MLP head."""

    def __init__(self, backbone_name: str):
        super().__init__()
        self.enc = AutoModel.from_pretrained(backbone_name)
        d = self.enc.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(self, input_ids, attention_mask, span_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        counts = span_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        pooled = torch.einsum("bst,btd->bsd", span_mask, h) / counts
        return self.head(pooled).squeeze(-1)  # [B, S]


def build_batch(tok, sents_tokens, max_seg, device):
    """One forward batch = all segments of one article (variable sentence count)."""
    segments, spans = pack_segments(tok, sents_tokens, max_seg)
    if not segments:
        return None
    T = max(len(ids) for ids in segments)
    S = max(len(sp) for sp in spans)
    B = len(segments)
    input_ids = torch.zeros((B, T), dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, T), dtype=torch.long, device=device)
    span_mask = torch.zeros((B, S, T), dtype=torch.float32, device=device)
    loc: dict[int, tuple[int, int]] = {}
    for b, (ids, sp) in enumerate(zip(segments, spans)):
        input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[b, : len(ids)] = 1
        for si, (gi, s, e) in enumerate(sp):
            span_mask[b, si, s:e] = 1.0
            loc[gi] = (b, si)
    return input_ids, attention_mask, span_mask, loc


def gather_logits(logits, loc, sent_indices):
    out = []
    b_idx, s_idx = [], []
    for gi in sent_indices:
        if gi in loc:
            b, s = loc[gi]
            b_idx.append(b)
            s_idx.append(s)
    if not b_idx:
        return None
    return logits[torch.tensor(b_idx), torch.tensor(s_idx)]


def tokenize_teacher_row(tok, row):
    """Return (sents_tokens, cand_indices, probs) or None."""
    q = row["questions"][0]
    sents = split_builder(row["state"])
    if len(sents) < 2:
        return None
    text_to_idx: dict[str, list[int]] = {}
    for i, s in enumerate(sents):
        text_to_idx.setdefault(s, []).append(i)
    cand: list[int] = []
    probs: list[float] = []
    used: dict[str, int] = {}
    for opt, p in zip(q["options"], q.get("probs") or []):
        t = opt["text"].strip()
        ids = text_to_idx.get(t)
        if not ids:
            continue
        k = used.get(t, 0)
        if k >= len(ids):
            continue
        used[t] = k + 1
        cand.append(ids[k])
        probs.append(float(p))
    if len(cand) < 2:
        return None
    tot = sum(probs)
    if tot <= 0:
        return None
    probs = [p / tot for p in probs]
    return tokenize_sentences(tok, sents), cand, probs


def auc_score(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC (Mann-Whitney U)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(ranks[i] for i, y in enumerate(labels) if y == 1)
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def spearman(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 3:
        return float("nan")

    def ranks(x):
        order = sorted(range(n), key=lambda i: x[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else float("nan")


@torch.no_grad()
def evaluate(model, tok, teacher_rows, bce_val_path, args, device):
    model.eval()
    top1 = n = 0
    sps, kls = [], []
    for row in teacher_rows:
        parsed = tokenize_teacher_row(tok, row)
        if parsed is None:
            continue
        sents_tokens, cand, probs = parsed
        batch = build_batch(tok, sents_tokens, args.max_seg_tokens, device)
        if batch is None:
            continue
        input_ids, attention_mask, span_mask, loc = batch
        logits = model(input_ids, attention_mask, span_mask)
        cand_logits = gather_logits(logits, loc, cand)
        if cand_logits is None or len(cand_logits) != len(probs):
            continue
        t = torch.tensor(probs, device=device)
        p = torch.softmax(cand_logits, dim=0)
        kl = float((t * (torch.log(t + 1e-9) - torch.log(p + 1e-9))).sum())
        kls.append(kl)
        if int(torch.argmax(p)) == int(torch.argmax(t)):
            top1 += 1
        sps.append(spearman([float(x) for x in p], probs))
        n += 1
    kd_metrics = {
        "top1": top1 / n if n else float("nan"),
        "spearman": sum(s for s in sps if not math.isnan(s)) / max(sum(0 if math.isnan(s) else 1 for s in sps), 1),
        "kl": sum(kls) / len(kls) if kls else float("nan"),
        "n": n,
    }

    bce_scores, bce_labels = [], []
    rows = 0
    for row in stream_jsonl(bce_val_path):
        sents = row.get("sents") or []
        labels = row.get("labels") or []
        if len(sents) < 2:
            continue
        st = tokenize_sentences(tok, sents)
        batch = build_batch(tok, st, args.max_seg_tokens, device)
        if batch is None:
            continue
        input_ids, attention_mask, span_mask, loc = batch
        logits = model(input_ids, attention_mask, span_mask)
        for gi, (b, s) in loc.items():
            if gi < len(labels):
                bce_scores.append(float(logits[b, s]))
                bce_labels.append(int(labels[gi]))
        rows += 1
        if rows >= args.val_bce_limit:
            break
    acc = (
        sum(1 for s, y in zip(bce_scores, bce_labels) if (1 if s > 0 else 0) == y)
        / max(len(bce_labels), 1)
    )
    bce_metrics = {"acc": acc, "auc": auc_score(bce_scores, bce_labels), "n": len(bce_labels)}
    model.train()
    return kd_metrics, bce_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--teacher", default="data/salience/salience_train.jsonl")
    ap.add_argument("--bce", default="data/doc_salience/train.jsonl")
    ap.add_argument("--val-teacher", default="data/salience/salience_eval.jsonl")
    ap.add_argument("--val-bce", default="data/doc_salience/val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--max-seg-tokens", type=int, default=512)
    ap.add_argument("--kd-ratio", type=float, default=0.5)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-bce-limit", type=int, default=300)
    ap.add_argument("--bce-index-limit", type=int, default=200000)
    ap.add_argument("--limit-teacher", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = SalienceDocModel(args.backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"backbone={args.backbone} params={n_params / 1e6:.1f}M device={device}", flush=True)

    teacher_all = list(stream_jsonl(args.teacher))
    if args.limit_teacher:
        teacher_all = teacher_all[: args.limit_teacher]
    teacher_ok = [r for r in teacher_all if tokenize_teacher_row(tok, r) is not None] or list(teacher_all)
    print(f"teacher rows: {len(teacher_all)} (usable {len(teacher_ok)})", flush=True)

    bce_index = JsonlIndex(args.bce, limit=args.bce_index_limit)
    print(f"bce index: {len(bce_index)} rows from {args.bce}", flush=True)

    val_teacher = list(stream_jsonl(args.val_teacher)) if args.val_teacher else []
    print(f"val teacher rows: {len(val_teacher)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(args.warmup, 1))
        * (0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))),
    )

    os.makedirs(args.out, exist_ok=True)
    pos_weight = torch.tensor(args.pos_weight, device=device)
    use_amp = device == "cuda"
    best = -1.0
    t0 = time.time()
    tok_seen = 0
    loss_ema = {"kd": 0.0, "bce": 0.0}
    opt.zero_grad(set_to_none=True)

    for step in range(1, args.steps + 1):
        do_kd = random.random() < args.kd_ratio
        batch = None
        kd_target = None
        if do_kd:
            row = teacher_ok[random.randrange(len(teacher_ok))]
            parsed = tokenize_teacher_row(tok, row)
            if parsed is not None:
                sents_tokens, cand, probs = parsed
                batch = build_batch(tok, sents_tokens, args.max_seg_tokens, device)
                kd_target = (cand, torch.tensor(probs, device=device))
        if batch is None:
            idx = random.randrange(len(bce_index)) if len(bce_index) else 0
            row = bce_index.get(idx)
            if row is None:
                continue
            sents = row.get("sents") or []
            labels = row.get("labels") or []
            if len(sents) < 2:
                continue
            st = tokenize_sentences(tok, sents)
            b = build_batch(tok, st, args.max_seg_tokens, device)
            if b is None:
                continue
            input_ids, attention_mask, span_mask, loc = b
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(input_ids, attention_mask, span_mask)
            lab = []
            lg = []
            for gi, (bi, si) in loc.items():
                if gi < len(labels):
                    lg.append(logits[bi, si])
                    lab.append(float(labels[gi]))
            if not lg:
                continue
            logits_f = torch.stack(lg)
            lab_t = torch.tensor(lab, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = F.binary_cross_entropy_with_logits(logits_f.float(), lab_t, pos_weight=pos_weight)
            loss_ema["bce"] = loss_ema["bce"] * 0.9 + float(loss.detach()) * 0.1
        else:
            input_ids, attention_mask, span_mask, loc = batch
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(input_ids, attention_mask, span_mask)
            cand, probs = kd_target
            cand_logits = gather_logits(logits, loc, cand)
            if cand_logits is None or len(cand_logits) != len(probs):
                continue
            T = args.kd_temp
            logp = torch.log_softmax(cand_logits.float() / T, dim=0)
            kl = -(probs * logp).sum() * (T * T)
            loss = kl
            loss_ema["kd"] = loss_ema["kd"] * 0.9 + float(loss.detach()) * 0.1
        tok_seen += int(attention_mask.sum())

        (loss / args.grad_accum).backward()
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(
                f"step {step}/{args.steps} kd={loss_ema['kd']:.4f} bce={loss_ema['bce']:.4f} "
                f"lr={sched.get_last_lr()[0]:.2e} tok/s={tok_seen / max(dt, 1e-9):.0f} t={dt:.0f}s",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            kd_m, bce_m = evaluate(model, tok, val_teacher, args.val_bce, args, device)
            print(
                f"eval step {step}: top1={kd_m['top1']:.3f} spearman={kd_m['spearman']:.3f} "
                f"kl={kd_m['kl']:.3f} | bce acc={bce_m['acc']:.3f} auc={bce_m['auc']:.3f}",
                flush=True,
            )
            score = kd_m["top1"] + kd_m["spearman"]
            torch.save(
                {"model": model.state_dict(), "backbone": args.backbone, "args": vars(args)},
                os.path.join(args.out, "ckpt_last.pt"),
            )
            if not math.isnan(score) and score > best:
                best = score
                torch.save(
                    {"model": model.state_dict(), "backbone": args.backbone, "args": vars(args)},
                    os.path.join(args.out, "ckpt_best.pt"),
                )
                print(f"  saved best (score={score:.3f})", flush=True)

    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
