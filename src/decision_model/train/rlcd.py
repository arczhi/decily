"""RLCD: RL for Calibrated Decisions (decider v10 recipe, non-generative form).

Losses per batch:
- belief:  proper scoring rule (log score) on soft-target rows (known laws)
- action:  REINFORCE with a baseline on hard-label rows (verifiable reward)
- kl:      KL(policy || reference policy) so the model stays near the SFT prior

The reference model is a frozen copy of the starting checkpoint; its logits are
computed under no_grad. Order consistency is structural in Route B (candidate
scoring is permutation-invariant), so no extra term is needed.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW

from ..data.mixture import MixtureSampler
from ..data.schema import DecisionExample
from ..data.transforms import add_abstention, drop_gold
from ..eval.belief import evaluate_belief
from ..eval.harness import evaluate


@dataclass
class RLCDConfig:
    steps: int = 800
    batch_size: int = 8
    grad_accum: int = 2
    lr: float = 2e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup_steps: int = 50
    scheduler: str = "cosine"
    belief_weight: float = 1.0
    action_weight: float = 0.3
    kl_weight: float = 0.1
    ce_weight: float = 1.0
    spherical_weight: float = 0.75
    brier_weight: float = 0.5
    kl_temperature: float = 1.0
    belief_ratio: float = 0.4
    abstention_ratio: float = 0.0
    log_every: int = 25
    eval_every: int = 200
    eval_batch_size: int = 16
    eval_max_batches: int | None = 60
    out_dir: str = "runs/rlcd"
    seed: int = 0
    device: str = "cuda"
    save_trainable_only: bool = True


def _to_device(batch, device: str):
    for field in batch.__dataclass_fields__:
        v = getattr(batch, field)
        if isinstance(v, torch.Tensor):
            setattr(batch, field, v.to(device))
    return batch


def _masked_log_softmax(logits: Tensor, valid: Tensor) -> Tensor:
    lg = logits.masked_fill(~valid, -1e9)
    return F.log_softmax(lg, dim=-1)


def rlcd_losses(
    out: dict,
    ref_out: dict,
    batch,
    cfg: RLCDConfig,
) -> dict[str, Tensor]:
    """v3 losses: joint CE (candidates + reject class), belief proper score
    (log + spherical), REINFORCE action with a Brier-shaped reward, KL anchor."""
    logits = out["logits"].float()
    if logits.dim() == 3:
        logits = logits[:, 0]
    ref_logits = ref_out["logits"].float()
    if ref_logits.dim() == 3:
        ref_logits = ref_logits[:, 0]

    valid = batch.option_valid[:, 0]
    defer = batch.defer.bool() if batch.defer is not None else torch.zeros_like(valid[:, 0])
    logp = _masked_log_softmax(logits, valid)
    probs = logp.exp()

    losses: dict[str, Tensor] = {}
    has_soft = batch.has_soft.bool()

    # ---- joint CE over candidates (+ reject class when a rejector exists) ----
    if cfg.ce_weight > 0:
        reject_logit = out.get("reject_logit")
        if reject_logit is not None:
            reject_logit = reject_logit.float()
            if reject_logit.dim() == 2:
                reject_logit = reject_logit[:, 0]
            full = torch.cat([logits, reject_logit.unsqueeze(-1)], dim=-1)
            full_valid = torch.cat(
                [valid, torch.ones_like(valid[:, :1])], dim=-1
            )
            k = full.size(-1)
            target = torch.where(
                defer,
                torch.full_like(batch.answer_index[:, 0], k - 1),
                batch.answer_index[:, 0].clamp(min=0),
            )
        else:
            full, full_valid = logits, valid
            k = full.size(-1)
            target = batch.answer_index[:, 0].clamp(min=0)

        row_mask = (~has_soft) & ((~defer) | (reject_logit is not None))
        if row_mask.any():
            lg = full[row_mask].masked_fill(~full_valid[row_mask], -1e9)
            tgt = target[row_mask]
            ls = cfg.label_smoothing if hasattr(cfg, "label_smoothing") else 0.0
            smooth = torch.full_like(lg, ls / max(k - 1, 1))
            smooth.scatter_(1, tgt.unsqueeze(1), 1.0 - ls)
            smooth = smooth.masked_fill(~full_valid[row_mask], 0.0)
            smooth = smooth / smooth.sum(-1, keepdim=True).clamp(min=1e-9)
            losses["ce"] = -(smooth * F.log_softmax(lg, dim=-1)).sum(-1).mean()

    # ---- belief: combined proper score (log + spherical) on soft targets ----
    if has_soft.any():
        soft = batch.soft_targets[:, 0][has_soft]
        soft = soft / soft.sum(-1, keepdim=True).clamp(min=1e-9)
        p = probs[has_soft]
        log_score = (soft * logp[has_soft]).sum(-1)
        spherical = (soft * p).sum(-1) / p.norm(dim=-1).clamp(min=1e-9)
        losses["belief"] = -(log_score + cfg.spherical_weight * spherical).mean()

    # ---- action: REINFORCE with reward = correct + brier (RLCR-style) ----
    hard = ~has_soft
    if hard.any() and cfg.action_weight > 0:
        gold = batch.answer_index[:, 0][hard].clamp(min=0)
        p = probs[hard]
        dist = torch.distributions.Categorical(probs=p.clamp(min=1e-9))
        action = dist.sample()
        y = (action == gold).float()
        q = p.gather(1, action.unsqueeze(1)).squeeze(1)
        reward = y + cfg.brier_weight * (-((q - y) ** 2))
        with torch.no_grad():
            onehot = F.one_hot(gold, num_classes=p.size(-1)).float()
            exp_reward = (
                p * (onehot + cfg.brier_weight * (-((p - onehot) ** 2)))
            ).sum(-1)
        adv = (reward - exp_reward).detach()
        losses["action"] = -(dist.log_prob(action) * adv).mean()

    # ---- KL to reference policy ----
    if cfg.kl_weight > 0:
        ref_logp = _masked_log_softmax(ref_logits, valid)
        kl = (probs * (logp - ref_logp)).sum(-1)
        losses["kl"] = kl.mean()

    total = torch.zeros((), device=logits.device)
    for name, w in (
        ("ce", cfg.ce_weight),
        ("belief", cfg.belief_weight),
        ("action", cfg.action_weight),
        ("kl", cfg.kl_weight),
    ):
        if name in losses:
            total = total + w * losses[name]
    losses["total"] = total
    return losses


def _make_scheduler(optimizer, cfg: RLCDConfig):
    warmup = max(cfg.warmup_steps, 0)

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(cfg.steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class RLCDTrainer:
    def __init__(
        self,
        model,
        reference_model,
        collator,
        sampler: MixtureSampler,
        belief_examples: list[DecisionExample],
        eval_examples: list[DecisionExample],
        belief_eval_examples: list[DecisionExample],
        cfg: RLCDConfig,
    ) -> None:
        self.model = model
        self.ref = reference_model
        self.collator = collator
        self.sampler = sampler
        self.belief_examples = belief_examples
        self.eval_examples = eval_examples
        self.belief_eval_examples = belief_eval_examples
        self.cfg = cfg

    def _mixed_batches(self) -> Iterator[list[DecisionExample]]:
        cfg = self.cfg
        rng = random.Random(cfg.seed)
        stream = self.sampler.sample(cfg.steps * cfg.batch_size * cfg.grad_accum, start_seed=cfg.seed)
        belief_pool = self.belief_examples
        for _ in range(cfg.steps * cfg.grad_accum):
            batch: list[DecisionExample] = []
            while len(batch) < cfg.batch_size:
                if rng.random() < cfg.belief_ratio:
                    batch.append(belief_pool[rng.randrange(len(belief_pool))])
                else:
                    ex = next(stream)
                    if cfg.abstention_ratio > 0 and rng.random() < cfg.abstention_ratio:
                        ex = drop_gold(ex)
                    batch.append(ex)
            yield batch

    def _save(self, step: int, tag: str) -> str:
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in self.model.named_parameters() if v.requires_grad}
        path = os.path.join(self.cfg.out_dir, f"ckpt_{tag}.pt")
        torch.save({"step": step, "state": state, "config": asdict(self.cfg)}, path)
        return path

    def run(self) -> dict:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        device = cfg.device
        self.model.to(device)
        self.ref.to(device).eval()
        for p in self.ref.parameters():
            p.requires_grad = False

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        print(f"[rlcd] trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M", flush=True)
        optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = _make_scheduler(optimizer, cfg)

        os.makedirs(cfg.out_dir, exist_ok=True)
        log_f = open(os.path.join(cfg.out_dir, "train_log.jsonl"), "a", encoding="utf-8")

        it = self._mixed_batches()
        running: dict[str, float] = {}
        window = 0
        opt_step = 0
        t0 = time.time()
        last_eval: dict = {}

        for micro, examples in enumerate(it, 1):
            batch = _to_device(self.collator(examples), device)
            self.model.train()
            out = self.model(batch)
            with torch.no_grad():
                ref_out = self.ref(batch)
            losses = rlcd_losses(out, ref_out, batch, cfg)
            (losses["total"] / cfg.grad_accum).backward()
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())

            if micro % cfg.grad_accum != 0:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            window += 1
            running["grad_norm"] = running.get("grad_norm", 0.0) + float(grad_norm)

            if window >= cfg.log_every:
                avg = {k: v / window for k, v in running.items()}
                running, window = {}, 0
                line = {"step": opt_step, "sec": round(time.time() - t0, 1), **avg}
                log_f.write(json.dumps(line) + "\n")
                log_f.flush()
                print(f"[rlcd] {line}", flush=True)

            if cfg.eval_every and opt_step % cfg.eval_every == 0:
                last_eval = evaluate(
                    self.model, self.eval_examples, self.collator,
                    batch_size=cfg.eval_batch_size, device=device,
                    max_batches=cfg.eval_max_batches,
                )
                belief_eval = evaluate_belief(
                    self.model, self.collator, self.belief_eval_examples,
                    batch_size=cfg.eval_batch_size, device=device,
                )
                log_f.write(
                    json.dumps({"step": opt_step, "eval": last_eval, "belief": belief_eval}) + "\n"
                )
                log_f.flush()
                print(f"[rlcd] step {opt_step} eval {last_eval}", flush=True)
                print(f"[rlcd] step {opt_step} belief {belief_eval}", flush=True)
                self._save(opt_step, f"step{opt_step}")

        self._save(cfg.steps, "last")
        log_f.close()
        return last_eval
