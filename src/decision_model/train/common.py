"""Shared training loop for Route A / Route B."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator

import torch
from torch.optim import AdamW

from ..data.batching import batched
from ..data.mixture import MixtureItem, MixtureSampler
from ..data.schema import DecisionExample
from ..eval.harness import evaluate
from ..models.decision_model import decision_loss


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100
    scheduler: str = "cosine"  # cosine | linear | none
    grad_accum: int = 1
    length_bucket: bool = False
    bucket_buffer_mult: int = 8
    optim: str = "adamw"  # adamw | adamw8bit
    save_dtype: str = "float32"
    log_every: int = 20
    eval_every: int = 250
    eval_batch_size: int = 64
    eval_max_batches: int | None = 50
    out_dir: str = "runs/mini_b0"
    seed: int = 0
    device: str = "cuda"
    label_smoothing: float = 0.05
    normalize_by_k: bool = False
    save_trainable_only: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_scheduler(optimizer, cfg: TrainConfig):
    warmup = max(cfg.warmup_steps, 0)

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if cfg.scheduler == "none":
            return 1.0
        progress = (step - warmup) / max(cfg.steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        if cfg.scheduler == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(
        self,
        model,
        collator,
        sampler: MixtureSampler,
        eval_examples: list[DecisionExample],
        eval_collator=None,
        cfg: TrainConfig | None = None,
        temperature: float = 1.0,
        probe_examples: list[DecisionExample] | None = None,
    ) -> None:
        self.model = model
        self.collator = collator
        self.sampler = sampler
        self.eval_examples = eval_examples
        self.eval_collator = eval_collator or collator
        self.cfg = cfg or TrainConfig()
        self.temperature = temperature
        self.probe_examples = probe_examples

    @torch.no_grad()
    def _probe_loss(self, device: str) -> float:
        was_training = self.model.training
        self.model.eval()
        batch = self.collator(self.probe_examples)
        batch = _to_device(batch, device)
        out = self.model(batch)
        loss = decision_loss(
            out["logits"], batch, self.cfg.label_smoothing, self.cfg.normalize_by_k
        )["total"]
        if was_training:
            self.model.train()
        return float(loss)

    def _save(self, step: int, tag: str) -> None:
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        cast = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }.get(self.cfg.save_dtype, torch.float32)
        if self.cfg.save_trainable_only:
            state = {
                k: v for k, v in self.model.named_parameters() if v.requires_grad
            }
            state = {k: v.detach().to(cast).cpu() for k, v in state.items()}
        else:
            state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        path = os.path.join(self.cfg.out_dir, f"ckpt_{tag}.pt")
        torch.save({"step": step, "state": state, "config": asdict(self.cfg)}, path)
        return path

    def run(self) -> dict:
        cfg = self.cfg
        set_seed(cfg.seed)
        device = cfg.device
        self.model.to(device)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        if n_train == 0:
            raise RuntimeError(
                "no trainable parameters; unfreeze the backbone or enable LoRA"
            )
        print(f"[train] trainable params: {n_train/1e6:.2f}M", flush=True)
        if cfg.optim == "adamw8bit":
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                trainable, lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        else:
            optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = _make_scheduler(optimizer, cfg)

        os.makedirs(cfg.out_dir, exist_ok=True)
        log_path = os.path.join(cfg.out_dir, "train_log.jsonl")
        log_f = open(log_path, "a", encoding="utf-8")

        accum = max(cfg.grad_accum, 1)
        it = batched(
            self.sampler.sample(cfg.steps * cfg.batch_size * accum, start_seed=cfg.seed),
            cfg.batch_size,
            bucket=cfg.length_bucket,
            buffer_mult=cfg.bucket_buffer_mult,
        )
        running: dict[str, float] = {}
        window = 0
        opt_step = 0
        t0 = time.time()
        last_eval: dict = {}
        for micro_step, examples in enumerate(it, 1):
            batch = self.collator(examples)
            batch = _to_device(batch, device)
            self.model.train()
            out = self.model(batch)
            losses = decision_loss(
                out["logits"], batch, cfg.label_smoothing, cfg.normalize_by_k
            )
            (losses["total"] / accum).backward()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())

            if micro_step % accum != 0:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            window += 1
            running["grad_norm"] = running.get("grad_norm", 0.0) + float(grad_norm)

            if window >= cfg.log_every:
                # running accumulates per MICRO-batch; divide by micro-batches, not steps
                avg = {k: v / (window * accum) for k, v in running.items()}
                running = {}
                window = 0
                line = {"step": opt_step, "sec": round(time.time() - t0, 1), **avg}
                if self.probe_examples:
                    line["probe_loss"] = self._probe_loss(device)
                log_f.write(json.dumps(line) + "\n")
                log_f.flush()
                print(f"[train] {line}", flush=True)
            if cfg.eval_every and opt_step % cfg.eval_every == 0:
                last_eval = evaluate(
                    self.model,
                    self.eval_examples,
                    self.eval_collator,
                    temperature=self.temperature,
                    batch_size=cfg.eval_batch_size,
                    device=device,
                    max_batches=cfg.eval_max_batches,
                )
                log_f.write(json.dumps({"step": opt_step, "eval": last_eval}) + "\n")
                log_f.flush()
                print(f"[eval] step {opt_step}: {last_eval}", flush=True)
                path = self._save(opt_step, f"step{opt_step}")
                print(f"[train] saved {path}", flush=True)
        self._save(cfg.steps, "last")
        log_f.close()
        return last_eval


def _to_device(batch, device: str):
    for field in batch.__dataclass_fields__:
        v = getattr(batch, field)
        if isinstance(v, torch.Tensor):
            setattr(batch, field, v.to(device))
    return batch
