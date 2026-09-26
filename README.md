# Decily

**Decily** is an explicit candidate-scorer decision model (Route B): given a
`state`, a runtime `question`, and an arbitrary set of candidates, it returns a
**calibrated probability distribution** over them. It does not generate text.
1.7B parameters, trainable on a single consumer GPU, deployable on-device.

> Design: [DESIGN.md](DESIGN.md) · Step-by-step training recipe:
> [TRAINING.md](TRAINING.md) · Experiment log: [TRAIN-REPORT.md](TRAIN-REPORT.md)
> · Hugging Face model card: [MODEL_CARD.md](MODEL_CARD.md)

```python
Decily.decide(
    state="Our app crashes on startup after the latest update.",
    question="What is the customer's intent?",
    options=["technical", "billing", "shipping", "returns"],
)  # -> {"technical": 0.9999, "billing": 0.0001, "returns": 0.0, "shipping": 0.0}
# measured with the released weights (bf16, T=0.45)
```

## Results

Same protocol everywhere (accuracy / NLL / ECE after temperature fitting;
details in TRAIN-REPORT.md §9.23–§9.24):

| Evaluation | Decily (1.7B, 24 tasks) | decider-2b (2B, 95 tasks) |
|---|---|---|
| In-task, shared training tasks | **0.863 / 0.390 / 0.034** | 0.811 / 0.453 / 0.032 |
| Zero-shot fair suite (8 tasks × 300) | 0.654 / 0.86 / 0.092 | **0.700 / 0.71 / 0.047** |
| Unseen label sets (60/77-class intents) | 0.581 / 1.451 / 0.068 | — (scores include trained tasks) |

- **Same-domain winner at a quarter of the task count:** +5.2 pt over
  decider-2b on tasks both were trained on, with 1.7B vs 2B parameters.
- **Zero-shot close, not equal:** 0.654 vs 0.700 on tasks neither model saw;
  the remaining gap sits in knowledge-heavy tasks (sciq / pubmedqa / quality).
- **Calibrated enough to threshold:** the most-confident 5% of predictions on
  unseen label sets are **93.3%** accurate (same protocol SFT baseline: 79%),
  and abstention thresholds can be calibrated to a target error rate.

## Small and deployable

| Variant | Format | Size | Target |
|---|---|---|---|
| [`Decily-1.7B`](https://huggingface.co/alexzhang0118/Decily-1.7B) | PyTorch bf16 safetensors | ~3.4 GB | server / GPU |
| [`Decily-MLX`](https://huggingface.co/alexzhang0118/Decily-MLX) | MLX bf16 safetensors | ~3.4 GB | Apple Silicon |
| [`Decily-MLX-4bit`](https://huggingface.co/alexzhang0118/Decily-MLX-4bit) | MLX 4-bit (group 64) | **968 MB** | on-device / low memory |
| [`Decily-ONNX-int8`](https://huggingface.co/alexzhang0118/Decily-ONNX-int8) | ONNX int8 (single file) | 1.66 GB | pure CPU / Windows / edge |

- Candidate sets are scored in one batched forward pass (~75 ms per sentence
  on M-series chips with the 4-bit build); 4-bit matches bf16 probabilities to
  0.003 with the same speed (TRAIN-REPORT §9.25).
- The 1.7B backbone is standard Qwen3, so it also exports to ONNX for CPU
  deployment (unlike Qwen3.5's linear attention, which cannot be exported).

## The recipe

What makes a 1.7B model competitive here is the training pipeline, not scale:

1. **Route B explicit candidate scorer** — +10.6 pt held-out over the
   LM-head-letter-logits route in a controlled same-backbone comparison.
2. **24 task families + 15% belief data** — synthetic stochastic processes
   with known probability laws teach calibration, not just accuracy.
3. **RLCD post-training** — belief proper scoring and confidence ordering;
   makes selective prediction / abstention usable.
4. **Probability-space ensemble → distillation** — four diverse models
   (NLL 1.455) distilled into one (NLL **1.451**) at 1× inference cost.
5. **Consumer hardware** — the v5 full fine-tune is 3000 steps ≈ **1.5 h on a
   single RTX 5090 32 GB**; the whole pipeline fits in well under a day.

## Scaling expectations

Task coverage is the documented main lever (TRAIN-REPORT §9.24): our 24-task
model already beats a 95-task 2B baseline on shared tasks, and the remaining
zero-shot gap is attributable to coverage rather than capacity. **Re-run this
recipe with 60–95 tasks and we expect the zero-shot gap to close and the
in-domain lead to grow — on one consumer GPU, in under a day.** If you have a
task set at hand, the honest path to a stronger Decily is the recipe in
[TRAINING.md](TRAINING.md), not a bigger backbone.

## Reproduce

[TRAINING.md](TRAINING.md) contains the complete, command-level recipe:
data conversion for all task families, the belief streams, the 45-task teacher,
the 1.7B SFT, the RLCD members, ensemble soft-label generation, the v5 full
fine-tune, evaluation (including the fair suite and abstention calibration),
exports, and the pitfalls we hit along the way.

```bash
pip install -e ".[train]" && pip install bitsandbytes
bash scripts/build_stage2_data.sh
PYTHONPATH=src python scripts/train_rlcd.py --config configs/rlcd_v5_distill.yaml
```

Reference checkpoints for every stage (expected metrics included) are listed in
TRAINING.md §6.3 so a reproduction can verify itself.

## Evaluation protocol

Every number in this README is reported **after temperature fitting**; raw T=1
logits are over-confident on unseen data (held-out ECE 0.34 → 0.068 fitted).
Estimates of external models are compared only on tasks that are held out for
both sides; `TRAIN-REPORT.md §9.23` explains why a naive held-out number for
decider-2b would be misleading.

## License and acknowledgements

Apache-2.0 (see [LICENSE](LICENSE)) for the code and the released weights.

- Base model: [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) (Apache-2.0).
- External baseline and Route A reference: [Mapika/decider](https://github.com/Mapika/decider).
