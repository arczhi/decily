# Training & Reproduction Guide

End-to-end recipe to reproduce the **v5 decision model** (Qwen3-1.7B explicit
candidate scorer) from public datasets on a single consumer GPU.
Design rationale: [DESIGN.md](DESIGN.md). Full results, ablations and pitfalls:
[TRAIN-REPORT.md](TRAIN-REPORT.md).

## 0. Hardware, environment, budget

- 1× RTX 5090 32 GB (a 24 GB card works with the same batch/grad-accum settings)
- Python ≥ 3.10; `pip install -e ".[train]"` plus `pip install bitsandbytes`
  (v5 uses `optim: adamw8bit`)
- China networks: `export HF_ENDPOINT=https://hf-mirror.com`
- Public base models: `Qwen/Qwen3-1.7B-Base` (student / SFT backbone),
  the Qwen3.5-2B base text tower (teacher backbone). Configs point at local
  paths under `/root/models/...`; replace with your download location.

Wall-clock on one 5090 (measured):

| Stage | Time |
|---|---|
| Data conversion (24 tasks + evals + belief + fair suite) | 30-60 min (download-bound) |
| Teacher: 45-task Qwen3.5-2B LoRA, 5000 steps | ~4.0 h (TRAIN-REPORT §9.27) |
| SFT: Qwen3-1.7B LoRA, 3000 steps | ~1 h |
| RLCD members v1 (800) + v2 (1000 steps, LoRA) | ~20 min together |
| fullFT@1500 (from merged SFT) | ~40 min |
| Ensemble soft labels for 24 tasks | ~15 min |
| **v5 full fine-tune, 3000 steps** | **~1.5 h** (5,414 s) |
| Evaluation (all suites) | minutes |

Total well under one day on a single GPU.

## 1. Pipeline at a glance

```
public datasets
  └─ convert_data.py ──► data/raw/*.jsonl                (stage 1: 4 tasks, stage 2: 24 tasks)
        └─ build_eval.py ──► data/eval/*.jsonl           (in-task / held-out / fair suite)
teacher:   train_rlcd.py --config configs/stage1_qwen35.yaml        (45-task Qwen3.5-2B LoRA)
SFT:       python -m decision_model.train.stage1 --config configs/cx_qwen17b_bigk.yaml
             └─ merge_sft_adapter.py ──► runs/merged_sft/model.pt   (full-weight warm start)
members:   train_rlcd.py --config configs/rlcd_17b.yaml             (v1)
           train_rlcd.py --config configs/rlcd_v2_17b.yaml          (v2)
           python -m decision_model.train.stage1 --config configs/fullft_from_sft.yaml
distill:   precompute_teacher.py (4 teachers) ──► data/distill5/*.jsonl   (soft labels)
v5:        train_rlcd.py --config configs/rlcd_v5_distill.yaml      (full FT, KD + belief)
export:    export_safetensors.py / export_stage1_mlx.py / mlx_lm.convert
```

All commands below assume the repo root as CWD and `PYTHONPATH=src`.

## 2. Data preparation

### 2.1 Format and tooling

Every example is a decision tuple (`state`, `question`, options, `answer_ids`,
optional `probs` for soft targets) defined in `src/decision_model/data/schema.py`.
Converters are registered by name; the CLI is `scripts/convert_data.py`:

```bash
python scripts/convert_data.py --dataset <name> --split <split> --limit <N> \
    --out data/raw/<name>_train.jsonl
```

The registry name equals the task name. Converter dataset ids and default
splits live in `src/decision_model/data/converters/`.

### 2.2 Student shards: 24 tasks (v5 training mixture)

Run `scripts/build_stage2_data.sh` (exact per-task splits and limits are in the
script). It converts train shards, in-task eval shards, then mixes the 24-task
in-task set:

```bash
bash scripts/build_stage2_data.sh
# -> data/raw/*_train.jsonl (24 tasks), data/raw/*_val.jsonl,
#    data/eval/stage2_intask.jsonl
```

Notes:
- `massive_scenario` is deliberately **not** trained (same utterances as the
  held-out massive-intent task; would leak).
- MMLU subsets use `split=test` for train shards and `split=validation` for eval
  so the two are disjoint (see the script comments).

### 2.3 Four basic shards and held-out set

`scripts/build_stage1_data.sh` produces the original 4 train shards plus the
held-out mix that is never trained:

```bash
bash scripts/build_stage1_data.sh
# -> data/eval/stage1_intask.jsonl, data/eval/stage1_heldout.jsonl
```

The evaluation sets used throughout the reports:

| File | Contents |
|---|---|
| `data/eval/stage2_intask.jsonl` | 24 trained tasks, ~250 each (in-task) |
| `data/eval/stage1_heldout.jsonl` | massive-intent (60) / banking77 (77) / emotion (6) — unseen classes |
| `data/eval/fair_suite.jsonl` | 8 tasks zero-shot for both us and decider-2b, 300 each |

### 2.4 Teacher shards: the 21 extension tasks (24 → 45)

The 45-task teacher also trains on: `hellaswag, openbookqa, commonsense_qa,
qasc, winogrande, boolq, cola, qnli, medmcqa, medqa, race, wiki_qa,
yelp_polarity, sst5, newsgroups20, civil_comments, sms_spam, bias_in_bios,
glaive_tools, hermes_tools, toolace`. Convert each with the same CLI and its
documented split, e.g.:

```bash
python scripts/convert_data.py --dataset hellaswag --split validation --limit 2000 \
    --out data/raw/hellaswag_train.jsonl
```

The authoritative task list, file paths and per-task weights (MMLU subsets 0.5,
rest 1.0) are the `data.manifest` of `configs/stage1_qwen35.yaml`.

### 2.5 Belief streams (calibration data)

Synthetic stochastic processes with known probability laws, carrying soft
targets (`probs`), used by the RLCD belief objective:

```bash
python scripts/convert_data.py --dataset belief      --split train --limit 20000 \
    --out data/raw/belief_v2_train.jsonl
python scripts/convert_data.py --dataset belief_eval --split train --limit 1000 \
    --out data/raw/belief_v2_eval.jsonl
```

Converters `goemotions_belief` / `goemotions_belief_eval` provide the real-world
ambiguity variant. What matters for reproduction is the mix ratio
(`belief_ratio` in each config), not the exact stream size.

### 2.6 Fair suite (zero-shot comparison)

Eight datasets marked held-out by both models (`src/decision_model/data/converters/fair_suite.py`
documents ids and splits): `paws, sciq, pubmedqa, bbc_news, quality,
fin_phrasebank, dolly_category, xstory_cloze`. Convert 300 each, then mix:

```bash
python scripts/convert_data.py --dataset paws --split test --limit 300 --out data/raw/paws_val.jsonl
# ... same for the other seven ...
python scripts/build_eval.py \
  --spec paws=300,sciq=300,pubmedqa=300,bbc_news=300,quality=300,fin_phrasebank=300,dolly_category=300,xstory_cloze=300 \
  --split val --out data/eval/fair_suite.jsonl
```

## 3. Teacher: 45-task Qwen3.5-2B (LoRA, frozen backbone)

```bash
PYTHONPATH=src python scripts/train_rlcd.py --config configs/stage1_qwen35.yaml
```

- frozen backbone + LoRA r32/α64, 45 tasks + 15% belief, 5000 steps,
  effective batch 16, lr 2e-4, ~4 h on a 5090
- outputs `runs/stage1_qwen35/` (LoRA adapter, ~290 MB)
- expected (temperature-fitted): in-task 0.754 / NLL 0.649 / ECE 0.036;
  held-out 0.583 / 1.458 / 0.093 (TRAIN-REPORT §9.27)

The teacher's role in the v5 recipe is to be one of the labelers for the reader
salience students; the v5 ensemble (below) uses the Qwen3-1.7B members instead.

## 4. SFT baseline: Qwen3-1.7B (LoRA, 24 tasks)

```bash
PYTHONPATH=src python -m decision_model.train.stage1 --config configs/cx_qwen17b_bigk.yaml
PYTHONPATH=src python scripts/merge_sft_adapter.py --config configs/cx_qwen17b_bigk.yaml \
    --ckpt runs/cx_qwen17b_bigk/ckpt_last.pt --out runs/merged_sft/model.pt
```

- LoRA r16/α32 on the frozen Qwen3-1.7B base, 3000 steps, lr 1e-4, ~1 h
- `merged_sft/model.pt` is the fp32 full-weight warm start used by both the
  fullFT member and v5

## 5. RLCD members (ensemble teachers)

```bash
PYTHONPATH=src python scripts/train_rlcd.py --config configs/rlcd_17b.yaml       # v1
PYTHONPATH=src python scripts/train_rlcd.py --config configs/rlcd_v2_17b.yaml    # v2
PYTHONPATH=src python -m decision_model.train.stage1 --config configs/fullft_from_sft.yaml
```

| Member | Start | Trainable | Steps | Key settings |
|---|---|---|---|---|
| SFT | Qwen3-1.7B | LoRA r16 | 3000 | hard labels, 24 tasks → `runs/cx_qwen17b_bigk` |
| v1 | SFT ckpt | LoRA | 800 | belief_ratio 0.4, lr 2e-5 |
| v2 | SFT ckpt | LoRA | 1000 | belief 0.25, abstention 0.2, KL 0.3, action 0.3 |
| fullFT@1500 | merged SFT | full FT | 3000 (use step 1500) | adamw8bit, lr 1e-5 → `runs/fullft_sft17b` |

The four distributions (SFT / v1 / v2 / fullFT@1500) are the ensemble teachers.
Their complementarity is the reason distillation works (TRAIN-REPORT §9.20).

## 6. Ensemble distillation → v5

### 6.1 Build teacher soft labels

```bash
PYTHONPATH=src python scripts/precompute_teacher.py \
  --config configs/rlcd_v5_distill.yaml \
  --ckpts runs/cx_qwen17b_bigk/ckpt_last.pt \
          runs/rlcd_17b/ckpt_last.pt \
          runs/rlcd_v2_17b/ckpt_last.pt \
          runs/fullft_sft17b/ckpt_step1500.pt \
  --configs configs/cx_qwen17b_bigk.yaml configs/cx_qwen17b_bigk.yaml \
            configs/cx_qwen17b_bigk.yaml configs/fullft_from_sft.yaml \
  --out-dir data/distill5
```

For every example of the 24-task manifest this writes
`Question.probs = mean(teacher softmaxes)`. `--argmax-gold` additionally sets
the hard label to the teacher argmax (use it when no human gold is desired).

### 6.2 Train v5 (full fine-tune)

```bash
PYTHONPATH=src python scripts/train_rlcd.py --config configs/rlcd_v5_distill.yaml
```

- full fine-tune from `runs/merged_sft/model.pt`, 3000 steps, effective batch 16,
  lr 1e-5, cosine, 8-bit AdamW, bf16 checkpoint (`save_dtype: bfloat16`)
- loss: `kd_alpha 0.5` Hinton KD against the soft labels (T=2) + 15% belief
  proper-score rows + CE on gold
- outputs `runs/rlcd_v5_distill/`; ~1.5 h on a 5090

### 6.3 Expected results (verification checkpoints)

| Metric | Target |
|---|---|
| stage2_intask acc / NLL / ECE (fitted) | 0.863 / 0.393 / 0.034 |
| held-out acc / NLL / ECE (fitted, T≈0.45) | 0.581 / 1.451 / 0.068 |
| fair suite macro acc (zero-shot, fitted) | 0.654 |
| belief excess (nats) | ~0.069 |

## 7. Evaluation protocol

### 7.1 Checkpoint eval (mandatory temperature fitting)

```bash
PYTHONPATH=src python scripts/eval_ckpt.py --config configs/rlcd_v5_distill.yaml \
    --ckpt runs/rlcd_v5_distill/ckpt_last.pt \
    --eval data/eval/stage2_intask.jsonl data/eval/stage1_heldout.jsonl \
    --fit-temperature --per-task --out runs/rlcd_v5_distill/eval_report.json
```

Raw T=1 numbers are over-confident (held-out ECE 0.34); always report the fitted
temperature alongside. Fine-grained per task comparisons:
`scripts/compare_reports.py runs_remote/*/eval_report.json --labels ...`.

### 7.2 Zero-shot fair suite

```bash
PYTHONPATH=src python scripts/eval_ckpt.py --config configs/rlcd_v5_distill.yaml \
    --ckpt runs/rlcd_v5_distill/ckpt_last.pt \
    --eval data/eval/fair_suite.jsonl --fit-temperature --per-task \
    --out runs/rlcd_v5_distill/fair_report.json
```

### 7.3 Selective prediction / abstention

```bash
PYTHONPATH=src python scripts/calibrate_abstention.py --config configs/rlcd_v2_17b.yaml \
    --ckpt runs/rlcd_v2_17b/ckpt_last.pt --eval data/eval/stage1_heldout.jsonl \
    --target-false-rate 0.05 --out runs/rlcd_v2_17b/abstention_calibration.json
PYTHONPATH=src python scripts/conformal_eval.py --config configs/rlcd_v3_17b.yaml \
    --ckpt runs/rlcd_v3_17b/ckpt_last.pt --eval data/eval/stage1_heldout.jsonl
```

### 7.4 Ensemble evaluation

```bash
PYTHONPATH=src python scripts/ensemble_eval.py \
    --labels SFT v1 v2 fullFT \
    --configs configs/cx_qwen17b_bigk.yaml configs/cx_qwen17b_bigk.yaml \
              configs/cx_qwen17b_bigk.yaml configs/fullft_from_sft.yaml \
    --ckpts runs/cx_qwen17b_bigk/ckpt_last.pt runs/rlcd_17b/ckpt_last.pt \
            runs/rlcd_v2_17b/ckpt_last.pt runs/fullft_sft17b/ckpt_step1500.pt \
    --evals data/eval/stage1_heldout.jsonl data/eval/stage2_intask.jsonl \
    --max-per-eval 1200 --out runs/ensemble_eval.json
```

### 7.5 Reporting rules

- always state the protocol: in-task vs held-out set, T=1 vs fitted temperature
- never compare "our in-task" against "another model's held-out" (TRAIN-REPORT §9.23)
- for external baselines, check their `heldout` flags before claiming a comparison

## 8. Export & on-device deployment

```bash
# portable safetensors bundle (HF-style): one model.safetensors (backbone +
# decision head, training key names) + config.json (with `decision_head` spec)
# + tokenizer files
PYTHONPATH=src python scripts/export_safetensors.py \
    --ckpt runs/rlcd_v5_distill/ckpt_last.pt \
    --backbone-dir /root/models/Qwen3-1.7B-Base \
    --out-dir models_hf/Decily-1.7B

# MLX bundle for Apple Silicon (bf16): merged backbone + head in one file,
#   keys `encoder.model.*` + `pool.*`/`head.*` — the layout `mlx/decision_mlx.py`
#   loads. (scripts/export_stage1_mlx.py is the reference exporter; adapt its key
#   mapping for the Qwen3 backbone.)
# 4-bit for on-device: quantize the tower with mlx_lm.convert and keep the head
#   as a separate decision_head.safetensors (group-64 affine)
mlx_lm.convert --hf-path models_mlx/v5_mlx -q --q-bits 4 --mlx-path models/v5_batched_4bit
```

- batched reader tower (state+question+candidate packing): `scripts/prep_reader_batched_model.py`
- ONNX: v5 uses the standard Qwen3 architecture and is exportable; the ONNX int8
  artifact is planned (TRAIN-REPORT §9.25 documents what fails for Qwen3.5's
  linear attention, which is why the teacher cannot be exported)
- verification: after MLX conversion, cross-check logits on a fixed sample
  (parity table in TRAIN-REPORT §9.22)

## 9. Pitfalls & reproducibility notes (from TRAIN-REPORT)

1. **Log accumulation bug** (§9.18): logged losses were `accum×` inflated; fixed
   by dividing by `window × accum`. Normalize before comparing runs.
2. **KD padding mask** (§9.21): when mixing soft targets with `-1e9` masks, apply
   the mask to the teacher probabilities too or the loss explodes.
3. **Checkpoint size**: full-FT checkpoints are 3.4 GB in bf16 — set
   `save_dtype: bfloat16` and keep only `ckpt_last`/best.
4. **`rsync --delete`**: never let it delete `data/distill5` (soft labels are
   expensive to regenerate); keep the exclude rule.
5. **Temperature protocol**: models need T≈1.1 (in-task) / T≈0.45–1.35
   (held-out/fair) — do not mix raw and fitted numbers in one table.
6. **Leakage**: `massive_scenario` must stay out of training; check eval-shard
   splits (`test` vs `validation`) when adding tasks.
7. **Seeds**: all configs are seeded (`seed: 0`); sampling in `MixtureSampler`
   derives from it, so runs are reproducible on the same data files.

## 10. File map

| Path | Purpose |
|---|---|
| `configs/stage1_qwen35.yaml` | 45-task teacher (LoRA on Qwen3.5-2B) |
| `configs/cx_qwen17b_bigk.yaml` | 1.7B SFT baseline (LoRA) |
| `configs/rlcd_17b.yaml`, `configs/rlcd_v2_17b.yaml` | RLCD members v1/v2 |
| `configs/fullft_from_sft.yaml` | full-FT member |
| `configs/rlcd_v5_distill.yaml` | **v5 ensemble distillation recipe** |
| `scripts/build_stage1_data.sh`, `build_stage2_data.sh` | data conversion |
| `scripts/convert_data.py`, `build_eval.py` | shard conversion / eval mixing |
| `scripts/precompute_teacher.py` | ensemble soft labels |
| `scripts/train_rlcd.py` | RLCD/KD trainer driver |
| `src/decision_model/train/stage1.py` | stage-1 / SFT trainer |
| `scripts/eval_ckpt.py`, `ensemble_eval.py`, `compare_reports.py` | evaluation |
| `scripts/calibrate_abstention.py`, `conformal_eval.py` | selective prediction |
| `scripts/export_safetensors.py`, `export_stage1_mlx.py` | export |
| `src/decision_model/train/rlcd.py` | loss/training implementation |
| `mlx/decision_mlx.py` | standalone MLX inference (no torch) |
| `TRAIN-REPORT.md` | results, ablations, pitfalls (§9.x) |
