# Training and Experiment Log (Train Report)

> Companion document: `DESIGN.md` (design and architecture). This document records all Stage results, ablations,
> calibration/generalization evaluations, deployment measurements, and engineering lessons.
>
> Section numbering follows the 9.x sequence of `DESIGN.md` before it was split (matching commit messages and historical discussions),
> so this report starts at 9.5; lookups by number are recommended. For external implementation reference facts see `DESIGN.md` §10.

## Index

| Section | Topic |
|---|---|
| 9.5 | Stage 1 results |
| 9.6 | Stage 2a controlled comparison: same backbone, A vs B |
| 9.7 | Two-stage inference for large candidate sets |
| 9.8 | Stage 2b: 24-task mixture |
| 9.9 | Stage 2c: training recipe comparison |
| 9.10 | Stage 3: scaling up to Qwen3-1.7B |
| 9.11 | Stage 4: RLCD calibration |
| 9.12 | Stage 4b: RLCD v2 |
| 9.13 | none threshold calibration results |
| 9.14 | RLCD enhancement plan |
| 9.15 | Selective prediction: confidence ranking and risk guarantees |
| 9.16 | RLCD v3 ablation: effective and ineffective components |
| 9.17 | Final comparison and recommended configuration |
| 9.18 | Engineering lesson: log accumulation division bug |
| 9.19 | Model Soup experiment log |
| 9.20 | Probability-space ensemble results |
| 9.21 | RLCD v5: ensemble distillation back to a single model |
| 9.22 | MLX conversion |
| 9.23 | decider-2b head-to-head evaluation |
| 9.24 | Fair suite: zero-shot generalization comparison for both sides |
| 9.25 | Deployment side: batched inference / ONNX / salience student distillation |
| 9.26 | Pure-CPU 5-second plan: segmented whole-document encoding + sentence span pooling |
| 9.27 | Qwen3.5-2B teacher model |

---

## 9.5 Stage 1 Results (2026-09-23)

Setup: 4 training tasks (sst2/ag_news/mnli/arc_easy, ~28k examples total), 3000 steps, batch 16,
warmup 100 + cosine; in-task evaluation on 1200 examples (300 per task), held-out 1200 examples
(massive 60 classes / banking77 77 classes / emotion 6 classes, none trained).

| Metric | B_frozen | B_lora | A_lora |
|---|---|---|---|
| Trainable params | 2.4M | 5.75M | 10.1M |
| Backbone | ModernBERT-base | ModernBERT-base | Qwen3-0.6B |
| in-task acc / NLL / ECE | 0.611 / 0.84 / 0.029 | 0.816 / 0.51 / **0.019** | **0.888** / **0.36** / 0.023 |
| held-out acc / NLL / ECE | 0.186 / 3.16 / 0.070 | **0.412 / 2.25 / 0.056** | 0.385 / 2.75 / 0.138 |

Conclusions:
- After adding LoRA, Route B is **better than Route A across the board on held-out** (acc +2.7pt, NLL −0.5,
  ECE 0.056 vs 0.138), slightly lower in-task but better calibrated → explicit candidate scoring + listwise
  loss is indeed more resistant to overfitting and generalizes better across schemas
- Route A fits the training tasks more strongly (0.888), but its held-out calibration is clearly worse (ECE 0.138)
- Note: the two use different backbones (ModernBERT vs Qwen3), not a strictly controlled variable; Stage 2 needs a
  same-backbone comparison (e.g. both Qwen3-0.6B, comparing letter logits vs an explicit scorer)
- Temperature fitting (in-task / held-out): B_frozen 1.25 / 1.43; B_lora 1.13 / 0.80;
  A_lora 1.07 / 1.70. A's held-out requires a significantly larger temperature (1.7) to be calibrated, indicating its
  cross-task confidence is too high (overconfident); B_lora goes the other way (0.8, slightly underconfident)

Key engineering lessons (now baked into config defaults):
- LoRA/adapter training needs lr ≤2e-4 + warmup + cosine; 1e-3 learns first and then regresses
- Frozen backbone + small scorer lacks capacity; held-out generalization needs trainable capacity in the encoder

## 9.6 Stage 2a Controlled Comparison: Same Backbone, A vs B (2026-09-23)

Same backbone Qwen3-0.6B-Base, same data (4 tasks), same LoRA/schedule (3000 steps):

| Metric | A (LM-head letter logits) | B (cross-encoder scorer) |
|---|---|---|
| Trainable params | 10.1M + lm_head | 14.3M |
| in-task acc / NLL / ECE | **0.888** / 0.36 / 0.023 | 0.881 / **0.35** / **0.020** |
| held-out acc / NLL / ECE | 0.385 / 2.75 / 0.138 | **0.491** / **1.95** / **0.129** |

Conclusions: after controlling the backbone, **Route B is significantly better on held-out** (acc +10.6pt, NLL −0.8),
ties in-task and calibrates slightly better. The generalization advantage of explicit candidate scoring is cleanly verified →
Stage 2b directly expands the task mixture (24 tasks) to further amplify this advantage.

## 9.7 Two-Stage Inference for Large Candidate Sets (measured 2026-09-23)

Implementation `decision_model.infer.two_stage`: candidates are scored independently in chunks (cross-encoder
logits are independent of the candidate set, so chunking is lossless) → Top-K → relative softmax.
This corresponds to the official "independent scoring first, then explicit choice at high cardinality".

120-example measurement (77-class banking77 + 60-class massive, Stage 2a model):

| Path | acc | mean_conf |
|---|---|---|
| Full softmax | 0.508 | 0.109 |
| Top-8 two-stage | 0.508 | 0.275 |

- **gold_in_top8 = 87.5%** → the shortlist ceiling, the basis for choosing K
- Accuracy is lossless (argmax preserved); confidence is more honest (0.275 vs 0.109)
- Note: the current chunk implementation re-encodes each chunk, so latency is higher; production should reuse prefix/KV
  (the cross-encoder's state+question prefix can be cached)

## 9.8 Stage 2b: 24-Task Mixture (completed 2026-09-23)

Data: 24 tasks (~75k training examples, including 8 MMLU subjects); model Qwen3-0.6B-Base + LoRA
(cross-encoder, 14.3M trainable); 3000 steps, effective batch 16 (micro 4 × accum 4),
bf16 + gradient checkpointing, state truncated to 256.

| Metric | A (4 tasks) | B (4 tasks) | B (24 tasks) |
|---|---|---|---|
| held-out acc | 0.385 | 0.491 | **0.542** |
| held-out NLL (fitted) | 2.75 | 1.95 | **1.86** |
| held-out ECE (fitted) | 0.138 | 0.129 | **0.107** |
| held-out fitted temperature | 1.70 | 0.50 | 0.40 |
| in-task acc (respective eval sets) | 0.888 | 0.881 | 0.850 |
| in-task ECE | 0.023 | 0.020 | **0.017** |

held-out per task (B4 → B24): massive 0.588→0.573 (slight drop), banking77 0.370→0.432,
emotion 0.512→0.620 → the main gains from mixture breadth are in **large candidate sets (77 classes) and fine-grained emotion**.

Conclusions:
- **Task coverage breadth works**: vs B4, held-out acc +5.1pt / NLL −0.09 / ECE −0.022
- B (24 tasks) beats Route A on all held-out metrics (acc +15.7pt, NLL −0.89, ECE −0.031)
- Calibration: held-out needs T≈0.4 (underconfident), in-task T≈1.0 (already good) → Stage 2c does
  per-task-family calibration
- massive did not improve: its distribution differs greatly from the training tasks (voice-assistant utterances), and scenario tasks were
  deliberately excluded for being same-source as the held-out intent

Evaluation hygiene fixes:
- MMLU training used the test split, evaluation switched to validation (to remove overlap); temperature grid lower bound 0.5→0.2
- Evaluation now collects logits in a single pass and reuses them at any temperature (the old implementation ran 3 forwards per file)

## 9.9 Stage 2c: Training Recipe Comparison (completed 2026-09-23)

Two recipe variants, everything else the same as s2b (24 tasks, Qwen3-0.6B + LoRA, 3000 steps):

| held-out | B(4 tasks) | B(24 tasks) | B_normk | **B_bigk** |
|---|---|---|---|---|
| acc | 0.491 | 0.542 | 0.487 | 0.538 |
| NLL | 1.95 | 1.86 | 1.86 | **1.60** |
| ECE | 0.129 | 0.107 | 0.065 | **0.071** |

- **bigk (max_options 10→16 + length_bucket) adopted**: accuracy is on par but NLL/ECE are significantly better;
  banking77 (77 classes) 0.432→**0.517** (+8.5pt) → training candidate-set size directly determines large-candidate-set generalization;
  cost: emotion 0.620→0.525 (capability reallocation)
- **normk (loss normalized by log K) not adopted as the main recipe**: held-out acc −5.5pt
  (banking77 collapses to 0.278), but it calibrates best (ECE 0.065) → kept as a
  "calibration fine-tuning" candidate for Stage 3
- Conclusion: Stage 3 (2B scale-up) uses the bigk recipe; mixture-ratio experiments (task weights) are left for Stage 2d/3

## 9.10 Stage 3: Scaling Up to Qwen3-1.7B (completed 2026-09-23)

bigk recipe unchanged, backbone 0.6B → 1.7B (34.2M LoRA trainable), 3000 steps.

| Metric | B_bigk 0.6B | **B_bigk 1.7B** |
|---|---|---|
| held-out acc | 0.538 | **0.565** (+2.7pt) |
| held-out NLL | 1.60 | **1.56** |
| held-out ECE | **0.071** | 0.094 (worse) |
| in-task acc | 0.823 | **0.854** (+3.1pt) |
| in-task ECE | 0.025 | 0.025 |

- All three held-out tasks rise: massive 0.573→0.585, banking77 0.517→0.525, emotion 0.525→0.585
- in-task improvements are broad: arc_easy 0.776→0.876, arc_challenge 0.636→0.748, mmlu subjects +0.2~0.35
- **The scale lever holds, but held-out calibration degrades (0.071→0.094)** → the core target of the next RLCD step

Engineering notes (pitfalls hit):
- 1.7B pure-bf16 training diverged at step ~400 (grad_norm 502) → fix: backbone bf16 +
  **LoRA/head kept in fp32 + forward autocast** (standard AMP/QLoRA practice), lr 2e-4→1e-4
- Mirror single-connection throttled to 2MB/s → **aria2c 16 connections ~10-80MB/s**, a 3.4GB model downloads in 5 minutes

## 9.11 Stage 4: RLCD Calibration (completed 2026-09-23)

Based on the Stage 3 (1.7B) checkpoint, 800 steps, belief:decision data ratio 0.4:0.6.
Loss = belief log-score (soft targets, known probability laws) + REINFORCE action (hard labels,
reward=correctness, baseline=model's own probability) + KL(policy‖reference policy, w=0.1).

Data: 6 classes of synthetic belief tasks (coins/dice/cards/ball draws/multi-dice sums), 20k training +
1k evaluation (same templates, unseen parameters).

| Metric | SFT 1.7B | **RLCD 1.7B** |
|---|---|---|
| held-out acc | 0.565 | **0.574** |
| held-out NLL | 1.56 | **1.55** |
| held-out ECE (T=1) | 0.280 | **0.124** |
| held-out ECE (after temperature scaling) | **0.094** | 0.111 |
| in-task acc | 0.854 | 0.854 |
| in-task ECE (T=1) | **0.027** | 0.087 |
| belief excess_nats | — | **0.0019** (decider v10: 0.22) |
| belief_ece | — | **0.0048** |

Conclusions:
- **The belief objective fully works**: excess 0.0019 nats on known-probability-law tasks (near perfect)
- **T=1 calibration on held-out improves substantially** (ECE 0.280→0.124), showing the model learned to "report probabilities honestly";
  acc/NLL also improve slightly
- **Cost**: in-task calibration is perturbed (0.027→0.087); after temperature scaling, held-out ECE is slightly worse than SFT
  (0.111 vs 0.094) → belief training lowers confidence overall, while SFT can be "rescued" by temperature fitting
- Next (RLCD v2): lower the belief ratio (0.4→0.2), raise the KL weight, or fit
  temperatures separately per task family; more ideal belief data should be close to the real decision distribution (not just synthetic laws)

## 9.12 Stage 4b: RLCD v2 (practicality first, completed 2026-09-23)

Three upgrades: ① GoEmotions 3-annotator vote distributions (8000 examples) as **real-ambiguity** belief
data (+5000 synthetic-law validation) ② abstention training (20% of decision examples have gold removed + "none of the
above") ③ calibration protection (belief ratio 0.4→0.25, KL weight 0.1→0.3). 1000 steps.

| Metric | SFT | RLCD_v1 | **RLCD_v2** |
|---|---|---|
| held-out acc | 0.565 | **0.574** | 0.559 |
| held-out ECE (after temperature scaling) | 0.094 | 0.111 | **0.050** |
| in-task ECE (after temperature scaling) | **0.025** | 0.031 | 0.037 |
| belief excess (synthetic + real ambiguity) | — | — | **0.030** |
| belief_ece | — | — | **0.005** |
| Correct abstention rate (gold absent) | — | — | 0.52 |
| **False abstention rate (gold present)** | — | — | **0.35 ⚠️** |
| Accuracy after adding none | — | — | 0.396 (down from 0.565) |

Conclusions:
- **Honest probabilities on real ambiguity hold**: excess 0.030 nats, belief_ece 0.005
- **Best held-out calibration after temperature scaling** (0.050), with clear per-task temperature differences
  (banking77 0.5 / emotion 0.4 / massive 0.45 → all need sharpening, indicating overall underconfidence)
- **Abstention is too aggressive**: a 0.35 false-abstention rate is not production-usable → threshold calibration for the none option is needed
  (fit a none bias on the validation set, push the false-abstention rate down to ≤5-10%, and maximize correct abstention)

Next steps (RLCD v3 / practicalization):
1. none threshold calibration (post-processing, no retraining needed)
2. abstention_ratio 0.2→0.1, or train abstention on data closer to the deployment distribution
3. Keep v1 as the "pure decision" version (highest accuracy) and v2 as the "calibration + abstention" version

### 9.13 none Threshold Calibration Results (2026-09-23)

Swept the none-logit bias on RLCD v2 (600 held-out examples, constructing gold-present /
gold-absent sets):

| False-abstention budget | bias | Correct abstention rate |
|---|---|---|
| ≤2% | -1.7 | 1.5% |
| ≤5% | -1.3 | 6.0% |
| ≤10% | -0.9 | 16.0% |
| ≤20% | -0.5 | 28.5% |

Conclusions: **the discrimination of an embedded none option in the softmax is weak**—once false abstention is pushed down, correct abstention is nearly zero,
showing the model did not truly learn "the answer is not among the candidates", it is merely biased toward none overall.

Fix directions (v3, by priority):
1. **Separate abstention head** (recommended): make "is the answer in the candidate set" a separate binary gate
   (drop-gold data + BCE + Platt threshold calibration), decoupled from "which option to pick"
   (the standard approach in selective classification / open-set settings)
2. Expand abstention training data (higher ratio, more steps, stratified construction by task difficulty)
3. Construct abstention examples from the real deployment distribution (currently gold is sampled randomly, too uniform)

## 9.14 RLCD Enhancement Plan (literature review 2026-09-23)

Sources:
- decoding-jev (reverse-engineering of the public Jev/Laya notebook): the RLCD implementation =
  belief reward **log score + 0.75 × spherical score**, multiple candidates sampled from noised logits,
  REINFORCE with a within-group mean baseline, a full-weight CE term, and post-hoc temperature fitting;
  and it warns that "the sampling objective ≠ the inference objective; a proper reward does not guarantee deployment calibration", recommending controlled ablations
- RLCR (arXiv 2507.16806): ordinary RL with binary rewards **harms calibration**; adding a
  bounded proper score (Brier) to the reward improves both accuracy and calibration; better than a post-hoc confidence classifier
- Mozannar & Sontag 2020 (Learning to Defer): the failure of confidence-threshold methods is theoretically known;
  one needs to **jointly train a separate rejector** (defer as a new class, CE over Y∪{⊥} is consistent)
- SelectiveNet (Geifman & El-Yaniv 2019): predict/select two heads + coverage constraint
- Conformal selective prediction: distribution-free thresholds + finite-sample coverage guarantees

Mapping to our measured weaknesses (§9.13 weak none discrimination) and improvements:

| Priority | Action | Expectation |
|---|---|---|
| P1 | Separate rejector head + joint training + conformal/cost-sensitive threshold | Correct abstention 6%→30-50% under a 5% false-abstention budget |
| P2 | belief reward: log + 0.75·spherical; add a Brier term to the action reward | RL no longer harms calibration |
| P3 | Controlled ablation (deterministic proper score vs sampled REINFORCE vs ±CE) | Confirm transferability of the training objective |

## 9.15 Selective Prediction: Confidence Ranking and Risk Guarantees (2026-09-24)

Literature pivot (per §9.14):
- Selective classification benchmark (44 datasets / 18 methods): dedicated reject heads (SELNET/DG) are
  not optimal; **softmax-response threshold + ensemble** is a strong baseline
- Measured confirmation: v3's separate rejector head brings only marginal improvement (6%→8.5% under a 5% budget),
  and held-out acc −4pt → **drop the rejector route, switch to threshold calibration**
- Theoretical correction: the set-coverage guarantee of split-conformal **cannot** directly constrain the "error rate of the accepted subset";
  switch to a PAC-style selective-risk upper bound (Hoeffding):
  `risk_hat(t) + sqrt(log(1/delta)/(2 n_acc)) <= alpha`

**Core measurement (risk-coverage, held-out unseen 60/77-class tasks)**:

| Selective risk at 10% coverage | SFT | RLCD_v1 | RLCD_v2 | RLCD_v3 |
|---|---|---|---|---|
| risk | 0.21 | 0.21 | **0.03** | **0.07** |

→ RLCD's belief training significantly improves **confidence ranking**: taking only the most confident 10%,
v2/v3 reach 97%/93% accuracy (SFT only 79%).

Key engineering conclusions:
- **Guarantee feasibility is dominated by calibration-set size**: the Hoeffding correction `sqrt(log(1/delta)/(2n_acc))`
  is 0.14 at n_acc=75 → with a 1500-example calibration set, α=0.2 is just infeasible (minimum upper bound 0.21)
- Measured diagnostics (v3, 1500 calibration examples, T=0.80):

| Coverage | n_acc | Empirical risk | PAC upper bound |
|---|---|---|---|
| 2% | 30 | 6.7% | 0.29 |
| 5% | 75 | **6.7%** | 0.21 |
| 10% | 150 | 12.0% | 0.22 |
| 20% | 300 | 23.7% | 0.31 |

- **Practical operating point (empirical)**: on unseen 60/77-class tasks, take the most confident 5% → 93.3% accuracy
  (SFT under the same protocol is only ~79%); this is a usable form of "high-precision mode + escalate to human"
- A formal PAC guarantee requires: a calibration set ≥5k (n_acc in the hundreds) + higher precision in the high-confidence regime;
  Hoeffding is conservative and can be tightened with exact binomial / CRC
- Conclusion: **RLCD's belief training makes selective prediction usable** (confidence-ranking quality is the premise),
  and thresholding itself does not need a rejector head (v3's rejector gives only marginal gains and hurts accuracy)

## 9.16 RLCD v3 Ablation: Effective and Ineffective Components (2026-09-24)

Cohort experiment (all based on SFT 1.7B, 1000 steps, rejector head, held-out after temperature scaling):

| Variant | held-out acc | ECE | NLL | Abstention@5% budget (correct rate) |
|---|---|---|---|---|
| v3 (full: CE+belief+action+KL) | 0.519 | **0.064** | **1.633** | 8.5% |
| v3b (no action RL) | 0.521 | 0.072 | 1.642 | 8.6% |
| v3c (abstention training 0.20→0.35) | 0.523 | 0.068 | 1.641 | 8.9% |
| v3d (belief without spherical, 500 steps) | to be tested | | | |

Conclusions (consistent with the literature):
- **action RL (REINFORCE) contributes almost nothing** → RLCR (RL with binary rewards does not improve calibration)
  and decoding-jev (sampling objective ≠ deployment objective) are reproduced in our measurements; belief + CE are the effective components
- **Raising the abstention training ratio does not improve rejector discrimination** → confirms §9.15 again:
  practical abstention should use "confidence thresholding + a large calibration set", not a rejector head
- The role of the spherical term awaits the v3d result (expected: faster belief convergence, little effect on final calibration)

## 9.17 Final Comparison and Recommended Configuration (2026-09-24 02:45)

All variants (1.7B + bigk, 3000-step SFT + RLCD stage), held-out = unseen
60/77-class intent/emotion tasks (1200 examples):

| Variant | held-out acc | ECE (fitted T) | in-task acc | Abstention@5% budget |
|---|---|---|---|---|
| SFT | **0.565** | 0.094 | 0.854 | — |
| **RLCD v2** (synthetic + real-ambiguity belief, abstention 0.2, KL 0.3) | 0.559 | **0.050** | 0.844 | — |
| RLCD v3 (+ separate rejector head) | 0.519 | 0.064 | 0.855 | 8.5% |
| v3b (no action RL) | 0.521 | 0.072 | 0.856 | 8.6% |
| v3c (abstention ratio 0.35) | 0.523 | 0.068 | 0.856 | 8.9% |
| v3d (no spherical, 500 steps) | 0.534 | 0.078 | 0.856 | 8.4% |

**Recommended configuration**:
1. **Accuracy first** → SFT 1.7B (held-out 0.565)
2. **Calibration first (recommended)** → **RLCD v2** (ECE 0.050, real-ambiguity belief excess 0.030,
   best confidence ranking: 93-95% accuracy on unseen tasks at top-5% coverage)
3. **Abstention / high-precision mode** → no rejector head; use v2's confidence + a threshold (with a large calibration set)
4. Things not needed (all with ablation evidence): rejector head, action RL, higher abstention ratio, spherical term

**Next step (Stage 5, incomplete)**: multimodal (Qwen3.5-2B VL + vision tower + text-path
isolation + 30% text replay + regression gate); training tasks 24→60+; full-FT comparison.

## 9.18 Engineering Lesson: Log Accumulation Division Bug (2026-09-24)

When recording loss, `Trainer` used `sum(all micro-batch losses) / window(optimization steps)`,
while each optimization step contains `accum` micro-batches → **logged values were inflated by accum×**.
Impact: the `choice_ce`/`grad_norm` logs of all historical runs were inflated (divide by accum for the true value).
The earlier 1.7B full FT was restarted twice due to a misjudged "divergence" of 2.95 vs SFT 1.29—the true values
0.37 vs 0.32, perfectly healthy (single-batch CE ranges 0.17~3.28; window-mean fluctuation is noise).
Fixed: divide by `window * accum`. Lesson: **unify accum before comparing loss across runs**.

## 9.19 Model Soup Experiment Log (2026-09-24, including bugs and conclusions)

Motivation (Wortsman et al., ICML 2022): **weight averaging** of fine-tuned models within the same error basin
can improve accuracy/robustness at zero inference cost; the greedy version adds models one by one if they help.

Protocol: candidates = {SFT, v1, v2, fullFT@1500}; LoRA is first merged into full weights (fp32);
selection set = first 3000 examples of stage1_heldout_big; test set = stage1_heldout 1200 examples (independent).

**Single models (selection set)**:

| Model | acc | ECE |
|---|---|---|
| SFT | 0.5453 | 0.096 |
| v1 | 0.5403 | 0.090 |
| v2 | 0.5327 | **0.067** |
| fullFT@1500 | **0.5607** | 0.164 |

**Greedy results**: seed = fullFT (0.5607) → **+SFT = 0.5670 ACCEPT** → +v1 = 0.0427
reject → +v2 = 0.0400 reject (the run then exited without writing a report)

**Two implementation bugs found**:
1. `evaluate_state` used the `lora=true` architecture to load **already-merged full weights** → key names did not match,
   so what was actually evaluated was the "base model + trained head" (v2 was misreported as 0.106; 0.533 after the fix).
   Fix: force `lora=false` when evaluating merged weights.
2. The greedy loop's baseline did not roll back after a reject (it used the previous attempt's acc instead of the current soup's acc).
   Not fixed (the experiment was aborted).

**Unexplained phenomenon**: weight-averaging trials that added v1/v2 had accuracy ≈ 0.04 (a NaN
signature of argmax always being 0; the single-model states themselves are normal at 0.53-0.54). Suspected to relate to memory pressure under 4 fp32 states (~28GB CPU) or a NaN in some tensor;
not localized.

**Conclusion**: even in the best case, the weight-soup gain is only
**{fullFT+SFT} 0.567 vs fullFT 0.561 (+0.6pt on the selection set)**, poor value for money.
Moreover, weight averaging is inherently sensitive to "cross-basin" models (fullFT vs the LoRA family).
→ Switch to a **probability-space ensemble** (Deep Ensembles, Lakshminarayanan 2017;
ENS+SR ranks first in the selective classification benchmark), requiring no weight alignment and robust across basins.

## 9.20 Probability-Space Ensemble Results (completed 2026-09-24)

Average of 4 model distributions (SFT / v1 / v2 / fullFT@1500), 1200 held-out + 1200 in-task examples:

**held-out (unseen 60/77-class tasks)**:

| Model | acc | NLL | ECE |
|---|---|---|---|
| SFT | 0.5608 | 1.572 | 0.097 |
| v1 | 0.5567 | 1.602 | 0.116 |
| v2 | 0.5425 | 1.584 | **0.039** |
| fullFT@1500 | **0.5933** | 1.596 | 0.201 |
| **ENSEMBLE** | **0.5750** | **1.455** | **0.061** |

**in-task**: ensemble acc 0.8467 (tied best), NLL 0.398 / ECE 0.032 (both best).

Conclusions:
- The ensemble's **NLL is 0.12 lower than the best single model** (the complementarity is real)
- **Accuracy close to fullFT (−1.8pt) + calibration close to v2 (+0.022)** → resolves the dilemma
- Cost: 4× inference. **Next first choice: distill using this ensemble as the teacher** (the root cause of v4's failure was a teacher
  that was too weak / highly correlated; the teacher ensemble's NLL is now clearly better than any single model, so distillation is worthwhile)
- By-product: this also explains v4—the teacher pool at the time {v1,SFT,v2,v3d} was highly correlated
  and lacked a heterogeneous member like fullFT

## 9.21 RLCD v5: Ensemble Distillation Back to a Single Model (completed 2026-09-24)

Recipe (literature-confirmed: Hinton 2015 original KD + Reich 2020 ensemble distillation):
- Teacher = arithmetic mean of the 4 model distributions (SFT/v1/v2/fullFT@1500)
- Student = 1.7B **full FT** (warm-started from merged SFT; match teacher capacity first)
- Loss = α·CE(gold, T=1) + (1−α)·T²·KL(p_student^T ‖ p_teacher^T), **T=2, α=0.5**
- Data = 24-task distillation set (66k, each example carries the teacher distribution) + belief 15%
- 3000 steps, lr 1e-5, 8-bit AdamW, bf16 checkpoint

**Final comparison (held-out = unseen 60/77-class tasks; in-task 1200 examples)**:

| Model | held-out acc | held-out NLL | held-out ECE | in-task acc |
|---|---|---|---|---|
| SFT | 0.561 | 1.572 | 0.097 | 0.847 |
| v1 | 0.557 | 1.602 | 0.116 | 0.840 |
| v2 | 0.543 | 1.584 | **0.039** | 0.843 |
| fullFT@1500 | **0.593** | 1.596 | 0.201 | 0.818 |
| Ensemble (4 models) | 0.575 | 1.455 | 0.061 | 0.847 |
| **v5_distill (single model)** | **0.581** | **1.451** | 0.068 | **0.863** |

Conclusions:
- **Distillation achieves "single model = ensemble quality"**: NLL 1.451 (best overall, even slightly better than the ensemble),
  acc 0.581 (beats all LoRA models, second only to fullFT), ECE 0.068 (close to the ensemble)
- **in-task acc 0.863 is the highest overall** (the regularization effect of distillation)
- Inference cost back to ×1 → v5 can serve as the deployment model
- The belief skill degrades slightly (excess 0.030→0.069), because belief is only 15% of KD training;
  add a short round of belief fine-tuning if recovery is needed

**Engineering lessons (pitfalls from this round, all fixed)**:
1. `rsync --delete` accidentally deleted the newly created remote data/distill5 (an exclusion rule was added)
2. RLCDTrainer lacked `save_dtype` → full-FT checkpoints at 6.9GB each filled the disk
   (bf16 is now supported, 3.4GB each)
3. RLCDTrainer lacked 8-bit optimizer support → fp32 Adam used 13.6GB and OOMed (adamw8bit is now supported)
4. KD mask bug: teacher probabilities at padding positions × the student's -1e9 mask = exploding loss (now masked by valid positions)
5. The v5 config's manifest once pointed to the original hard-label data + distill_ratio=0 → KD had no effect at all
   (now the manifest points directly to the teacher-distribution data)

## 9.22 MLX Conversion (Mac-side inference, completed 2026-09-24)

Artifacts: `models_mlx/v5_mlx/` (model.safetensors 3.47GB + config + tokenizer)
+ `mlx/decision_mlx.py` (pure mlx.core implementation, no torch needed)

Parity check (fixed ids / same example):
- logit: MLX **-10.278** vs torch -10.25 (difference 0.03)
- example probabilities: {tech 0.568, shipping 0.370, returns 0.054, billing 0.009}
  vs torch {0.562, 0.374, 0.056, 0.009}

**Key pitfall**: MLX `mx.fast.rope(traditional=False)` and HF's rotate_half convention are
**inconsistent** (causing a 3% hidden-state std deviation and a logit deviation of 6 after 28 layers) → a hand-written
rotate_half RoPE aligns perfectly. Recorded in the code comments.

Other implementation points: RMSNorm computes internally in fp32 matching HF convention; causal mask constructed explicitly;
GQA expanded via repeat; pool/head computed in fp32.

Usage:
```
python mlx/decision_mlx.py --model-dir models_mlx/v5_mlx \
  --state "..." --question "..." --options "a,b,c" --temperature 0.45
```

## 9.23 decider-2b Head-to-Head Evaluation (completed 2026-09-24)

Using their own `decider` package (eager mode, `use_graphs=False` to avoid per-shape compile
+capture), ran Mapika/decider-2b (2B, Qwen3.5 hybrid-attention backbone,
95-task training + 384 steps of calibration-aware RL, official temperature 1.3) on our two evaluation sets:

| Evaluation set | decider-2b | v5 (ours) |
|---|---|---|
| Our in-task (24 tasks, mostly trained by both) | 0.811 / 0.453 / 0.032 | **0.863 / 0.390 / 0.034** |
| Our held-out (banking77/massive/emotion) | **0.867 / 0.420 / 0.020** | 0.581 / 1.45 / 0.068 |

**Key facts (from the heldout flags in their eval_results.json)**:
- decider-2b **trained on our held-out tasks** (banking77/massive_intent/emotion
  are all heldout=false) → its 0.867 is an in-task score (official same-task scores 0.843-0.96)
- Our zero-shot score on the same tasks is 0.581 → not directly comparable
- **On the commonly trained in-task tasks, v5 leads decider-2b by +5.2pt** (1.7B/24 tasks
  vs 2B/95 tasks)

Conclusion: on overlapping tasks our model is stronger; their 0.4-0.98 scores on their held-out (28 tasks: paws/sciq/trec/
boolq/strategyqa/quality/ade/fin_sentiment, etc.) cannot be aligned with our
current numbers—**a fair comparison requires building a "zero-shot for both sides" suite** (selected from tasks where they have heldout=true
and that are not among our 24 tasks).

## 9.24 Fair Suite: Zero-Shot Generalization Comparison for Both Sides (completed 2026-09-25)

Built an 8-task suite that is "unseen by both sides" (all in decider-2b's heldout=true list and
none among our 24 training tasks), 300 examples each:
paws / sciq / pubmedqa / bbc_news / quality / fin_phrasebank / dolly_category /
xstory_cloze

| Task | v5 (ours) | decider-2b |
|---|---|---|
| paws | 0.450 | 0.443 |
| sciq | 0.947 | **0.973** |
| pubmedqa | 0.470 | **0.660** |
| bbc_news | **0.960** | 0.947 |
| quality | 0.303 | **0.433** |
| fin_phrasebank | 0.863 | **0.897** |
| dolly_category | **0.323** | 0.277 |
| xstory_cloze | 0.917 | **0.970** |
| **Macro-average acc** | **0.654** | **0.700** |
| NLL (after temperature scaling) | 0.86 | **0.71** |
| ECE (after temperature scaling) | 0.092 | **0.047** |
| Fitted temperature | 1.35 | 1.50 |

Conclusions (honest):
- **Zero-shot generalization: decider-2b leads by +4.6pt + better calibration**; the gap is concentrated in knowledge-intensive tasks
  (pubmedqa −19, quality −13), while we overtake on topic/instruction classification
- Sources of their advantage: 95 tasks (vs our 24) + a 2B Qwen3.5 hybrid-attention backbone +
  calibration-aware RL
- Contrast: on commonly trained tasks we lead by +5.2pt (§9.23) → **the "seen vs unseen" gap
  clearly points to training-task coverage breadth**, consistent with the 4→24 task conclusion of §9.8
- Next levers (by priority): ① tasks 24→60-95 (main lever) ② a stronger backbone (Qwen3.5-2B
  or larger) ③ long-context capability (our 256-token truncation is a real weakness on quality-type tasks)


## 9.25 Deployment Side: Batched Inference / ONNX / Salience Student Distillation (2026-09-25)

**Batched inference (MLX, Qwen3-1.7B v5)**:
- Candidate batching, one forward per batch: 117ms → **75ms/sentence (1.55x)**; probability consistency 0.003
- 4-bit (mlx_lm.convert): 3.2GB → **934MB** (saves memory, same speed)
- Conclusion: **a compute bottleneck, not a bandwidth bottleneck**; batching only improves utilization, speedups must come from quantization/smaller models
- Artifacts: `reader/models/v5_batched[_4bit]` (the original model `models_mlx/v5_mlx` is untouched)

**ONNX cross-platform** (CPU / CUDA / DirectML):
- Qwen3.5 **cannot be exported to ONNX** (its linear attention uses `linalg_solve_triangular`; both generations of exporter fail)
- The student model (standard ModernBERT architecture) exports smoothly: fp32 606MB → **int8 153MB**, numerical error 2e-5

**Salience student distillation** (reader scenario):
- Teacher = stage1 (Qwen3.5-2B/45 tasks) scores 5347 articles sentence by sentence (`--argmax-gold` soft labels)
- Student = ModernBERT-base (150M) full FT, KD(T=2, α=0.5) + 24-task replay + belief;
  5000 salience examples + 4000 steps, 37 minutes (0.56s/step)
- Evaluation: teacher agreement top1 **18%**, spearman **0.27** (random 10%) → rather weak, needs more data
- End-to-end (Mac CPU, int8): **131ms/sentence (14 sentences 1.9s)**; CUDA/DirectML expected 10-30ms/sentence

Conclusion: the ONNX pipeline (including int8) is verified for cross-platform deployment; student quality needs more data
(5000→20000) and a higher salience weight (1.5→3.0) for another training round.

## 9.26 Pure-CPU 5-Second Plan: Segmented Whole-Document Encoding + Sentence Span Pooling (2026-09-25)

**Measurements refuted the assumption of "whole document 8192 in one forward"** (Mac M5 CPU, ModernBERT-base int8,
`onnxruntime 1.30` CPU EP):

| Input | Time | Throughput |
|---|---|---|
| Single 4096 token | 6.1 s | 0.46k tok/s |
| Single 8192 token | 75 s | 0.11k tok/s |
| 8 × 512 token batch | 2.3 s | 1.8k tok/s |

The reason is that attention grows quadratically on long sequences and is memory-bound; ModernBERT's local attention only enters the
linear efficient regime on short sequences ≤512. **Conclusion: the whole document must be cut into ~512-token segments and batched through the graph**,
rather than concatenated into one long sequence.

**New architecture (student)**: encoder (MiniLM/ModernBERT) → greedily pack sentences into 512-token segments →
sentence span mean-pool → LayerNorm/MLP head → per-sentence logit. A single ONNX graph
(`input_ids + attention_mask + span_mask[B,S,T] → logits[B,S]`), after int8
MiniLM-L6 is only **22.9MB** (the old cross-encoder was 153MB).

**Training** (`scripts/train_salience_doc.py`): teacher-probs soft-label KL (T=2) +
CNN/DailyMail ROUGE-1≥0.5 binary BCE (pos_weight 3), KD:BCE = 1:1,
20000 steps, full FT, 512-token segments; three backbones trained in parallel (5090 32G has ample headroom).

**Teacher agreement (347 held-out teacher probs; compared with the old deployed model on the same eval set;
torch best = evaluation by the training script, int8 = quantized ONNX)**:

| Model | top1 | spearman | KL | ROUGE AUC | int8 size |
|---|---|---|---|---|---|
| Old deployed cross-encoder (ModernBERT) | 0.49 | 0.587 | 0.283 | — | 153MB |
| MiniLM-L6 whole-document | 0.50 / int8 0.48 | 0.55 / int8 0.55 | 0.40 | 0.80 | 23MB |
| MiniLM-L12 whole-document | 0.545 / int8 0.53 | 0.609 / int8 0.60 | 0.34 | 0.83 | 34MB |
| ModernBERT whole-document | 0.608 / int8 0.57 | 0.680 / int8 0.68 | 0.31 | 0.84 | 152MB |

**CPU end-to-end (M5, int8, 8000 characters)**: Chinese 7869 tokens / 16 segments →
MiniLM-L6 **0.55s** (14k tok/s), MiniLM-L12 **1.08s**, ModernBERT **4.9s**
(the Chinese tokenizer splits more finely, 12.3k tokens); English 8000 characters → MiniLM-L12 0.25s,
ModernBERT 0.52s. The old cross-encoder measured 6.2s for 140 sentences on the same machine (≈8-11s for 8000 characters).
The int8 vs torch quality gap is ≤0.01 (spearman); the primary deployment choice is **MiniLM-L12** (CPU-safe),
with ModernBERT as the quality option (GPU / short documents).

**Deployment**: reader gains a `salience` backend (`reader/salience_model.py`,
auto-selected `models_onnx/salience_*/model.int8.onnx`), segment packing is done on the inference side;
app end-to-end measured 8000 Chinese characters at **1.29s** (including HTTP and rendering). Known trade-off: the whole-document model is
goal-free (the teacher query is fixed to "understanding this text"), and the reader's goal
input is disabled under this backend; keeping it would require regenerating goal-conditioned teacher data.
Artifacts: `models_onnx/salience_{minilm,minilm12,modernbert}`,
on-device reader release v1.1.0 (minilm12 default 34MB + modernbert optional).

## 9.27 Qwen3.5-2B Teacher Model (stage1_qwen35): Training, Evaluation, and Positioning (2026-09-26)

**Role**: a 45-task general decision model + soft-label teacher for the salience student (§9.25/§9.26's
`probs` in `salience_train.jsonl` were generated by it scoring 5347 articles sentence by sentence).

**Training config** (`configs/stage1_qwen35.yaml`):
- Backbone `Qwen3.5-2B-Base-Text`, **frozen** + LoRA r=32/α=64 (only adapters trainable, bf16)
- route cx (explicit candidate scoring + attention pooling), sdpa + grad checkpointing
- Data: 45 tasks (8 MMLU subsets weighted 0.5, the rest 1.0) + belief 15%;
  collator 256/96/64 tokens, 2-16 candidates
- 5000 steps, batch 4 × grad_accum 4 (effective 16), lr 2e-4, warmup 200, cosine,
  label smoothing 0.05, spherical 0.75, belief 1.0, ce 1.0
- Training time **14,315s ≈ 4.0h** (2.86s/step); train loss: CE 4.75→1.64,
  belief 0.66→0.29, total 5.42→1.92
- Artifacts: `runs/stage1_qwen35` (LoRA adapter ~290MB, ckpt_last/2000/4000, bf16)

**Evaluation (unified temperature-fitting protocol)**:

| Evaluation set | T=1 acc / NLL / ECE | fitted acc / NLL / ECE | fitted T |
|---|---|---|---|
| in-task (`stage1_v2_intask`, 45 tasks) | 0.754 / 0.664 / 0.044 | 0.754 / 0.649 / 0.036 | 1.3 |
| held-out (`stage1_heldout`, unseen 60/77 classes, 1200 examples) | 0.583 / 1.640 / 0.238 | 0.583 / 1.458 / 0.093 | 0.6 |
| fair suite (zero-shot for both, 8 tasks × 300) | 0.643 / 0.906 / 0.069 | 0.643 / 0.864 / 0.075 | 1.55 |

**Same-protocol comparison with v5** (same held-out, same temperature fitting): acc 0.583 vs 0.581,
NLL 1.458 vs 1.451, ECE 0.093 vs 0.068 —— **essentially tied, v5 calibrates slightly better**;
fair suite 0.643 vs 0.654 (+1.1pt, noise-level on 2400 examples).
Note that in-task 0.754 **cannot** be compared directly with v5's 0.862 (different eval sets: 45 tasks
vs 24 tasks). The earlier statement that "3.5 is worse" came from a mixed protocol (comparing v5's fitted numbers
against 3.5's T=1 numbers); this is hereby corrected: **the two are tied in zero-shot, each with its own strengths**.

**Fair suite vs decider-2b (per-task acc)**:

| Task | stage1 (3.5-2B) | v5 (1.7B) | decider-2b |
|---|---|---|---|
| paws | 0.447 | 0.450 | 0.443 |
| sciq | 0.847 | 0.947 | **0.973** |
| pubmedqa | 0.573 | 0.470 | **0.660** |
| bbc_news | 0.933 | **0.960** | 0.947 |
| quality | 0.367 | 0.303 | **0.433** |
| fin_phrasebank | 0.863 | 0.863 | **0.897** |
| dolly_category | 0.250 | **0.323** | 0.277 |
| xstory_cloze | 0.860 | 0.917 | **0.970** |
| **Macro average** | **0.643** | **0.654** | **0.700** |

3.5 did not win a single task on this suite (paws is tied), trailing decider by 5.8pt overall;
however, on raw T=1 calibration 3.5 is actually better than decider (ECE 0.069 vs 0.113),
and decider only overtakes after fitting with its official temperature 1.3 (0.047 vs 0.075).

**Why the 2B did not show a scale advantage (analysis)**:
1. **Capacity bottleneck of LoRA + frozen backbone** (already measured in §9.5) → in effective capacity,
   "full-FT 1.7B > LoRA 2B"; most of the 2B's pretrained knowledge is not exploited by task adaptation
2. Training signal difference: 45-task hard labels vs v5's 24-task ensemble soft-label KD + RLCD
3. Backbone architecture: Qwen3.5's hybrid/linear attention adapts only so-so to "hidden + attention pooling +
   linear head", and it cannot be ONNX-exported (§9.25, `linalg_solve_triangular`)
4. The extra 21 tasks did not translate into a zero-shot advantage (fair 0.643 < v5 0.654),
   which does not contradict §9.8's conclusion that "task coverage is the main lever"—coverage must be paired with sufficient trainable capacity

**Release positioning**: not promoted as the general-purpose model (v5 is better and has a complete set of deployment forms); kept as the
salience student's teacher and a reproduction source; if released separately, only the LoRA adapter (290MB) would be released, with the backbone
`Qwen3.5-2B-Base-Text` and license declared (to be verified).
