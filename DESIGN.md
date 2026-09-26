# Route B: Explicit Candidate-Scorer Decision Model — Design

> Goal: train a text decision model that "reads state + a runtime-arbitrary question + an arbitrary candidate set, and outputs a calibrated probability distribution".
> This design corresponds to Route B (explicit candidate scorer) of Jev / System One, with Mapika/decider's Route A (LM-head letter logits) as the comparison baseline.

> The complete record of training/evaluation/deployment is in `TRAIN-REPORT.md` (the design document keeps only the design itself).

---

## 0. Background and Positioning

### 0.1 Task Definition

The model does no text generation; it only outputs probabilities over the candidate set given at runtime:

```
F(state, question, candidates) → P(candidate | state, question)
```

- `state`: arbitrary context (string / JSON / object / array, training limit ~8k tokens, inference supports 16k–32k)
- `question`: a semantic question defined arbitrarily at runtime (category name, criterion, rating dimension)
- `candidates`: semantic objects defined arbitrarily at runtime (2–255), supporting plain strings or objects with a description
- Output: Choice (softmax distribution) / Score (independent probability per level) / Noul (independent yes-no per criterion)

The vocabulary is defined dynamically at inference time; schemas unseen during training must be usable — this is the core generalization goal of this project.

### 0.2 Differences from Route A

Route A (validated by decider): write candidates into the prompt, take the LM-head logits of the A/B/C letter tokens at `Answer: (`, and softmax with temperature.

Route B (this design): explicitly encode the candidates, score with a learnable interaction layer, then normalize.

| | Route A | Route B |
|---|---|---|
| Candidate representation | compressed into the embedding row of a single letter token | each candidate has a full contextual representation |
| State reuse | recomputed on every forward | encoded once, cached and reused |
| Order bias | causal LM is position-sensitive | structurally permutation-invariant |
| Independent scores | none (only a normalized distribution) | sigmoid independent scores, supporting thresholds/two-stage |
| Long option description | prompt bloat, information bottleneck | directly used as candidate encoding input |
| Multimodal extension | must be a generative VLM | swap the state encoder; candidates remain text |

### 0.3 Hypothesis to Validate (top priority of this project)

> With the same data and the same order of parameter count, is an explicit candidate scorer better than LM-head readout on **held-out schema generalization** and **calibration**?

The Stage 1 A/B controlled experiment exists to answer this one question. If B is not better than A, the project promptly shrinks back to engineering polish of Route A.

---

## 1. Core Abstractions

### 1.1 Scoring and Normalization

```
h_S  = Enc_state(S)                      # [B, Ls, d]
h_Q  = Enc_question(Q_i)                 # [B, Q, Lq, d]
h_C  = Enc_candidate(C_ij)               # [B, Q, K, Lc, d]
s_ij = Scorer(h_S, h_Q, h_C)             # [B, Q, K] scalar logit
```

Two normalizations, chosen by question type:

```
Choice :  P_ij = softmax_j( s_ij / T_choice[task_type] )         # within-group relative probability
Indep  :  σ_ij = sigmoid( s_ij / T_indep[task_type] )            # absolute probability
```

Semantic energy function perspective: a larger `E(S, Q, C)` means the candidate fits the current question better; softmax is a conditional distribution, sigmoid is a per-candidate marginal probability.

### 1.2 Two-Stage for Large Candidate Sets

Corresponding to the official 255 options claim:

```
Stage 1  independent scoring:  a_k = sigmoid(s_k)        k = 1..N
Stage 2  Top-K:                softmax( s_topk / T )      K = 8~16
```

Probabilities outside Top-K are zeroed (or merged into an explicit "none of the above" bucket). Training already uses sub-sampled candidates (see §3.2) to match this inference form.

### 1.3 Permutation Invariance

CandidateEncoder encodes each candidate independently, Scorer scores each candidate independently, and there is no attention interaction between candidates → permutation-invariance is structurally guaranteed. This also eliminates the order augmentation Route A needs (but the data layer still does label shuffle to match upstream habits).

---

## 2. Architecture Design

### 2.1 Overall Structure

```
                 STATE (text/JSON)
                      │
                 StateEncoder
                      │
              state latents  [B, Ls, d] ────────────────┐
                      │                                 │
              QuestionEncoder                           │
                      │                                 │
             question latents [B,Q,Lq,d] ──────┐        │
                      │                        │        │
     candidates C_1..C_K                       │        │
                      │                        ▼        ▼
              CandidateEncoder        CrossAttentionScorer
              (Siamese, shared)       (cand as query, [state;question] as kv)
                      │                        │
                      └────────────────────────┤
                                               ▼
                                        Pooling + MLP
                                               │
                                    s_ij [B, Q, K] scalar logits
                                               │
                          ┌────────────────────┼───────────────────┐
                          ▼                    ▼                   ▼
                    Choice softmax       Score sigmoid       Noul sigmoid
```

### 2.2 Encoder Organization (three variants, recommended to adopt by stage)

| Variant | Structure | Trainable parameters (at 0.6B) | Use |
|---|---|---|---|
| V0 shared | single encoder + role embedding (three roles: state/question/candidate), fully frozen | ~50–150M (scorer only) | Stage 1 hypothesis validation, fastest |
| V1 two-tower | one shared by State+Question, one for Candidate (initialized from the same ckpt) | ~150–300M | Stage 2 default |
| V2 three-tower | independent encoder for each of the three | full parameters | Stage 3 upper-bound exploration (optional) |

Recommended configuration:
- Backbone candidates (text, 0.5B–2B): `ModernBERT-base/large` (bidirectional, long context, high encoding efficiency) or `Qwen3-0.6B/1.7B` (stronger semantic prior, but the causal mask is unfriendly to encoding and the attention mask must be changed)
- Stage 1 gets things working with the V0 frozen backbone; Stage 2 then decides whether to unfreeze and adopt V1

### 2.3a Known Issue and Literature Basis: Representation Anisotropy (measured 2026-09-23)

Measurement: frozen ModernBERT's candidate pooled representations have pairwise cosine similarity **0.975** (norm≈31),
and the shared component dominates the gradient → training collapses to a uniform distribution (loss stalls at ln K).

Literature:
- Ethayarajh 2019: contextual representations are highly anisotropic; the average cos between any two words is ≈0.99
- Su et al. 2021 "Whitening Sentence Representations": the mature solution is whitening /
  removing the dominant principal component; Arora et al. 2017 and Mu & Viswanath 2018 share the same idea
- SimCSE (Gao et al. 2021): contrastive objectives flatten the representation space and improve uniformity
- RankT5 / monoBERT: the scoring head is trained directly with listwise softmax / ranking loss

Design decisions:
- For choice/score rows, the scorer performs **per-question candidate mean centering** (`center_pooled`),
  i.e. "removing the dominant principal component" within the set; noul rows keep the absolute component (independent probabilities need absolute information)
- The loss keeps using listwise softmax CE (= RankT5's listwise ranking loss)
- Later optional: fixed/learned whitening, SimCSE-style contrastive regularization (re-evaluate in Stage 2)

### 2.3b Architecture Decision: Default Changed to Cross-Encoder Scorer (measured 2026-09-23)

Measurement conclusion (300-step controlled comparison on a 4-task mixture):

| Variant | Single task | 4-task mixture |
|---|---|---|
| late-interaction + centering | learns (loss 0.56@200) | **collapses** (loss≈chance, gradient→0) |
| per-task grouped batching | - | slightly better on 2 tasks, still poor on 4 tasks |
| **cross-encoder (frozen backbone)** | - | **learns normally** (3000 steps, in-task acc 0.608 / ECE 0.034) |

Cause: late-interaction conditions too weakly on the question text (the question is only in the kv and the cos feature, and
the pooled representation is dominated by the state); with mixed tasks the shared scorer cannot route by question text, and gradients conflict with each other
(the negative transfer described by Kendall 2018 / CAGrad 2021).

Solution (RankT5/monoBERT style): `[state; question; candidate]` concatenated into the encoder →
attention pooling → scalar score → listwise CE. Default `route: cx`.

- `center_pooled` is still kept for the late-interaction variant (effective on a single task)
- late-interaction's state-caching efficiency advantage is left for further optimization in Stage 2/3
  (it needs stronger question conditioning, e.g. injecting the question text as a query into attention)

### 2.3 Scorer Interaction Strength (core ablation variable)

| Level | Method | Cost | Notes |
|---|---|---|---|
| L1 late-interaction | MaxSim: `max_i max_j (c_i · [s;q]_j)` | lowest | ColBERT-style, usable for short candidates |
| L2 cross-attention (default) | candidate tokens as query, state+question latents as kv, 2–4 layers | medium | FiD-style; a candidate can re-read the parts of the state relevant to itself |
| L3 full cross-encoder | `[S; Q; C]` concatenated through a full transformer | highest | compared with L2 on the same data, as an upper-bound reference |

L2 concrete structure:

```python
class CrossAttentionScorer(nn.Module):
    # layers: 2-4 x (MHA cross-attn + FFN + LayerNorm)
    # query: candidate latents [B*Q*K, Lc, d]
    # key/value: concat(state, question) latents [B*Q, Ls+Lq, d]
    # pooling: attention pooling or [CLS]; then concat a dot-product similarity feature
    # head: MLP(d -> 4d -> 1)
    def forward(self, h_cand, h_cond, cand_mask, cond_mask) -> logits  # [B, Q, K]
```

Design points:
- The scorer is shallow per layer (2–4 layers); most computation is in the encoder, making inference cost easy to control
- Cross-state reuse: `h_S` is computed only once per state; all Q×K scoring for the same state is fully batched
- During training Q≤16, K≤10; during inference Q is unlimited, K≤255 (using the §1.2 two-stage scheme)

### 2.4 Output Heads

A single scalar head + task-type embedding, rather than building a separate head per task:

```python
logit = MLP(concat[pooled, task_type_emb, dot_feature])
```

- `choice`: masked softmax (invalid candidates set to -inf)
- `score`: each level goes through its own sigmoid (the level number and neighbors do not enter the scoring, ensuring level isolation)
- `noul`: one independent sigmoid per criterion

### 2.5 Calibration Layer

- Temperature is fitted separately per task family (decider uses 1.08 / 1.30 as reference, showing temperature is a necessary step)
- Temperature fitting data: in-task regression split (the same protocol as decider, ensuring comparability) + additionally report uncalibrated metrics
- Training-time soft labels: label smoothing ε≈0.05–0.1 or mixup with a uniform distribution, explicitly suppressing overconfidence
- Independent probabilities of Noul/Score are calibrated separately with Platt scaling / sigmoid temperature

### 2.6 Recommended M1 Configuration

```yaml
# configs/mini_b0.yaml
backbone: answerdotai/ModernBERT-base    # or Qwen3-0.6B
frozen_backbone: true                    # V0
role_embeddings: [state, question, candidate]
scorer:
  type: cross_attention                  # L2
  layers: 3
  heads: 8
  pooling: attention
  dot_feature: true
heads:
  choice: masked_softmax
  score: independent_sigmoid
  noul: independent_sigmoid
train:
  max_state_tokens: 8192
  max_question_tokens: 128
  max_candidate_tokens: 64               # 128 when the description is long
  questions_per_example: 1
  candidates_per_example: [2, 10]        # subset sampling cap (aligned with decider)
  loss: [ce, bce]                        # + label_smoothing 0.05
  optimizer: adamw
  lr: 1e-3                               # scorer only (frozen backbone)
  batch_states: 32
eval:
  metrics: [acc, nll, brier, ece_15bin, acc_at_80, aurc]
```

---

## 3. Data Design

### 3.1 Unified Data Format

```json
{
  "task": "ticket_routing",
  "task_type": "choice",
  "state": "I ordered a laptop three weeks ago...",
  "questions": [
    {
      "text": "Which department should handle this?",
      "options": [
        {"id": "returns",  "text": "returns",  "description": "exchanges, wrong items"},
        {"id": "billing",  "text": "billing",  "description": "payments, invoices"},
        {"id": "shipping", "text": "shipping", "description": "delivery problems"}
      ],
      "answer": "returns"
    }
  ]
}
```

- `choice`: answer is an option id
- `score`: options are levels (1–5), answer is a level; each level is independently judgeable
- `noul`: question is a criterion, options are the objects to be judged, answer is yes/no
- One example can carry multiple questions (the same state shares its encoding); during training Q=1 in general

### 3.2 Training Example Construction

1. **Candidate source**: the dataset's full original label space serves as the candidate pool (gold must be included)
2. **Subset sampling**: each example randomly draws `K ∈ [2, 10]` candidates; gold is always kept
3. **Order**: candidate order is still randomly shuffled during training (a defensive practice, even though the structure is invariant)
4. **Hard negatives (optional enhancement)**: use a teacher LLM to generate candidates semantically close to gold but incorrect, raising discrimination difficulty (preferred for QA/intent tasks)
5. **Textualization**: `option = text + " : " + description` (when description supports objects/arrays, serialize to compact JSON)

### 3.3 Task Mixture (start with 20–30 classes at Stage 0, target 95 classes)

| Task family | Data source (examples) | Suggested weight |
|---|---|---|
| Text classification/topic | AG News, BBC, Dolly categories | 10% |
| Sentiment/emotion | SST2, Financial PhraseBank, tweet irony | 10% |
| NLI/entailment | MNLI, SNLI, CommitmentBank, PAWS | 10% |
| Intent/tool selection | CLINC, MASSIVE, Hermes tool selection | 15% |
| QA/multiple choice | ARC, SciQ, MMLU subset, QuALITY | 15% |
| Fact checking/moderation | FEVER, Hateful Memes (text part), moderation | 10% |
| Preference/reward | RewardBench, arena preferences (3-way) | 10% |
| Refusal/absence | abstention construction (choose "none of the above" when gold is not among the candidates) | 10% |
| Rating/ordinal | CR reviews, ADE (1–5) | 10% |

### 3.4 Held-out Generalization Set (never appears in training)

The decider evaluation list is copied verbatim to ensure comparability:

```
TREC, BBC news, PAWS, SciQ, Social IQa, StrategyQA, PubMedQA, TruthfulQA,
tweet irony, financial sentiment, ADE, MASSIVE scenario, student question
categories, Dolly categories, CR reviews, Financial PhraseBank,
CommitmentBank, QuALITY, XStoryCloze, RewardBench, arena preferences,
Hermes tool selection, abstention probe
```

Plus two self-built probe categories:

1. **Unseen-schema probe**: a new class set for a training task (e.g. replacing ticket departments with hospital departments), where all candidate names are unseen
2. **Distractor probe**: insert highly semantically similar near-synonyms among the candidates to test discrimination resolution

### 3.5 Data Pipeline Structure

```
datasets → converters (one converter per dataset) → decision tuples (jsonl, unified format)
        → mixture sampler (weighted sampling + subset sampling) → collate → train
```

The converter output is the audit point: each dataset can be converted, validated, and counted with a single command, avoiding a black box.

---

## 4. Training Plan

### 4.1 Stage 0: Pipeline and Evaluation Baseline (no training)

- Data conversion + mixture + collate + evaluation harness
- **Evaluation harness parity check first**: directly evaluate the public `Mapika/decider-2b`; if the in-task/held-out metrics we reproduce match the model card, the harness is trustworthy
- Build the held-out schema probe and the abstention probe

### 4.2 Stage 1: Route B-Mini (core validation)

- Freeze the backbone (V0), train only the scorer + heads (~50–300M)
- Train **Route B** and the **Route A readout** on the same data (same backbone, same prompt format, letter logits)
- Produce a comparison report: acc, NLL, Brier, ECE, acc@80, AURC on in-task / held-out
- Ablations: L1 vs L2 vs L3; shared vs two-tower; softmax vs sigmoid; label smoothing on/off
- Decision gate: if B is no worse than A on held-out or calibration, proceed to Stage 2; otherwise shrink

### 4.3 Stage 2: End-to-End Fine-Tuning

- Unfreeze the backbone (V1 two-tower), full-parameter or LoRA, 0.6B–2B
- Expand data to 50–95 task classes
- Training recipe: CE (choice, label smoothing 0.05) + BCE (score/noul) + optional InfoNCE (representation shaping)
- Use held-out NLL as the early-stopping metric (not training loss)

### 4.4 Stage 3: Calibration and Two-Stage

- Fit temperature per task family; apply Platt scaling to the independent heads
- Implement Top-K two-stage inference (K=8/16) and evaluate latency and accuracy with 255 candidates
- Goal: ECE ≤ 0.03 (in-task), ≤ 0.08 (held-out), matching the decider-2b v10 level

### 4.5 Stage 4: RLCD-Style RL (optional)

- PPO action objective (verifiable environments, such as game/browser actions)
- log proper score of belief (checked against known probability laws)
- option-order consistency, KL regularization
- Refer to decider's `docs/RL.md` (v10 has proven effective: belief excess 0.22 nats vs v8 0.47)

### 4.6 Loss and Objective Summary

```
L = L_choice + L_score + L_noul + λ·L_aux

L_choice = CE( softmax(s/T) , y )                    # proper scoring rule family
L_score  = Σ_l BCE( sigmoid(s_l/T) , 1[y ≥ l] or 1[y = l] )
L_noul   = Σ_c BCE( sigmoid(s_c/T) , y_c )
L_aux    = InfoNCE( h_C, h_S )  (optional, small weight)
```

Key principle: **the main loss must be a proper scoring rule** (log score / Brier), otherwise the model has no incentive to report probabilities honestly.

---

## 5. Evaluation Plan

### 5.1 Metric Definitions

| Metric | Definition | Notes |
|---|---|---|
| Accuracy | correct when gold has the highest probability | |
| NLL | -log P(gold) | lower is better |
| Brier | Σ(p_i - y_i)² | multi-class |
| ECE | 15 bins, |acc - conf| weighted average | core calibration metric |
| acc@80 | accuracy on the most confident 80% of decisions | selective prediction |
| AURC | area under the risk-coverage curve | selective performance across all thresholds |

### 5.2 Evaluation Protocol

- Report in separate columns: in-task (test sets of the training tasks) and held-out (§3.4), the same protocol as decider
- Temperature: report both uncalibrated and calibrated (temperature fitting uses only the in-task regression split)
- Chance baseline: held-out is mostly 3-way, chance 0.33
- Fixed seed, macro average, a separate table per task

### 5.3 Required Comparisons

| Row | Model | Description |
|---|---|---|
| 1 | backbone zero-shot (letter logits) | first validate the backbone's capability |
| 2 | Route A (trained on the same data) | main comparison |
| 3 | Route B (this design) | main experiment |
| 4 | decider-2b/35b public models | external reference (numbers taken from the model card) |

---

## 6. Inference and Serving

### 6.1 Compute Reuse

- Multiple questions / multiple candidates for the same state are all batched into a single forward
- `h_S` caching: deduplicated by state within a batch; the server can cache by state hash
- Both training/inference use SDPA / flash-attention

### 6.2 Two-Stage and Thresholds

```
score_all = independent_scores(...)        # [N] sigmoid absolute probabilities
topk_idx  = score_all.topk(K)
probs     = softmax(logits[topk_idx]/T)
abstain   = score_all.max() < threshold    # or all < threshold
```

### 6.3 API Shape

Compatible with the request shape of TypeSafe's `POST /v1/systemone` (the same as decider, making it easy to run parity checks with the official SDK):

```
Request:  { state, questions: [ {type: choice|score|noul, text, options, criteria} ] }
Response: { questions: [ {distribution: {option_id: prob}, } ] }
```

---

## 7. Repository Structure

```
decision-model/
├── DESIGN.md                  # this document
├── README.md
├── pyproject.toml
├── configs/
│   ├── mini_b0.yaml           # Stage 1 default
│   └── full_b0.yaml           # Stage 2
├── src/decision_model/
│   ├── data/
│   │   ├── schema.py          # DecisionExample / Question / Option
│   │   ├── converters/        # one conversion script per dataset
│   │   ├── mixture.py         # weighted sampling + subset sampling
│   │   └── collate.py
│   ├── models/
│   │   ├── encoders.py        # role embedding / shared and two-tower
│   │   ├── scorer.py          # L1 / L2 / L3
│   │   ├── heads.py           # choice / score / noul
│   │   └── decision_model.py
│   ├── train/
│   │   ├── stage1_mini.py
│   │   ├── stage2_full.py
│   │   └── calibrate.py
│   ├── eval/
│   │   ├── metrics.py         # acc/nll/brier/ece/acc@80/aurc
│   │   ├── harness.py
│   │   └── probes.py          # held-out schema / abstention
│   └── serve/
│       └── api.py
├── scripts/                   # entry points for conversion, training, evaluation
└── tests/                     # schema / sampling / metric unit tests
```

---

## 8. Milestones and Acceptance Criteria

| Milestone | Content | Acceptance criteria |
|---|---|---|
| M0 | data pipeline + evaluation harness | reproduced metrics on the public decider-2b model within <0.5pt of the model card |
| M1 | Route B-Mini (V0+L2) | converges when training on 20–30 task classes; in-task acc ≥ 0.70 |
| M2 | A/B comparison report | B ≥ A on held-out metrics or ECE; otherwise trigger a shrink review |
| M3 | Stage 2 end-to-end | held-out acc ≥ 0.75, ECE ≤ 0.08 |
| M4 | calibration + two-stage | 255 candidates end-to-end <500ms (single A100/4090) |
| M5 | (optional) RLCD | belief log score significantly better than SFT-only |
| M6 | (optional) multimodal state encoder | text metrics do not degrade after swapping the encoder |

---

## 9. Risks and Open Questions

| Risk | Mitigation |
|---|---|
| Route B does not beat Route A | Stage 1 falsifies at low cost; A continues as a fallback and keeps being engineered |
| High interaction-layer cost (Q×K cross-attns) | state caching, candidate parallelism, two-stage pruning, shallow scorer |
| Candidate encoder capacity/memorization | subset sampling + held-out schema probes for continuous monitoring |
| Absolute vs relative probability inconsistency | train both heads, calibrate both, evaluate separately |
| Large data engineering effort (95 class conversions) | Stage 0 does only 20–30 classes; expandable once the converters are standardized |
| Long state (32k) | hierarchical encoding / chunk + cross-attn compression; train at 8k first |

Open questions (answered together in Stage 1):
1. backbone: bidirectional encoder (ModernBERT) vs generative LLM (Qwen3) as the frozen backbone — which has the stronger semantic prior on decision tasks?
2. Is the gap between L2 and L3 worth the inference cost?
3. What is the optimal combination of label smoothing and temperature calibration on held-out?
4. Threshold strategy for the independent sigmoid head (per-task or global)?

---

## 10. Reference Facts (from public implementations)

- decider Route A readout: prompt up to `Answer: (`, take letter-token logits, softmax with temperature; the letter is never generated, and multiple questions share one forward
- During training, large label sets are subsampled to ≤10 candidates, gold always kept, order shuffled
- Temperature is fitted on in-task data: 2b v10 uses 1.30, 35b uses 1.08
- decider-2b v10 public metrics: in-task 0.805 / 0.474 / 0.037 (acc/NLL/ECE), held-out 0.755 / 0.622 / 0.084
- v10 RL: belief proper log score (checked against exact laws) + PPO action + option-order consistency + KL
- Official Jev features: 255 options, multiple questions parallel and in isolation, adding a question barely increases latency; at high cardinality, independent scoring first and then explicit choice
- Reference repository: https://github.com/Mapika/decider
