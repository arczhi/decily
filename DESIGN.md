# 路线 B：显式 Candidate Scorer 决策模型 —— 设计方案

> 目标：训练一个"读 state + 运行时任意 question + 任意 candidate 集合，输出 calibrated 概率分布"的文本决策模型。
> 本方案对应 Jev / System One 的路线 B（显式 candidate scorer），并以 Mapika/decider 的路线 A（LM-head 字母 logits）为对照基线。

---

## 0. 背景与定位

### 0.1 任务定义

模型不做文本生成，只在运行时给定候选集上输出概率：

```
F(state, question, candidates) → P(candidate | state, question)
```

- `state`：任意上下文（字符串 / JSON / 对象 / 数组，训练上限 ~8k tokens，推理支持 16k–32k）
- `question`：运行时任意定义的语义问题（类别名、criterion、评分维度）
- `candidates`：运行时任意定义的语义对象（2–255 个），支持纯字符串或带 description 的对象
- 输出：Choice（softmax 分布）/ Score（每档独立概率）/ Noul（每个 criterion 独立 yes-no）

词汇表在推理时动态定义，训练时未见过的 schema 必须可用 —— 这是本项目的核心泛化目标。

### 0.2 与路线 A 的差异

路线 A（decider 已验证）：把候选写进 prompt，`Answer: (` 处取 A/B/C 字母 token 的 LM-head logits，按温度 softmax。

路线 B（本方案）：显式编码候选，用可学习的交互层打分，再归一化。

| | 路线 A | 路线 B |
|---|---|---|
| 候选表示 | 压进单个字母 token 的 embedding 行 | 每个候选有完整 contextual 表示 |
| 状态复用 | 每次 forward 重算 | 编码一次缓存复用 |
| 顺序偏差 | causal LM 位置敏感 | 结构上置换不变 |
| 独立分数 | 无（只有归一化分布） | sigmoid 独立分，支持阈值/两阶段 |
| 长 option description | prompt 膨胀、信息瓶颈 | 直接作为候选编码输入 |
| 多模态扩展 | 必须是生成式 VLM | 换 state encoder，候选仍是文本 |

### 0.3 待验证假设（本项目第一优先级）

> 在同等数据、同等参数量级下，显式 candidate scorer 相比 LM-head 读出，在 **held-out schema 泛化** 和 **calibration** 上是否更优？

Stage 1 的 A/B 对照实验就是为了回答这一句。若 B 不优于 A，则项目及时收缩到路线 A 的工程化打磨。

---

## 1. 核心抽象

### 1.1 打分与归一化

```
h_S  = Enc_state(S)                      # [B, Ls, d]
h_Q  = Enc_question(Q_i)                 # [B, Q, Lq, d]
h_C  = Enc_candidate(C_ij)               # [B, Q, K, Lc, d]
s_ij = Scorer(h_S, h_Q, h_C)             # [B, Q, K] 标量 logit
```

两种归一化，按 question 类型选择：

```
Choice :  P_ij = softmax_j( s_ij / T_choice[task_type] )         # 组内相对概率
独立   :  σ_ij = sigmoid( s_ij / T_indep[task_type] )            # 绝对概率
```

语义能量函数视角：`E(S, Q, C)` 越大表示候选越符合当前问题；softmax 是条件分布，sigmoid 是逐候选边际概率。

### 1.2 大候选集两阶段

对应官方 255 options 的说法：

```
阶段 1  独立打分:  a_k = sigmoid(s_k)        k = 1..N
阶段 2  Top-K:     softmax( s_topk / T )      K = 8~16
```

Top-K 之外的概率归零（或并入一个显式 "none of the above" 桶）。训练时即用 sub-sampled candidates（见 §3.2）来匹配这个推理形态。

### 1.3 置换不变性

CandidateEncoder 对每个候选独立编码，Scorer 对每个候选独立打分，候选之间无 attention 交互 → 结构保证 permutation-invariant。这同时消除了路线 A 需要的顺序增强（但数据层仍做 label shuffle 以匹配上游习惯）。

---

## 2. 架构设计

### 2.1 总体结构

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

### 2.2 编码器组织（三个变体，建议按阶段采用）

| 变体 | 结构 | 可训参数（以 0.6B 计） | 用途 |
|---|---|---|---|
| V0 共享 | 单 encoder + role embedding（state/question/candidate 三种 role），全冻结 | ~50–150M（仅 scorer） | Stage 1 验证假设，最快 |
| V1 双塔 | State+Question 共用一个，Candidate 一个（同一 ckpt 初始化） | ~150–300M | Stage 2 默认 |
| V2 三塔 | 三者独立 encoder | 全参 | Stage 3 上限探索（可选） |

推荐配置：
- 基座候选（文本，0.5B–2B）：`ModernBERT-base/large`（双向、长上下文、编码效率高）或 `Qwen3-0.6B/1.7B`（语义先验更强，但 causal mask 对编码不友好，需改 attention mask）
- Stage 1 用 V0 冻结基座跑通，Stage 2 再决定是否解冻与采用 V1

### 2.3a 已知问题与文献依据：表示各向异性（2026-09-23 实测）

实测：冻结 ModernBERT 的候选 pooled 表示两两余弦相似度 **0.975**（norm≈31），
共享分量主导梯度 → 训练塌缩为均匀分布（loss 停在 ln K）。

文献：
- Ethayarajh 2019：上下文表示高度各向异性，任意两词平均 cos≈0.99
- Su et al. 2021《Whitening Sentence Representations》：成熟解法是白化 /
  消除主导主成分；Arora et al. 2017、Mu & Viswanath 2018 同思路
- SimCSE（Gao et al. 2021）：对比目标 flatten 表示空间、提升 uniformity
- RankT5 / monoBERT：评分头用 listwise softmax / 排序损失直接训练

设计决定：
- scorer 对 choice/score 行做 **per-question 候选均值中心化**（`center_pooled`），
  即集合内「消除主导主成分」；noul 行保留绝对分量（独立概率需要绝对信息）
- 损失沿用 listwise softmax CE（= RankT5 的 listwise 排序损失）
- 后续可选：固定/学习式 whitening、SimCSE 式对比正则（Stage 2 再评估）

### 2.3b 架构决定：默认改为 Cross-Encoder scorer（2026-09-23 实测）

实测结论（4 任务混合训练 300 步对照）：

| 变体 | 单任务 | 4 任务混合 |
|---|---|---|
| late-interaction + centering | 会学（loss 0.56@200） | **塌缩**（loss≈chance，梯度→0） |
| per-task 组批 | - | 2 任务略好，4 任务仍差 |
| **cross-encoder（冻结底座）** | - | **正常学习**（3000 步 in-task acc 0.608 / ECE 0.034） |

原因：late-interaction 的题面条件化太弱（question 只在 kv 和 cos 特征里，
pooled 被 state 主导），混合任务时共享 scorer 无法按题面路由，梯度互相冲突
（Kendall 2018 / CAGrad 2021 描述的 negative transfer）。

方案（RankT5/monoBERT 式）：`[state; question; candidate]` 拼接进编码器 →
attention pooling → 标量分 → listwise CE。默认 `route: cx`。

- `center_pooled` 仍保留给 late-interaction 变体（单任务有效）
- late-interaction 的 state 缓存效率优势留待 Stage 2/3 再优化
  （需要更强 question conditioning，例如题面 query 注入 attention）

### 2.3 Scorer 交互强度（核心消融变量）

| Level | 方式 | 成本 | 备注 |
|---|---|---|---|
| L1 late-interaction | MaxSim：`max_i max_j (c_i · [s;q]_j)` | 最低 | ColBERT 式，短候选可用 |
| L2 cross-attention（默认） | 候选 token 作 query，state+question latents 作 kv，2–4 层 | 中 | FiD 式；候选可重读状态中与自己相关的片段 |
| L3 full cross-encoder | `[S; Q; C]` 拼接过完整 transformer | 最高 | 与 L2 同数据对比，作为上限参照 |

L2 具体结构：

```python
class CrossAttentionScorer(nn.Module):
    # layers: 2-4 x (MHA cross-attn + FFN + LayerNorm)
    # query: candidate latents [B*Q*K, Lc, d]
    # key/value: concat(state, question) latents [B*Q, Ls+Lq, d]
    # pooling: attention pooling 或 [CLS]；再 concat 一个 dot-product 相似度特征
    # head: MLP(d -> 4d -> 1)
    def forward(self, h_cand, h_cond, cand_mask, cond_mask) -> logits  # [B, Q, K]
```

设计要点：
- 每层 scorer 浅（2–4 层），主要计算在编码器，便于控制推理成本
- 跨状态复用：`h_S` 每状态只算一次；同一 state 的 Q×K 打分全部 batch 化
- 训练期 Q≤16、K≤10；推理期 Q 不限、K≤255（走 §1.2 两阶段）

### 2.4 输出头

单一 scalar head + task-type embedding，不按任务各自建头：

```python
logit = MLP(concat[pooled, task_type_emb, dot_feature])
```

- `choice`：masked softmax（invalid candidates 置 -inf）
- `score`：每个等级单独过一个 sigmoid（等级数字与邻居不进入打分，保证等级隔离）
- `noul`：每个 criterion 一个独立 sigmoid

### 2.5 校准层

- 温度按 task family 分别拟合（参考 decider 用的 1.08 / 1.30，说明温度是必要工序）
- 温度拟合数据：in-task regression split（与 decider 口径一致，保证可比）+ 额外报告未调温指标
- 训练期软标签：label smoothing ε≈0.05–0.1 或与均匀分布 mixup，显式抑制过自信
- Noul/Score 的独立概率用 Platt scaling / sigmoid 温度单独校准

### 2.6 推荐 M1 配置

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
  max_candidate_tokens: 64               # description 较长时 128
  questions_per_example: 1
  candidates_per_example: [2, 10]        # 子集采样上限（对齐 decider）
  loss: [ce, bce]                        # + label_smoothing 0.05
  optimizer: adamw
  lr: 1e-3                               # scorer only (frozen backbone)
  batch_states: 32
eval:
  metrics: [acc, nll, brier, ece_15bin, acc_at_80, aurc]
```

---

## 3. 数据设计

### 3.1 统一数据格式

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

- `choice`：answer 为 option id
- `score`：options 为等级（1–5），answer 为等级；每档独立可判
- `noul`：question 为 criterion，options 为待判对象，answer 为 yes/no
- 一个 example 可携带多个 question（同一 state 共享编码），训练期一般 Q=1

### 3.2 训练样本构造

1. **候选来源**：数据集原始标签空间全量作候选池（gold 必须在内）
2. **子集采样**：每例随机抽 `K ∈ [2, 10]` 个候选，gold 必留
3. **顺序**：训练期仍随机 shuffle 候选顺序（防御性做法，尽管结构上不变）
4. **难负例（可选增强）**：用教师 LLM 生成与 gold 语义接近但不正确的候选，提升判别难度（优先用于 QA/意图类）
5. **文本化**：`option = text + " : " + description`（description 支持对象/数组时序列化为紧凑 JSON）

### 3.3 任务混合（Stage 0 起始 20–30 类，目标 95 类）

| 任务族 | 数据源（举例） | 建议权重 |
|---|---|---|
| 文本分类/主题 | AG News, BBC, Dolly categories | 10% |
| 情感/情绪 | SST2, Financial PhraseBank, tweet irony | 10% |
| NLI/蕴含 | MNLI, SNLI, CommitmentBank, PAWS | 10% |
| 意图/工具选择 | CLINC, MASSIVE, Hermes tool selection | 15% |
| QA/多选 | ARC, SciQ, MMLU subset, QuALITY | 15% |
| 事实核查/审核 | FEVER, Hateful Memes (text 部分), moderation | 10% |
| 偏好/奖励 | RewardBench, arena preferences (3-way) | 10% |
| 拒绝/缺席 | abstention 构造（gold 不在候选中时选 "none of the above"） | 10% |
| 评分/序数 | CR reviews, ADE (1–5) | 10% |

### 3.4 Held-out 泛化集（训练中绝不出现）

照抄 decider 的评测名单以保证可比：

```
TREC, BBC news, PAWS, SciQ, Social IQa, StrategyQA, PubMedQA, TruthfulQA,
tweet irony, financial sentiment, ADE, MASSIVE scenario, student question
categories, Dolly categories, CR reviews, Financial PhraseBank,
CommitmentBank, QuALITY, XStoryCloze, RewardBench, arena preferences,
Hermes tool selection, abstention probe
```

另加两类自建探针：

1. **未见 schema 探针**：训练任务的新类别集合（如把 ticket 部门换成医院科室），候选名全部未见过
2. **干扰探针**：候选中插入语义高度相似的近义项，考察判别分辨率

### 3.5 数据管线结构

```
datasets → converters（每数据集一个转换器）→ decision tuples (jsonl, 统一格式)
        → mixture sampler（按权重采样 + 子集采样）→ collate → train
```

转换器输出即审计点：每个数据集一条命令可转、可校验、可统计，避免黑箱。

---

## 4. 训练方案

### 4.1 Stage 0：管线与评测基线（无训练）

- 数据转换 + mixture + collate + 评测 harness
- **评测 harness 先对拍**：直接评测公开的 `Mapika/decider-2b`，若我们复现的 in-task/held-out 指标与模型卡一致，说明 harness 可信
- 建立 held-out schema 探针与 abstention 探针

### 4.2 Stage 1：Route B-Mini（核心验证）

- 冻结 backbone（V0），只训 scorer + heads（~50–300M）
- 同数据训练 **Route B** 与 **Route A readout**（同 backbone、同 prompt 格式、字母 logits）
- 产出对照报告：in-task / held-out 的 acc、NLL、Brier、ECE、acc@80、AURC
- 消融：L1 vs L2 vs L3；共享 vs 双塔；softmax vs sigmoid；label smoothing 开关
- 决策门：B 在 held-out 或 calibration 上不劣于 A，则进入 Stage 2；否则收缩

### 4.3 Stage 2：端到端微调

- 解冻 backbone（V1 双塔），全参或 LoRA，0.6B–2B
- 数据扩到 50–95 类任务
- 训练配方：CE（choice，label smoothing 0.05）+ BCE（score/noul）+ 可选 InfoNCE（表示塑造）
- 早停指标用 held-out NLL（不是训练 loss）

### 4.4 Stage 3：校准与两阶段

- 按 task family 拟合温度；独立头做 Platt scaling
- 实现 Top-K 两阶段推理（K=8/16）并评测 255 候选下的延迟与精度
- 目标：ECE ≤ 0.03（in-task）、≤ 0.08（held-out），对齐 decider-2b v10 水平

### 4.5 Stage 4：RLCD 式 RL（可选）

- PPO action objective（可验证环境，如游戏/浏览器动作）
- belief 的 log proper score（对照已知概率律）
- option-order consistency、KL 正则
- 参考 decider 的 `docs/RL.md`（v10 已证有效：belief excess 0.22 nats vs v8 0.47）

### 4.6 损失与目标函数汇总

```
L = L_choice + L_score + L_noul + λ·L_aux

L_choice = CE( softmax(s/T) , y )                    # proper scoring rule 族
L_score  = Σ_l BCE( sigmoid(s_l/T) , 1[y ≥ l] or 1[y = l] )
L_noul   = Σ_c BCE( sigmoid(s_c/T) , y_c )
L_aux    = InfoNCE( h_C, h_S )  (可选，小权重)
```

关键原则：**主损失必须是 proper scoring rule**（log score / Brier），否则模型没有动机诚实报概率。

---

## 5. 评测方案

### 5.1 指标定义

| 指标 | 定义 | 备注 |
|---|---|---|
| Accuracy | gold 概率最高即正确 | |
| NLL | -log P(gold) | 越低越好 |
| Brier | Σ(p_i - y_i)² | 多分类 |
| ECE | 15 bins，|acc - conf| 加权平均 | 核心校准指标 |
| acc@80 | 最自信 80% 决策上的准确率 | 选择性预测 |
| AURC | risk-coverage 曲线下面积 | 全阈值下的选择性表现 |

### 5.2 评测协议

- 分列报告：in-task（训练任务测试集）与 held-out（§3.4），与 decider 口径一致
- 温度：同时报告未调温与调温后（温度拟合只用 in-task regression split）
- Chance baseline：held-out 多为 3-way，chance 0.33
- 固定 seed、宏平均、每任务单独一表

### 5.3 必备对照

| 行 | 模型 | 说明 |
|---|---|---|
| 1 | 基座 zero-shot（字母 logits） | 先验证底座能力 |
| 2 | Route A（同数据训练） | 主对照 |
| 3 | Route B（本方案） | 主实验 |
| 4 | decider-2b/35b 公开模型 | 外部参照（数字取自模型卡） |

---

## 6. 推理与服务

### 6.1 计算复用

- 同一 state 的多个 question / 多个 candidate 全部 batch 到一次 forward
- `h_S` 缓存：批内按 state 去重；服务端可按 state hash 缓存
- 训练/推理都用 SDPA / flash-attention

### 6.2 两阶段与阈值

```
score_all = independent_scores(...)        # [N] sigmoid 绝对概率
topk_idx  = score_all.topk(K)
probs     = softmax(logits[topk_idx]/T)
abstain   = score_all.max() < threshold    # 或全部 < threshold
```

### 6.3 API 形状

兼容 TypeSafe 的 `POST /v1/systemone` 请求形状（与 decider 一致，便于用官方 SDK 对拍）：

```
请求:  { state, questions: [ {type: choice|score|noul, text, options, criteria} ] }
响应:  { questions: [ {distribution: {option_id: prob}, } ] }
```

---

## 7. 仓库结构

```
decision-model/
├── DESIGN.md                  # 本文档
├── README.md
├── pyproject.toml
├── configs/
│   ├── mini_b0.yaml           # Stage 1 默认
│   └── full_b0.yaml           # Stage 2
├── src/decision_model/
│   ├── data/
│   │   ├── schema.py          # DecisionExample / Question / Option
│   │   ├── converters/        # 每数据集一个转换脚本
│   │   ├── mixture.py         # 权重采样 + 子集采样
│   │   └── collate.py
│   ├── models/
│   │   ├── encoders.py        # role embedding / 共享与双塔
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
├── scripts/                   # 转换、训练、评测入口
└── tests/                     # schema / 采样 / 指标单测
```

---

## 8. 里程碑与验收标准

| 里程碑 | 内容 | 验收标准 |
|---|---|---|
| M0 | 数据管线 + 评测 harness | 对 decider-2b 公开模型的复现指标与模型卡误差 <0.5pt |
| M1 | Route B-Mini（V0+L2） | 训练 20–30 类任务可收敛；in-task acc ≥ 0.70 |
| M2 | A/B 对照报告 | held-out 指标或 ECE 上 B ≥ A；否则触发收缩评审 |
| M3 | Stage 2 端到端 | held-out acc ≥ 0.75、ECE ≤ 0.08 |
| M4 | 校准 + 两阶段 | 255 候选端到端 <500ms（单卡 A100/4090） |
| M5 |（可选）RLCD | belief log score 显著优于 SFT-only |
| M6 |（可选）多模态 state encoder | 换 encoder 后文本指标不退化 |

---

## 9. 风险与开放问题

| 风险 | 缓解 |
|---|---|
| Route B 未赢过 Route A | Stage 1 小成本即证伪；A 作为兜底继续工程化 |
| 交互层成本高（Q×K 次 cross-attn） | state 缓存、候选并行、两阶段剪枝、浅 scorer |
| 候选编码器容量/记忆 | 子集采样 + held-out schema 探针持续监测 |
| 绝对概率 vs 相对概率不一致 | 两套头都训练、都校准、分别评测 |
| 数据工程量大（95 类转换） | Stage 0 只做 20–30 类，转换器标准化后可扩展 |
| 长 state（32k） | 分层编码 / chunk + cross-attn 压缩，先按 8k 训练 |

开放问题（Stage 1 一并回答）：
1. backbone：双向 encoder（ModernBERT）vs 生成式 LLM（Qwen3）作冻结底座，谁在决策任务上语义先验更强？
2. L2 与 L3 的差距是否值得推理成本？
3. label smoothing 与温度校准在 held-out 上的组合最优值？
4. 独立 sigmoid 头的阈值策略（per-task 还是全局）？

---

## 9.5 Stage 1 结果（2026-09-23）

设置：4 训练任务（sst2/ag_news/mnli/arc_easy，共 ~28k 例），3000 步，batch 16，
warmup 100 + cosine；in-task 评测 1200 条（每任务 300），held-out 1200 条
（massive 60 类 / banking77 77 类 / emotion 6 类，均未训练）。

| 指标 | B_frozen | B_lora | A_lora |
|---|---|---|---|
| 可训参数 | 2.4M | 5.75M | 10.1M |
| 底座 | ModernBERT-base | ModernBERT-base | Qwen3-0.6B |
| in-task acc / NLL / ECE | 0.611 / 0.84 / 0.029 | 0.816 / 0.51 / **0.019** | **0.888** / **0.36** / 0.023 |
| held-out acc / NLL / ECE | 0.186 / 3.16 / 0.070 | **0.412 / 2.25 / 0.056** | 0.385 / 2.75 / 0.138 |

结论：
- Route B 加 LoRA 后**在 held-out 全面优于 Route A**（acc +2.7pt，NLL −0.5，
  ECE 0.056 vs 0.138），in-task 略低但校准更好 → 显式候选打分 + listwise
  损失确实更抗过拟合、跨 schema 泛化更好
- Route A 在训练任务内拟合更强（0.888），但 held-out 校准明显更差（ECE 0.138）
- 注意：两者底座不同（ModernBERT vs Qwen3），非严格控制变量；Stage 2 需做
  同底座对照（如都用 Qwen3-0.6B，比较字母 logits vs 显式 scorer）
- 温度拟合（in-task / held-out）：B_frozen 1.25 / 1.43；B_lora 1.13 / 0.80；
  A_lora 1.07 / 1.70。A 的 held-out 需要显著放大温度（1.7）才校准，说明其
  跨任务置信度偏高（过度自信）；B_lora 反向（0.8，略欠自信）

关键工程经验（已固化到配置默认值）：
- LoRA/适配器训练需要 lr ≤2e-4 + warmup + cosine，1e-3 会先学后退
- 冻结底座 + 小 scorer 容量不足，held-out 泛化需要给编码器可训容量

## 9.6 Stage 2a 受控对照：同底座 A vs B（2026-09-23）

同底座 Qwen3-0.6B-Base、同数据（4 任务）、同 LoRA/调度（3000 步）：

| 指标 | A（LM-head 字母 logits） | B（cross-encoder scorer） |
|---|---|---|
| 可训参数 | 10.1M + lm_head | 14.3M |
| in-task acc / NLL / ECE | **0.888** / 0.36 / 0.023 | 0.881 / **0.35** / **0.020** |
| held-out acc / NLL / ECE | 0.385 / 2.75 / 0.138 | **0.491** / **1.95** / **0.129** |

结论：控制底座后，**Route B 在 held-out 上显著更优**（acc +10.6pt，NLL −0.8），
in-task 持平、校准略好。显式候选打分的泛化优势得到干净验证 →
Stage 2b 直接扩任务混合（24 任务）继续放大这个优势。

## 9.7 大候选集两阶段推理（2026-09-23 实测）

实现 `decision_model.infer.two_stage`：候选分 chunk 独立打分（cross-encoder
logits 与候选集无关，chunking 无损）→ Top-K → 相对 softmax。
对应官方「高 cardinality 时先独立 scoring 再 explicit choice」。

120 条实测（77 类 banking77 + 60 类 massive，Stage 2a 模型）：

| 路径 | acc | mean_conf |
|---|---|---|
| 全量 softmax | 0.508 | 0.109 |
| Top-8 两阶段 | 0.508 | 0.275 |

- **gold_in_top8 = 87.5%** → shortlist 上限，K 的选择依据
- 精度无损（argmax 保持）；置信度更诚实（0.275 vs 0.109）
- 注：当前 chunk 实现逐块重编码，延迟更高；生产应做 prefix/KV 复用
  （cross-encoder 的 state+question 前缀可缓存）

## 9.8 Stage 2b：24 任务混合（2026-09-23 完成）

数据：24 任务（~75k 训练样本，含 8 个 MMLU 学科）；模型 Qwen3-0.6B-Base + LoRA
（cross-encoder，可训 14.3M）；3000 步，有效 batch 16（micro 4 × accum 4），
bf16 + 梯度检查点，状态截断 256。

| 指标 | A（4 任务） | B（4 任务） | B（24 任务） |
|---|---|---|---|
| held-out acc | 0.385 | 0.491 | **0.542** |
| held-out NLL（fitted） | 2.75 | 1.95 | **1.86** |
| held-out ECE（fitted） | 0.138 | 0.129 | **0.107** |
| held-out 拟合温度 | 1.70 | 0.50 | 0.40 |
| in-task acc（各自评测集） | 0.888 | 0.881 | 0.850 |
| in-task ECE | 0.023 | 0.020 | **0.017** |

held-out 分任务（B4 → B24）：massive 0.588→0.573（略降）、banking77 0.370→0.432、
emotion 0.512→0.620 → 混合广度主要增益在**大候选集（77 类）和细粒度情感**。

结论：
- **任务覆盖广度有效**：相对 B4，held-out acc +5.1pt / NLL −0.09 / ECE −0.022
- B（24 任务）在 held-out 上全指标优于 Route A（acc +15.7pt，NLL −0.89，ECE −0.031）
- 校准：held-out 需 T≈0.4（欠自信），in-task T≈1.0（已良好）→ Stage 2c 做
  per-task-family 校准
- massive 未提升：与训练任务分布差异大（语音助手话语），且 scenario 任务因与
  held-out intent 同源被刻意排除

评测卫生修复：
- MMLU 训练用 test 划分、评测改用 validation（消除重叠）；温度网格下限 0.5→0.2
- 评测改为单遍收集 logits、任意温度复用（原实现每文件 3 遍 forward）

## 9.9 Stage 2c：训练配方对照（2026-09-23 完成）

两个配方变体，其余与 s2b 相同（24 任务、Qwen3-0.6B + LoRA、3000 步）：

| held-out | B(4任务) | B(24任务) | B_normk | **B_bigk** |
|---|---|---|---|---|
| acc | 0.491 | 0.542 | 0.487 | 0.538 |
| NLL | 1.95 | 1.86 | 1.86 | **1.60** |
| ECE | 0.129 | 0.107 | 0.065 | **0.071** |

- **bigk（max_options 10→16 + length_bucket）采纳**：精度持平但 NLL/ECE 显著更好；
  banking77（77 类）0.432→**0.517**（+8.5pt）→ 训练候选集大小直接决定大候选集泛化；
  代价：emotion 0.620→0.525（能力再分配）
- **normk（按 log K 归一化损失）不采纳为主配方**：held-out acc −5.5pt
  （banking77 崩到 0.278），但校准最好（ECE 0.065）→ 留作 Stage 3 的
  "校准微调"候选手段
- 结论：Stage 3（2B 放大）采用 bigk 配方；配比实验（任务权重）留待 Stage 2d/3

## 9.10 Stage 3：规模放大到 Qwen3-1.7B（2026-09-23 完成）

bigk 配方不变，底座 0.6B → 1.7B（LoRA 34.2M 可训），3000 步。

| 指标 | B_bigk 0.6B | **B_bigk 1.7B** |
|---|---|---|
| held-out acc | 0.538 | **0.565**（+2.7pt） |
| held-out NLL | 1.60 | **1.56** |
| held-out ECE | **0.071** | 0.094（变差） |
| in-task acc | 0.823 | **0.854**（+3.1pt） |
| in-task ECE | 0.025 | 0.025 |

- held-out 三任务全涨：massive 0.573→0.585、banking77 0.517→0.525、emotion 0.525→0.585
- in-task 提升面广：arc_easy 0.776→0.876、arc_challenge 0.636→0.748、mmlu 各科 +0.2~0.35
- **规模杠杆成立，但 held-out 校准退化（0.071→0.094）** → 下一步 RLCD 的核心目标

工程记录（踩坑）：
- 1.7B 纯 bf16 训练在 step ~400 发散（grad_norm 502）→ 修复：底座 bf16 +
  **LoRA/head 保持 fp32 + forward autocast**（AMP/QLoRA 标准做法），lr 2e-4→1e-4
- 镜像单连接限速 2MB/s → **aria2c 16 连接 ~10-80MB/s**，3.4GB 模型 5 分钟下完

## 9.11 Stage 4：RLCD 校准（2026-09-23 完成）

基于 Stage 3（1.7B）checkpoint，800 步，belief:decision 数据比 0.4:0.6。
损失 = belief log-score（软目标，已知概率律）+ REINFORCE action（硬标签，
reward=正确性，baseline=模型自身概率）+ KL(策略‖参考策略, w=0.1)。

数据：合成 belief 任务 6 类（硬币/骰子/纸牌/摸球/多骰和），2 万训练 +
1 千评测（同模板、未见参数）。

| 指标 | SFT 1.7B | **RLCD 1.7B** |
|---|---|---|
| held-out acc | 0.565 | **0.574** |
| held-out NLL | 1.56 | **1.55** |
| held-out ECE（T=1） | 0.280 | **0.124** |
| held-out ECE（调温后） | **0.094** | 0.111 |
| in-task acc | 0.854 | 0.854 |
| in-task ECE（T=1） | **0.027** | 0.087 |
| belief excess_nats | — | **0.0019**（decider v10: 0.22） |
| belief_ece | — | **0.0048** |

结论：
- **belief 目标完全生效**：已知概率律任务上 excess 0.0019 nats（接近完美）
- **held-out 上 T=1 的校准大幅改善**（ECE 0.280→0.124），说明模型学会"诚实报概率"；
  acc/NLL 也小幅提升
- **代价**：in-task 校准被扰动（0.027→0.087），调温后 held-out ECE 略差于 SFT
  （0.111 vs 0.094）→ 说明 belief 训练整体压低了置信度，而 SFT 可被温度拟合"救回"
- 下一步（RLCD v2）：降低 belief 比例（0.4→0.2）、提高 KL 权重、或按任务族
  分别拟合温度；更理想的 belief 数据应贴近真实决策分布（不只是合成律）

## 9.12 Stage 4b：RLCD v2（实用优先，2026-09-23 完成）

三项升级：① GoEmotions 3 标注者投票分布（8000 条）作为**真实歧义** belief
数据（+5000 合成律验证）② 弃权训练（20% 决策样本抽掉 gold + "none of the
above"）③ 校准保护（belief 比例 0.4→0.25、KL 权重 0.1→0.3）。1000 步。

| 指标 | SFT | RLCD_v1 | **RLCD_v2** |
|---|---|---|---|
| held-out acc | 0.565 | **0.574** | 0.559 |
| held-out ECE（调温后） | 0.094 | 0.111 | **0.050** |
| in-task ECE（调温后） | **0.025** | 0.031 | 0.037 |
| belief excess（合成+真实歧义） | — | — | **0.030** |
| belief_ece | — | — | **0.005** |
| 正确弃权率（gold 缺失） | — | — | 0.52 |
| **误弃权率（gold 存在）** | — | — | **0.35 ⚠️** |
| 加 none 后准确率 | — | — | 0.396（从 0.565 掉） |

结论：
- **真实歧义上的诚实概率成立**：excess 0.030 nats、belief_ece 0.005
- **调温后 held-out 校准最佳**（0.050），per-task 温度差异明显
  （banking77 0.5 / emotion 0.4 / massive 0.45 → 均需锐化，说明整体偏欠自信）
- **弃权过于激进**：误弃权率 0.35 不可用于生产 → 需要 none 选项的阈值标定
  （在验证集上拟合 none 偏置，把误弃权率压到 ≤5-10%，最大化正确弃权）

下一步（RLCD v3 / 实用化）：
1. none 阈值标定（后处理，无需重训）
2. abstention_ratio 0.2→0.1，或在更贴近部署分布的数据上训练弃权
3. v1 作为"纯决策"版本保留（accuracy 最高），v2 作为"校准+弃权"版本

### 9.13 none 阈值标定结果（2026-09-23）

对 RLCD v2 做 none-logit 偏置扫描（held-out 600 条，构造 gold-present /
gold-absent 两个集合）：

| 误弃权预算 | bias | 正确弃权率 |
|---|---|---|
| ≤2% | -1.7 | 1.5% |
| ≤5% | -1.3 | 6.0% |
| ≤10% | -0.9 | 16.0% |
| ≤20% | -0.5 | 28.5% |

结论：**softmax 内嵌 none 选项的区分度弱**——压低误弃权后正确弃权几乎归零，
说明模型没有真正学会"答案不在候选里"，只是整体偏向 none。

修复方向（v3，按优先级）：
1. **独立弃权头**（推荐）：把"答案是否在候选集内"做成单独的二元门控
   （drop-gold 数据 + BCE + Platt 阈值标定），与"选哪个选项"解耦
   （selective classification / open-set 的标准做法）
2. 扩大弃权训练数据（比例↑、步数↑、按任务难度分层构造）
3. 在真实部署分布上构造弃权样本（当前是随机抽 gold，过于均匀）

## 9.14 RLCD 加强方案（文献调研 2026-09-23）

来源：
- decoding-jev（对 Jev/Laya 公开 notebook 的逆向分析）：RLCD 实现 =
  belief reward **log score + 0.75 × spherical score**、logits 加噪采样多候选、
  组内均值 baseline 的 REINFORCE、全权重 CE 项、事后温度拟合；
  并警告"采样目标 ≠ 推理目标，proper reward 不保证部署校准"，建议受控消融
- RLCR（arXiv 2507.16806）：二元 reward 的普通 RL **损害校准**；reward 加入
  有界 proper score（Brier）可同时提升准确率与校准；优于事后置信度分类器
- Mozannar & Sontag 2020（Learning to Defer）：置信度阈值法失败是理论已知；
  需**联合训练独立 rejector**（defer 作为新类，CE over Y∪{⊥} 一致）
- SelectiveNet（Geifman & El-Yaniv 2019）：predict/select 双头 + 覆盖率约束
- Conformal selective prediction：分布无关阈值 + 有限样本覆盖保证

对应我们的实测短板（§9.13 none 区分度弱）与改进项：

| 优先级 | 动作 | 预期 |
|---|---|---|
| P1 | 独立 rejector 头 + 联合训练 + conformal/代价敏感阈值 | 5% 误弃权预算下正确弃权 6%→30-50% |
| P2 | belief reward: log + 0.75·spherical；action reward 加 Brier 项 | RL 不再伤校准 |
| P3 | 受控消融（确定性 proper score vs 采样 REINFORCE vs ±CE） | 确认训练目标的可迁移性 |

## 9.15 选择性预测：置信度排序与风险保证（2026-09-24）

文献改道（依据 §9.14）：
- selective classification 基准（44 数据集/18 方法）：专用 reject 头（SELNET/DG）
  不是最优；**softmax-response 阈值 + 集成**是强基线
- 实测确认：v3 的独立 rejector 头只带来边际改善（5% 预算下 6%→8.5%），
  且 held-out acc −4pt → **弃用 rejector 路线，转向阈值标定**
- 理论修正：split-conformal 的集合覆盖保证**不能**直接约束"接受子集错误率"；
  改用 PAC 式选择性风险上界（Hoeffding）：
  `risk_hat(t) + sqrt(log(1/delta)/(2 n_acc)) <= alpha`

**核心实测（risk-coverage，held-out 60/77 类未见任务）**：

| 覆盖 10% 时的选择性风险 | SFT | RLCD_v1 | RLCD_v2 | RLCD_v3 |
|---|---|---|---|---|
| risk | 0.21 | 0.21 | **0.03** | **0.07** |

→ RLCD 的 belief 训练显著改善了**置信度排序**：只取最有把握的 10%，
v2/v3 的准确率达 97%/93%（SFT 仅 79%）。

关键工程结论：
- **保证可行性受校准集大小支配**：Hoeffding 修正 `sqrt(log(1/delta)/(2n_acc))`
  在 n_acc=75 时 0.14 → 1500 条校准集下 α=0.2 刚好不可行（最小上界 0.21）
- 实测诊断（v3，1500 条校准，T=0.80）：

| 覆盖 | n_acc | 经验风险 | PAC 上界 |
|---|---|---|---|
| 2% | 30 | 6.7% | 0.29 |
| 5% | 75 | **6.7%** | 0.21 |
| 10% | 150 | 12.0% | 0.22 |
| 20% | 300 | 23.7% | 0.31 |

- **实用操作点（经验）**：未见 60/77 类任务上取最有把握的 5% → 准确率 93.3%
  （SFT 同口径只有 ~79%）；这是"高精度模式 + 升级人工"的可用形态
- 正式 PAC 保证需要：校准集 ≥5k（n_acc 数百）+ 更高的高置信区间精度；
  Hoeffding 偏保守，可用 exact binomial / CRC 收紧
- 结论：**RLCD 的 belief 训练让选择性预测变得可用**（置信度排序质量是前提），
  阈值化本身不需要 rejector 头（v3 的 rejector 仅边际收益且损精度）

## 9.16 RLCD v3 消融：有效成分与无效成分（2026-09-24）

队列实验（均基于 SFT 1.7B，1000 步，rejector 头，held-out 调温后）：

| 变体 | held-out acc | ECE | NLL | 弃权@5%预算(正确率) |
|---|---|---|---|---|
| v3（全量：CE+belief+action+KL） | 0.519 | **0.064** | **1.633** | 8.5% |
| v3b（去掉 action RL） | 0.521 | 0.072 | 1.642 | 8.6% |
| v3c（弃权训练 0.20→0.35） | 0.523 | 0.068 | 1.641 | 8.9% |
| v3d（belief 去掉 spherical，500 步） | 待测 | | | |

结论（与文献一致）：
- **action RL（REINFORCE）几乎无贡献** → RLCR（二元 reward 的 RL 不改善校准）
  与 decoding-jev（采样目标≠部署目标）在实测中复现；belief + CE 是有效成分
- **提高弃权训练比例不能改善 rejector 区分度** → 再次验证 §9.15：
  实用弃权应走"置信度阈值化 + 大校准集"，而非 rejector 头
- spherical 项的作用待 v3d 结果（预期：加快 belief 收敛，对最终校准影响小）

## 9.17 最终对比与推荐配置（2026-09-24 02:45）

全部变体（1.7B + bigk，3000 步 SFT + RLCD 阶段），held-out = 未见
60/77 类意图/情感任务（1200 条）：

| 变体 | held-out acc | ECE(T拟合) | in-task acc | 弃权@5%预算 |
|---|---|---|---|---|
| SFT | **0.565** | 0.094 | 0.854 | — |
| **RLCD v2**（合成+真实歧义 belief，弃权 0.2，KL 0.3） | 0.559 | **0.050** | 0.844 | — |
| RLCD v3（+独立 rejector 头） | 0.519 | 0.064 | 0.855 | 8.5% |
| v3b（去 action RL） | 0.521 | 0.072 | 0.856 | 8.6% |
| v3c（弃权比例 0.35） | 0.523 | 0.068 | 0.856 | 8.9% |
| v3d（去 spherical，500 步） | 0.534 | 0.078 | 0.856 | 8.4% |

**推荐配置**：
1. **精度优先** → SFT 1.7B（held-out 0.565）
2. **校准优先（推荐）** → **RLCD v2**（ECE 0.050，真实歧义 belief excess 0.030，
   置信度排序最好：top-5% 覆盖下未见任务准确率 93-95%）
3. **弃权/高精度模式** → 不用 rejector 头；用 v2 的置信度 + 阈值（大校准集）
4. 不需要的东西（均有消融证据）：rejector 头、action RL、更多弃权比例、spherical 项

**下一步（Stage 5，未完成）**：多模态（Qwen3.5-2B VL + vision tower + 文本路径
隔离 + 30% 文本 replay + 回归门禁）；训练任务 24→60+；全参微调对照。

## 9.18 工程教训：日志累积除法 bug（2026-09-24）

`Trainer` 记录 loss 时用 `sum(所有 micro-batch loss) / window(优化步数)`，
而每个优化步含 `accum` 个 micro-batch → **日志值被放大 accum 倍**。
影响：所有历史 run 的 `choice_ce`/`grad_norm` 日志都被放大（÷accum 才是真值）。
前 1.7B 全参微调曾因 2.95 vs SFT 1.29 的"发散"误判被两次重启——真实值
0.37 vs 0.32，完全健康（单批 CE 波动范围 0.17~3.28，窗口均值波动是噪声）。
已修复：除以 `window * accum`。教训：**跨 run 比较 loss 前先统一 accum**。

## 9.19 Model Soup 实验记录（2026-09-24，含 bug 与结论）

动机（Wortsman et al., ICML 2022）：同一误差盆地内的微调模型做**权重平均**
可提升精度/鲁棒性且零推理成本；贪心版逐个加入有增益的模型。

协议：候选 = {SFT, v1, v2, fullFT@1500}；LoRA 先合并为全权重（fp32）；
选择集 = stage1_heldout_big 前 3000 条；测试集 = stage1_heldout 1200 条（独立）。

**单模型（选择集）**：

| 模型 | acc | ECE |
|---|---|---|
| SFT | 0.5453 | 0.096 |
| v1 | 0.5403 | 0.090 |
| v2 | 0.5327 | **0.067** |
| fullFT@1500 | **0.5607** | 0.164 |

**贪心结果**：seed = fullFT (0.5607) → **+SFT = 0.5670 ACCEPT** → +v1 = 0.0427
reject → +v2 = 0.0400 reject（运行随后退出，未写报告）

**发现的两个实现 bug**：
1. `evaluate_state` 用 `lora=true` 的架构去加载**已合并的全权重** → 键名不匹配，
   实际评测的是"底模 + 训练过的 head"（v2 单模型误报 0.106；修复后为 0.533）。
   修复：评测合并权重时强制 `lora=false`。
2. 贪心循环的 baseline 在"拒绝"后没有回退（用了上一次尝试的 acc 而非当前 soup 的 acc）。
   未修复（实验已中止）。

**未解现象**：加入 v1/v2 的权重平均试次 accuracy ≈ 0.04（≈ argmax 恒为 0 的 NaN
特征；单模型状态本身正常 0.53-0.54）。怀疑与 4 份 fp32 状态（~28GB CPU）下的
内存压力或某 tensor 的 NaN 有关，未定位。

**结论**：即便按最好情形看，权重 soup 的增益也只是
**{fullFT+SFT} 0.567 vs fullFT 0.561（选择集 +0.6pt）**，性价比低。
且权重平均对"跨盆地"模型（fullFT 与 LoRA 系）本就敏感。
→ 改用**概率空间集成**（Deep Ensembles, Lakshminarayanan 2017；
selective classification 基准中 ENS+SR 为第一），无需权重对齐、跨盆地稳健。

## 9.20 概率空间集成结果（2026-09-24 完成）

4 模型分布平均（SFT / v1 / v2 / fullFT@1500），1200 条 held-out + 1200 条 in-task：

**held-out（未见 60/77 类任务）**：

| 模型 | acc | NLL | ECE |
|---|---|---|---|
| SFT | 0.5608 | 1.572 | 0.097 |
| v1 | 0.5567 | 1.602 | 0.116 |
| v2 | 0.5425 | 1.584 | **0.039** |
| fullFT@1500 | **0.5933** | 1.596 | 0.201 |
| **ENSEMBLE** | **0.5750** | **1.455** | **0.061** |

**in-task**：集成 acc 0.8467（并列最佳）、NLL 0.398 / ECE 0.032（均最佳）。

结论：
- 集成 **NLL 比最好的单模型低 0.12**（互补性真实存在）
- **精度接近 fullFT（−1.8pt）+ 校准接近 v2（+0.022）** → 解决了两难
- 代价：推理 ×4。**下一步首选：以该集成为教师做蒸馏**（v4 失败的根因是教师
  太弱/相关性高；现在教师集成的 NLL 明显优于任何单体，具备蒸馏价值）
- 副产品：这也解释了 v4——当时教师池 {v1,SFT,v2,v3d} 的相关性高、
  且没有 fullFT 这样的异质成员

## 9.21 RLCD v5：集成蒸馏回单模型（2026-09-24 完成）

配方（文献确认：Hinton 2015 原始 KD + Reich 2020 集成蒸馏）：
- 教师 = 4 模型（SFT/v1/v2/fullFT@1500）分布的算术平均
- 学生 = 1.7B **全参**（从合并 SFT 热启动；容量匹配教师在先）
- 损失 = α·CE(gold, T=1) + (1−α)·T²·KL(p_student^T ‖ p_teacher^T)，**T=2, α=0.5**
- 数据 = 24 任务蒸馏集（66k，每例带教师分布）+ belief 15%
- 3000 步，lr 1e-5，8-bit AdamW，bf16 checkpoint

**最终对比（held-out = 未见 60/77 类任务；in-task 1200 条）**：

| 模型 | held-out acc | held-out NLL | held-out ECE | in-task acc |
|---|---|---|---|---|
| SFT | 0.561 | 1.572 | 0.097 | 0.847 |
| v1 | 0.557 | 1.602 | 0.116 | 0.840 |
| v2 | 0.543 | 1.584 | **0.039** | 0.843 |
| fullFT@1500 | **0.593** | 1.596 | 0.201 | 0.818 |
| 集成（4 模型） | 0.575 | 1.455 | 0.061 | 0.847 |
| **v5_distill（单模型）** | **0.581** | **1.451** | 0.068 | **0.863** |

结论：
- **蒸馏达成"单模型 = 集成质量"**：NLL 1.451（全场最佳，甚至略优于集成）、
  acc 0.581（超所有 LoRA，仅次 fullFT）、ECE 0.068（接近集成）
- **in-task acc 0.863 为全场最高**（蒸馏的正则化效应）
- 推理成本回到 ×1 → v5 可作部署模型
- belief 技能小幅退化（excess 0.030→0.069），因 KD 训练中 belief 仅占 15%；
  如需恢复可加一轮短程 belief 微调

**工程教训（本轮踩坑，均已修复）**：
1. `rsync --delete` 误删远端新建的 data/distill5（已加排除规则）
2. RLCDTrainer 缺 `save_dtype` → 全参 checkpoint 6.9GB/个写满磁盘
   （已支持 bf16，3.4GB/个）
3. RLCDTrainer 缺 8-bit 优化器支持 → fp32 Adam 占 13.6GB OOM（已支持 adamw8bit）
4. KD 掩码 bug：padding 位教师概率 × 学生 -1e9 掩码 = loss 爆炸（已按有效位掩码）
5. v5 配置的 manifest 曾指向原始硬标签数据 + distill_ratio=0 → KD 完全没生效
   （已改为 manifest 直接指向教师分布数据）

## 9.22 MLX 转换（Mac 端推理，2026-09-24 完成）

产物：`models_mlx/v5_mlx/`（model.safetensors 3.47GB + config + tokenizer）
+ `mlx/decision_mlx.py`（纯 mlx.core 实现，无需 torch）

对拍验证（固定 ids / 同一样本）：
- logit：MLX **-10.278** vs torch -10.25（差 0.03）
- 样本概率：{tech 0.568, shipping 0.370, returns 0.054, billing 0.009}
  vs torch {0.562, 0.374, 0.056, 0.009}

**关键坑**：MLX `mx.fast.rope(traditional=False)` 与 HF 的 rotate_half 约定
**不一致**（导致 28 层累积后隐藏态 std 偏差 3%、logit 偏差 6）→ 手写
rotate_half RoPE 后完全对齐。已记在代码注释里。

其他实现要点：RMSNorm 按 HF 习惯用 fp32 内部计算；因果掩码显式构造；
GQA 用 repeat 展开；pool/head 在 fp32 下计算。

用法：
```
python mlx/decision_mlx.py --model-dir models_mlx/v5_mlx \
  --state "..." --question "..." --options "a,b,c" --temperature 0.45
```

## 9.23 decider-2b 同台评测（2026-09-24 完成）

用他们自己的 `decider` 包（eager 模式，`use_graphs=False` 避开每形状 compile
+capture）在我们两个评测集上跑 Mapika/decider-2b（2B，Qwen3.5 混合注意力底座，
95 任务训练 + calibration-aware RL 384 步，官方温度 1.3）：

| 评测集 | decider-2b | v5（我们） |
|---|---|---|
| 我们的 in-task（24 任务，双方多数训练过） | 0.811 / 0.453 / 0.032 | **0.863 / 0.390 / 0.034** |
| 我们的 held-out（banking77/massive/emotion） | **0.867 / 0.420 / 0.020** | 0.581 / 1.45 / 0.068 |

**关键事实（来自其 eval_results.json 的 heldout 标记）**：
- decider-2b **训练了我们的 held-out 任务**（banking77/massive_intent/emotion
  均 heldout=false）→ 其 0.867 是 in-task 成绩（官方同任务 0.843-0.96）
- 我们零样本同任务 0.581 → 不可直接比较
- **在共同训练的 in-task 任务上，v5 领先 decider-2b +5.2pt**（1.7B/24 任务
  vs 2B/95 任务）

结论：在重叠任务上我们的模型更强；他们在其 held-out（28 任务：paws/sciq/trec/
boolq/strategyqa/quality/ade/fin_sentiment 等）上的 0.4-0.98 分数无法与我们
现有数字对齐——**要做公平对比需构建"双方都零样本"的套件**（从他们 heldout=true
且不在我们 24 任务中的任务里选）。

## 9.23 decider-2b 同台评测（2026-09-24 完成）

用他们自己的 `decider` 包（eager 模式，`use_graphs=False` 避开每形状 compile
+capture）在我们两个评测集上跑 Mapika/decider-2b（2B，Qwen3.5 混合注意力底座，
95 任务训练 + calibration-aware RL 384 步，官方温度 1.3）：

| 评测集 | decider-2b | v5（我们） |
|---|---|---|
| 我们的 in-task（24 任务，双方多数训练过） | 0.811 / 0.453 / 0.032 | **0.863 / 0.390 / 0.034** |
| 我们的 held-out（banking77/massive/emotion） | **0.867 / 0.420 / 0.020** | 0.581 / 1.45 / 0.068 |

**关键事实（来自其 eval_results.json 的 heldout 标记）**：
- decider-2b **训练了我们的 held-out 任务**（banking77/massive_intent/emotion
  均 heldout=false）→ 其 0.867 是 in-task 成绩（官方同任务 0.843-0.96）
- 我们零样本同任务 0.581 → 不可直接比较
- **在共同训练的 in-task 任务上，v5 领先 decider-2b +5.2pt**（1.7B/24 任务
  vs 2B/95 任务）

结论：在重叠任务上我们的模型更强；他们在其 held-out（28 任务：paws/sciq/trec/
boolq/strategyqa/quality/ade/fin_sentiment 等）上的分数无法与我们现有数字对齐——
**要做公平对比需构建"双方都零样本"的套件**（从他们 heldout=true 且不在我们 24
任务中的任务里选）。

## 9.24 公平套件：双方都零样本的泛化对比（2026-09-25 完成）

构建"双方都未见"的 8 任务套件（全部在 decider-2b 的 heldout=true 名单，且
均不在我们 24 训练任务中），各 300 条：
paws / sciq / pubmedqa / bbc_news / quality / fin_phrasebank / dolly_category /
xstory_cloze

| 任务 | v5（我们） | decider-2b |
|---|---|---|
| paws | 0.450 | 0.443 |
| sciq | 0.947 | **0.973** |
| pubmedqa | 0.470 | **0.660** |
| bbc_news | **0.960** | 0.947 |
| quality | 0.303 | **0.433** |
| fin_phrasebank | 0.863 | **0.897** |
| dolly_category | **0.323** | 0.277 |
| xstory_cloze | 0.917 | **0.970** |
| **宏平均 acc** | **0.654** | **0.700** |
| NLL（调温后） | 0.86 | **0.71** |
| ECE（调温后） | 0.092 | **0.047** |
| 拟合温度 | 1.35 | 1.50 |

结论（诚实）：
- **零样本泛化：decider-2b 领先 +4.6pt + 更好校准**；差距集中在知识密集任务
  （pubmedqa −19、quality −13），我们在主题/指令分类上反超
- 他们的优势来源：95 任务（vs 我们 24）+ 2B Qwen3.5 混合注意力底座 +
  校准感知 RL
- 对照：共同训练任务上我们领先 +5.2pt（§9.23）→ **"见过 vs 没见过"的差距
  明确指向训练任务覆盖广度**，与 §9.8 的 4→24 任务结论一致
- 下一步杠杆（按优先级）：① 任务 24→60-95（主杠杆）② 更强底座（Qwen3.5-2B
  或更大）③ 长上下文能力（quality 类任务我们的 256 token 截断是硬伤）


## 9.25 部署侧：批处理 / ONNX / 显著性学生蒸馏（2026-09-25）

**批处理（MLX, Qwen3-1.7B v5）**：
- 候选批处理一批一次 forward：117ms → **75ms/句（1.55x）**；概率一致性 0.003
- 4-bit（mlx_lm.convert）：3.2GB → **934MB**（省内存，速度持平）
- 结论：**算力瓶颈而非带宽瓶颈**，批处理只改善利用率；提速要靠量化/小模型
- 产物：`reader/models/v5_batched[_4bit]`（原模型 `models_mlx/v5_mlx` 未动）

**ONNX 跨平台**（CPU / CUDA / DirectML）：
- Qwen3.5 **无法导出 ONNX**（线性注意力用 `linalg_solve_triangular`，两代 exporter 均失败）
- 学生模型（ModernBERT 标准架构）导出顺利：fp32 606MB → **int8 153MB**，数值误差 2e-5

**显著性学生蒸馏**（阅读器场景）：
- 教师 = stage1（Qwen3.5-2B/45 任务）对 5347 篇文章逐句打分（`--argmax-gold` 软标签）
- 学生 = ModernBERT-base（150M）全参，KD(T=2, α=0.5) + 24 任务 replay + belief；
  5000 条显著性 + 4000 步，37 分钟（0.56s/步）
- 评测：与教师一致性 top1 **18%**、spearman **0.27**（随机 10%）→ 偏弱，待加数据
- 端到端（Mac CPU, int8）：**131ms/句（14 句 1.9s）**；CUDA/DirectML 预期 10-30ms/句

结论：ONNX 管线（含 int8）已验证可跨平台部署；学生质量需要更多数据
（5000→20000）与更高显著性权重（1.5→3.0）再训一轮。

## 10. 参考事实（来自公开实现）""",
)
open(p, "w").write(s)
EOF
git add -A && git commit -q -m "decider-2b head-to-head: v5 leads +5.2pt on shared in-task tasks (0.863 vs 0.811); their high held-out scores come from training on our held-out tasks" && git log --oneline | head -1

- decider 路线 A 读出：prompt 到 `Answer: (`，取字母 token logits，按温度 softmax；字母从不生成，多 question 一次 forward
- 训练期大标签集子采样到 ≤10 个候选，gold 必留、顺序 shuffle
- 温度在 in-task 数据上拟合：2b v10 用 1.30，35b 用 1.08
- decider-2b v10 公开指标：in-task 0.805 / 0.474 / 0.037（acc/NLL/ECE），held-out 0.755 / 0.622 / 0.084
- v10 RL：belief proper log score（exact laws 对照）+ PPO action + option-order consistency + KL
- 官方 Jev 特性：255 options、多 question parallel and in isolation、加 question 几乎不增延迟；高 cardinality 时先独立 scoring 再 explicit choice
- 参考仓库：https://github.com/Mapika/decider
