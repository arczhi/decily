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

## 10. 参考事实（来自公开实现）

- decider 路线 A 读出：prompt 到 `Answer: (`，取字母 token logits，按温度 softmax；字母从不生成，多 question 一次 forward
- 训练期大标签集子采样到 ≤10 个候选，gold 必留、顺序 shuffle
- 温度在 in-task 数据上拟合：2b v10 用 1.30，35b 用 1.08
- decider-2b v10 公开指标：in-task 0.805 / 0.474 / 0.037（acc/NLL/ECE），held-out 0.755 / 0.622 / 0.084
- v10 RL：belief proper log score（exact laws 对照）+ PPO action + option-order consistency + KL
- 官方 Jev 特性：255 options、多 question parallel and in isolation、加 question 几乎不增延迟；高 cardinality 时先独立 scoring 再 explicit choice
- 参考仓库：https://github.com/Mapika/decider
