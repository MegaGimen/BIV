# refs：世界模型当下层、agent 当上层

本目录只收 **arXiv HTML 转出的纯文本**，不收 PDF。正文在 `papers/`。
`python3 refs/fetch_html_text.py` 可重拉。下面的分组对应一个具体问题：

英文理解能当 agent 的底座，是因为 agent 仍在同一套 next-token 目标上往上长。
代码 / 医疗 / 法律能 merge，是因为目标函数同类、领域平行。
世界模型 \(P(o\mid h,a)\) 和 agent \(P(a\mid h)\) 抢同一助手位，目标相悖，平行 merge 的前提不成立。
想做的事：把 agent 做成世界模型的上层建筑，而不是把两个相悖向量加在一起。

## 怎么读

先读本组 2–3 篇，不必按发表年从头扫。

| 组 | 问的是什么 | 先看 |
|----|------------|------|
| A | 为什么 Chat Vector / task arithmetic 在目标相悖时会塌 | Chat Vector、TATR、CAT Merging、LATA |
| B | 经典「先世界、后策略」长什么样（真分层，不是共参数硬加） | DreamerV3、UniPi、DreamZero / WAM、LED-WM |
| C | 语言 agent 里，动作和下一状态怎样共存而不互相洗掉 | DyMo、PaW、ECHO、RWML、RAP |
| D | 穷的时候怎样只加一层、不改底座 | O-LoRA、Ortho-LoRA、LST side-tuning、LIMA |
| E | 官方世界模型自己怎么用底座 | Qwen-AgentWorld §6.2 |

文件名：`papers/{arxiv}-{slug}.txt`。文内第一行是来源 URL。

## A. Merge 在目标相悖时为什么失败

平行领域 merge 默认：各任务向量近似正交、都从同一 Base 出发、输出语义同类。
WM vs agent 三条都不满足。

- **Chat Vector**（[2310.04799](https://arxiv.org/abs/2310.04799)）— 你们现在的公式来源：\(\theta_{\text{CPT}} + (\theta_{\text{chat}}-\theta_{\text{base}})\)。它能把「新语言 CPT」接上 chat，是因为 CPT 仍是语言建模，和 chat 同族。不是给「观察生成器 + 决策器」用的。
- **Task Arithmetic**（[2212.04089](https://arxiv.org/abs/2212.04089)）— 任务向量能加，前提是微调位移小、任务不太对着干。
- **TATR**（[2501.15065](https://arxiv.org/abs/2501.15065)）— 把 merge 失败写成 *knowledge conflict*：向量里顺着对方任务梯度的分量会互相伤害；只保留近似正交的「信任域」。
- **CAT Merging**（[2505.06977](https://arxiv.org/abs/2505.06977)）— 训练无关地剪掉冲突分量。
- **LATA**（[2502.20186](https://arxiv.org/abs/2502.20186)）— 按层拆「指令遵循」和「任务知识」。和「IF 是薄上层、领域是底座」同构，可借鉴分层加权，不要整网一个 \(\lambda\)。
- **Demystifying Mergeability**（[2601.22285](https://arxiv.org/abs/2601.22285)）— merge 成不成功不是模型固有属性，跟方法和表征冲突有关。
- **TIES**（[2306.01708](https://arxiv.org/abs/2306.01708)）/ **DARE**（[2311.03099](https://arxiv.org/abs/2311.03099)）— 符号选举 / 随机稀疏。目标对着干时，选举往往是「选边」而不是「合成」。

## B. 先世界、后策略（上层建筑的正例）

这些工作不把两个相悖目标塞进同一套全参、同一个助手位。世界模型先成为动力学底座，策略在其上生成。

- **DreamerV3**（[2301.04104](https://arxiv.org/abs/2301.04104)）— 先学 \(P(s_{t+1},r\mid s,a)\)，策略只在世界模型的想象里训练。这是最干净的「WM 底座 + actor 上层」。
- **UniPi**（[2305.16309](https://arxiv.org/abs/2305.16309)）— 视频世界模型想象未来，**逆动力学** \(P(a\mid s,s')\) 再把未来译成动作。agent 明确是世界模型的反函数，不是再训一个抢同一输出头的 LM。
- **DreamZero / World Action Model**（[2602.15922](https://arxiv.org/abs/2602.15922)）— 口号就是 *pretrained to imagine, fine-tuned to act*：视频世界模型当先验，再把动作当成另一种模态接上去。和「英文先验上长 agent」同构，只是底座换成世界动力学。
- **LED-WM**（[2511.22904](https://arxiv.org/abs/2511.22904)）— 语言条件世界模型训完，策略完全在该模型里学，不在真实环境里对 WM 反传。
- **PriorZero**（[2605.12289](https://arxiv.org/abs/2605.12289)）— LLM 先验和世界模型 **分开目标、交替更新**，明确写了「LLM 先验不得干扰动力学保真」。

语言 agent 很难做成 Dreamer 那种潜空间 actor；可借的是结构：底座冻结或少动，上层另开一条动作通道（adapter / 逆动力学 / 交替优化）。

## C. 同一套 LM 里动作和观察怎么共存

若必须共用一个 35B，就不要 merge，要 **分 token、分损失、分更新幅度**。

- **DyMo**（[2506.02918](https://arxiv.org/abs/2506.02918)）— 同一次生成既出 tool call 又出下一状态；训练信号是联合的，推理时用内部模拟做 Self-Verification。这是「不要让两个目标抢独享助手位，让它们成为一次推理的两段」。
- **PaW**（[2606.02388](https://arxiv.org/abs/2606.02388)）— 政策 RL + 辅助下一观察损失，\(\lambda\) 随回报降。处理的就是你说的目标冲突，但是 **agent 底座上加 WM**，和你们顺序相反；损失平衡可以倒过来用。
- **ECHO**（[2605.24517](https://arxiv.org/abs/2605.24517)）— GRPO 只打 action token，观察 token 另加 CE；同一前向、两套 mask。TB2.1 上有数。结构可反：WM 底座冻结，只对 action 开 LoRA。
- **RWML**（[2602.05842](https://arxiv.org/abs/2602.05842)）— 纯 WM-SFT 会冲掉 IF / 数学 / 代码；用 RL 学世界模型忘得少。说明 **SFT 把助手位拧成观察生成器** 才是相悖的来源，不是「学了世界」本身。
- **RL vs SFT circuits**（[2605.28860](https://arxiv.org/abs/2605.28860)）— 机制上：SFT 改电路快、忘得多；RL 贴着原策略、底座留得住。若要在 AgentWorld 上长 agent，优先小步 RL / 近端更新，而不是大力 SFT。
- **RAP**（[2305.14992](https://arxiv.org/abs/2305.14992)）— 同一 LM 既当策略又当世界模型，用在规划里，权重不必 merge。

## D. 只加一层、不改底座

对应「英文理解几乎不动、agent 是上层建筑」在参数上的做法。

- **LIMA**（[2305.11206](https://arxiv.org/abs/2305.11206)）— 指令遵循可以很薄，**前提是底座还是通用 LM**。AgentWorld 的助手位已被观察 SFT 钉死，不能直接拿 LIMA 当「5k 就够」的证据。
- **O-LoRA**（[2310.14152](https://arxiv.org/abs/2310.14152)）— 新任务 LoRA 与旧任务子空间正交，减轻续训覆盖。WM LoRA 冻住，agent LoRA 走正交补空间。
- **Ortho-LoRA**（[2601.09684](https://arxiv.org/abs/2601.09684)）— 多任务梯度投影到彼此正交补，专门打 *gradient conflict*。
- **LST side-tuning**（[2206.06522](https://arxiv.org/abs/2206.06522)）— 主干冻结，侧网吃中间激活再出新任务。最接近「WM 当表征、agent 当侧头」。

## E. Qwen 自己的底座用法

- **Qwen-AgentWorld**（[2606.24597](https://arxiv.org/abs/2606.24597)）— §6.2 Table 9：从 **已经会当 agent 的 SFT 模型** 做 LWM RL，再零微调上 TB2.1。这是 agent 底座 + 世界上层。你们要测的是反序；文里 **没有**「开源模拟器权重再 SFT 成通用 agent」的成功对照。

## 检索词

按问题搜，不要只搜 `world model agent merge`。

**目标相悖 / merge 塌掉**

```
task arithmetic knowledge conflict
model merging collapse representational incompatibility
task arithmetic trust region TATR
CAT merging conflict
layer-aware task arithmetic instruction following
demystifying mergeability
chat vector continual pretraining instruction following
```

**世界当下层、策略当上层**

```
pretrained to imagine fine-tuned to act
world action model inverse dynamics
frozen world model actor-critic Dreamer
UniPi video inverse dynamics policy
LED-WM language-conditioned world model then policy
decoupled world model LLM policy alternating update
```

**同一 LM：动作 vs 观察**

```
auxiliary next-observation loss agent RL
ECHO environment cross-entropy GRPO
policy world modeling co-training PaW
DyMo joint tool-call next-state
RWML world model SFT catastrophic forgetting
separate action observation token mask LLM
```

**薄上层、不冲底座**

```
O-LoRA orthogonal subspace continual learning
orthogonal gradient projection LoRA task conflict
side-tuning frozen backbone new head
OPLoRA orthogonal projection pretrained singular vectors
LIMA instruction following thin layer
RL preserves circuits better than SFT
```

**和「英文→agent」类比直接对齐**

```
language prior as substrate policy as superstructure
world model as foundation model then agent SFT
hierarchical world model policy
forward model then inverse dynamics
```

## 更新

```bash
python3 refs/fetch_html_text.py
```

只拉 HTML（arXiv `/html/`，不行再试 ar5iv），用 `w3m -dump` 转纯文本。
