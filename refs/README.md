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

## F. 经典 / 3D 物理世界模型：把规律写进参数

下一观察 CE 只拟合表面；物理 WM 用结构或损失把 \(F=ma\) 一类约束写进参数，外观只是渲染。

- **PINNs**（[1711.10561](https://arxiv.org/abs/1711.10561)）— PDE 残差进 loss，不是只拟合观测点。
- **HNN / LNN**（[1906.01563](https://arxiv.org/abs/1906.01563), [2003.04630](https://arxiv.org/abs/2003.04630)）— 网络输出能量，动力学由方程生成，守恒是结构。
- **Neural ODE**（[1806.07366](https://arxiv.org/abs/1806.07366)）— 连续时间状态转移。
- **Ha World Models / PlaNet**（[1803.10122](https://arxiv.org/abs/1803.10122), [1811.04551](https://arxiv.org/abs/1811.04551)）— 潜动力学是世界，像素是渲染；策略在潜空间上训。
- **GNS**（[2002.09405](https://arxiv.org/abs/2002.09405)）— 状态是粒子图，不是像素。
- **NeuMA / DeformMaster**（[2410.08257](https://arxiv.org/abs/2410.08257), [2605.09586](https://arxiv.org/abs/2605.09586)）— \(\mathcal{M}=\mathcal{M}_0+\Delta\mathcal{M}\)：冻物理先验，残差/LoRA 只补未建模项。
- **PhysGaussian / Physically Native WM**（[2311.12198](https://arxiv.org/abs/2311.12198), [2605.00412](https://arxiv.org/abs/2605.00412)）— 外观跟物理轨迹走；核心是潜动力学不是视频生成。
- **Cosmos / PhyWorld / Phys4D**（[2501.03575](https://arxiv.org/abs/2501.03575), [2605.19242](https://arxiv.org/abs/2605.19242), [2603.03485](https://arxiv.org/abs/2603.03485)）— 外观预训练会幻觉物理，再用仿真监督把规律压进去。
- **PSG-JEPA**（[2608.06799](https://arxiv.org/abs/2608.06799)）— 只做前向预测不够，要用物理状态把 latent 钉住。

## G. 从观测发现律，再编进可提前触发的表征

- **发现律：** AI Feynman / SINDy / Othello-GPT / grokking / 因果表征
- **提前躲：** Dyna 把想象编进策略；后继表征把未来占用编进特征；Dreamer/I2A/MuZero；婴儿 VoE

## H. 补三个算法缺口（可识别、显式 z、T→π）

- **律可识别：** CITRIS / iVAE / ICP / Noether / MDP homomorphism
- **状态是预测不是隐故事：** PSR、DeepMDP、DBC、Denoised MDP、SLAC、KG-A2C
- **编进策略：** Successor Features、MuZero、GVF/Horde、MBPO、Value Prediction Networks

---

# 2026-08 检索轮：把 H 的三个缺口逐条收口

下面 I–O 是一次系统检索（Consensus，六轮）的结果，目的只有一个：
**让「世界理解 → agent 变强」这条因果故事的每一段都挂在已有定理上**，并把真正还没人证过的部分单独列出来（见 O）。
H 里那三条「缺口」有两条已经不成立了，改写在 J / L。

## I. 牛顿问题：预测得准 ≠ 学到律（正反两面都有）

用户的原问题是「从苹果掉下来总结出 G=mg，而不是拟合轨迹」。这条现在有直接的实验回答。

- **反面 —— 通用 Transformer 会变成曲线拟合器。**
  [What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](https://arxiv.org/abs/2507.06952)（Vafa 2025）：
  在轨道数据上训练的模型能把行星位置预测得很准，但换到新的物理任务上完全用不出牛顿力学，学到的是**任务专用启发式**。
  同组更早的 [Evaluating the World Model Implicit in a Generative Model](https://arxiv.org/abs/2406.03689) 用 Myhill–Nerode 定理造了一套「世界模型恢复度」指标，
  结论一致：现有诊断（比如探针能读出棋盘）通过了，但世界模型其实支离破碎，换个相近任务就崩。
  → 对我们：`eval_wm.py` 的下一观察 CE 降低**不构成**「学到 OS 规律」的证据，必须换成这类恢复度 / 迁移指标。
- **正面 —— 加三个归纳偏置就能从开普勒走到牛顿。**
  [From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers](https://arxiv.org/abs/2602.06923)：
  在同样的轨道数据上，加 (1) 空间平滑（预测做连续回归而非离散 token）、(2) 稳定性（训练时喂带噪上下文，抑制误差累积）、
  (3) **时间局部性**（把注意力窗口限制在最近状态，强制「未来只依赖当前状态」）——模型就放弃拟合曲线，转而长出牛顿力场表征。
  → 这三条里第三条最关键，它和 K 组的 strict mediation 是同一件事的两种说法：**不许模型回头翻历史，它才会去建状态。**
- **旁证：** [Intuitive physics understanding emerges from self-supervised pretraining on natural videos](https://arxiv.org/abs/2502.11831)（V-JEPA 系）——
  在**表征空间**做预测的模型有直觉物理，**像素空间**预测和**用文本推理的多模态 LLM** 都在随机水平。
  这是「别在观察 token 上算 CE，要在潜空间上算」的最强经验证据。
- **孤证反驳：** [Verification of the Implicit World Model in a Generative Model via Adversarial Sequences](https://arxiv.org/abs/2602.05903)——
  象棋模型里棋盘状态探针**对下一步预测没有因果作用**。探针能读出 ≠ 模型在用。见 N。

## J. 缺口一收口：未知多点干预的可识别性（H 里的「CITRIS 要已知干预目标」已过时）

H 说「CITRIS 需要已知干预目标，而 `python foo.py` 一次动很多变量，所以对不上」。**这条现在不成立了**：
未知目标 + 多点干预的可识别性已经有一批结果。

- [Linear Causal Representation Learning from Unknown Multi-node Interventions](https://arxiv.org/abs/2406.05937)——
  未知**多点**干预下，软干预可识别到祖先、硬干预可完美识别，和单点干预的最好结果持平。
- [Identifying Linearly-Mixed Causal Representations from Multi-Node Interventions](https://arxiv.org/abs/2311.02695)——
  第一个允许一个环境里同时干预多个变量的可识别性结果，靠干预在方差上留下的痕迹 + 稀疏正则。
- [Nonparametric Identifiability of Causal Representations from Unknown Interventions](https://arxiv.org/abs/2306.00542)（von Kügelgen）——
  混合函数和因果模型都非参数、干预目标未知的一般设定下的第一个可识别性结果。
- [Beyond identifiability: Learning causal representations with few environments and finite samples](https://arxiv.org/abs/2603.25796)——
  **对数**数量的未知多点干预就够，且干预目标**不需要事先设计**，同时恢复图、混合矩阵和未知干预目标。有限样本界。
- [Score-based Causal Representation Learning](https://arxiv.org/abs/2402.00849) / [General Identifiability and Achievability for CRL](https://arxiv.org/abs/2310.15450)——
  「非配对 / uncoupled 干预」也能识别，附带可执行算法。
- **谁是干预目标，在 shell 里可以估。** [ShIOEnv](https://arxiv.org/abs/2505.18374) 造了一个 Gym 兼容的 Bash 环境和 210 万条 shell 输入-输出对，
  并提出 **irreducibility 信号**：估计一条命令里有多大比例的参数真正影响了观察到的执行行为。
  这就是「一条命令动了哪些变量」的可操作代理——正好是上面那批定理要的干预目标估计。
- **硬边界还在：** [The Causal-Neural Connection](https://arxiv.org/abs/2107.00793) 证明 Causal Hierarchy Theorem 对神经网络同样成立——
  **再大的模型、再多的观察数据，也推不出干预层的答案**。所以「纯旁观够不够」这个问题，答案是数学上的「不够」，不是工程上的「再练练」。

## K. 缺口二收口：文本里的显式状态 z，且策略只读 z

H 说「PSR 没解决用哪些 test；stdout 是无界文本」。现在文本域有了直接工作。

- **[Textual Belief States for World Models: Identifiable Representation Learning Under Strict Mediation](https://arxiv.org/abs/2606.27681) —— 这篇几乎是给我们写的。**
  它点名了 LLM 世界模型的病：**history bypass**——解码器太强，直接回头读历史，绕过瓶颈，于是预测再准，潜状态也是**不可识别**的。
  解法是 strict mediation（预测只准依赖 z 和 a）。文本潜状态是离散、变长、不可微的，没法变分训练，于是提出 **fGRPO**（树结构 RL）来强制中介。
  TextWorld / ScienceWorld 上一步预测精度不掉，表征质量 +57%，rollout +98%，且任务越长收益越大。
- [Next-Latent Prediction Transformers Learn Compact World Models](https://arxiv.org/abs/2511.05963)（NextLat）——
  在 next-token 之外加一个「预测自己下一个潜状态」的自监督项，**可证收敛到 belief state**，不改架构、不改推理、不影响并行训练。
  → 这是能直接加进我们现有 TRL LoRA 训练的最低成本改动。
- [Guaranteed Discovery of Control-Endogenous Latent States with Multi-Step Inverse Models](https://arxiv.org/abs/2207.08229)（AC-State）——
  多步逆模型 + 信息瓶颈，**有理论保证地**只留下「agent 能控制的」状态，把无关噪声全丢掉。
  → 对应 stdout 里大量与任务无关的日志噪声。
- [Learning World Models with Identifiable Factorization](https://arxiv.org/abs/2306.06561)（IFactor）——
  按「与动作有关 / 与奖励有关」把潜变量分成四块，块级可识别。
- **PSR 的「用哪些 test」也有进展：**
  [Spectral Learning of PSRs with Insufficient Statistics](https://aaai.org/papers/…)（[AAAI 2015](https://doi.org/10.1609/aaai.v29i1.9635)）给出一个**谱界**作为选 test 的判据；
  [PAC Reinforcement Learning for Predictive State Representations](https://arxiv.org/abs/2207.05738) 证明 PSR 在函数逼近下有多项式样本复杂度，
  且**不显式依赖状态/观察空间大小**——对「stdout 无界」这条正好对症。

## L. 缺口三收口：「世界理解 → agent 变强」是有定理的

H 说「DeepMDP 是奖励/控制的界，SF 假设 φ 已给定，循环论证」。这两条现在都能补。

- **低秩 MDP 这条线就是这个命题的形式化。** 如果转移核可以写成 \(P(s'|s,a)=\langle\phi(s,a),\mu(s')\rangle\)，
  那么**从转移数据里学 φ** 就能让后续策略学习变得样本高效：
  [FLAMBE](https://arxiv.org/abs/2006.10814) → [REP-UCB](https://arxiv.org/abs/2110.04652)（样本复杂度从 \(\epsilon^{-10}\) 砍到 \(\epsilon^{-2}\)）→
  [SPEDER](https://arxiv.org/abs/2208.09515)（谱分解出 state-action 抽象，不受采样策略污染）→
  [Spectral Representation-based RL](https://arxiv.org/abs/2512.15036)（把谱视角推广到部分可观测）。
- **「上游多领域学表征、下游省样本」有明确的界。**
  [Provable Benefit of Multitask Representation Learning in RL](https://arxiv.org/abs/2206.05900)：
  上游多任务 **reward-free** 学共享表征，任务数超过阈值后就严格优于单任务各自学；
  下游新任务的次优性上界 = **上游表征估计误差 + 一个随下游样本增大而消失的项**。
  → 这正是我们 `wm_code : wm_os` 多领域混合的理论依据，也说明「上游表征学得越准，下游 agent 上限越高」不是比喻。
  另见 [Provable Benefits of Representational Transfer in RL](https://arxiv.org/abs/2205.14571)。
- **但要老实报边界。** [On the Power of Pre-training for Generalization in RL](https://arxiv.org/abs/2210.10464)：
  如果允许在目标环境里继续交互，**渐近**上预训练的好处最多是个常数因子；真正的收益在**非渐近**区间。
  → 对应到我们：不要宣称「世界模型底座能无限拉高 agent 天花板」，要宣称「在有限数据/有限 RL 预算下更快更高」。
- **φ 的循环论证已被打破。**
  [Does Zero-Shot Reinforcement Learning Exist?](https://arxiv.org/abs/2209.14935) 指出 SF 的表现严重依赖基础特征的选法，
  而 **Forward-Backward (FB) 表征从单一准则里同时学出基础特征和后继特征**，在 URL benchmark 上稳定拿到有监督 RL 的 85%。
  [Which Features are Best for Successor Features?](https://arxiv.org/abs/2502.10790)（Ollivier）第一次**非同义反复地**给出最优基础特征，
  且对三类下游任务族答案相同。[Proto Successor Measure](https://arxiv.org/abs/2411.19418) 给出「所有可能行为」的基底集。
- **自预测损失做的正是谱分解。** [Understanding Self-Predictive Learning for RL](https://arxiv.org/abs/2212.03319) 证明
  自预测学习的动力学在理想情形下**对状态转移矩阵做谱分解**，并指出防塌缩靠的是「predictor 更新更快 + 表征上用半梯度（stop-gradient）」。
  [Bridging State and History Representations](https://arxiv.org/abs/2401.08898)（Ni 2024）把一堆看似不同的状态/历史抽象统一成「自预测抽象」，给出极简算法。
  → 把 L 的第一条和这条接起来：**自预测辅助损失 → 转移算子的谱特征 → 低秩 MDP 的 φ → 下游策略样本复杂度下降。这就是缺失的组合链条的主干。**
- **T→π 的编译器也有保证版本。**
  [Theoretically Guaranteed Policy Improvement Distilled from Model-Based Planning](https://arxiv.org/abs/2307.12933)（MPDP）给出**单调改进 + 收敛**保证；
  [SAVE](https://arxiv.org/abs/1912.02807) 把 MCTS 的价值摊销回 Q 网络；
  [Amortized Planning with Large-Scale Transformers](https://arxiv.org/abs/2406.11907)（ChessBench）说明搜索**可以但不能完美**蒸进 Transformer；
  [Latent Geometry Beyond Search](https://arxiv.org/abs/2605.08732) 显示潜空间足够规整时，规划可以摊销成一个逆动力学映射，单步决策成本降 100 倍。
- **LLM 侧已经有人做同一件事。**
  [Dyna-Mind](https://arxiv.org/abs/2510.09577)（ReSim + Dyna-GRPO）、[ProAct](https://arxiv.org/abs/2602.05327)（把搜索树压成因果推理链再 SFT）、
  [Internalizing the Future](https://arxiv.org/abs/2606.27483)（WM-AMT → FE-SFT → FC-RL，并点名 **format-capability gap**：
  直接在 look-ahead 轨迹上微调只会学到「假装有远见」的格式）、
  [SPA](https://arxiv.org/abs/2510.15047)（自博弈 SFT 先学世界模型再 RL）、[COMAP](https://arxiv.org/abs/2606.02372)（世界模型与策略共演化）。
  → **`Internalizing the Future` 的三阶段几乎就是用户想要的顺序**，且它自己承认「先注入能力、再对格式」是必须的——
  这对我们「AgentWorld 上小规模 agent SFT 只是格式胶水」的判断是正面印证。

## M. 目标冲突不是玄学：辅助任务何时帮、何时伤

- [When does Self-Prediction help? Understanding Auxiliary Tasks in RL](https://arxiv.org/abs/2406.17718)（Voelcker 2024）——
  线性模型下的学习动力学分析：**潜自预测是好的辅助任务，而观察重建单独用时特征更好、当辅助任务时反而拖累**。
  → 直接解释了为什么「在助手位上算下一观察 token 的 CE」会和策略打架，而「在潜空间上自预测」不会。
- [RWML](https://arxiv.org/abs/2602.05842) 在 LLM agent 上给出对应的经验版本：
  **token 级下一状态预测会导致模型塌缩**（去抠字面措辞而非语义），改成在预训练嵌入空间里对齐模拟状态与真实状态就稳，
  而且比 LLM-as-a-judge 更抗 reward hacking。ALFWorld / τ²-Bench 上纯自监督也有明显增益。
- [Understanding and Improving Information Transfer in MTL](https://arxiv.org/abs/2005.00944)——
  共享主干 + 各自输出头的设定下，**任务数据是否对齐**决定正/负迁移，并给出正迁移的充分条件。
- [Capacity–Redundancy Trade-offs in Multi-Task Learning](https://arxiv.org/abs/2607.16554)——
  给出「分簇共享何时优于全局共享」的**充要条件**，并用 gradient–total-correlation 桥证明**梯度余弦相似度**可以当冗余度代理。
  → 这是「WM 和 agent 到底该不该共享同一套参数」的可测判据，比我们之前拍脑袋的 λ 强。
- [Adapting Auxiliary Losses Using Gradient Similarity](https://arxiv.org/abs/1812.02224)——
  用梯度余弦相似度动态调辅助损失权重，**保证收敛到主任务的临界点**。PaW 的 λ 调度可以换成这个有保证的版本。
- [ECHO](https://arxiv.org/abs/2605.24517)（终端域）与 [PaW](https://arxiv.org/abs/2606.02388)（通用 agent RL）是这条思路在我们目标域上的两个实例。

## N. 评测：怎么证明学到的是律，不是表面统计

- **VoE 范式（看到不可能事件会不会「惊讶」）：**
  [IntPhys 2](https://arxiv.org/abs/2506.09849)、[X-VoE](https://doi.org/10.1109/iccv51070.2023.00369)、[Piloto 2022](https://doi.org/10.1038/s41562-022-01394-8)。
  [A Probabilistic Explanation for VoE-based Evaluation] 指出传统 surprise 分数设计是经验性的，给了似然比理论下的两个更合理的分数。
- **零成本反事实：** [YoCausal](https://arxiv.org/abs/2605.30346) 把真实视频**时间反转**当天然反事实，定义 Reverse Surprise Index；
  并证明「能感知时间箭头」**不等于**「理解因果」。
  → 可直接搬到文本：把工具轨迹的 (a,o) 顺序反转/打乱，看模型的 surprise 是否上升。这比我们现有的 shuffled-observation twin 更细。
- **探针 ≠ 因果，这条必须写进评测协议：**
  [Latent Planning Emerges with Scale](https://arxiv.org/abs/2604.12493) 把「潜规划」定义成必须同时满足**可解码**和**因果驱动生成**；
  [Where's the Plan?](https://arxiv.org/abs/2605.07984) 发现在十几个模型里，只有 Gemma-3-27B 在行末真正**因果依赖**那个编码，其他模型探针信号强但因果效应近零；
  [Causality is Key for Interpretability Claims to Generalise](https://arxiv.org/abs/2602.16698) 把 Pearl 三层直接映射到「观察相关 / activation patching / 反事实」，说明哪一层证据支持哪一类主张。
  → 我们的「铡刀举起就侧身」实验，**光有线性探针不够，必须补 activation patching**，否则只能声称相关。
- **别高估潜规划深度：** [The Depth Ceiling](https://arxiv.org/abs/2604.06427)——
  从零训的小 Transformer 只能发现 3 步潜规划，微调过的 GPT-4o / Qwen3-32B 到 5 步，GPT-5.4 few-shot 到 7 步；
  训练期能学到的上限是 5 步，但学会后测试时能外推到 8 步。
  → 「潜意识提前躲」在 3–7 步这个量级内是现实的，超出就得外化成 CoT。
- **System-2 压进 System-1 的现状：** [Implicit CoT via Knowledge Distillation](https://arxiv.org/abs/2311.01460)、
  [CODI](https://arxiv.org/abs/2502.21074)、[SIM-CoT](https://arxiv.org/abs/2509.20317)（指出潜 token 变多会**同质化塌缩**，要加步级监督）、
  [Do Latent-CoT Models Think Step-by-Step?](https://arxiv.org/abs/2602.00449)（CODI 在 2–3 跳上是真的逐步算，跳数一多就退化成捷径）。

## P. 自由文本动作怎么进定理（O.1 的补丁）

上面所有低秩 MDP / 可识别性结果都要求动作可嵌入。bash 命令能不能算「可嵌入动作」，这条线其实有现成办法：
**不要把动作当成 token 串，先学一个动作表征，再让策略在表征空间里出手。**

- [Learning Action Representations for RL](https://arxiv.org/abs/1902.00183)（Chandak 2019）——
  把策略拆成「在低维动作表征空间里选点」+「把点还原成真实动作」两段，**给出了收敛条件**。
  好处是能对**没见过的动作**做外推：见过 `rm -f a.txt`，就能推断 `rm -rf b/` 的后果。
- [Jointly-Learned State-Action Embedding](https://arxiv.org/abs/2010.04444)（Pritz 2020，[CIKM 版](https://doi.org/10.1145/3459637.3482357)）——
  **给出了「用嵌入后的状态和动作训 RL 是有效的」这件事的理论基础**，正是 P 这一节要的那块砖。
- [Deep RL in Large Discrete Action Spaces](https://arxiv.org/abs/1512.07679)（Dulac-Arnold）——
  百万级动作，靠嵌入 + 近似最近邻做到**对数时间**查找。
- [The Natural Language of Actions](https://arxiv.org/abs/1902.01119)（Act2Vec）——
  直接用「动作的上下文」学动作向量，等于给 shell 命令做 word2vec。
- [Generalization to New Actions in RL](https://arxiv.org/abs/2011.01928) / [AGLO](https://arxiv.org/abs/2503.08867)——
  零样本泛化到训练时没有的动作集合，两阶段：先从动作自身的信息推表征，再训一个对动作集合不敏感的策略。
- [For SALE](https://arxiv.org/abs/2306.02451)（Fujimoto）——state-action 联合嵌入的强经验版本。
- [Outcome-Predictive State Representations](https://arxiv.org/abs/2604.07016)（OPSR, 2026）——
  把状态定义成「对任务无关的 outcome 的预测」，**形式化证明了它带来的迁移是最优但有限的**，
  再用基于 OPSR 的 skill（抽象动作）突破这个上限。这是 PSR 思路 + 动作抽象的合流，和 K 组直接接得上。
- [Language Representations for Generalization in RL](https://consensus.app/papers/details/10901a6c273b5bd699c3e1a3e69b78c1/)——
  用语言当状态/动作表征的 agent 泛化更好，作者归因于**语言的组合性**。对 shell 这种高度组合的动作空间是正面信号。

→ 结论：O.1 不再是「没人做过」，而是「没人在 shell 规模上做过」。可做的最小实验是给命令学一个动作嵌入，
看下一观察预测能否对**没见过的 flag 组合**外推。

## Q. 把律写成程序：文本域的可执行世界模型（含一条对我们最要命的实证）

这一组和 WorldCoder 是同一家族，但 2025–2026 已经推进到**文本 agent 环境**和**随机、无引导探索**了。
注意：这类做法把律放在**程序**里而不是**参数**里，对我们不是主线（AGENTS.md 已经否掉硬编码 tracker），
但它们的**评测协议和失败模式**对我们极有价值。

- **[PatchWorld: Gradient-Free Optimization of Executable World Models](https://arxiv.org/abs/2605.30880) —— 这篇给了我们最想要的那条实证。**
  在七个 AgentGym 文本环境上，它发现一个明确的 trade-off：
  **提高「表面观察保真度」会削弱「动作可判别的动力学」，反之亦然。**
  换句话说，用户担心的「世界模型目标和 agent 目标打架」不是猜想，在文本世界模型里被直接测出来了。
  → 这条应该写进我们的实验假设：`eval_wm.py` 的观察保真度指标和 agent 指标**可能负相关**，必须同时报。
- [OneLife](https://arxiv.org/abs/2510.12088)——随机环境、**只有一条命**、无人引导，把动力学建成
  「前置条件 → 效果」的**条件激活式程序律**，只有相关的律参与推理。
  评测拆成 **state ranking**（能不能把合理的未来排在不合理的前面）和 **state fidelity**（生成的未来像不像）两项。
  → 这个二分正好对应 N 组的 VoE：**ranking 才是「懂律」，fidelity 只是「像」。**
- **[Baba in Wonderland / Alice](https://arxiv.org/abs/2605.16725) —— 「律 vs 表面词汇」的干净判据。**
  把 Baba Is You 的规则词换成**无关词**，保留底层动力学，看模型是靠语义捷径还是靠交互证据。
  → 直接可搬到我们这里：把 shell 命令名做**一致重命名**（`rm`→`zaq`），律学到了就该照样预测「文件没了」。
  这比 shuffled-observation twin 更能分辨「记住了 rm 的语义先验」和「学到了删除这个转移」。
- [NeSyS](https://arxiv.org/abs/2602.10480)——符号 WM 与神经 WM 交替训练，**符号规则直接改 LLM 的输出概率分布**，
  神经 WM 只在符号规则覆盖不到的轨迹上微调，数据量减半而精度不掉。
  注意：这是**规则被学出来**的混合体，不是我们否掉的手写 tracker；但律仍在程序里，不在参数里。
- [TheoryCoder](https://arxiv.org/abs/2503.20124) / [TheoryCoder-2](https://arxiv.org/abs/2602.00929)——
  Theory-Based RL：把「理论」当前向模拟器，TheoryCoder-2 进一步**自己学抽象**而不是靠人给。
- [PoE-World](https://arxiv.org/abs/2505.10819)（程序专家的乘积）、[Mind-Studio](https://arxiv.org/abs/2606.16070)（K 步前瞻保真度协议）、
  [OPINE-World](https://arxiv.org/abs/2607.01531)（ontology error 引导探索，ARC-AGI-3 上 25 局解 20 局）。
- [Combining Functional and Automata Synthesis](https://doi.org/10.1145/3571249)（Autumn/Das 2023）——
  在函数综合和自动机综合之间迭代，**自动补出解释观察所必需的隐状态**。
  → 这是 K 组「z 该有多大」的一个可判定性版本答案：隐状态不是拍脑袋定的，是被观察逼出来的。
- [NSI](https://arxiv.org/abs/2605.01293)（把 agent 轨迹提升成带控制流的逻辑程序技能）、
  [CodeARC](https://arxiv.org/abs/2503.23145)（可交互查询隐藏函数的归纳综合 benchmark，o3-mini 也只有 52.7%）。

## R. 谁来选 `do(a)`：主动干预 / 好奇心（CHT 硬墙的唯一出口）

CHT 说观察数据推不出干预层。我们的语料是**离线**的，`do(a)` 是别人选的。要过这堵墙只能让 agent 自己选实验。

- **「探索学得好的 agent，后来做任务也更好」是被实验证过的。**
  [Learning and exploration in action-perception loops](https://doi.org/10.3389/fncir.2013.00037)（Little & Sommer 2013）
  提出 **Predicted Information Gain (PIG)**，在无外部奖励下纯为提升内部模型质量而探索，
  并明确验证：**探索期学得更高效的 agent，之后在一系列目标导向任务上表现更好。**
  → 这是「世界理解 → agent 变强」在受控环境里的直接证据，虽然是小规模，但因果方向是我们要的那个。
- **白噪声陷阱 = stdout 里的时间戳和哈希。**
  [Active World Model Learning with Progress Curiosity](https://arxiv.org/abs/2007.07853)（γ-Progress）用「学习进度」而不是「预测误差」当信号，
  从而不去追那些**不可预测但也学不会**的东西；
  [Curiosity-Critic](https://arxiv.org/abs/2604.18701)（2026）更进一步，用一个协同训练的 critic 在线估计每个转移的
  **不可约噪声下界**，把 epistemic（可学）和 aleatoric（不可学）误差在线分开，
  并指出 Schmidhuber 1991 以来的各种预测误差好奇心都是它的特例。
  → 对我们**极其对症**：shell 输出里一大半 token（时间戳、PID、哈希、进度条）是纯 aleatoric，
  在观察 token 上算 CE 等于把大量算力浪费在噪声上。这给「不要在观察 token 上直接算 CE」又添一条理由。
- [MaxInfoRL](https://arxiv.org/abs/2412.12098)（信息增益内在奖励 + Boltzmann，简化设定下 sublinear regret）、
  [PTS-BE](https://arxiv.org/abs/2507.02639)（证明 epistemic bonus 在模型确定后收敛到零，之前这类方法没有理论保证）。
- 主动实验设计侧：[CAASL](https://arxiv.org/abs/2405.16718)（摊销式主动干预设计，训练一个策略直接输出下一个该做的干预）、
  [Active ICP](https://arxiv.org/abs/2006.05690)、[Interventions, Where and How?](https://arxiv.org/abs/2203.02016)。
- **一条重要的负面结果：** [Can In-Context Learning Support Intrinsic Curiosity?](https://arxiv.org/abs/2606.19476)（2026）
  证明在**一般 MDP** 里，用 in-context learner 的预测误差去无偏地估计「学习进度」是**不可能**的；
  但在**非时序**子类（主动学习 / 贝叶斯实验设计）里可以，且有收敛保证。
  → 意思是：想让 LLM「靠 in-context 预测误差」自己决定下一步做什么实验，在完整 agent 轨迹上没有无偏保证；
  但如果把探索缩成「选一条独立的命令去试」这种非时序设定，理论是站得住的。这直接框定了我们该怎么设计干预采集。
- [Active Inference, Curiosity and Insight](https://doi.org/10.1162/neco_a_00999)（Friston 2017）——
  用贝叶斯模型约简解释「顿悟」：agent 主动检验关于**对称性/不变量**的假设。
  这是「牛顿式发现」在自由能框架下的形式化，和 I 组的归纳偏置、J 组的不变量是同一件事的第三种说法。

## S. 参数里的律 vs 上下文里的律：这是有实证、也有阈值的

用户第二问是「能不能把律编进参数、变成潜意识，而不是每次在 CoT 里现算」。
这条在 ICL/IWL（in-context / in-weights learning）这条线上有非常直接的回答。

- **[Transformers generalize differently from information stored in context vs in weights](https://arxiv.org/abs/2210.05675)（Chan 2022）——直接回答用户的问题。**
  在受控刺激上，**从权重里泛化更偏「规则式」，从上下文里泛化更偏「样例式」**。
  换句话说：想要「像牛顿一样总结出律」，就得让它进权重；留在上下文里，模型更倾向于比对见过的例子。
  （补充一句：在自然语言预训练过的大模型上，in-context 也变得相当规则化，作者认为是语言里稀疏的规则结构带来的涌现。）
- **「什么时候从死记切到学律」有阈值，而且阈值由数据多样性定，不是由数据量定。**
  [Data Distributional Properties Drive Emergent ICL](https://arxiv.org/abs/2205.05055)（burstiness、长尾类别、动态词义）；
  [Pretraining task diversity and the emergence of non-Bayesian ICL](https://arxiv.org/abs/2306.15063)——存在一个明确的**任务多样性阈值**，
  低于它模型就退化成「以预训练分布为先验的贝叶斯估计器」（= 只会拟合见过的），高于它才解得了全新任务；
  [Differential learning kinetics](https://arxiv.org/abs/2412.00104)——给出**记忆化 scaling law**，直接算出多样性阈值，
  并指出转变的原因是记忆子回路和泛化子回路的**学习速率之差**，不是容量不够；
  [From Shortcut to Induction Head](https://arxiv.org/abs/2512.18634)——单层 transformer 上**严格证明**了这个相变，
  给出显式的数据多样性度量（trigger 间距的 max-sum 比），并推导出使 OOD 泛化的最优预训练分布。
  → **这是对我们语料设计最可执行的一条**：决定我们学到「rm 会删文件」还是「背下这条轨迹」的，
  是**环境/任务的多样性**（多少种不同的仓库、不同的失败模式、不同的路径布局），不是 SWE-Hero 有多少行。
  语料配比该按多样性阈值来调，而不是按总 token 数。
- [Toward Understanding In-context vs. In-weight Learning](https://arxiv.org/abs/2410.23042)（理论刻画两者何时涌现、何时消失）、
  [The Transient Nature of Emergent ICL](https://arxiv.org/abs/2311.08360)（ICL 会先出现后消失，最终让位给 IWL；两套回路在竞争）。
  → 后者对我们是好消息：**继续训下去，模型的默认倾向就是把东西沉进权重。**
- [Dual Process Learning: Controlling ICL vs IWL with Weight Forgetting](https://arxiv.org/abs/2406.00053)——
  用主动遗忘把两种策略**同时**保住，形成 dual-process。这是「世界律进权重、任务适应留上下文」共存的一个可操作机制。
- [Shared sensitivity to data distribution in humans and transformers](https://doi.org/10.1038/s41562-025-02359-3)（Nature Human Behaviour 2025）——
  冗余度驱动 in-weights、多样性驱动 in-context，二者权衡；混合分布能让两种策略并用。
- 机制侧：[Birth of a Transformer](https://arxiv.org/abs/2306.00802)（全局 bigram 学得快、induction head 学得慢）、
  [Reddy 2023](https://arxiv.org/abs/2312.03002)（induction head 的突现与内在课程）、
  [Distinct mechanisms underlying ICL](https://arxiv.org/abs/2604.12151)（四个算法相位，两条边界分别由动力学竞争和表征瓶颈决定）。

## T. 架构硬约束：不写 CoT 就想追踪 OS 状态，得看底座是什么

这是「潜意识提前躲」这条路上最容易被忽略、但**最硬**的一条限制。

- **坏消息：普通 Transformer 和普通 SSM 都做不到。**
  常数深度 Transformer 的表达力上界是 $\mathsf{TC}^0$（[Merrill & Sabharwal 2021](https://doi.org/10.1162/tacl_a_00493)），
  [The Illusion of State in State-Space Models](https://arxiv.org/abs/2404.08819) 证明 Mamba 类 SSM **也**卡在 $\mathsf{TC}^0$，
  文中原话就是它们**无法可靠地「evaluate code」或「track entities in a long narrative」**——正是我们要它做的事。
  [Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](https://arxiv.org/abs/2402.12875) 与
  [The Expressive Power of Transformers with CoT](https://arxiv.org/abs/2310.07923) 说明：
  要突破，要么写 CoT（线性步数才能认所有正则语言），要么加深度
  （[log-depth 就够](https://arxiv.org/abs/2503.03961)，且比加 CoT 步数更省），要么加 padding/looping（[2505.18948](https://arxiv.org/abs/2505.18948)）。
  → 这给 N 组那个「潜规划深度天花板 3–7 步」提供了**理论解释**，不是经验巧合。
- **好消息：我们这条线的底座恰好在正确的一侧。**
  [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)（Grazzi 2024）证明：
  只要状态转移矩阵是「单位阵减去向量外积」的乘积、特征值取值放宽到 $[-1,1]$，**LRNN 可以学任意正则语言**；
  把 Mamba / DeltaNet 的特征值范围扩到负值，parity 就解得了，state-tracking 全面提升，且 1.3B 规模能稳定预训练。
  [Why Are Linear RNNs More Parallelizable?](https://arxiv.org/abs/2603.03612)（Merrill 2026）给出更细的分级：
  **对角 + 低秩的 LRNN（DeltaNet / Gated DeltaNet 家族）是 $\mathsf{PNC}^1$-complete，严格强于置换-对角型的 $\mathsf{NC}^1$-complete。**
  → Qwen3.5-35B-A3B 是 **Gated DeltaNet + attention 混合**（[Gated Delta Networks](https://arxiv.org/abs/2412.06464)），
  正好落在「有可能在一次前向里追踪状态」的那一档。这不是宣传语，是可查的表达力分级。
- **[Learning State-Tracking from Code Using Linear RNNs](https://arxiv.org/abs/2602.14814)（Siems 2026）—— 这篇就是我们的设定。**
  它把置换合成任务改写成 **REPL 轨迹**：穿插 `print` 的状态揭示和变量变换，从而变成 next-token 预测格式。
  结果：**能做 state-tracking 的线性 RNN 在这个格式下也行，Transformer 仍然失败。**
  它还点出代码里追踪状态难的真正原因是**动作并非总是完全可观测**，并形式化为
  「带确定性状态揭示的概率有限状态机」——这正是 shell（命令的副作用不全打印出来）的准确刻画。
  → 直接可用：把我们的 shell 轨迹按它的 REPL 协议改造（穿插 `pwd` / `ls` 当状态揭示），
  就得到一个能判定「模型是否在无 CoT 情况下真的在追踪 OS 状态」的干净实验。
- 其他补强路线：[DeltaProduct](https://arxiv.org/abs/2502.10297)（多步 Householder 乘积，可调表达力）、
  [SLiCEs](https://arxiv.org/abs/2505.17761)（块对角/稀疏/Walsh-Hadamard 达到稠密矩阵的最大表达力，单层解 $A_5$）、
  [Fixed-Point RNNs](https://consensus.app/papers/details/22e1ea9b37195e2aadfa03fd27931301/)（稠密线性 RNN 表示成可并行对角 RNN 的不动点，$A_5$/$S_5$ 上 SOTA）、
  [M²RNN](https://arxiv.org/abs/2603.14360)（非线性矩阵值状态，
  在 7B MoE 上比同规模 Gated DeltaNet 混合低 0.4–0.5 PPL，且只换一层就有明显收益）、
  [Sparse Delta Memory](https://arxiv.org/abs/2607.07386)（把 Gated DeltaNet 的状态扩成稀疏寻址的大显式记忆，
  并可把初始状态当**参数化记忆**训——和「律进参数」这件事直接同构）。

## U. 「世界模型目标和 agent 目标冲突」在 MBRL 里有正式名字：objective mismatch

这一组是整份文档里**最贴近用户原始问题**的。用户说的「训世界模型和训 agent 的目标函数根本相悖」，
在 model-based RL 里叫 **objective mismatch**，有名字、有定理、也有诚实的失败记录。

- **问题陈述（几乎是用户原话的学术版）：**
  [VaGraM](https://arxiv.org/abs/2204.01464)（Voelcker 2022）开篇即说：
  模型通常只被拟合去**重建动力学、尤其是状态观察**，而**模型误差对策略的影响根本没进训练目标**，
  于是「让策略和价值学好」这个真实目的，和「预测未来状态」这个实际损失之间产生错配。
  → 我们在助手位上算下一观察 token 的 CE，就是这句话的字面实例。
- **正式的替代目标与界：**
  [Value-Aware Loss Function for MBRL](https://consensus.app/papers/details/b0f6ce8be2c65cad9395b655aa4342f0/)（Farahmand 2017）——
  论证「最小化 log-loss 学生成模型是 overkill」，给出把价值函数结构纳入的损失和**有限样本上界**；
  [Iterative VAML](https://consensus.app/papers/details/14e6dd2e07ee51108dfbdf5849a185c5/)（Farahmand 2018）给出可解版本和有限样本保证；
  [Model-Advantage Optimization](https://arxiv.org/abs/2106.14080)——用「同一策略在两个模型下的性能差」的上界当目标，
  第一次让 value-aware MBRL 在连续控制上真正跑起来。
- **[The Value Equivalence Principle](https://arxiv.org/abs/2011.03506)（Grimm 2020）—— 这是「要多少世界模型才够」的答案。**
  两个模型只要对一组函数和策略给出**相同的 Bellman 更新**，就是价值等价的；
  随着考虑的策略/函数集合变大，价值等价的模型类逐渐收缩，极限才收到真实模型。
  意思是：**逐状态预测得准既困难又往往不必要**；有限的表征资源应该花在「对基于价值的规划直接有用」的模型上。
  这条原理同时是 MuZero / VPN / Predictron 等经验成功的第一个理论支撑。
  → 对我们：这给了一个原则性的理由，说明为什么不该把参数预算砸在把 stdout 一字不差地生成出来。
- **[Deciding What to Model](https://arxiv.org/abs/2206.02072)（Arumugam 2022）—— 用率失真理论决定「压到多简」。**
  当 agent 容量根本装不下真实环境时，迭代地算出一个**近似价值等价的有损压缩**去替代真实模型，
  并给出信息论贝叶斯 regret 界，可写成两种形式：给定次优容忍度求最简模型，或给定容量求最好模型。
- **[Error Bounds of Imitating Policies and Environments](https://doi.org/10.1109/tpami.2021.3096966)（Xu 2020）—— 对我们最扎心的一条。**
  把「学环境模型」当成对环境做模仿学习：行为克隆式（= 我们现在做的下一观察 CE）会有**复合误差**，
  而对抗式模仿把策略评估误差压到**关于有效规划步长线性**（而不是平方）依赖模型偏差。
  → 直白说：**逐 token 拟合观察在长轨迹上误差是复利增长的，这不是工程细节，是有界可证的。**
- **[An Optimal Tightness Bound for the Simulation Lemma](https://arxiv.org/abs/2406.16249)（Lobel 2024）——
  把 RL 的地基定理 simulation lemma 收紧到常数因子都最优，且对转移误差是**次线性**依赖；
  旧界在折扣因子大时会退化成空话。这是「模型误差 → 价值误差」这一环现在最好的砖。
- **诚实的反面记录（必须写进来，否则会踩坑）：**
  [Decision-Aware Model Learning for Actor-Critic: When Theory Does Not Meet Practice](https://consensus.app/papers/details/df54c74ed6165977957689b47d7bd7dd/)（Lovatto 2020）——
  在连续域里，朴素的 MLE 常常**打赢**价值感知模型，且更省算力；理论保证不等于端到端更好。
  [Calibrated Value-Aware Model Learning](https://arxiv.org/abs/2505.22772)（Voelcker 2025）——
  包括 MuZero loss 在内的这一族价值感知损失**是未校准的代理损失**，不保证恢复正确的模型和价值函数，并给出修正。
  → 结论不是「换成 value-aware 就赢」，而是：**观察 CE 是错的损失这件事有定理；换什么损失才对，还没定论。**
- 相关：[Control-Oriented MBRL with Implicit Differentiation](https://doi.org/10.1609/aaai.v36i7.20758)（把模型参数经隐函数直接对回报求导）、
  [Policy-Aware Simulator Learning](https://arxiv.org/abs/2605.29032)（2026；模型 vs 对抗策略的零和博弈，
  给出 sublinear regret、critic 局部损失界住全局策略价值差、以及 **Error-MDP 对偶**——
  「找最坏策略」形式上对偶于「以一步 critic 误差为奖励的标准 RL」，由此得到可证收敛的**主动数据选择**算法。
  这条把 R 组的主动干预和本组的目标错配缝在了一起）、
  [The Central Role of the Loss Function in RL](https://arxiv.org/abs/2409.12799)（不同回归损失如何改变样本效率与自适应性）。

## W. 已经有人在代码域做过我们这件事：执行感知预训练（最接近的经验先例）

前面 I–U 大多是理论。**这一组是「学环境动力学 → 下游任务变强」在 LLM 规模上已经跑通的经验证据**，
只不过环境是「Python 解释器」而不是「shell」。这是我们假设最直接的先例，之前的 paper shelf 里完全没有。

- **[TRACED: Execution-Aware Pre-Training for Source Code](https://doi.org/10.1145/3597503.3608140)（ICSE 2024）—— 结构上和我们一模一样。**
  用「源码 + 可执行输入 + 对应执行轨迹」做预训练，目标是让模型**不再重复执行代码**就能静态估计动态属性。
  结果：完整执行路径预测 +12.4%、运行时变量值预测 +25.2%，而且在**它没被直接训练的下游任务**
  （克隆检索、漏洞检测）上也显著超过纯静态预训练模型。
  → 这就是「世界理解 → 下游能力」的一个已完成的实例，只是域是解释器不是 OS。
- **[SemCoder](https://arxiv.org/abs/2406.01006)（6.7B）—— 用「自言自语式推理」把静态代码和动态执行状态连起来。**
  训模型不只写代码，还要用自然语言讲清楚每条语句的**局部执行效果**和整体输入输出行为（rubber-duck debugging）。
  6.7B 在 HumanEval / CRUXEval 上打平或超过 GPT-3.5-turbo，
  并且**明确对比了 monologue 式和 concrete scratchpad 式执行推理**，前者整合多维语义更顺——
  这条对我们「潜意识 vs 显式 CoT」这一问有直接参考价值。
- **[What I cannot execute, I do not understand（Execution Tuning, E.T.）](https://arxiv.org/abs/2503.05703)—— 长轨迹这条最关键。**
  在真实执行轨迹上训练（行级 / 指令级两种粒度），CruxEval / MBPP 上约 80% 输出预测准确率；
  关键发现是**动态 scratchpad**（模型自己维护并更新一份自包含的中间计算）在长执行（最长 14k 步）上
  明显优于**把历史一路累积下去**的做法。
  → 这在经验上说了和 K 组 strict mediation 同一件事：**维护「状态」比堆「历史」好**，而且在 14k 步这种我们关心的长度上被验证过。
- **[StepCodeReasoner](https://arxiv.org/abs/2605.11922)（2026）—— 执行建模同时提升了代码生成。**
  自动往代码里插入**基于 print 的执行轨迹锚点**（和 T 组 REPL state-tracking 那篇是同一个技巧），
  把代码推理变成可验证的逐步执行建模问题，再用 Bi-Level GRPO 做两级信用分配。
  7B 在 CRUXEval 91.1% / LiveCodeBench 86.5%，超过 GPT-4o；REval 82.9% 超过自己的 14B 版本。
  论文明确写：**显式的执行建模同时改善了代码推理和代码生成。**
  → 这是「世界模型目标提升 agent 目标」而不是抢权重的一个正面样本，且规模在我们够得着的范围。
- [Self-Execution Simulation Improves Coding Models](https://arxiv.org/abs/2604.03253)（2026）——
  执行轨迹 SFT + 可验证奖励 RL，两个互补目标（给代码和输入预测输出 / 用自预测的执行反馈解题），
  让模型能对多个候选解自验证、迭代自修。
- [CodeExecutor](https://arxiv.org/abs/2305.05383)（执行预训练 + 课程学习）、
  [TraceFixer](https://arxiv.org/abs/2304.12743)（用部分执行轨迹训修复模型，比只学代码编辑高 13–20%）。
- **反面：把轨迹塞进 prompt 基本没用。**
  [Towards Effectively Leveraging Execution Traces for Program Repair](https://doi.org/10.18653/v1/2025.knowledgenlp-1.17)（2025）——
  在 6 个「数据集 × 模型」组合里只有 2 个有提升，**而且轨迹越复杂收益越小**；
  但**在小数据上微调一个小模型又不如 prompt**。
  → 把这条和 S 组放在一起看结论很干净：**执行动力学放进上下文效果有限，放进权重才有跨任务的收益。**
  这正是用户「把律编码进参数」的经验依据。

## X. 嫁接：从 Instruct 搬一段层过来当 token 头的初始值，不是 merge

这组回答的问题不是「WM 和 agent 的目标该怎么共处」——那件事已经交给 K-draft/JEPA/selector/token 头处理了。
这组回答的是更窄的一件事：**Stage 2 的草稿头/token 头要不要随机初始化**。

用户 2026-08-29 指出的关键事实，已经核对：`Qwen-AgentWorld-35B-A3B` 和 `Qwen3.5-35B-A3B`
（我们的 agent/Instruct checkpoint）**同源**——都是 `Qwen3.5-35B-A3B-Base` 往下的两条后训练路径。
直接查了 hub 上两边的 `config.json` / `tokenizer_config.json`：`hidden_size=2048`、`num_hidden_layers=40`，
`layer_types` 逐位置相同（Gated DeltaNet linear-attention + full-attention，同样每 4 层一次 full attention）、
`vocab_size=248320`、`tie_word_embeddings=false`（LM head 是独立张量，不和输入 embedding 共享，嫁接对象明确）、
`added_tokens_decoder` / special tokens / chat_template 骨架几乎逐字相同（只有 `model_max_length` 和几处 jinja
分支——比如是否保留 `preserve_thinking`、参数序列化的判断顺序——不同）。这比下面几篇 Layer Swapping 论文的前提更干净：
不仅同 Base，连词表和大部分模板都没变。

- **Layer Swapping for Zero-Shot Cross-Lingual Transfer** ([2410.01335](https://arxiv.org/abs/2410.01335)) ——
  同源两专家（数学 vs 目标语言微调），直接把数学专家的顶层+底层**换成**语言专家的对应层，不做向量加减，
  比模型汤 / DARE 在 MGSM 上平均高 10 个点。
- **The Unreasonable Effectiveness of Model Merging for Cross-Lingual Transfer** ([2505.18356](https://arxiv.org/abs/2505.18356)) ——
  解释为什么换层赢过任务向量加减：数学能力和语言能力依赖的参数子集基本不重叠；
  还发现「训完再把没用的更新还原」比「一开始就冻住不训」效果更好。
- **Rethinking the Multilingual Reasoning Gap with Layer Swap** ([2605.26735](https://arxiv.org/abs/2605.26735)) ——
  权重级分析：微调更新在**中间层**高度一致（语言无关的推理核心），在**外层**分歧大（语言特定）；
  只换中间层就能补上大部分推理差距，同时保留目标语言的表达层。
- **The Remarkable Robustness of LLMs: Stages of Inference?** ([2406.19384](https://arxiv.org/abs/2406.19384)) ——
  删层/换邻层实验：模型对动**中间层**极其稳健（72–95% 准确率不掉），对动**最早**和**最后**几层最敏感；
  提出四段划分 detokenization → feature engineering → prediction ensembling → residual sharpening，
  后两段负责「把内部状态收敛成具体输出」——对应我们要嫁接的「怎么把状态写成命令」这一段。
- **Revisiting Model Stitching in the Foundation Model Era** ([2603.12433](https://arxiv.org/abs/2603.12433)) ——
  异构模型才需要拼接层；拼接层要用「目标模型倒数第二层的特征匹配损失」单独训，而不是端到端任务损失，
  否则在浅拼接点会失败。**我们大概率不需要这一步**（同源、张量形状逐一相同），但接上之前应该照它的协议测一下，不要假设能直接接。
- **Fine-Tuning can Distort Pretrained Features and Underperform OOD**（LP-FT，[2202.10054](https://arxiv.org/abs/2202.10054)） ——
  头没配好就整体微调，头的噪声梯度会反向污染主干已经学好的特征；正确顺序是先只训头（或拼接点），
  等头不再失配，再解冻主干一起训。直接决定 Stage 2 的训练顺序。
- **Parameter-Efficient Tuning Makes a Good Classification Head** ([2210.16771](https://arxiv.org/abs/2210.16771)) ——
  用已经训过的头替换随机头本身就能稳定提升，支持「别随机初始化 token 头」这个直觉本身是对的。

→ **结论：层替换/头嫁接解决的是初始化问题，不是目标冲突问题。**
目标冲突（WM 的 \(P(o\mid h,a)\) vs agent 的 \(P(a\mid h)\)）仍然要靠 K-draft + JEPA + argmax + token 头架构处理；
嫁接只是让 Stage 2 的 token 头一开始就会写话，不用从零学「怎么把动作向量变成一句通顺命令」，
预期能减少 Stage 2 需要的数据/步数（省的是这笔钱，不是省掉整个 Stage 2）。
**老实的边界**：Layer Swapping 论文验证的是两个**平行**目标（数学 vs 语言）之间换层；
我们这两个目标是**相冲**的（这正是 Chat Vector merge 失败的原因）。层替换在目标相冲时是否依然有效，
**没有人验证过**——这是 Stage “−1” 那个零训练 sanity check 要自己测的地方，不是抄论文结论。

## V. 把上面这些串成一条链（每段可挂定理，缺的地方明说）

这是把 I–U 拼起来后能写出的算法骨架。**它不是「突破性堆栈」，是一张标注了哪里有砖、哪里是空的图。**

1. **语料按多样性阈值配，而不是按 token 数配。** 挂 S 组：[2306.15063](https://arxiv.org/abs/2306.15063) 的任务多样性阈值、
   [2412.00104](https://arxiv.org/abs/2412.00104) 的 memorization scaling law、[2512.18634](https://arxiv.org/abs/2512.18634) 的相变证明。
   *缺*：这些阈值都在合成任务上算的，shell/SWE 语料的「多样性」怎么度量没人定义。
2. **动作侧：给命令学动作表征，而不是当 token 串。** 挂 P 组：[Chandak](https://arxiv.org/abs/1902.00183) 的收敛条件、
   [Pritz](https://arxiv.org/abs/2010.04444) 的「嵌入状态-动作训 RL 有效」的理论基础。
   *缺*：bash 动作空间是否线性可嵌入，没人证也没人测。
3. **状态侧：显式潜状态 z，且强制 strict mediation（预测只准读 z 和 a，不准回头翻历史）。**
   挂 K 组：[2606.27681](https://arxiv.org/abs/2606.27681) 的可识别性 + fGRPO、[NextLat](https://arxiv.org/abs/2511.05963) 的收敛到 belief state、
   [AC-State](https://arxiv.org/abs/2207.08229) 滤掉不可控噪声。
   兼容 I 组「时间局部性」这条归纳偏置（[2602.06923](https://arxiv.org/abs/2602.06923)）。
   *缺*：这些保证都在 TextWorld / 低维控制域上，真实 shell 的无界 stdout 没验过。
4. **损失侧：别在观察 token 上算 CE 当主损失。** 挂 U 组（objective mismatch、价值等价、模仿环境的复合误差界）
   和 M 组（[潜自预测是好辅助任务、观察重建当辅助任务反而拖累](https://arxiv.org/abs/2406.17718)）
   以及 [RWML](https://arxiv.org/abs/2602.05842) 的经验版（token 级下一状态预测会塌缩，嵌入空间对齐才稳）。
   *缺*：换成什么损失才对，literature 自己也没定论（[2505.22772](https://arxiv.org/abs/2505.22772) 指出价值感知损失未校准）。
5. **自预测辅助损失 → 转移算子的谱特征 → 低秩 MDP 的 φ → 下游策略样本复杂度下降。**
   挂 L 组：[2212.03319](https://arxiv.org/abs/2212.03319)（自预测=谱分解，防塌缩靠 stop-gradient + 更快的 predictor）、
   [SPEDER](https://arxiv.org/abs/2208.09515)、[REP-UCB](https://arxiv.org/abs/2110.04652)、
   [多任务表征迁移界](https://arxiv.org/abs/2206.05900)（下游次优性 = 上游表征误差 + 随下游样本消失的项）。
   **这是「世界理解 → agent 变强」这句话唯一一条完整的形式化路径。**
   *缺*：整条链没人端到端证过，尤其是动作为自由文本时。这是我们能贡献的位置。
6. **φ 不预先给定：用 Forward-Backward 同时学基础特征和后继特征。**
   挂 L 组：[2209.14935](https://arxiv.org/abs/2209.14935)、[2502.10790](https://arxiv.org/abs/2502.10790)。
7. **T→π：用摊销规划/想象 backup 编译，而不是在同一个未冻结的 LM head 上同时更新观察 CE 和策略 CE。**
   挂 L 组：[MPDP](https://arxiv.org/abs/2307.12933) 的单调改进保证、[SAVE](https://arxiv.org/abs/1912.02807)、
   LLM 侧的 [Internalizing the Future](https://arxiv.org/abs/2606.27483)（WM-AMT → FE-SFT → FC-RL，并点名 format-capability gap）。
   训 π 时冻结 z 的转移。
8. **架构：确认底座能在一次前向里追踪状态。** 挂 T 组：Gated DeltaNet 是 $\mathsf{PNC}^1$-complete，
   前提是特征值范围覆盖负值（[2411.12537](https://arxiv.org/abs/2411.12537)、[2603.03612](https://arxiv.org/abs/2603.03612)）；
   用 [REPL 状态追踪协议](https://arxiv.org/abs/2602.14814)去实测。
   *缺*：没人在 Qwen3.5-35B-A3B 的实际权重上查过特征值范围。
9. **评测：三件事必须同时报。** (a) 观察保真度；(b) agent 指标；(c) 二者的**相关方向**
   （[PatchWorld](https://arxiv.org/abs/2605.30880) 测出可能是负的）。
   律的判据用 Q 组的**一致重命名**（`rm`→`zaq` 仍能预测「文件没了」）和 I 组的**世界模型恢复度**，
   而不是 CE；因果主张必须做 activation patching（N 组）。
10. **干预：离线语料里没有我们选的 `do(a)`，这是 CHT 的硬墙。**
    工具有（R 组的 γ-Progress、Curiosity-Critic、CAASL；U 组 Policy-Aware Simulator Learning 的 Error-MDP 对偶给出可证收敛的主动数据选择），
    但 [2606.19476](https://arxiv.org/abs/2606.19476) 证明在一般 MDP 上靠 in-context 预测误差估学习进度必然有偏，
    所以要把 shell 上的探索**降成非时序的实验设计问题**才有保证。

## O. 现在真正还缺的（三轮检索后剩下的硬洞）

1. **组合本身（最硬，也是我们能贡献的位置）。**
   「自预测 → 转移算子谱分解 → 低秩 MDP 的 φ → 下游策略样本复杂度界」这条链每一环都有定理（L 组），
   动作嵌入（P 组）和显式状态（K 组）也各自有定理，
   但**没有任何一篇把它们串起来证过端到端的界**，尤其是动作为自由文本、观察为无界字符串时。
2. **无界观察 + 真实 OS。** strict mediation 只在 TextWorld / ScienceWorld 验过；
   AC-State 的控制内生性保证在低维控制域证的；Curiosity-Critic 的噪声下界估计在网格世界证的。
   真实 shell 的 stdout 无界、外生噪声占比高，这些保证有没有迁移，没人测。
3. **保真度与效用的负相关有多严重。** PatchWorld 在 AgentGym 上测出了这个 trade-off（Q 组），
   但只在程序式世界模型上；**参数式**世界模型（我们这条路）上它是不是同样成立、有多强，没人量化过。
   这恰好是我们 real vs shuffled 双跑之外应该加的第三个观测量。
4. **离线语料里没有我们选的 `do(a)`。** R 组给了主动干预的工具和理论，
   但全部在低维 SCM / 网格世界上；而且 [2606.19476](https://arxiv.org/abs/2606.19476) 证明在一般 MDP 上
   基于 in-context 预测误差的学习进度估计**必然有偏**。
   要在 shell 上做主动干预，得先把它降成非时序的实验设计问题。
5. **底座的表达力是「可能」，不是「已经」。** T 组只说明 Gated DeltaNet 混合架构**在原理上**可能在一次前向里追踪状态
   （$\mathsf{PNC}^1$-complete，前提是特征值范围覆盖负值），
   但没人验过 Qwen3.5-35B-A3B 的实际权重是否落在那个范围，也没人在真实 shell 长度上测过。
   [2602.14814](https://arxiv.org/abs/2602.14814) 的 REPL 协议是现成的测法，但它只在合成置换任务上跑过。

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

**把规律写进参数（物理 WM）**

```
physics-informed neural networks residual PDE loss
Hamiltonian Lagrangian neural network energy conservation
residual physics M0 plus delta M LoRA constitutive
Dreamer PlaNet latent dynamics not pixel reconstruction
Graph Network Simulator particle state
world foundation model post-train Cosmos
physics-faithful video RL after appearance pretrain
physical state grounding JEPA
```

## 更新

```bash
python3 refs/fetch_html_text.py
```

只拉 HTML（arXiv `/html/`，不行再试 ar5iv），用 `w3m -dump` 转纯文本。
