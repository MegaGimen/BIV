This file provides guidance to AI coding agents working with this repository.

## 给下一个 agent 的入口

当前主线在分支 **`agentworld-JEPALLM-Qwen3.5-35B-A3B`**。要做的事：把 OS / 代码世界的转移律编进参数，再在这套表征上长出写命令的能力，用同一套脚手架看 agent 是否变强。旧 MLP JEPA 线在 **`agentworld-JEPA-Qwen3.5-35B-A3B`**（停在 `12751879`），不要和本线混对照。

读完下面三块就能动手。文末 **灵感来源** 写最终方案各零件对应哪些论文；**论文链接** 给出全部编号条目（含检索过但方案未直接引用的）。HTML 全文在 [`refs/`](./refs/)（只提交文本，不提交 PDF）。

1. **目标与叙事** — 两个研究问题、用哪个 checkpoint、什么叫成功。
2. **模型架构** — 两条独立 backbone（AgentWorld / Instruct）、草稿 / JEPA / 打分器 / 线性层 \(W\) / `lm_head` 怎么接、每个 Stage 跑哪一段。切鱼为什么放弃见本节开头。
3. **Stage + Step** — 从切回合到评测的走法。

运行时产品（缸中之脑 / Demon）和上游 nanobot 在文后 **BIV 运行时**、**开发命令**。`Muse` 上的 Glimmer-30B LoRA 是**另一条** checkpoint 线，不要和本线混对照。

---

## 仓库是什么

**BIV**（Brain In a Vat / 缸中之脑）叠在 [nanobot](https://github.com/HKUDS/nanobot) 上。

- **运行时：** Agent A 以为在用真工具；Agent B（Demon）截获碰世界的工具，按 Matrix Law 返回自洽的假世界。见根目录 `README.md`、`cartesian/`。
- **研究 / 训练（`train/`）：** 用真实环境转移提高对 OS 和代码世界的理解，再看这套理解会不会转到控制台 / 编码工具 agent。改 BIV 行为优先动 `cartesian/` 和 `train/`；上游 `nanobot/` 只在分叉或修 bug 时动。

上游 nanobot 仍是轻量 Python agent（channels → bus → agent loop → LLM → tools → memory）加 React/TypeScript WebUI。

## 怎么和用户说话（始终）

用户明确要求过。不要电报体、不要论文摘要体。

- **讨论训练 / 评测 / 论文时的北极星：** 靠提升世界理解来提升 agent。手段不限（共训、后续 agent 阶段、RL、当模拟器、改 mix 都可），配方不神圣。
- **语言：** 用户用中文就简体中文。
- **篇幅：** 先给短结论，再写够让没读过论文和代码的人跟得上。比较、机制、决策写成连贯段落或一个算过的例子，不要三行术语子弹当全文。
- **用词：** 日常话。若必须出现 \(P(o\mid h,a)\)、Terminus、SFT、LoRA，立刻用一句话说清在本项目里是什么。不要把论文名叠成解释。
- **「有没有类似研究 / 这是不是 X」：** (1) 结论；(2) 一个类比或本仓库例子；(3) 别人怎么训、怎么测、和我们差在哪；(4) 若用户问了下一步，再写意味着什么。
- **按意图读，不要逐字死扣。** 「哪篇论文 / 哪个」通常是「哪些」。不要发明用户没问的议程。
- **没问的不要答。** 不要自行「分清两件事」或塞没要求的建议。
- **不要先把用户的想法极端化再否定。** 需要澄清时直接陈述事实。

---

## 当前研究：目标与叙事

**把世界的律写进参数。** 类比牛顿：看苹果落地，自己发现 \(G=mg\)，不是去拟合那条轨迹，也不是有人把公式塞进提示词。律进参数之后，铡刀举起来就该侧身——观察还没写进上下文，内部状态或下一步动作已经按「文件没了」来对待，不必在 CoT 里把物理重算一遍。然后在这套编码上长「会做事」，世界目标和 agent 目标不要抢同一张嘴（助手位上的同一套 token）。

用户当成研究的两问：

1. **从观察里能不能掌握底下的律？** 接受的答案：能，但只当观众不够，需要不变性和干预（同一段历史上的 `do(a)`，不是换一句 prompt）。
2. **掌握的律能不能编成「潜意识」，让模型提早闪避？** 接受的答案：原则上能——把 System-2 的心智模拟编进 System-1 / 后继特征一类的东西里。Qwen-AgentWorld 论文 Table 9 的 Postfix CoT 是昂贵的显式形式，不是我们要的形态。

**成功怎么量：** 同一套 Harbor / Terminal-Bench 2.1 脚手架，Stage 2 之后比之前高；再加「把 `rm` 改成 `zaq` 是否还跟得上转移」和「截断轨迹、命中观察还没进上下文时是否已经当文件没了」。世界侧同时报 \(\hat z\) 的**排序**和**表面保真度**。对照：同一批命令行打在 Instruct 底座上，看世界主干有没有帮忙。打乱观察的孪生集作因果对照。

**本线用的三个同源 checkpoint**（同一 Base、同一套 40 层盒子、词表一致）：

| 角色 | Hub 名 | 在本实验里干什么 |
|------|--------|------------------|
| 世界底座 | `Qwen/Qwen-AgentWorld-35B-A3B` | 独立的一整条 40 层 backbone，专属 JEPA；该线经过 CPT → 下一状态 SFT → RL |
| 写命令 | `Qwen/Qwen3.5-35B-A3B`（Instruct） | 独立的一整条 40 层 backbone + 原来的 `lm_head`，专属 agent（草稿 / 打分 / \(W\)） |
| 参照 | `Qwen/Qwen3.5-35B-A3B-Base` | `compare.py` 逐层量两条后训练相对 Base 差在哪；**不再用来定切点** \(\ell\)（切鱼已放弃，见「模型架构」） |

层数 40，`hidden_size=2048`，`vocab_size=248320`，`tie_word_embeddings=false`（`lm_head` 是独立张量）。层类型是 Gated DeltaNet 线性注意和完整注意力的混合，大约每 4 层一次 full attention。评测入口：`train/eval/`、`merge/eval.py`。

贯穿例子：日志里 `rm a.txt` 成功之后，下一条命令是 `git checkout a.txt`。

---

## 模型架构

**两条完全独立的 40 层 backbone，不切、不合并。** AgentWorld 和 Instruct 各自保留自己完整的权重，草稿 / JEPA / 打分 / 线性层 \(W\) 架在两者中间当连接件。这是放弃切鱼之后的现状；切鱼为什么放弃、放弃前试过什么，见下面「现在的方案」一节开头。

### 出厂 Qwen3.5（还没切、还没 JEPA）

Instruct 和 AgentWorld 是同一套盒子：

\[
\text{token} \;\to\; \mathrm{emb} \;\to\; L_0 \;\to\; \cdots \;\to\; L_{39} \;\to\; h_{39} \;\to\; \texttt{lm\_head} \;\to\; \text{下一个 token}
\]

残差流一路走到第 39 层，`lm_head` 是 \(2048 \times 248320\) 的表（约 \(5\times 10^8\) 参数），本来吃的就是 \(h_{39}\)。

### 现在的方案：为什么放弃切鱼，改成两条独立 backbone

切鱼假设 AgentWorld 和 Instruct 相对 Base 的改动是**低熵**的——某几层动得凶、某几层几乎不动，才谈得上「照抄没被碰过的那部分」。`compare.py` 测出来的实际情况是**处处都动，只是幅度不同**（AgentWorld 幅度是 Instruct 的 5~8 倍，且这个比例几乎不随层数变化），找不出一批天然无冲突的层。TIES/DARE 这类按幅度投票的合并方法在只有两个专家（\(N{=}2\)）时投票机制退化，而且幅度差这么大会出现 Asymmetric Collapse——大的一方几乎全程压过小的一方，和两者是不是正交没关系。闭式投影/加权合并算出来的张量是没被训练或评测验证过的新值，事后冻住它也保不住任何一边已经验证过的能力，等于丧失了硬切本来最大的好处（冻住的部分=精确复刻已验证行为）。这三条路都是在这对具体 checkpoint 上跑出的结论，不是理论上可能有坑。

现在的方案是**完全不切、不合并**：AgentWorld 和 Instruct 各自保留自己完整的 40 层权重，谁都不碰谁，中间架一个折叠进已有模块、不单独起名的连接件。

| 名字 | 盒子 | 权重从哪来 | 干什么 |
|------|------|------------|--------|
| World 分支 | \(\mathrm{emb}+L_0\ldots L_{39}\) | AgentWorld，整条不动 | Stage 1 训好 JEPA；Stage 2 里整体冻死，只作为打分时被查询的对象 |
| Agent 分支 | \(\mathrm{emb}+L_0\ldots L_{39}\) | Instruct，整条不动 | 真正写命令；Step 1 冻，Step 2 才小步解冻 |
| `lm_head` | 词表读出 | Instruct，保留 | 出字的嘴；经 \(W\) 吃动作向量 |
| 线性层 \(W\) | \(2048\times 2048\)（约 \(4\times 10^6\) 参数） | 新建 | 把胜出分支的动作向量映到 Instruct `lm_head` 熟悉的门口 |
| 草稿头 | 新模块，一个共享主干 + 两个小输出头 | 新建 | 从 Agent 分支的 \(c_t\) 提出 \(K\) 个候选：一路 \(u_k^W\) 喂给 JEPA，一路 \(u_k^A\) 喂给 \(W\)。这两个输出头顺带兼职当连接件，见下 |
| JEPA | \(\hat z=\) 对历史+命令**单独**走 chat 模板再编码，取 `len(unpad)+last_token` 处的隐藏向量（Qwen 默认 `last_token=-3`） | Stage 1 训好，Stage 2 冻死 | 不再另挂小 MLP。接线抄 LLM-JEPA：Text 和 Code 各编一次、互不看见 |
| 打分器 | 新模块 | 新建 | 读每个候选的 \((u_k^W,\hat z_k)\)，给 \(K{+}1\) 个候选（\(K\) 个草稿 + 1 个真实分支）打分；输入层顺带兼职当连接件另一半 |

**连接件焊在哪，不是单独一个模块**。草稿头的 \(u^W\) 输出头（把 Agent 分支的 \(c_t\) 翻成 World 分支/JEPA 认得的动作向量）和打分器读 \(\hat z\) 的那层输入（把 World 分支的预测换算成对选择有用的分数），各自顶着自己本来就要挂的损失去训——不是先离线做一个「让分布看起来像」的对齐、再指望它管用。这条离线对齐的路已经有反例：两个独立模型之间做到 0.97 余弦相似度的线性映射，注入进真实前向计算之后照样没用（[Pythia 多跳负结果](https://consensus.app/papers/details/3882637eac3059fa87366e0661f009b7/)）；反过来端到端地、挨着真实下游损失训连接件才有用（[Bicameral](https://consensus.app/papers/details/95cf6a71ffee5d18b78189605ec37f2d/)、[KV-Cache Alignment](https://consensus.app/papers/details/798e52a7c2835eeaa12313a7cc9d83a5/)）。AgentWorld 和 Instruct 同出一个 Base、同一套 40 层盒子，分叉前表征空间完全相同，这个起点比上面两篇论文验证过的场景都好，但好起点不能替代对的训练方法——这一点是这条设计里最容易踩空的地方，训连接件时要留意。

### 出字这条路怎么接

\[
u_i^A \;\xrightarrow{W}\; \texttt{Instruct lm\_head} \;\to\; \text{命令 token}
\]

`lm_head` 吃的是 \(W(u_i^A)\)。World 分支的 \(h_{39}\)、打分 \(s[i]\)、\(\hat z\) 都不进 **Instruct** 这张表。Stage 1 会用 **AgentWorld 自己的** `lm_head` 写观察（见下），那是另一张表、另一条损失；命令的 token 交叉熵只打在 Instruct 的 \(W\)、`lm_head`、草稿头的 \(u^A\) 输出头上。两条 backbone 天生分开，命令交叉熵进不了 World 分支。

Stage 1 的「预测下一观察」就是 **World 分支自己**把历史+命令当成一轮对话单独编码（不是从完整序列上切前缀），取 `last_token` 那个位置的隐藏向量。Stage 2 训练时对每个候选命令同样跑一遍冻住的 AgentWorld，同一套取向量规则当 \(\hat z_k\)，和真实观察的独立编码一起送进打分器。

### 每个 Stage 实际接通的图

**Stage 1（世界，只碰 AgentWorld）。** 配方抄 LLM-JEPA 的 `RepresentationTrainer`：两路损失同时训。左边把历史+命令单独套 chat 模板再编码（他们的 Enc(Text)），右边把真观察单独套 chat 模板再编码（他们的 Enc(Code)），向量取在 `len(unpad)+last_token`（Qwen 默认 `-3`）。写字路用 AgentWorld 自己的 `lm_head` 对完整对话做移位交叉熵，只解开观察正文那些标签。对齐是 `1 - cosine`，两边都带梯度。没有 JEPA MLP、没有逆动力学、没有 SimCSE/bank。Agent 分支、草稿、打分、Instruct `lm_head`、\(W\) 全部不参与。更新：AgentWorld 全部 40 层的 LoRA。

**Stage 2（出字，两条分支都上）。** Agent 分支算出 \(c_t\) → 草稿头提出 \(K\) 个候选命令向量 \(u_k^A\)（喂 \(W\)）以及对应的命令 token 串 → 冻死的 AgentWorld 对每个候选把 \((h, a_k)\) 编一遍，末尾隐藏向量就是 \(\hat z_k\) → 打分器给 \(K{+}1\) 个候选打分 → 训练时走教师强制的真实分支解码，推理时才用 \(\arg\max\) 选出 \(u_i^A\) → \(W\) → Instruct `lm_head`。Step 1 冻两条 backbone 和 Instruct `lm_head`，只训草稿 / 打分 / \(W\)；Step 2 才小步解开 Instruct `lm_head` 和 Agent 分支。

```mermaid
flowchart TD
    subgraph WORLD["World 分支 — Stage 1 训好，Stage 2 整段冻住"]
        H1["历史 h + 候选命令 a_k"] --> AWBB["AgentWorld 自己的 40 层（冻）"]
        AWBB --> ZHAT["ẑ_k = 末尾隐藏向量"]
    end

    subgraph AGENT["Agent 分支 — Instruct 自己的 40 层"]
        H2["历史 h（同一份输入）"] --> INSBB["Instruct 自己的 40 层<br/>Step1 冻 / Step2 小LoRA解冻"]
        INSBB --> CT["c_t"]
        CT --> DRAFT["草稿头（新，可训）"]
        DRAFT --> UA["u_k^A，K个，喂给W/lm_head"]
        DRAFT --> AK["候选命令 token 串"]
    end

    AK --> AWBB
    ZHAT --> SCORER["打分器（新，可训）"]
    SCORER --> PICK["argmax 选 i（仅推理时；训练时走教师强制，见「训练全过程」Stage 2）"]
    UA --> WMAT["线性 W 2048x2048（可训）"]
    PICK -. 推理时的选择 .-> WMAT
    WMAT --> LMHEAD["Instruct 原 lm_head<br/>Step1 冻 / Step2 小LR解冻"]
    LMHEAD --> CMD["命令 token"]
```

```mermaid
flowchart LR
    SD["Stage Data"] --> S1["Stage 1：AgentWorld + LLM-JEPA 两路损失"]
    S1 --> S2["Stage 2 Step1 门口 / Step2 嘴巴"]
    S2 --> EV["Stage Eval"]
```

### 训练库

本线用 **`torch` + HuggingFace `transformers` + `peft` + `accelerate`（FSDP2）**。训练循环自己写。FSDP wrap `Qwen3_5MoeDecoderLayer`。

Muse 线用的是同一组库里的 **TRL `SFTTrainer`**（观察 token 交叉熵）。旧 9B 线是 Unsloth。Coder-Next 是 Axolotl。这三条都不覆盖切鱼和 JEPA，本线不用它们当 trainer。

`probe.py`/`cut_stage1.py`/`compare.py` 是切鱼方案留下的测量工具，切鱼本身已经放弃（见「模型架构」），但这几个脚本作为**诊断/历史对照**继续保留、可以继续跑：`python train/scripts/probe.py`（读 `merge/output/cache` 里已下载的三个 checkpoint，写出 `train/outputs/probe/`，逐层量相对 Base 的位移比）；改得狠度的文本柱图：`python train/scripts/compare.py`（Layer Swapping 行均值绝对差，Base 缺权重会删残目录再下）。它们不再决定任何切点 \(\ell\)——现在的 Stage 1/2 流水线里没有切点这个概念。

盒子是 HuggingFace 的 `Qwen3_5MoeForConditionalGeneration`：40 层文本主干在 `model.language_model` 里，每层都是 MoE；30 层 Gated DeltaNet（`linear_attn`）+ 10 层完整注意力（`self_attn`，层号 3,7,…,39）；256 专家、每 token 8 个加 1 个共享专家。`lm_head` 独立。Instruct 另外还有 `model.visual`（ViT）和 `mtp.*`（官方投机解码草稿）。AgentWorld 的 `language_model_only=true`。

**Stage 1 直接读 AgentWorld 自己的 checkpoint，不经过任何切鱼步骤。** `train_jepallm.py` 用 `merge/download.py` 的 `resolve_model(role="world")` 解析 `configs/jepa/jepallm.yaml` 里的 `model_dir`（默认 `Qwen/Qwen-AgentWorld-35B-A3B`），缺失会自动下载进 `merge/output/cache`。**`lm_head` 留在活模块里**（AgentWorld 自己那张表）：主干 `forward` 只出隐藏状态（65k 全序列乘词表会爆显存），写字路只对观察标签那些位置再乘这张表，移位方式和 HuggingFace CausalLM 相同。对齐路是两次独立前向：`Enc(chat(h+a))` 和 `Enc(chat(o))`，各取 `last_token`（默认 `-3`），`1 - cosine`，两边都带梯度。三段序列分开跑、按 batch 实际长度 pad，不把三条拼成一个 3 倍 batch（那是他们 2k 序列上的写法）。LoRA 打在**全部 40 层**。启动时打印架构：`biv_wm.arch.log_world_architecture(..., expect_lm_head="attached")`。命令：`cd train && CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_jepallm.sh`（4 卡 FSDP2+CP，序列 65536）。检查点节奏与原来相同：2 个 epoch、每 25 步滚存 `checkpoint-e{epoch}-s{step}`（只留最新 3 个）、每个 epoch 结束永久存 `checkpoint-epoch{N}-end-s{step}`。损失和坍缩检查共用 `log_steps`（默认 5）。`--resume` 在 `output_dir`（默认 `outputs/jepallm_stage1`）里按 epoch 再 steps 选最新完整检查点。FSDP 下只 gather LoRA 到 CPU 再写 `adapter_model.safetensors`，没有 `jepa.pt` / `inv.pt`。控制台和 TensorBoard 前缀一律 `jepallm`。旧脚本 `train_jepa.sh` 仍在，是被替换掉的 MLP+SimCSE 配方，不要和新跑混对照。官方对照代码在 `train/vendor/llm-jepa/`（不进训练循环）。数据仍用 `wm_code` / `wm_os`（`mix_v2` 优先；**不要** `anti_forget`）。LoRA 只打 2D 线性叶子。FSDP wrap `Qwen3_5MoeDecoderLayer`。主干 LoRA `5e-5`，warmup 50 步后恒定。

Stage 2 单独加载 Instruct 自己的完整 checkpoint（不读任何切鱼输出），挂上草稿头 / 打分器 / \(W\)；Stage 1 训好的 AgentWorld LoRA 作为另一份独立权重加载进来，整段冻死，前向时对 `(h, 候选命令)` 编一遍，末尾隐藏向量当 \(\hat z\)，不参与 Stage 2 任何一步的参数更新。

Stage 2 单独加载 Instruct 自己的完整 checkpoint（不读 Stage 1 产物、不读任何切鱼输出），挂上草稿头 / 打分器 / \(W\)；Stage 1 训好的 JEPA（连同它挂着的 AgentWorld）作为另一份独立权重加载进来，整段冻死，前向时被当顾问查询，不参与 Stage 2 任何一步的参数更新。

---

## 训练全过程（Stage / Step）

顺序就是实验。Stage 1 的观察 token 交叉熵打在 **AgentWorld 自己的** `lm_head` 上，用来挡住坍缩；命令仍不在这一阶段写。Instruct 的 `lm_head` 要到 Stage 2 才碰。**不再有「Stage −1 切鱼」这一步**——AgentWorld 和 Instruct 各自保留完整权重，`probe.py`/`compare.py` 只是诊断工具，不产出任何要接进训练图的切好的 checkpoint。

**Stage Data — 切成回合**

- **Step 1.** 从原始轨迹里，把每一条命令和它后面那次沙箱观察配成一对。训练原子是 \((h, a, o)\)：发命令前的历史、这条真命令、真打回来的下一观察。
- **Step 2.** 按「上下文 / 动作 / 下一状态」来用。打乱 \(o\) 的拷贝当对照臂。

**Stage 1 — 世界（只碰 AgentWorld，LLM-JEPA 两路损失）**

套的是 [LLM-JEPA](https://arxiv.org/abs/2509.14252) 的 `RepresentationTrainer` 接线，配对换成我们的转移：左边 = 历史+命令（他们的 issue / Text），右边 = 真观察（他们的 diff / Code）。对照代码在 `train/vendor/llm-jepa/finetune.py`，训练**不调用**那份脚本；35B 仍走本仓库的 FSDP2+CP，三段序列分开前向、不整段过词表。

- **Step 1.** `model_dir` 解析成 AgentWorld（`train/configs/jepa/jepallm.yaml`）。**`lm_head` 留着**：底座冻住，LoRA 不打在这张表上，但写字损失的梯度穿过它回到骨干 LoRA。`forward` 主干只出隐藏状态。启动时确认 `lm_head: attached`。
- **Step 2.** 每个 \((h,a,o)\) 做成三段，和他们 user / assistant / full 一样各自走 chat 模板：完整对话给交叉熵，`chat(h+a)` 给左边，`chat(o)` 给右边。mix JSONL 里 user = 工具调用、assistant = 真观察。观察必须单独编码，不能从完整序列里抠。
- **Step 3.** 两路损失加在一起，一次反传：

  \[
  \texttt{loss} = \gamma\cdot\texttt{lm\_loss} + \lambda\cdot\texttt{jepa\_loss}
  \]

  `lm_loss`：HuggingFace 那种移位交叉熵，标签只解开观察正文。`jepa_loss`：`1 - mean(cosine(Enc(left)[last_token], Enc(right)[last_token]))`，**目标不 detach**。默认 \(\gamma=1\)、\(\lambda=0.1\)（他们的 `--gamma` / `--lbd`），Qwen 的 `last_token=-3`。没有 JEPA MLP、没有逆动力学、没有 SimCSE/bank。可训练的只有全部 40 层 LoRA（`rank=16`，`lr=5e-5`）。`cd train && CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_jepallm.sh`（不要加 `--resume` 才是从零开）。冒烟：`bash scripts/train_jepallm.sh --save-steps 1 --max-steps 2`。不要 `--resume` 旧的 `jepa_stage1` 检查点。训完这一步，AgentWorld+LoRA 整体冻死，Stage 2 只查询、不更新。

**32768 speed variant（截断，只为快速验证配方，不是正式跑）。** `train_jepallm_32k.sh` + `configs/jepa/jepallm_32k.yaml`：4 卡切成两组各 2 卡，每组内部按 `cp_size=2` 切序列，两组吃不同数据（`dp_replicate_size=2`）。比 65536/4 卡整组 CP 快一倍左右，代价是长样本被切得更狠。`output_dir=outputs/jepallm32k_stage1`、TensorBoard 前缀 `jepallm32k-`、控制台前缀 `[jepallm32k]`，和 65536 跑的 `outputs/jepallm_stage1` / `jepallm-` 互不覆盖。命令：`cd train && CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_jepallm_32k.sh`。两组各吃不同数据靠 `dp_replicate_info()`（`train_jepallm.py`）读 accelerate 的设备网格，第一次跑起来时终端会打印每个 rank 的 `dp_replicate rank=x/2`——GPU0/1 应该是 `0/2`、GPU2/3 应该是 `1/2`，不是这样说明这套映射没猜对，需要另外核实。这条路径没在这台开发机上跑过，第一次在 GPU 机上跑请顺手看一眼这行日志。

accelerate 的 `ParallelismConfig` 自带一条校验，只要 `dp_replicate_size>1` 又开了 `cp_size>1`，就要求 `dp_shard_size` 也必须 `>1`（否则报"pure data parallelism...cannot be used with...context parallelism"），按这条字面要求得 8 卡才能凑出「两组各自 CP+同步梯度」。但往下看它自己怎么建 FSDP 用的网格（`fsdp_dim_names`／`dp_shard_cp_dim_names`）：`cp` 自己就会被折进那张切分维度表，跟现在已经跑通的单组 4 卡 CP（`dp_shard=1, cp=4`）用的是同一套机制，只是没叠 `dp_replicate` 这层。所以 `train_jepallm.py` 的 `build_parallelism_config()` 手动绕开这条校验：构造时先塞个假的 `dp_shard_size=2` 骗过检查，构造完再把它改回真实值 `1`——后面用到的都是实时读属性，不受这次事后修改影响。没在 GPU 上验证过这个绕法，出问题应该是 `fully_shard` 直接报网格形状不对，不会静默训错。

**只有一份 TensorBoard 记录，代表两组合起来的整个任务，不是两份并行实验。** 之前试过每组各写各的 `-g0`/`-g1` 两条曲线，但那看起来像"跑了两份独立实验"，不是用户要的东西：用户要的是"card0 跑 5 步、card1 同时跑 5 步，合起来代表跑了 10 步"这种合并视图，方便直接跟不开并行的单卡曲线对比。做法是 `merge_group_stats()`（`train_jepallm.py`）：每次到 `log_steps`，**所有 rank**（不只是 is_main）都要跑一次 `accelerator.reduce(..., reduction="sum")`，把这一窗口内两组各自的 `running/run_ce/run_jepa/run_h/run_a/run_o/n_loss` 累加值加总。因为同一个 CP 组内每张卡的本地值完全相同（CP 是在算 loss 之前就把隐藏状态 all-gather 好了，loss 本身没被切），全局求和之后再取比值（比如 `running_g/denom_g`）会自动抵消 `cp_size` 这个重复因子，正好落在"两组真实平均"上——不需要额外除以 `dp_size`。只有 `is_main` 用这个合并后的值写一份 `SummaryWriter`（目录仍是单一的 `jepallm32k-<stamp>`，没有 `-g0`/`-g1` 后缀）。x 轴也换成 `eff_step = step * dp_size`（"单卡等效步数"），这样这条曲线走到 6816×2=13632 时，正好能跟不开并行、直接用 65536 序列跑的那条曲线在同一把 x 轴刻度上对比。`collapse/*` 也用同一个 `eff_step`，同一份记录里所有标量共享一套 x 轴。存档用的文件名（`checkpoint-e{epoch}-s{step}`）和 resume 逻辑仍然用没乘过 `dp_size` 的原始 `step`，跟这份"给人看的"x 轴是两回事，不要混着比。

**两组的 step 不会错位，不是靠运气。** `ReplicaSampler` 切数据前先按 `(n // dp_size) * dp_size` 截掉多出来的尾巴，保证每组分到的行数完全相等（最多扔掉 `dp_size-1` 行/epoch），`steps_per_epoch` 对两组永远算出同一个数，`save_steps`/`log_steps` 触发的 `step` 在所有 rank 上是同一个整数，不需要临时协商。另外加了 `assert_equal_loader_len()` 作为保险丝：训练开始前用一次 `all_gather_object` 把所有 rank 的 `len(loader)` 收集起来，只要有一个不一样就直接报错退出（把哪个 rank 长度是多少打出来），不会等到训到一半、某组数据先耗尽、another 组的梯度同步集合通信永远等不到对端，卡死在那里都不知道为什么。

写字这一路就是挡坍缩的锚：如果编码器把所有观察编成同一个方向，观察正文写不对。对齐这一路两边都活着，编码器没法把右边焊死再让预测头朝万能方向吐。这是他们成功案例里的做法，不是再叠对比损失。

坍缩检查只负责报，不产生梯度。损失和检查共用 `log_steps`（默认 5），存盘仍是 `save_steps`（默认 25）。检查用的是这一段已经训过的行上缓存的左边向量 / 右边向量。配对中位数明显高于错配 → 至少在见过的例子上分开了；两者都高 → 坍缩。

**Stage 1 坍缩检查（和损失一起，每 `log_steps` 步，默认 5）**

这些数字**不是**再跑一遍前向。训练每条样本时已经算出了左边向量 \(\hat z\)（独立编码的历史+命令，`last_token` 位置）和右边向量 \(z^*\)（独立编码的观察），主进程把它们缓存在 CPU 上；碰到 `log_steps` 时一边写 `train/loss*`，一边拿这一段刚训过的行做余弦统计。同一份结果出现三处：终端一行 `[jepallm] collapse ...`、`output_dir/collapse.json`、TensorBoard 的 `collapse/*`。

| 字段 | TensorBoard | 在问什么 |
|------|-------------|----------|
| `n` | `collapse/n` | 这一段拿了几条来比 |
| `skipped_same_o` | `collapse/skipped_same_o` | 两条观察原文完全一样、换位时丢掉的对数 |
| `paired_med` | `collapse/paired_median` | 左边向量靠不靠近自己这条的真观察编码。对应 `train/loss_jepa`（余弦高 ↔ 对齐损失低） |
| `mismatch_med` / `mismatch_p90` | `collapse/mismatch_median`、`collapse/mismatch_p90` | 左边向量靠不靠近别人的真观察编码 |
| `z_self_med` | `collapse/z_self_median` | 两条真观察编完之后彼此像不像 |
| `pred_self_med` | `collapse/pred_self_median` | 几条左边向量彼此像不像 |
| `verdict` | `collapse/verdict`（text） | `collapse_like`＝对自己和对别人都高；`paired_ahead`＝对自己高、对别人低；`unclear`＝两个都低；`no_mismatch_pairs`＝去掉相同文本后没有负例 |

期望方向：`paired_median` 升高、同时 `mismatch_median` 压低。只升配对、错配跟着升 → 万能方向。两项都低 → 还没拟合。差 ≥ 0.2 标 `paired_ahead`；配对 > 0.7 且差距 < 0.05 标 `collapse_like`。

**底座抽查（不训练）：相近向量把原文记下来。** 32k 配方里 `save_steps=25`、`grad_accum=8`、`batch_size=1`、两组卡。一次 optimizer step 在一组卡上吃 8 条；25 step 就是这一组 200 条。两组一起吃不重复的数据时是 400 条。训练时 TensorBoard 的 `collapse/n=40` 是 `log_steps=5` × 8 条，只覆盖更短的一窗。抽查默认跑 **200 条**（25 step、一组卡、和存盘节奏对齐），硬顶 **400**（两组 25 step 的全部不重复行）。只前向、不反传、不加 LoRA，编码方式和训练一样：独立 `Enc(历史+命令)` / `Enc(观察)`，取 `last_token=-3`。余弦规则与 `collapse_stats` 相同（观察原文逐字相同的不算负例）。向量余弦 ≥ 0.7 的对会把两边文本写进 JSON（观察对观察 = `z_self`，命令对命令 = `pred_self`，命令对别人的观察 = `mismatch`）。

```bash
cd train
CUDA_VISIBLE_DEVICES=0 bash scripts/probe_jepallm_collapse.sh          # 1 卡：直接 python，不开 CP
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/probe_jepallm_collapse.sh    # 4 卡：和训练一样 2x2
# 只看一窗 40 条：bash scripts/probe_jepallm_collapse.sh --max-rows 40
```

1 卡时不要再走 `qwen35_moe_fsdp2_cp.yaml`：那份 yaml 写死了 `num_processes: 4` / `cp_size: 4`，进程只有 1 个时会报 `Mesh should not be bigger than default world size 1, but found 4 ranks`。脚本会清掉训练留下的 `PARALLELISM_CONFIG_*`。32k 整段在单卡上可能 OOM，把 `--max-length` 降下来即可。

结果在 `outputs/jepallm32k_collapse_probe/collapse_probe-<stamp>.json`（并复制一份 `collapse_probe_latest.json`）。

**TensorBoard `train/*`**

| 标量 | 是什么 |
|------|--------|
| `train/loss` | 每 micro-batch 反传的加权和：\(\gamma\times\)CE \(+ \lambda\times\)JEPA |
| `train/loss_ce` | 观察 token 移位交叉熵（还没乘 \(\gamma\)） |
| `train/loss_jepa` | 左右 `last_token` 隐藏 `1 - cosine`（还没乘 \(\lambda\)） |
| `train/lr_backbone` | 主干 LoRA 的学习率，热身到 `5e-5` |
| `train/len_h` / `len_a` / `len_o` | 这段样本里完整对话 / 左边 / 右边的非 pad token 数 |

**JEPA 怎么防坍缩（现在跟 LLM-JEPA 同一套）**

图像 JEPA 常用 EMA 教师和学生停梯度。我们 35B、只训 LoRA 时，完整教师存不下，而且旧配方把观察编码焊在 `no_grad` 里之后，逆动力学 / SimCSE / bank 都没挡住「预测头吐万能方向」。LLM-JEPA 的挡法更简单：对齐从来不是单独训的，旁边始终挂着「把右边写成字」。字写不对，编码器就不能把所有样本编成一个样。我们把这一条换成写**观察**。旧的 `train_jepa.sh`（MLP + inv + simcse + bank）留着对照，新跑走 `train_jepallm.sh`。

梯度检查点保持全开。没开量化。`max_length=65536` 下关掉检查点，长尾样本会 OOM。

**Stage 2 — 出字（Instruct 自己的 backbone + 草稿 / 打分 / \(W\)，冻死的 AgentWorld 当顾问）**

单独加载 Instruct 自己完整的 checkpoint，不读切鱼输出；Stage 1 训好的 AgentWorld LoRA 作为另一份独立权重加载进来，整段冻死。查询接口不再是小 MLP：对每个候选把 \((h, a_k)\) 送进冻住的 AgentWorld，末尾隐藏向量就是 \(\hat z_k\)。本次不改 Stage 2 代码，只把这个接口写在这里。

- **Step 1（门口）。** 冻住 Instruct backbone 和 Instruct `lm_head`。加上草稿头、打分器、\(W\)。前向：Instruct → \(c_t\) → 草稿提出 \(K\) 个候选命令 → 冻死的 AgentWorld 对每个候选编 \((h,a_k)\) 得到 \(\hat z_k\)，真实命令同样编一遍当真实分支。两条损失：
  - **打分器的排序损失**：\(K{+}1\) 路 softmax 交叉熵，标签是真实分支。
  - **解码器的教师强制交叉熵**：真实命令过 \(W\to\) Instruct `lm_head`。AgentWorld 全程冻死，命令交叉熵不回流到它。
- **Step 2（嘴巴和主干）。** Instruct `lm_head` 用很小的学习率或 LoRA 解开，Instruct backbone 也开一点 LoRA。AgentWorld 永远冻死。
- **推理时**才 \(\arg\max\) 选 \(u_i^A\)，过 \(W\to\) Instruct `lm_head` 生成命令。

**Stage Eval — 同一套脚手架**

- **Step 1.** TB2.1 / Harbor：Stage 2 之前对之后（头条）。
- **Step 2.** 语料里 `rm` 改成 `zaq`：还能跟上转移才像律。
- **Step 3.** 截断轨迹，命中观察不在上下文里：\(c_t\) / \(\hat z\) / 下一步命令是否已经当文件没了。打乱 \(o\) 的孪生作对照。
- **Step 4.** 对照臂（与主线并行）：同一批命令行打在 Instruct 上。排序和保真度分开报。

**用例子串一遍。** Stage Data 切出删除回合和恢复回合。Stage 1 吃删除：AgentWorld 编出「文件还在」和 `rm a.txt`，JEPA 对齐「没了」，训完整体冻死。Stage 2 吃恢复：Instruct 的 backbone 给出「已经没了」，草稿头提出恢复 / 再删 / 去 cat 三个候选（各自一份 \(u^W\)/\(u^A\)），冻死的 AgentWorld+JEPA 看三条未来打出 \(\hat z\)，打分器（训练时靠 \(K{+}1\) 路 softmax 交叉熵、教师强制的真实分支走交叉熵）学会给恢复这条打最高分，推理时 \(\arg\max\) 选中它，\(W(u_i^A)\) 写出 `git checkout a.txt`。Stage Eval Step 3 把删除成功的观察藏起来，问恢复是不是已经被选中。

---

## 数据怎么对应到 Stage

仓库里已经接好的三份原料：SWE-Hero（代码沙箱工具 I/O，`wm_code`）、ISETrace（真实 OS 工具 I/O，`wm_os`）、SWE-Zero（整段 agent 路径，`anti_forget`，按 `instance_id` 相对 Hero 去重）。`prepare_data.py --wm-code --wm-os` 写到 `data/processed/mix_v1/`（或你指定的 `--out-dir`，本线优先读已有的 `mix_v2`）。默认一条轨迹一行。标签始终是沙箱真实 I/O。`data/global_demon_prompt.txt` 只给运行时 Demon 用。

三份都能切出 \((h,a,o)\)。Stage 1 用带真观察的回合，经 AgentWorld 自己的 backbone 编码后只训 JEPA。Stage 2 用同一类回合里的**命令字符串**当嘴巴目标，草稿头/打分器在真 \((c_t,u^{\star W},z^{\star W})\) 上把真实分支当 \(K{+}1\) 个候选之一训排序损失，解码器走教师强制。Zero 补写命令的面。Terminal 域语料仍可选。Muse 线上的 1:1:0.35 mix 属于另一条 checkpoint，不要直接当成本线配方。

系统提示若还要用短世界角色，见 `train/src/biv_wm/formatting.py` 的 `DEFAULT_WM_SYSTEM`。

---

## BIV 运行时（Cartesian）

| 角色 | 位置 | 工作 |
|------|------|------|
| Agent A | nanobot 循环 + 配置的 provider | 规划、工具、用户对话 |
| Agent B（Demon） | `cartesian/demon.py` | 在 Matrix Law 下伪造工具结果 |
| 代理 | `cartesian/tool_proxies.py` | `exec` / FS / web 交给 B；`create_goal` / `update_goal` 保持真；丢掉逃生工具 |

- 仪表盘 + API：`cartesian-dashboard/`、`cartesian/server.py`
- Matrix Law 提示：`data/global_demon_prompt.txt`（仅运行时）
- 现场说明：根目录 `README.md`

## 世界模型训练操作（`train/`，含另一条 Muse 线）

北极星仍是：agent 变强，原因是世界理解变好。同时保持模型仍能写命令、选工具。详细 runbook：[`train/README.md`](./train/README.md)。

**本线（Qwen3.5-35B 两条独立 backbone + JEPA）**，GPU 机上 mix JSONL 已有的话，Stage 1 不再需要切鱼这一步：

```bash
# 可选诊断：相对 Base 的行-MAV 文本表（不再决定任何切点，纯测量）
python train/scripts/compare.py

# 若还没有 mix：不要 --all（Stage 1 不用 anti_forget）
python train/scripts/prepare_data.py --wm-code --wm-os --out-dir train/data/processed/mix_v2

# Stage 1 LLM-JEPA：直接读 AgentWorld 自己的 checkpoint（缺失自动下载），
# 4 卡 FSDP2+CP，序列 65536。两路损失：写观察 + 左右隐藏对齐。
cd train
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_jepallm.sh

# 可选：AgentWorld 词表下三段序列在 65536 会砍掉多少（CPU，只加载 tokenizer；
# 长度写入 train/outputs/stat_cache/jepallm/，再跑会命中）
python scripts/stat.py --max-length 65536
```

`probe.py`/`cut_stage1.py` 仍在仓库里，留作历史对照/实验用，不在这条 live 流水线里。

旧 9B 线可用 `configs/pilot.yaml` 先筛（保持 8k，少采样行，不要截断序列把观察切掉）。缓存根：`outputs/ds_cache/`。GPU 上再跑 `train_sft.py` / `swift` / `axolotl`；`prepare_data.py` 等可在 CPU。

```bash
cd train
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
python scripts/prepare_data.py --swe-hero --out-dir data/processed --eval-ratio 0.05
python scripts/train_sft.py --config configs/pilot.yaml
python scripts/eval_wm.py --config configs/pilot.yaml
```

**Muse Glimmer-30B**（分支 `Muse`，TRL + PEFT LoRA；与 Qwen3.5-35B 本线分开）：

```bash
cd train
pip install -r requirements-muse.txt
python scripts/prepare_model.py
python scripts/tokenize_data.py
CUDA_VISIBLE_DEVICES=0 bash scripts/trainmodel.sh --max-length 8192 --choice 1
```

数据 TODO：Nebius / Terminal 域；anti_forget 的 Coder-XML 渲染检查；OS held-out 面板。

旧假设评测协议（Muse / 9B LoRA 线）：held-out 下一观察指标；同一脚手架上 base vs 真 I/O LoRA vs 打乱 LoRA；打乱若同样涨则拒绝「世界理解→迁移」；agent 指标相对 base 不崩。

## 上游 nanobot 开发命令

```bash
pytest tests/test_openai_api.py::test_function -v
ruff check nanobot/
uv sync --all-extras --dev
uv run --no-sync python -m scripts.install_channel_dependencies --all-channels
uv run --no-sync basedpyright
cd webui && bun run dev
nanobot gateway
./start-biv.sh
```

消息经 `nanobot/bus/queue.py` 的异步总线：Channels 发 `InboundMessage` → `AgentLoop` → `AgentRunner`（LLM ↔ 工具）→ `OutboundMessage`。BIV 里碰现实的工具在返回 A 之前被 Demon 代理替换。

子系统：`nanobot/agent/`、`providers/`、`channels/`、`agent/tools/`、`agent/memory.py`、`session/`、`config/`（常见 `~/.nanobot/config.json`；BIV 还有 `config/cartesian.json`）、`webui/`、`cartesian-dashboard/`、`nanobot/api/server.py`、`cartesian/`、`train/`。

入口：`nanobot/cli/commands.py`、`nanobot/nanobot.py`、`./start-biv.sh`、以及 `train/scripts/` 下的 prepare / tokenize / train。

项目笔记：`.agent/design.md`、`.agent/security.md`、`.agent/gotchas.md`、`train/README.md`、`refs/README.md`、`merge/merge.py`。贡献见 `CONTRIBUTING.md`。

代码风格：Python 3.11+，运行时全 asyncio；行宽 100；ruff E,F,I,N,W（忽略 E501）；pytest `asyncio_mode = auto`。`train/` 自备 venv。

常见路径：`nanobot/config/schema.py`、`providers/base.py`、`channels/base.py`、`agent/tools/registry.py`、`cartesian/demon.py`、`cartesian/tool_proxies.py`、`train/src/biv_wm/`、`train/configs/trl/muse_glimmer_30b_lora.yaml`（Muse）、`refs/papers/`、`merge/merge.py`、`webui/vite.config.ts`。测试目录镜像 `nanobot/` 包结构。

---

## 灵感来源

编号对下面 **论文链接**。[1]–[42] 是最终方案直接用到的零件；同一编号只出现一次。两条 backbone 之间连接件怎么训这一条结论额外引了 [198]–[200]。Stage 1 现在套 LLM-JEPA 的两路损失 [213]：观察 token 交叉熵挡住坍缩，左右末尾隐藏向量对齐是 JEPA 项。先前 MLP + 停梯度 + 逆动力学 + SimCSE/bank 的防坍缩尝试及其文献 [201]–[212] 是被替换掉的配方，旧脚本 `train_jepa.sh` 仍在仓库里，不是当前训练入口。

世界这一头走 JEPA：在表征空间里预测下一状态。潜预测的框架来自 [1][2]；视频上的同类预训练对应直觉物理 [3]。把「下一潜状态」接到 Transformer 来自 [6][7]。Stage 1 的具体损失抄 LLM-JEPA [213]：给定左边（历史+命令），一边用 AgentWorld 自己的 `lm_head` 把右边（真观察）写成字，一边让两边最后一层末尾隐藏向量靠近。写字挡住「全编成一个样」，对齐是 JEPA 项本身。观察损失和命令损失仍然分家：观察写在 AgentWorld 的嘴上，命令写在 Instruct 的嘴上 [8][9][10]。

出字这一头走「先在向量里提案，在潜空间里看未来，只把一个赢家写成字」。ω-EVA 是提案 → 潜未来 → 改写 [11]；I2A 先编码想象轨迹再交给策略 [12]；Coconut 让字只在最后出现 [13]。动作侧先学表征、再还原成真命令 [14]。负例在同一条轨迹内采 [15]；转移损失用排序形式 [16][17]，现在具体落地成打分器的 \(K{+}1\) 路 softmax 交叉熵（真实分支当正例，草稿当负例，不需要经过 \(\arg\max\)）。\(K\) 个草稿全部过冻住的 AgentWorld（对每个候选编 \((h,a_k)\)），\(\arg\max\) 只在推理时用一次；训练时解码器走教师强制的真实分支，和打分器的排序损失彼此独立。

主干**不做**同源整层替换：Layer Swapping 这条路径 [18][19][20] 试过，`compare.py` 测出来的结果是处处都动、只是幅度不同，否定了它「某些层几乎不动」的低熵前提，已经放弃，两条 backbone 各自保留、不合并（细节见「模型架构」）。末端层负责把内部状态收成具体输出这条证据 [21] 仍然支持「出字这一路窄」的设计。两条 backbone 之间的连接件走的是模型缝合（model stitching）的思路 [22][23]，具体到两个独立训练的模型之间怎么训连接件才真的管用而不是看起来像，见下一段。先冻只训头、再小步解冻主干仍按 LP-FT 的顺序走 [24][25]，只是现在冻的是 Instruct 自己完整的 backbone，不是切鱼之后的半截。世界 checkpoint 与 Harbor / TB2.1 口径来自 Qwen-AgentWorld [26]。底座是 Gated DeltaNet 与完整注意力的混合 [27]。

连接件（草稿头的 \(u^W\) 输出头、打分器的输入层）不能只靠离线做「让两边分布看起来像」的对齐：两个独立模型之间做到 0.97 归一化余弦相似度的线性映射，注入真实前向计算之后完全没用、甚至有害，这是一个明确的负结果 [198]。反过来，让连接件的参数直接挨真实下游任务的梯度、端到端训过完整的使用路径，才是有效的做法 [199][200]。AgentWorld 和 Instruct 同出一个 Base、同一套 40 层盒子，分叉前表征空间完全相同，这个起点比 [198][199][200] 验证过的场景都更接近同源，但好起点不能替代对的训练方法，这也是负结果论文真正想说明的事。

「律进参数、提早闪避」对齐权重里的规则式泛化 [34]，以及时间局部性一类归纳偏置 [33]。后继特征把未来占用编进现在的向量 [35][36]。多领域上游表征对下游策略的样本收益见 [41]。先世界、后策略的顺序对齐 Dreamer 和 Internalizing the Future [38][39]。评测同时报排序和保真度 [28][29]，用命令重命名（`rm`→`zaq`）[30]、世界模型恢复度 [31][32] 和 VoE [40]。辅助损失权重可用梯度余弦调度 [37]。stdout 里与控制无关的噪声用控制内生状态来滤 [42]。

分组索引和 HTML 下载仍在 [`refs/README.md`](./refs/README.md)。刷新：`python3 refs/fetch_html_text.py`。不要提交 `refs/pdfs/`。

## 论文链接

[1]–[42] 对应上一节。[43] 起为检索中读过、最终方案未直接引用的篇目。

[1] [A Path Towards Autonomous Machine Intelligence (LeCun, JEPA)](https://consensus.app/papers/details/376c7ec2fb015a48bacc8b62901a860a/)
[2] [V-JEPA 2](https://arxiv.org/abs/2506.09985)
[3] [Intuitive physics understanding emerges from self-supervised pretraining on natural videos](https://arxiv.org/abs/2502.11831)
[4] [When does Self-Prediction help? Understanding Auxiliary Tasks in RL](https://arxiv.org/abs/2406.17718)
[5] [Understanding Self-Predictive Learning for RL](https://arxiv.org/abs/2212.03319)
[6] [Next-Latent Prediction Transformers Learn Compact World Models (NextLat)](https://arxiv.org/abs/2511.05963)
[7] [Textual Belief States for World Models: Identifiable Representation Learning Under Strict Mediation](https://arxiv.org/abs/2606.27681)
[8] [RWML](https://arxiv.org/abs/2602.05842)
[9] [Error Bounds of Imitating Policies and Environments](https://arxiv.org/abs/2010.11876)（期刊版 [TPAMI](https://doi.org/10.1109/tpami.2021.3096966)）
[10] [The Value Equivalence Principle](https://arxiv.org/abs/2011.03506)
[11] [ω-EVA](https://arxiv.org/abs/2606.09457)
[12] [Imagination-Augmented Agents (I2A)](https://arxiv.org/abs/1707.06203)
[13] [Coconut: Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
[14] [Learning Action Representations for RL (Chandak)](https://arxiv.org/abs/1902.00183)
[15] [In-trajectory negatives for dynamics (contrastive WM)](https://arxiv.org/abs/2606.07770)
[16] [Ranking noise-contrastive estimation for transitions](https://arxiv.org/abs/2311.01388)
[17] [R-NCE](https://arxiv.org/abs/2309.05803)
[18] [Layer Swapping for Zero-Shot Cross-Lingual Transfer](https://arxiv.org/abs/2410.01335)
[19] [The Unreasonable Effectiveness of Model Merging for Cross-Lingual Transfer](https://arxiv.org/abs/2505.18356)
[20] [Rethinking the Multilingual Reasoning Gap with Layer Swap](https://arxiv.org/abs/2605.26735)
[21] [The Remarkable Robustness of LLMs: Stages of Inference?](https://arxiv.org/abs/2406.19384)
[22] [Model stitching](https://arxiv.org/abs/2106.07682)
[23] [Revisiting Model Stitching in the Foundation Model Era](https://arxiv.org/abs/2603.12433)
[24] [Fine-Tuning can Distort Pretrained Features and Underperform OOD (LP-FT)](https://arxiv.org/abs/2202.10054)
[25] [Parameter-Efficient Tuning Makes a Good Classification Head](https://arxiv.org/abs/2210.16771)
[26] [Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)
[27] [Gated Delta Networks](https://arxiv.org/abs/2412.06464)
[28] [PatchWorld: Gradient-Free Optimization of Executable World Models](https://arxiv.org/abs/2605.30880)
[29] [OneLife](https://arxiv.org/abs/2510.12088)
[30] [Baba in Wonderland / Alice](https://arxiv.org/abs/2605.16725)
[31] [What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](https://arxiv.org/abs/2507.06952)
[32] [Evaluating the World Model Implicit in a Generative Model](https://arxiv.org/abs/2406.03689)
[33] [From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers](https://arxiv.org/abs/2602.06923)
[34] [Transformers generalize differently from information stored in context vs in weights](https://arxiv.org/abs/2210.05675)
[35] [Does Zero-Shot Reinforcement Learning Exist? (Forward-Backward)](https://arxiv.org/abs/2209.14935)
[36] [Which Features are Best for Successor Features?](https://arxiv.org/abs/2502.10790)
[37] [Adapting Auxiliary Losses Using Gradient Similarity](https://arxiv.org/abs/1812.02224)
[38] [DreamerV3](https://arxiv.org/abs/2301.04104)
[39] [Internalizing the Future](https://arxiv.org/abs/2606.27483)
[40] [IntPhys 2](https://arxiv.org/abs/2506.09849)
[41] [Provable Benefit of Multitask Representation Learning in RL](https://arxiv.org/abs/2206.05900)
[42] [Guaranteed Discovery of Control-Endogenous Latent States (AC-State)](https://arxiv.org/abs/2207.08229)

[43] [Deep RL in Large Discrete Action Spaces](https://arxiv.org/abs/1512.07679)
[44] [Physics-Informed Neural Networks (PINNs)](https://arxiv.org/abs/1711.10561)
[45] [World Models (Ha & Schmidhuber)](https://arxiv.org/abs/1803.10122)
[46] [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
[47] [PlaNet](https://arxiv.org/abs/1811.04551)
[48] [The Natural Language of Actions (Act2Vec)](https://arxiv.org/abs/1902.01119)
[49] [Hamiltonian Neural Networks](https://arxiv.org/abs/1906.01563)
[50] [SAVE](https://arxiv.org/abs/1912.02807)
[51] [Graph Network Simulator (GNS)](https://arxiv.org/abs/2002.09405)
[52] [Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630)
[53] [Understanding and Improving Information Transfer in MTL](https://arxiv.org/abs/2005.00944)
[54] [Active ICP](https://arxiv.org/abs/2006.05690)
[55] [FLAMBE](https://arxiv.org/abs/2006.10814)
[56] [Active World Model Learning with Progress Curiosity (γ-Progress)](https://arxiv.org/abs/2007.07853)
[57] [Jointly-Learned State-Action Embedding](https://arxiv.org/abs/2010.04444)
[58] [Generalization to New Actions in RL](https://arxiv.org/abs/2011.01928)
[59] [arXiv:2105.01136](https://arxiv.org/abs/2105.01136)
[60] [arXiv:2106.04379](https://arxiv.org/abs/2106.04379)
[61] [Model-Advantage Optimization](https://arxiv.org/abs/2106.14080)
[62] [The Causal-Neural Connection (Causal Hierarchy Theorem)](https://arxiv.org/abs/2107.00793)
[63] [REP-UCB](https://arxiv.org/abs/2110.04652)
[64] [Interventions, Where and How?](https://arxiv.org/abs/2203.02016)
[65] [VaGraM](https://arxiv.org/abs/2204.01464)
[66] [Data Distributional Properties Drive Emergent ICL](https://arxiv.org/abs/2205.05055)
[67] [Provable Benefits of Representational Transfer in RL](https://arxiv.org/abs/2205.14571)
[68] [Deciding What to Model](https://arxiv.org/abs/2206.02072)
[69] [LST side-tuning](https://arxiv.org/abs/2206.06522)
[70] [PAC Reinforcement Learning for Predictive State Representations](https://arxiv.org/abs/2207.05738)
[71] [SPEDER](https://arxiv.org/abs/2208.09515)
[72] [On the Power of Pre-training for Generalization in RL](https://arxiv.org/abs/2210.10464)
[73] [Task Arithmetic](https://arxiv.org/abs/2212.04089)
[74] [TraceFixer](https://arxiv.org/abs/2304.12743)
[75] [CodeExecutor](https://arxiv.org/abs/2305.05383)
[76] [LIMA](https://arxiv.org/abs/2305.11206)
[77] [RAP](https://arxiv.org/abs/2305.14992)
[78] [UniPi](https://arxiv.org/abs/2305.16309)
[79] [Nonparametric Identifiability of Causal Representations from Unknown Interventions](https://arxiv.org/abs/2306.00542)
[80] [Birth of a Transformer](https://arxiv.org/abs/2306.00802)
[81] [TIES-Merging](https://arxiv.org/abs/2306.01708)
[82] [For SALE](https://arxiv.org/abs/2306.02451)
[83] [Learning World Models with Identifiable Factorization (IFactor)](https://arxiv.org/abs/2306.06561)
[84] [Pretraining task diversity and the emergence of non-Bayesian ICL](https://arxiv.org/abs/2306.15063)
[85] [MPDP: Theoretically Guaranteed Policy Improvement Distilled from Model-Based Planning](https://arxiv.org/abs/2307.12933)
[86] [Chat Vector](https://arxiv.org/abs/2310.04799)
[87] [The Expressive Power of Transformers with CoT](https://arxiv.org/abs/2310.07923)
[88] [O-LoRA](https://arxiv.org/abs/2310.14152)
[89] [General Identifiability and Achievability for CRL](https://arxiv.org/abs/2310.15450)
[90] [Implicit CoT via Knowledge Distillation](https://arxiv.org/abs/2311.01460)
[91] [Identifying Linearly-Mixed Causal Representations from Multi-Node Interventions](https://arxiv.org/abs/2311.02695)
[92] [DARE](https://arxiv.org/abs/2311.03099)
[93] [The Transient Nature of Emergent ICL](https://arxiv.org/abs/2311.08360)
[94] [PhysGaussian](https://arxiv.org/abs/2311.12198)
[95] [The mechanistic basis of data dependence and abrupt learning in an in-context classification task (Reddy)](https://arxiv.org/abs/2312.03002)
[96] [Bridging State and History Representations](https://arxiv.org/abs/2401.08898)
[97] [Score-based Causal Representation Learning](https://arxiv.org/abs/2402.00849)
[98] [Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](https://arxiv.org/abs/2402.12875)
[99] [The Illusion of State in State-Space Models](https://arxiv.org/abs/2404.08819)
[100] [CAASL](https://arxiv.org/abs/2405.16718)
[101] [Dual Process Learning: Controlling ICL vs IWL with Weight Forgetting](https://arxiv.org/abs/2406.00053)
[102] [SemCoder](https://arxiv.org/abs/2406.01006)
[103] [Linear Causal Representation Learning from Unknown Multi-node Interventions](https://arxiv.org/abs/2406.05937)
[104] [Amortized Planning with Large-Scale Transformers](https://arxiv.org/abs/2406.11907)
[105] [An Optimal Tightness Bound for the Simulation Lemma](https://arxiv.org/abs/2406.16249)
[106] [The Central Role of the Loss Function in RL](https://arxiv.org/abs/2409.12799)
[107] [NeuMA](https://arxiv.org/abs/2410.08257)
[108] [Toward Understanding In-context vs. In-weight Learning](https://arxiv.org/abs/2410.23042)
[109] [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
[110] [Proto Successor Measure](https://arxiv.org/abs/2411.19418)
[111] [Differential learning kinetics](https://arxiv.org/abs/2412.00104)
[112] [MaxInfoRL](https://arxiv.org/abs/2412.12098)
[113] [Cosmos](https://arxiv.org/abs/2501.03575)
[114] [TATR](https://arxiv.org/abs/2501.15065)
[115] [DeltaProduct](https://arxiv.org/abs/2502.10297)
[116] [LATA](https://arxiv.org/abs/2502.20186)
[117] [CODI](https://arxiv.org/abs/2502.21074)
[118] [Log-depth transformers](https://arxiv.org/abs/2503.03961)
[119] [What I cannot execute, I do not understand (Execution Tuning)](https://arxiv.org/abs/2503.05703)
[120] [AGLO](https://arxiv.org/abs/2503.08867)
[121] [TheoryCoder](https://arxiv.org/abs/2503.20124)
[122] [CodeARC](https://arxiv.org/abs/2503.23145)
[123] [CAT Merging](https://arxiv.org/abs/2505.06977)
[124] [PoE-World](https://arxiv.org/abs/2505.10819)
[125] [SLiCEs](https://arxiv.org/abs/2505.17761)
[126] [ShIOEnv](https://arxiv.org/abs/2505.18374)
[127] [Looping / padding for serial computation](https://arxiv.org/abs/2505.18948)
[128] [Calibrated Value-Aware Model Learning](https://arxiv.org/abs/2505.22772)
[129] [DyMo](https://arxiv.org/abs/2506.02918)
[130] [PTS-BE](https://arxiv.org/abs/2507.02639)
[131] [SIM-CoT](https://arxiv.org/abs/2509.20317)
[132] [Dyna-Mind](https://arxiv.org/abs/2510.09577)
[133] [SPA](https://arxiv.org/abs/2510.15047)
[134] [LED-WM](https://arxiv.org/abs/2511.22904)
[135] [Spectral Representation-based RL](https://arxiv.org/abs/2512.15036)
[136] [From Shortcut to Induction Head](https://arxiv.org/abs/2512.18634)
[137] [Ortho-LoRA](https://arxiv.org/abs/2601.09684)
[138] [Demystifying Mergeability](https://arxiv.org/abs/2601.22285)
[139] [Do Latent-CoT Models Think Step-by-Step?](https://arxiv.org/abs/2602.00449)
[140] [TheoryCoder-2](https://arxiv.org/abs/2602.00929)
[141] [arXiv:2602.02900](https://arxiv.org/abs/2602.02900)
[142] [ProAct](https://arxiv.org/abs/2602.05327)
[143] [Verification of the Implicit World Model in a Generative Model via Adversarial Sequences](https://arxiv.org/abs/2602.05903)
[144] [NeSyS](https://arxiv.org/abs/2602.10480)
[145] [Learning State-Tracking from Code Using Linear RNNs](https://arxiv.org/abs/2602.14814)
[146] [DreamZero / World Action Model](https://arxiv.org/abs/2602.15922)
[147] [Causality is Key for Interpretability Claims to Generalise](https://arxiv.org/abs/2602.16698)
[148] [arXiv:2603.02862](https://arxiv.org/abs/2603.02862)
[149] [Phys4D](https://arxiv.org/abs/2603.03485)
[150] [Why Are Linear RNNs More Parallelizable?](https://arxiv.org/abs/2603.03612)
[151] [M²RNN](https://arxiv.org/abs/2603.14360)
[152] [Beyond identifiability: Learning causal representations with few environments and finite samples](https://arxiv.org/abs/2603.25796)
[153] [Self-Execution Simulation Improves Coding Models](https://arxiv.org/abs/2604.03253)
[154] [The Depth Ceiling](https://arxiv.org/abs/2604.06427)
[155] [Outcome-Predictive State Representations (OPSR)](https://arxiv.org/abs/2604.07016)
[156] [Distinct mechanisms underlying ICL](https://arxiv.org/abs/2604.12151)
[157] [Latent Planning Emerges with Scale](https://arxiv.org/abs/2604.12493)
[158] [Curiosity-Critic](https://arxiv.org/abs/2604.18701)
[159] [Physically Native World Models](https://arxiv.org/abs/2605.00412)
[160] [NSI](https://arxiv.org/abs/2605.01293)
[161] [Where's the Plan?](https://arxiv.org/abs/2605.07984)
[162] [Latent Geometry Beyond Search](https://arxiv.org/abs/2605.08732)
[163] [DeformMaster](https://arxiv.org/abs/2605.09586)
[164] [StepCodeReasoner](https://arxiv.org/abs/2605.11922)
[165] [PriorZero](https://arxiv.org/abs/2605.12289)
[166] [PhyWorld](https://arxiv.org/abs/2605.19242)
[167] [ECHO](https://arxiv.org/abs/2605.24517)
[168] [RL vs SFT circuits](https://arxiv.org/abs/2605.28860)
[169] [Policy-Aware Simulator Learning](https://arxiv.org/abs/2605.29032)
[170] [YoCausal](https://arxiv.org/abs/2605.30346)
[171] [COMAP](https://arxiv.org/abs/2606.02372)
[172] [PaW](https://arxiv.org/abs/2606.02388)
[173] [Mind-Studio](https://arxiv.org/abs/2606.16070)
[174] [Can In-Context Learning Support Intrinsic Curiosity?](https://arxiv.org/abs/2606.19476)
[175] [OPINE-World](https://arxiv.org/abs/2607.01531)
[176] [Sparse Delta Memory](https://arxiv.org/abs/2607.07386)
[177] [Capacity–Redundancy Trade-offs in Multi-Task Learning](https://arxiv.org/abs/2607.16554)
[178] [PSG-JEPA](https://arxiv.org/abs/2608.06799)
[179] [X-VoE](https://doi.org/10.1109/iccv51070.2023.00369)
[180] [Piloto 2022, Violation of Expectation](https://doi.org/10.1038/s41562-022-01394-8)
[181] [Shared sensitivity to data distribution in humans and transformers](https://doi.org/10.1038/s41562-025-02359-3)
[182] [Jointly-Learned State-Action Embedding (CIKM)](https://doi.org/10.1145/3459637.3482357)
[183] [Combining Functional and Automata Synthesis (Autumn/Das)](https://doi.org/10.1145/3571249)
[184] [TRACED: Execution-Aware Pre-Training for Source Code](https://doi.org/10.1145/3597503.3608140)
[185] [Active Inference, Curiosity and Insight (Friston)](https://doi.org/10.1162/neco_a_00999)
[186] [The Expressive Power of Transformers (TACL, Merrill & Sabharwal)](https://doi.org/10.1162/tacl_a_00493)
[187] [Spectral Learning of PSRs with Insufficient Statistics](https://doi.org/10.1609/aaai.v29i1.9635)
[188] [Control-Oriented MBRL with Implicit Differentiation](https://doi.org/10.1609/aaai.v36i7.20758)
[189] [Towards Effectively Leveraging Execution Traces for Program Repair](https://doi.org/10.18653/v1/2025.knowledgenlp-1.17)
[190] [Learning and exploration in action-perception loops (Little & Sommer)](https://doi.org/10.3389/fncir.2013.00037)
[191] [BMAS](https://doi.org/10.3390/a17020060)
[192] [Language Representations for Generalization in RL](https://consensus.app/papers/details/10901a6c273b5bd699c3e1a3e69b78c1/)
[193] [Iterative VAML (Farahmand 2018)](https://consensus.app/papers/details/14e6dd2e07ee51108dfbdf5849a185c5/)
[194] [Fixed-Point RNNs](https://consensus.app/papers/details/22e1ea9b37195e2aadfa03fd27931301/)
[195] [Value-Aware Loss Function for MBRL (Farahmand 2017)](https://consensus.app/papers/details/b0f6ce8be2c65cad9395b655aa4342f0/)
[196] [Decision-Aware Model Learning for Actor-Critic: When Theory Does Not Meet Practice](https://consensus.app/papers/details/df54c74ed6165977957689b47d7bd7dd/)
[197] [MDP homomorphisms (van der Pol)](https://consensus.app/papers/details/1cfe004e0ff557c79871865825e0a21c/)

[198] [A Negative Result on Cross-Model Activation Transfer in a Pythia Multi-Hop Setting](https://consensus.app/papers/details/3882637eac3059fa87366e0661f009b7/)
[199] [The Bicameral Model: Bidirectional Hidden-State Coupling Between Parallel Language Models](https://consensus.app/papers/details/95cf6a71ffee5d18b78189605ec37f2d/)
[200] [Latent Space Communication via K-V Cache Alignment](https://consensus.app/papers/details/798e52a7c2835eeaa12313a7cc9d83a5/)

[201] [I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://consensus.app/papers/details/55bce9858cb654c68da164ab93faf9bb/)
[202] [C-JEPA: Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning](https://consensus.app/papers/details/4ef0e26b71ca58e788e85bbb28f72015/)
[203] [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://consensus.app/papers/details/ce639618c566518c98a2d7af76e0234f/)
[204] [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://consensus.app/papers/details/9eb96593274e541c950622292cd05348/)
[205] [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://consensus.app/papers/details/38896c0cf07758818f46cbbe31bfe177/)
[206] [Delta-JEPA: Learning Action-Sensitive World Models via Latent Difference Decoding](https://consensus.app/papers/details/113822ab3e8d5d2a998bc66dc83f69a7/)
[207] [Sensorimotor World Models: Perception for Action via Inverse Dynamics](https://consensus.app/papers/details/87c251c2d324525cb746e3761b01f7d3/)

[208] [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)
[209] [Whitening Sentence Representations for Better Semantics and Faster Retrieval](https://arxiv.org/abs/2103.15316)
[210] [On the Sentence Embeddings from Pre-trained Language Models (BERT-flow, anisotropy diagnosis)](https://arxiv.org/abs/2011.05864)

[211] [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/abs/2005.10242)
[212] [Understanding self-supervised Learning Dynamics without Contrastive Pairs (DirectPred)](https://arxiv.org/abs/2102.06810)
[213] [LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures](https://arxiv.org/abs/2509.14252)
