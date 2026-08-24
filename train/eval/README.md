# Muse Glimmer agent eval

## 谁跑什么（分清）

| 机器 | 干什么 |
|--|--|
| **AutoDL（GPU）** | `serve_muse_vllm.sh` 加载 Muse + **默认最新 LoRA ckpt**，监听 **6006** → 公网 `:8443` |
| **本机（这台，有 Docker）** | Harbor `--env docker` 起考题沙箱；`test.py` 只选远程 model id（**无本地 ckpt**） |

不需要 E2B。AutoDL 套不了 Docker 也不影响——沙箱在本机。

```
本机 Harbor (Docker 沙箱)  ──HTTP──►  AutoDL vLLM (:6006 → 公网 :8443)
```

## AutoDL：起推理（无 Docker）

在已有 **`.venv-muse`** 上覆盖装 Muse day-0 vLLM（`--no-deps`，不重下 torch）：

```bash
cd ~/autodl-tmp/BIV/train
source .venv-muse/bin/activate
bash scripts/install_muse_vllm.sh   # 只需一次

# 默认：自动选 train out_dir 下最新完整 LoRA ckpt
MAX_LENGTH=65536 CHOICE=1 bash scripts/serve_muse_vllm.sh
# 或显式路径 / 只跑 base：
# bash scripts/serve_muse_vllm.sh --ckpt outputs/.../checkpoint-e0-s2150
# bash scripts/serve_muse_vllm.sh --base
```

脚本会加：`--reasoning-parser muse_glimmer`、`--tool-call-parser muse_glimmer`、`--generation-config auto`；挂 LoRA 时再加 `--limit-mm-per-prompt '{"image":0,"video":0}'`。  
元数据写到 `outputs/.muse_vllm_serve.json`；banner 会打印建议的 `MUSE_EVAL_ARM` / `MUSE_EVAL_STEP`（本机 TB 对齐用）。

公网默认（实例变了就改）：  
`https://u741253-ltqs-498e6237.westd.seetacloud.com:8443/v1`

## 本机：跑分

```bash
cd train
source .venv-eval/bin/activate   # Harbor ≥3.12
# 默认已指向上面公网 URL；实例变了再 export：
# export MUSE_BASE_URL=https://…:8443/v1

# 可选：与 AutoDL serve 的 ckpt 步数对齐（看 serve banner）
# export MUSE_EVAL_ARM=checkpoint-e0-s2150
# export MUSE_EVAL_STEP=2150

python scripts/test.py --dry-run
python scripts/test.py              # 请求 muse-lora（AutoDL 应已挂最新 LoRA）
python scripts/test.py --base       # 请求 Muse-Glimmer-30B（AutoDL 需 --base）
```

默认 `--env docker`。可选 `-n` 控制并发。

**Terminus 停止条件（默认已改）：**

| 项 | 默认 | 说明 |
|--|--|--|
| `--max-turns` | **300** | Terminus LLM 回合上限（`--ak max_turns`）。**题目 `task.toml` 不写这个**；Harbor 原生默认 ~1e6。常见取值：Harbor 文档示例 100；AgentCompass TB2.1 默认 300。`--max-turns 0` = 不限制。 |
| `--agent-timeout-multiplier` | **100** | 题集里的 `agent.timeout_sec`（多数 900s）×100，避免慢远端推理先撞墙钟。 |

题目只给定 **墙钟** `timeout_sec`（89 题里约一半是 900s，其余 600～12000 不等），**没有**给定 max_turns。

**`qemu-alpine-ssh` + 本机 tmux：** 不是「Harbor 套在宿主机 tmux 里」导致的。该题镜像是 Debian bullseye，自带 **tmux 3.1c**；Terminus 用 `tmux new-session -e`（需要 ≥3.2）注入 `--ae` 的 `OPENAI_*`，于是 agent 还没上场就 `Failed to start tmux session`。`test.py` / `run_harbor.py` 会把 `eval/harbor_runtime` 加进 Harbor 的 `PYTHONPATH`，改成在 pane 里 `export` 而不是 `-e`。resume 已失败的 trial 需要 `-f RuntimeError`。

## 实时看 agent 轨迹

Harbor 会把完整轨迹写到 job 目录（边跑边更新）：

```text
outputs/agent_eval/<stamp>_<arm>/<arm>_<suite>/<task>__*/agent/
  trajectory.json   # ATIF：analysis/plan/commands + observation
  terminus_2.pane   # 终端面板快照
  recording.cast    # asciinema
```

当前正在跑的 job，另开一个终端：

```bash
cd train && source .venv-eval/bin/activate
python -m eval.follow_traj outputs/agent_eval/20260814T170504Z_checkpoint-e0-s1100/checkpoint-e0-s1100_terminal_bench_2_1
# 或直接跟某个 trial：
# python -m eval.follow_traj …/write-compressor__kKRDyks/agent/trajectory.json
```

下次跑分时加 `--follow-traj` 会在同一终端流式打印新 step；`--debug` 打开 Harbor debug；`--raw-traj` 把原始 LLM 回复写进 trajectory。

## Suites

| Suite | Dataset | Agent | Meta |
|--|--|--|--|
| TB 2.1 | `terminal-bench/terminal-bench-2-1` | terminus-2 | 51.7 |
| SWE Verified | `swe-bench/swe-bench-verified` | mini-swe-agent | 76.0 |
| SWE Pro | `scale-ai/swe-bench-pro` | mini-swe-agent | 51.2 |

## 检查点 / resume

Harbor 每个 suite 的 job 目录里有 `config.json` + 已完成的 `trial_result.json`。中断后**不要**再开一轮默认 `test.py`（会新建时间戳目录），用：

```bash
# 续跑整个 stamp 下所有未完成 suite（默认写入 65536 上下文，避免 LiteLLM 1e6 fallback）
python scripts/test.py --resume outputs/agent_eval/20260814T170504Z_checkpoint-e0-s2150 --max-model-len 65536

# 只续某一个 suite job
python scripts/test.py --resume outputs/agent_eval/.../checkpoint-e0-s2150_terminal_bench_2_1

# stamp 根下只续指定 suite
python scripts/test.py --resume outputs/agent_eval/<stamp>_… --suite terminal_bench_2_1
```

底层是 `harbor job resume -p <job_dir>`：已完成 trial 保留；Harbor 默认会清掉 `CancelledError` 的 trial 再重跑它们。自定义：`--filter-error-type SomeError`（可重复）。

训练 LoRA 检查点仍在 AutoDL `checkpoint-e*-s*`，与这里无关。

## TensorBoard

默认开着（`--no-tensorboard` 关）。与训练共用 `LOGGING_DIR`（默认 `/root/tf-logs`）：

```bash
export LOGGING_DIR=/root/tf-logs
export MUSE_EVAL_ARM=checkpoint-e0-s2150
export MUSE_EVAL_STEP=2150
python scripts/test.py
# → /root/tf-logs/{n}_eval_agent_…_s2150/
```

**实时进度怎么写：** Harbor 边跑边落盘 `**/trial_result.json`；`test.py` 开后台线程约每 15s `glob` 这些文件，算出当前 pass% / `n_trials`，有变化就 `SummaryWriter.flush()` 到 TB。不是 Harbor 推事件，是我们轮询磁盘。

| 标签 | 何时写 | x 轴 |
|--|--|--|
| `eval_agent_live/<suite>/score_percent` | **跑分过程中**（新 trial 落盘后） | 已完成 `n_trials` |
| `eval_agent/<suite>/score_percent` 等 | 每个 suite **结束时** + session 收尾 | 训练 ckpt step（跨 arm 对比） |

自定义根目录：`--log-dir /path/to/tf-logs`。
