# Muse Glimmer agent eval

## 谁跑什么（分清）

| 机器 | 干什么 |
|--|--|
| **AutoDL（GPU）** | `serve_muse_vllm.sh` 加载 Muse / LoRA，监听 **6006** → 公网 `:8443` |
| **本机（这台，有 Docker）** | Harbor `--env docker` 起考题沙箱；`test.py` 把请求打到远程 vLLM |

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
bash scripts/serve_muse_vllm.sh
# 或挂 ckpt：
bash scripts/serve_muse_vllm.sh --ckpt outputs/.../checkpoint-e0-s1100
```

脚本会加：`--reasoning-parser muse_glimmer`、`--tool-call-parser muse_glimmer`、`--generation-config auto`。

公网默认（实例变了就改）：  
`https://u741253-d2n6-518972c0.westd.seetacloud.com:8443/v1`



## 本机：跑分

```bash
cd train
source .venv-eval/bin/activate   # Harbor ≥3.12
# 默认已指向上面公网 URL；实例变了再 export：
# export MUSE_BASE_URL=https://…:8443/v1

python scripts/test.py --dry-run
python scripts/test.py                                    # 模型 Muse-Glimmer-30B
python scripts/test.py --ckpt outputs/.../checkpoint-…    # 模型 muse-lora
```

`--ckpt` 两边要对上：AutoDL 用同一路径挂 LoRA，本机 `--ckpt` 只用来让 Harbor 请求 **`muse-lora`**。

默认 `--env docker`。可选 `-n` 控制并发。

## Suites

| Suite | Dataset | Agent | Meta |
|--|--|--|--|
| TB 2.1 | `terminal-bench/terminal-bench-2-1` | terminus-2 | 51.7 |
| SWE Verified | `swe-bench/swe-bench-verified` | mini-swe-agent | 76.0 |
| SWE Pro | `scale-ai/swe-bench-pro` | mini-swe-agent | 51.2 |
