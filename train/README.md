# World-model SFT for BIV
#
# Hypothesis (GPT-style):
#   Fit next real tool observation P(o_t | history, tool_call)
#   → improve environment consistency
#   → indirectly improve coding-agent performance.
#
# Inspired by next-token prediction → emergent skills (e.g. translation).
# Fitting tool I/O is the means; agent lift is the measured end.

## Layout

```text
train/
├── README.md
├── requirements.txt          # pin stack for GPU training hosts
├── configs/
│   ├── default.yaml          # real-observation SFT
│   └── control_shuffled.yaml # shuffled-o control arm
├── src/biv_wm/               # data formatting + metrics
├── scripts/
│   ├── prepare_data.py       # CPU OK — build JSONL
│   ├── smoke_cpu.py          # CPU OK — sanity checks
│   ├── train_sft.py          # CUDA required — Unsloth LoRA
│   └── eval_wm.py            # CUDA for generation; metrics dry-run on CPU
├── data/
│   ├── examples/             # tiny checked-in trajectories
│   └── processed/            # generated JSONL (gitignored)
└── outputs/                  # adapters / preds (gitignored)
```

## What we train

| Item | Choice |
|------|--------|
| Base checkpoint | `Qwen/Qwen3.5-9B` (agent-capable Instruct, not Base) |
| Objective | assistant = real tool observation JSON only (response-masked CE) |
| Method | Unsloth LoRA bf16 (~22GB; fits 40GB cards) |
| Not included | Matrix Law persona; full agent policy SFT from scratch |

Control arm: identical setup on **shuffled observations**. If agent metrics rise on control too, the GPT-style causal story fails.

## Setup (GPU training machine)

```bash
cd train
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If Unsloth wheels fail for your CUDA/torch combo, install Unsloth first from
https://unsloth.ai/docs/get-started/install then re-run `pip install -r requirements.txt`.

### Downloads (HuggingFace default)

Configs default to `model.source: huggingface` (`Qwen/Qwen3.5-9B`).

| Asset | Where |
|-------|--------|
| **Qwen/Qwen3.5-9B** | HuggingFace (or set `source: modelscope`) |
| **ISETrace** | ModelScope `LambdaLinker/ISETrace` via `--isetrace` (HF: `--isetrace-source huggingface`) |

```bash
# optional: ModelScope login for private datasets
# modelscope login --token $MODELSCOPE_API_TOKEN
```

Upload existing local data → ModelScope (no re-download):

```bash
export MODELSCOPE_API_TOKEN=ms-xxxxxxxx
# list guessed HF hub snapshots, or point at processed JSONL:
python scripts/upload_isetrace_modelscope.py --list-guesses
python scripts/upload_isetrace_modelscope.py \
  --local-dir data/processed \
  --ms-repo LambdaLinker/ISETrace
```

## Pipeline

### 1) Prepare data (CPU OK)

```bash
# Smoke with checked-in examples
python scripts/smoke_cpu.py
python scripts/prepare_data.py \
  --local data/examples/sample_trajectories.jsonl \
  --out-dir data/processed

# Real run (needs network): ISETrace from ModelScope (default)
python scripts/prepare_data.py \
  --isetrace \
  --out-dir data/processed \
  --eval-ratio 0.05
```

`--isetrace` pulls ~5GB once (ModelScope cache). Without `--isetrace-max-rows` it processes all
**23,132** trajectories but **does not** load them all into RAM at once. Start with
`--isetrace-max-rows 2000` for a smoke subset. Legacy alias: `--hf-isetrace`.
Add more Docker/terminal or SWE execution traces as local JSONL with either:

```json
{"turns":[{"tool":"exec","arguments":{"command":"..."},"observation":"...","is_error":false}]}
```

or OpenAI-style `messages` with `tool_calls` / `role=tool`.

### 2) Train (CUDA required)

```bash
# Full-scale (slow)
python scripts/train_sft.py --config configs/default.yaml
python scripts/train_sft.py --config configs/control_shuffled.yaml

# Fast pilot (keep seq=8192; sample ~half + packing + 1 epoch) — hypothesis screen
python scripts/train_sft.py --config configs/pilot.yaml
python scripts/train_sft.py --config configs/pilot_shuffled.yaml

# Resume from latest checkpoint under output_dir (ds_cache still hits):
python scripts/train_sft.py --config configs/pilot.yaml --resume
```

Pilot writes adapters to `outputs/wm_sft_pilot/` but **dataset caches are shared** under
`outputs/ds_cache/` (content-addressed; not tied to `output_dir`). If a full-corpus
ready cache exists (e.g. legacy `outputs/wm_sft/ds_cache` with the same 8192 seq),
pilot will **subset only** (no seq truncate) instead of re-tokenizing. `packing`
only applies when building from text; pretokenized reuse ignores packing.

`train_sft.py` exits immediately if `torch.cuda.is_available()` is false.

### 2c) Upload train/eval to Bailian via API (no OSS)

Uses DashScope Files API (`purpose=fine-tune`). Fine-tune file size limit **300MB**
→ script shards JSONL to ≤190MB. Train + eval both uploaded; `file_id`s written to
`data/export_bailian/bailian_file_ids.json`.

```bash
pip install requests python-dotenv pyyaml tqdm
# train/.env → DASHSCOPE_API_KEY=sk-...
python scripts/upload_bailian.py
python scripts/upload_bailian.py --reuse-shards
python scripts/upload_bailian.py --upload-only   # reuse export_bailian/*.jsonl
# PrivateLink (run on ECS in the endpoint VPC; default ep-* host is HTTP-only):
python scripts/upload_bailian.py --reuse-shards \
  --base-url http://ep-xxxx.dashscope.cn-beijing.privatelink.aliyuncs.com
# or .env DASHSCOPE_BASE_URL=... / DASHSCOPE_FILES_URL=.../api/v1/files
```

Create fine-tune later with:
`training_datasets` / `validation_datasets` → `data_source_type: file_id`
([文档](https://help.aliyun.com/zh/model-studio/create-fine-tuning-job-api)).
Console「数据管理」单次上传约 10 个文件；大量 `file_id` 请用 **API 创建调优任务**
（`training_datasets` 数组可列多项），勿指望面板勾选上百个分片。

官方上传：https://help.aliyun.com/zh/model-studio/upload-file-api

费用粗估（本地 `outputs/ds_cache/*/train_ready` Token × epoch × 单价）：

```bash
python scripts/cost.py
python scripts/cost.py --epochs 1 --max-length 8192 --price-per-k 0.02
# 若开启百炼「混合训练」data_augmentation（额外预置语料也计费）:
python scripts/cost.py --mix-ratios 0.1,0.05,0.15
```

`qwen3.5-9b` 列表价约 ¥0.02/千Token（以官方计费页为准）。混合训练说明见 `scripts/cost.py` 文档字符串。

### 2c-legacy) Optional: upload via Aliyun OSS

See `scripts/upload.py` + `configs/oss.yaml` (Ulanqab internal by default).

### 2b) Qwen3.5 fast kernels (Scheme B: fla + causal-conv1d)

Unsloth may print *Falling back to torch implementation* until these are installed.
`requirements.txt` intentionally does **not** `pip`-pin `causal-conv1d`: with
`torch 2.13+cu130` there is often no wheel, and source builds fail if system
`nvcc` (e.g. 12.8) ≠ `torch.version.cuda` (13.0).

```bash
source .venv/bin/activate
pip install -U ninja packaging "flash-linear-attention[cuda]"
pip install -U "nvidia-cuda-nvcc"   # do NOT use nvidia-cuda-nvcc-cu13 (deprecated stub)

# Prefer pip CUDA-13 nvcc over system /usr/local/cuda-12.8
export CUDA_HOME=$(python - <<'PY'
from importlib.util import find_spec
from pathlib import Path
spec = find_spec("nvidia.cuda_nvcc")
root = Path(spec.submodule_search_locations[0])
# some wheels nest bin/ under root or root/bin
for cand in (root, root / "bin", *root.glob("**/nvcc")):
    p = cand if cand.name != "nvcc" else cand.parent.parent
    if (Path(p) / "bin" / "nvcc").exists():
        print(p)
        break
else:
    raise SystemExit(f"nvcc not found under {root}")
PY
)
export PATH="$CUDA_HOME/bin:$PATH"
which nvcc && nvcc -V   # must be 13.x for torch *+cu130

CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install -U "causal-conv1d>=1.4.0" --no-build-isolation
python -c "import fla, causal_conv1d; print('fast path ok')"
```

Restart training afterward (`--resume` if a checkpoint exists).
Do not set `FLA_CONV_BACKEND=triton` if causal-conv1d imported successfully
(default CUDA conv backend is what Scheme B wants).

### 3) World-model eval

```bash
python scripts/eval_wm.py --config configs/default.yaml \
  --adapter outputs/wm_sft/lora_adapter
```

Metrics: normalized exact match, token F1, `isError` accuracy.

### 4) Agent transfer (decision protocol)

Keep the **same agent scaffold** and compare:

1. Vanilla `Qwen/Qwen3.5-9B`
2. + real-observation LoRA
3. + shuffled-observation LoRA (control)

Suggested suites: Terminal-Bench / Harbor, SWE-bench-style tasks.  
Claim holds only if (2) beats (1) **and** (3) does not explain the gain.

Optional inference use of the world model (DyMo/SVS-style): sample candidate tool calls, predict observations, pick the coherent one — separate from training.

## Hardware notes

| Setting | ~VRAM |
|---------|-------|
| Qwen3.5-9B bf16 LoRA, seq 8k, bs=1 | ~22–28GB |
| Same with 4-bit QLoRA | lower; enable `load_in_4bit: true` |

40GB is enough for the default bf16 config with `gradient_accumulation_steps: 8`.

## Non-goals on this repo host

The BIV application server is **not** a training box. Do not run `train_sft.py` here.
Use it only for editing, `prepare_data.py`, and `smoke_cpu.py`.
