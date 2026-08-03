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

## Pipeline

### 1) Prepare data (CPU OK)

```bash
# Smoke with checked-in examples
python scripts/smoke_cpu.py
python scripts/prepare_data.py \
  --local data/examples/sample_trajectories.jsonl \
  --out-dir data/processed

# Real run (needs network): ISETrace trajectories
# Correct HF API: config name="trajectories", split="train"
python scripts/prepare_data.py \
  --hf-isetrace \
  --hf-max-rows 10000 \
  --local data/examples/sample_trajectories.jsonl \
  --out-dir data/processed \
  --eval-ratio 0.05
```

`--hf-isetrace` downloads ~5GB once (cached under `~/.cache/huggingface`). Without `--hf-max-rows` it converts all **23,132** trajectories (hundreds of thousands of prefix samples); start with a cap for smoke tests.
Add more Docker/terminal or SWE execution traces as local JSONL with either:

```json
{"turns":[{"tool":"exec","arguments":{"command":"..."},"observation":"...","is_error":false}]}
```

or OpenAI-style `messages` with `tool_calls` / `role=tool`.

### 2) Train (CUDA required)

```bash
python scripts/train_sft.py --config configs/default.yaml
# Control:
python scripts/train_sft.py --config configs/control_shuffled.yaml
```

`train_sft.py` exits immediately if `torch.cuda.is_available()` is false.

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
