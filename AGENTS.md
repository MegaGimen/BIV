This file provides guidance to AI coding agents working with this repository.

## Project Overview

This repository is **BIV** (Brain In a Vat / 缸中之脑): a Cartesian-demon experiment **on top of** [nanobot](https://github.com/HKUDS/nanobot).

- **Runtime product:** Agent A believes it uses real tools; Agent B (Demon) intercepts world-touching tools and returns a coherent simulated world (see root `README.md`, `cartesian/`).
- **Research / training track (`train/`):** GPT-style hypothesis — fit **real** next tool observations so environment consistency emerges and **indirectly** improves coding-agent skill. Runtime nanobot code paths are largely upstream; BIV adds the Cartesian layer + world-model SFT scaffold.

Upstream nanobot remains a lightweight Python agent framework (channels → bus → agent loop → LLM → tools → memory) with a React/TypeScript WebUI. Prefer changing `cartesian/` and `train/` for BIV-specific behavior; touch `nanobot/` only when necessary for forks or bugs.

## BIV Runtime (Cartesian layer)

| Role | Location | Job |
|------|----------|-----|
| Agent A | nanobot loop + configured provider | Plans, tool calls, user chat |
| Agent B (Demon) | `cartesian/demon.py` | Fabricates tool results under Matrix Law |
| Proxies | `cartesian/tool_proxies.py` | Route `exec` / FS / web tools to B; keep `create_goal` / `update_goal` real; drop escape tools |

- Dashboard + API: `cartesian-dashboard/`, `cartesian/server.py`
- Matrix Law prompt: `data/global_demon_prompt.txt` (runtime only; **not** used as SFT supervision in `train/`)
- Live demo notes: root `README.md`

## World-model training (`train/`) — hypothesis

**Analogy (GPT):** next-token prediction on text → emergent skills (e.g. translation).  
**Here:** next-observation prediction on real tool I/O → hoped-for lift in coding-agent benchmarks.

\[
P(o_t \mid h_{<t}, a_t)
\]

| Item | Choice |
|------|--------|
| Checkpoint | `Qwen/Qwen3.5-9B` (agent-capable Instruct; **not** Base) |
| Update | Unsloth LoRA; loss **only** on assistant / observation tokens (`labels=-100` elsewhere) |
| Labels | Real sandbox tool outputs (from execution-grounded trajectories) |
| Task system prompt | Short WM role in `train/src/biv_wm/formatting.py` (`DEFAULT_WM_SYSTEM`) — **not** Matrix Law |
| Not in SFT | Matrix Law / `data/global_demon_prompt.txt`; full agent policy SFT from scratch |
| Control | shuffled-observation twin configs — identical setup on **shuffled** \(o_t\) |

**Claim only if:** world-model metrics improve **and** same-scaffold agent coding metrics improve vs base **and** the shuffled control does **not** explain the gain.

Detailed runbook: [`train/README.md`](./train/README.md).

### Pilot vs full-scale (prefer pilot to screen the hypothesis)

Single-GPU full corpus @ 8k × 2 epochs can be **months** of wall-clock. To **falsify/screen** the causal story cheaply, use pilot configs **before** burning a full run:

| | Full (`configs/default.yaml`) | Pilot (`configs/pilot.yaml`) |
|--|--|--|
| `max_seq_length` | 8192 | **8192 (do not shorten)** |
| Data | full ready set | **deterministic ~half sample** (`max_train_samples: 320000`, eval 16000) |
| Epochs | 2 | **1** |
| Packing | optional | `true` when building from text |
| Output adapters | `outputs/wm_sft/` | `outputs/wm_sft_pilot/` |
| Control twin | `control_shuffled.yaml` | `pilot_shuffled.yaml` |

**Why keep 8k and sample rows instead of truncating sequences:** left-truncating 8k→4k often chops the **end** of ChatML (assistant observation / late history), which hurts exactly \(P(o\mid h,a)\). Prefer **fewer full-length rows**.

**Subset seeding:** `shuffle(seed).select(range(N))` with `train.seed` (default **42**) for train; eval uses **seed+1**. Same config + same JSONL ⇒ reproducible subset. Seed also flows to `SFTConfig` and LoRA `random_state`.

**Pilot validates** the real-vs-shuffled causal structure at reduced cost. It does **not** by itself claim full-corpus / 2-epoch 9B transfer — confirm with full config if pilot succeeds.

### Dataset cache (shared; reuse across `output_dir`)

- Shared root: `data.ds_cache_dir` → `outputs/ds_cache/` (content-addressed; **not** tied to `train.output_dir`).
- Also reads legacy `outputs/wm_sft/ds_cache` etc.
- Same source files + same seq: pilot **subsets** a full ready cache (no re-tokenize / re-mask when possible).
- Format runs **before** subset so HF map caches from full format can hit.
- `packing` applies only when tokenizing from text; pretokenized cache reuse ignores packing.
- Resume: `python scripts/train_sft.py --config … --resume` (or `--resume-from`).

### Training commands (GPU host)

```bash
cd train
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
# Optional Qwen3.5 fast kernels (match nvcc to torch.version.cuda; see train/README.md §2b)

# Data: streams one trajectory at a time (avoid full-corpus to_list OOM)
# ISETrace is HF-hosted (~5GB raw); optional: export HF_ENDPOINT=https://hf-mirror.com
python scripts/prepare_data.py --hf-isetrace --out-dir data/processed --eval-ratio 0.05

huggingface-cli login   # or: hf auth login

# Preferred first: hypothesis screen
python scripts/train_sft.py --config configs/pilot.yaml
python scripts/train_sft.py --config configs/pilot_shuffled.yaml

# Full-scale (slow)
python scripts/train_sft.py --config configs/default.yaml
python scripts/train_sft.py --config configs/control_shuffled.yaml

# World-model held-out metrics
python scripts/eval_wm.py --config configs/pilot.yaml
```

- Do **not** run `train_sft.py` on CPU-only app servers; `prepare_data.py` / `smoke_cpu.py` are OK without GPU.
- Disk: `~/.cache/huggingface/datasets` (tokenize/map Arrow) + `outputs/ds_cache/` can be large; clear HF caches if root fills. Keep `hub/` model weights if possible.
- VRAM: L20-class (~46GB) can use larger micro-batch (e.g. 8×accum 2, eff≈16). Keep **effective batch** stable if comparing runs; raising only micro-batch with fixed eff batch barely changes optimization.
- Checkpoints: every `save_steps` (default 35) + forced save/eval each epoch; `save_total_limit: 3`.

### Data

Primary corpus: **[ISETrace](https://huggingface.co/datasets/valiere/ISETrace)** — multi-turn OS-agent trajectories with **real** tool execution (paper: [ISE / arXiv:2606.11520](https://arxiv.org/abs/2606.11520), code: [Valiere01/ISE-Trace](https://github.com/Valiere01/ISE-Trace)). Domains include `code-runtime`, `file-io`, `system-infrastructure`, etc. (Linux `exec`, write/run Python, plus multimedia/web — not pure SWE-bench).

`prepare_data.py` extracts tool turns → causal prefixes → chat JSONL (`data/processed/`). That JSONL is what `train_sft.py` consumes.

### Evaluation protocol (hypothesis)

1. **World-model:** held-out observation CE / token-F1 / `isError` accuracy (`scripts/eval_wm.py`).
2. **Agent transfer:** same scaffold, compare base `Qwen3.5-9B` vs real-I/O LoRA vs shuffled LoRA on Terminal-Bench / Harbor, SWE-style suites, etc.
3. **Causal check:** if shuffled rises as much as real I/O, reject the GPT-style story.
4. **Order:** run pilot real vs pilot shuffled first; only then invest in full-scale if the gap supports the story.

### Related world-model literature (reference — not fully reimplemented)

Current `train/` is a **minimal** next-observation LoRA SFT (plus shuffled control), inspired by these lines of work. Full CPT→RL stacks, joint \(L_{FC}+L_{SP}\), or test-time SVS/planning are **future** options:

| Work | Links | Idea to steal later |
|------|-------|---------------------|
| Qwen-AgentWorld | [GitHub](https://github.com/QwenLM/Qwen-AgentWorld), [paper HTML](https://arxiv.org/html/2606.24597) | Native LWM: CPT→SFT→RL; next-state prediction; AgentWorldBench consistency dimensions; LWM warmup transfers to agent benches |
| DyMo + SVS | [arXiv:2506.02918](https://arxiv.org/abs/2506.02918), [OpenReview](https://openreview.net/forum?id=DALpFQM3rE) | Joint tool-call + next-state loss; sample → predict state → proceed without oracle env |
| RAP | [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.507/), [arXiv:2305.14992](https://arxiv.org/abs/2305.14992) | Same LM as agent + world model; MCTS-style planning over predicted states |
| Word2World | [GitHub](https://github.com/X1AOX1A/Word2World) | Text WM fidelity vs agent utility; WM2Real rollout consistency |
| WorldCoder | [GitHub](https://github.com/haotang1995/WorldCoder), [arXiv:2402.12275](https://arxiv.org/abs/2402.12275) | Explicit code as world-model / transition law |
| TerminalTraj | [arXiv:2602.01244](https://arxiv.org/abs/2602.01244) | Docker-grounded terminal trajectories at scale |
| V-JEPA (consistency analogy) | [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) | Spatial/temporal consistency objectives (map to cross-tool / multi-step env consistency in text) |
| Survey / index | [awesome-world-model-evolution](https://github.com/OpenRaiser/awesome-world-model-evolution) | Broader WM taxonomy |

**Design stance for this repo:** prefer **environment consistency** (causal continuity across tool turns) as the first-class training/eval object; agent coding uplift is the **transfer** metric. Do not confuse runtime Matrix Law (BIV product) with world-model SFT labels (must stay real I/O). Screen with **pilot (sample rows, keep 8k)**; confirm with full-scale only if justified.
## Development Commands (upstream nanobot)

```bash
# Python: run single test / lint
pytest tests/test_openai_api.py::test_function -v
ruff check nanobot/

# Strict type checking (matches CI)
uv sync --all-extras --dev
uv run --no-sync python -m scripts.install_channel_dependencies --all-channels
uv run --no-sync basedpyright

# WebUI: dev server (proxies API/WS to gateway :8765), build, test
cd webui && bun run dev
cd webui && bun run build
cd webui && bun run test

# Gateway / BIV
nanobot gateway
./start-biv.sh
```

## High-Level Architecture (nanobot runtime)

### Core Data Flow

Messages flow through an async `MessageBus` (`nanobot/bus/queue.py`) that decouples chat channels from the agent core:

1. **Channels** (`nanobot/channels/`) publish `InboundMessage` events to the bus.
2. **`AgentLoop`** (`nanobot/agent/loop.py`) consumes inbound messages and coordinates the turn.
3. **`AgentRunner`** (`nanobot/agent/runner.py`) runs the LLM ↔ tool loop and streams responses.
4. Responses are published as `OutboundMessage` events back to the channel.

In BIV, tool execution for reality-touching tools is replaced by Demon proxies before results return to A.

### Key Subsystems

- **Agent Loop** (`nanobot/agent/loop.py`, `runner.py`)
- **LLM Providers** (`nanobot/providers/`)
- **Channels** (`nanobot/channels/`)
- **Tools** (`nanobot/agent/tools/`) — FS, shell/sandbox, web, MCP, cron, subagents, long tasks, etc.
- **Memory / sessions** (`nanobot/agent/memory.py`, `nanobot/session/`)
- **Config** (`nanobot/config/schema.py`, `loader.py`) — typically `~/.nanobot/config.json`; BIV also uses `config/cartesian.json`
- **WebUI** (`webui/`), **Cartesian dashboard** (`cartesian-dashboard/`)
- **API** (`nanobot/api/server.py`, `cartesian/server.py`)
- **Cartesian / Demon** (`cartesian/`)
- **World-model SFT** (`train/`)

### Entry Points

- **CLI**: `nanobot/cli/commands.py`
- **Python SDK**: `nanobot/nanobot.py`
- **BIV start**: `./start-biv.sh`
- **WM prepare / train**: `train/scripts/prepare_data.py`, `train/scripts/train_sft.py`

## Project-Specific Notes

- Architecture constraints: [`.agent/design.md`](.agent/design.md)
- Security boundaries: [`.agent/security.md`](.agent/security.md)
- Common gotchas: [`.agent/gotchas.md`](.agent/gotchas.md)
- Training runbook: [`train/README.md`](./train/README.md)

## Contribution Flow

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for contribution flow and PR guidelines.

## Code Style

- Python 3.11+, asyncio throughout (runtime).
- Line length: 100.
- Linting: `ruff` with rules E, F, I, N, W (E501 ignored).
- pytest with `asyncio_mode = "auto"`.
- `train/` uses its own venv + `requirements.txt` (Unsloth/TRL); do not assume the app `.venv` has training deps.

## Common File Locations

- Config schema: `nanobot/config/schema.py`
- Provider base: `nanobot/providers/base.py`
- Channel base: `nanobot/channels/base.py`
- Tool registry: `nanobot/agent/tools/registry.py`
- Demon / proxies: `cartesian/demon.py`, `cartesian/tool_proxies.py`
- WM data + metrics: `train/src/biv_wm/`
- WM configs: `train/configs/default.yaml`, `train/configs/control_shuffled.yaml`
- WebUI proxy: `webui/vite.config.ts`
- Tests mirror the `nanobot/` package structure.
