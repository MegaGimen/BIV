This file provides guidance to AI coding agents working with this repository.

## Project Overview

This repository is **BIV** (Brain In a Vat / 缸中之脑): a Cartesian-demon experiment **on top of** [nanobot](https://github.com/HKUDS/nanobot).

- **Runtime product:** Agent A believes it uses real tools; Agent B (Demon) intercepts world-touching tools and returns a coherent simulated world (see root `README.md`, `cartesian/`).
- **Research / training track (`train/`):** raise **general world understanding** (OS + code environments) via real next-observation SFT, hoping that transfers **indirectly** to agent console / coding tool-use — while guarding against catastrophic forgetting into an observation-only model. Runtime nanobot is largely upstream; BIV adds the Cartesian layer + world-model SFT scaffold.

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

## World-model training (`train/`) — goals & hypothesis

### Research goals (optimize methods around these)

1. **Primary — world understanding → indirect agent gain:** improve the model’s **general understanding of the world**, including **OS** and **code/repo environments**, by fitting real environment transitions; test whether that capacity **transfers** to better console / coding **tool-use agent** performance (same scaffold vs base).
2. **Constraint — anti-forgetting:** avoid catastrophic forgetting into a model that **only** emits / completes tool **observations** (env-simulator shell) and loses agentic coding / tool-*selection* skill. Anti-forgetting is a **regularizer**, not a second equal training objective or a substitute claim channel.

**Analogy (GPT):** next-token prediction on text → emergent skills.  
**Here:** next-observation prediction on real multi-domain tool I/O → hoped-for **transfer** to agent benchmarks (not “train the policy hard and call it world modeling”).

\[
P(o_t \mid h_{<t}, a_t)
\]

| Item | Choice |
|------|--------|
| Checkpoint (current) | `Qwen/Qwen3.5-9B` Instruct (**not** Base). Stronger coding-agent Instructs (e.g. Qwen3-Coder-*) are optional later bases; changing base invalidates prior controls unless re-run. |
| Primary update | Unsloth LoRA on **observation** tokens (`labels=-100` elsewhere): learn env dynamics, not Demon / Matrix Law |
| Labels | **Real** sandbox / execution-grounded tool outputs only |
| Task system prompt | Short WM role in `train/src/biv_wm/formatting.py` (`DEFAULT_WM_SYSTEM`) — **not** Matrix Law |
| Not the claim | Matrix Law / `data/global_demon_prompt.txt`; heavy policy SFT that could alone explain agent uplift |
| Control | shuffled-observation twin — identical setup on **shuffled** \(o_t\) |

**Claim only if all hold:** (a) world-model / env-consistency metrics improve; (b) same-scaffold agent metrics (console + coding tools) improve vs base; (c) shuffled control does **not** explain the agent gain; (d) agent capability does **not** collapse vs base (anti-forgetting check).

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

# Data: streams one SWE-Hero trajectory at a time (avoid full-corpus to_list OOM)
# Default: ModelScope `nv-community/SWE-Hero-openhands-trajectories`
python scripts/prepare_data.py --swe-hero --out-dir data/processed --eval-ratio 0.05
# HF fallback: --swe-hero-source huggingface  (optional: export HF_ENDPOINT=https://hf-mirror.com)

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

Two **roles**, one shared LoRA (when mix is enabled). Do **not** confuse them.

#### A. Primary — multi-domain environment understanding (WM)

Learn \(P(o\mid h,a)\) from **real** \((a,o)\) in OS + code worlds:

| Domain | Corpus (intent) | Status |
|--------|-----------------|--------|
| **Code / SWE tool I/O** | **SWE-Hero** OpenHands trajectories — execution-grounded | **Wired now** ([HF](https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories), [paper](https://arxiv.org/abs/2604.01496); ModelScope [`nv-community/SWE-Hero-openhands-trajectories`](https://www.modelscope.cn/datasets/nv-community/SWE-Hero-openhands-trajectories)) |
| **OS / desktop agent** | **ISETrace** (and similar real OS interaction logs) | **Planned** — primary OS-world signal once adapter exists (not “bad coding-agent mix”) |
| **Terminal** (optional) | Shell / Terminal-domain env trajectories (AgentWorld-style or self-collected) | Optional breadth for console physics |

`prepare_data.py` (SWE-Hero): keeps OpenHands env tools (`execute_bash`, `str_replace_editor`, …), drops non-env tools (`think`, `finish`), emits **one JSONL row per trajectory** (response-only loss on observations once; no causal-prefix expansion by default). Optional legacy: `--expand-prefixes`. That JSONL is what `train_sft.py` consumes today.

#### B. Auxiliary — anti-forgetting (regularizer only)

Small mix of **native agentic coding / multi-tool agent trajectories** so updates do not wipe tool-*selection* priors. This path is **not** the hypothesis channel; keep **token share modest** and prove agent gains are not “just more policy SFT.”

| Prefer | Why |
|--------|-----|
| [SWE-Zero OpenHands trajectories](https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories) | Same SWE/OpenHands **semantics**, different corpus from Hero (avoid replaying WM rows as policy) |
| [Nebius SWE-rebench OpenHands trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | Extra OpenHands agent rollouts; **dedupe `instance_id` vs Hero** |
| Light Instruct / chat replay or KL to base | Classic capability anchor if policy mix is costly |

**Do not** reuse the **same SWE-Hero rows** as both WM and anti-forget policy mix (duplicate semantics + muddy controls).  
**Do not** treat ISETrace as the anti-forget “coding agent” mix — it belongs in **primary OS-world** understanding once converted.

Suggested starting mix (when implemented): WM-dominant (e.g. ~70–90% env-obs tokens) vs small policy replay (~10–30%); tune by agent retention, not by maximizing agent SFT score.

### Future TODO (training data)

- [ ] **ISETrace → WM primary (OS):** conversion layer mapping ISE tools (`exec` / `read` / `write` / …) into shared canonical / OpenHands-native WM turns. Skeleton: `train/src/biv_wm/adapters/isetrace.py` + prepare flag. No raw ISE in prepare without adapter.
- [ ] **Anti-forget mix:** prepare + interleave SWE-Zero (and optional Nebius) full agent trajectories at low ratio; `instance_id` decontam vs Hero; optional KL/Instruct replay.
- [ ] Optional Terminal-domain env trajectories for broader console world modeling.

### Evaluation protocol (hypothesis)

1. **World understanding:** held-out next-obs metrics on **code-env** (and later **OS**) — CE / token-F1 / `isError` (`scripts/eval_wm.py`); prefer cross-domain held-out when OS corpus lands.
2. **Agent transfer (goal 1):** same scaffold — Terminal-Bench / Harbor / SWE-style — base vs real-I/O LoRA vs shuffled LoRA (console + coding tools).
3. **Causal check:** if shuffled rises as much as real I/O on agent metrics, reject the “world understanding → transfer” story.
4. **Anti-forgetting (goal 2):** agent metrics must not collapse vs base; optional general-capability spot-checks if useful.
5. **Order:** pilot real vs pilot shuffled first; full-scale only if the gap supports the story. Report **both** WM and agent metrics — do not pick only one train path at eval time.

### Related literature (reference — not fully reimplemented)

Current `train/` is a **minimal** next-observation LoRA SFT (plus shuffled control). Multi-domain CPT→RL, anti-forget mixes, or joint loss designs are **future**:

| Work | Links | Relevance to our goals |
|------|-------|------------------------|
| Qwen-AgentWorld | [GitHub](https://github.com/QwenLM/Qwen-AgentWorld), [arXiv:2606.24597](https://arxiv.org/abs/2606.24597) | Native multi-domain LWM (Terminal/SWE/OS/…); CPT→SFT next-state→RL; LWM warm-up transfers to agent benches — closest framing to goal 1 |
| PaW | [arXiv:2606.02388](https://arxiv.org/abs/2606.02388) | Policy + world-model co-training / loss balancing — borrow for goal 2 (we keep WM primary, policy auxiliary) |
| RWML | [arXiv:2602.05842](https://arxiv.org/abs/2602.05842) | Warns pure WM-token SFT can hurt retention; motivates anti-forget checks |
| DyMo + SVS | [arXiv:2506.02918](https://arxiv.org/abs/2506.02918) | Joint tool-call + next-state; optional later |
| RAP | [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.507/), [arXiv:2305.14992](https://arxiv.org/abs/2305.14992) | Same LM as agent + WM; planning over predicted states |
| Word2World | [GitHub](https://github.com/X1AOX1A/Word2World) | WM fidelity vs agent utility |
| WorldCoder | [GitHub](https://github.com/haotang1995/WorldCoder), [arXiv:2402.12275](https://arxiv.org/abs/2402.12275) | Code as explicit transition law |
| TerminalTraj | [arXiv:2602.01244](https://arxiv.org/abs/2602.01244) | Docker-grounded terminal trajectories |
| SWE-Zero → SWE-Hero | [arXiv:2604.01496](https://arxiv.org/abs/2604.01496) | Hero = current code-env WM corpus; Zero = candidate anti-forget OpenHands mix (not Hero replay) |
| Survey / index | [awesome-world-model-evolution](https://github.com/OpenRaiser/awesome-world-model-evolution) | Broader WM taxonomy |

**Design stance:** first-class object = **environment / world consistency** across OS + code tool turns; agent console/coding uplift = **transfer** metric; anti-forget mix = **capability anchor**. Runtime Matrix Law ≠ SFT labels (labels stay real I/O). Screen with **pilot (sample rows, keep 8k)**; full-scale only if justified.

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
