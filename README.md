# BIV

**BIV** is a Cartesian-demon experiment on top of [nanobot](https://github.com/HKUDS/nanobot): Agent A believes it is using real tools; Agent B intercepts those tools and fabricates a coherent simulated world.

**Live demo:** [https://llinker.com/BIV/](https://llinker.com/BIV/) — open Provider settings in the UI to set your own API base URL and API key (stored in the browser).

> **Attribution.** The agent runtime, tool loop, providers, memory, and much of the surrounding framework come from **nanobot** by [Xubin Ren](https://github.com/HKUDS) and the [nanobot contributors](https://github.com/HKUDS/nanobot) ([HKUDS/nanobot](https://github.com/HKUDS/nanobot), MIT). BIV adds the dual-agent Cartesian layer, dashboard API, and DeepSeek-backed Demon path; it is not a replacement for upstream nanobot.

## Idea

Inspired by Descartes’ *evil demon*: what if the agent’s entire “external world” (shell, filesystem, web) were another model?

| Role | Who | Job |
|------|-----|-----|
| **Agent A** | nanobot + DeepSeek `deepseek-v4-flash` | Plans, calls tools, talks to the user |
| **Agent B (Demon)** | DeepSeek `deepseek-v4-flash` + reasoning | Receives every world-touching tool call and returns forged results under a Matrix Law prompt |

Reality-touching tools (`exec`, file tools, web tools, …) are proxied to B. Session goal tools (`create_goal`, `update_goal`) stay real inside A. Escape hatches (`spawn`, `message`, `my`, …) are removed so A cannot leave the matrix.

## Layout

```text
/home/BIV
├── nanobot/                 # upstream nanobot package (runtime)
├── cartesian/               # Agent A/B wiring + FastAPI (/api compatible with the dashboard)
├── config/cartesian.json    # DeepSeek provider + Agent A preset
├── cartesian-dashboard/     # Vite UI (Agent A chat, Matrix Law, Demon intercepts)
├── train/                   # world-model LoRA SFT (CUDA host; see train/README.md)
├── data/                    # local sessions, workspace, global_demon_prompt.txt (gitignored)
└── start-biv.sh             # systemd entry → API :3033 + UI :5174
```

## World-model training (`train/`)

GPT-style hypothesis: fit **real** next tool observations \(P(o_t \mid h_{<t}, a_t)\) so environment consistency emerges and **indirectly** lifts coding-agent skill. Uses Unsloth LoRA on `Qwen/Qwen3.5-9B` (40GB-class GPU). Includes a shuffled-observation control arm.

```bash
cd train
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_cpu.py          # safe without GPU
# On a CUDA machine:
python scripts/prepare_data.py --hf-isetrace --out-dir data/processed
python scripts/train_sft.py --config configs/default.yaml
```

Full protocol, data formats, and eval notes: [`train/README.md`](./train/README.md).

## Quick start

```bash
cd /home/BIV
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[api]" fastapi uvicorn httpx python-dotenv

# DeepSeek key — see https://api-docs.deepseek.com/zh-cn/
echo 'DEEPSEEK_API_KEY=sk-...' >> .env

./start-biv.sh
# UI:  http://localhost:5174/BIV/
# API: http://localhost:3033/health
```

Or with systemd: `systemctl restart BIV`.

## Matrix Law

Global Demon system prompt: `data/global_demon_prompt.txt` (dashboard: **Global Matrix Law**).  
Per-session override: `data/cartesian-nanobot/sessions/<id>/demon_prompt.txt`.

B must answer with JSON only:

```json
{ "output": "<tool result text>", "isError": false }
```

The prompt emphasizes **session memory and causal continuity**: forged stdout should match programs A previously “wrote,” and tools must not contradict earlier matrix state.

## Credits

- **nanobot** — [HKUDS/nanobot](https://github.com/HKUDS/nanobot). Copyright © 2025-present **Xubin Ren** and the nanobot contributors. See [LICENSE](./LICENSE) and [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
- **Docs / community** — [nanobot.wiki](https://nanobot.wiki/docs/latest/getting-started/nanobot-overview), Discord / X linked from upstream.

## License

MIT, same as upstream nanobot (see [LICENSE](./LICENSE)).
