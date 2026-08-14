# Muse Glimmer external agent eval (Meta three-way alignment)

**Runs on the GPU train server.** `test.py` loads Muse weights itself:

| Flag | What loads |
|--|--|
| (none) | base `Muse-Glimmer-30B` |
| `--ckpt PATH` / `--ckpt auto` | base + that PEFT LoRA checkpoint |

Then it starts a **local** OpenAI shim on `127.0.0.1:8000`, and Harbor (Terminus-2 / mini-swe-agent) calls that. No remote `--base-url` needed.

| Suite | Harbor dataset | Agent | Meta score |
|--|--|--|--|
| Terminal-Bench 2.1 | `terminal-bench/terminal-bench-2-1` | `terminus-2` | 51.7 |
| SWE-Bench Verified | `swe-bench/swe-bench-verified` | `mini-swe-agent` | 76.0 |
| SWE-Bench Pro | `scale-ai/swe-bench-pro` | `mini-swe-agent` | 51.2 |

## Setup (GPU train host)

```bash
cd train
# Train stack (loads Muse)
source .venv-muse/bin/activate
pip install fastapi uvicorn 'harbor>=0.21.0'   # if missing; harbor needs Python ≥3.12

# Or keep Harbor in .venv-eval — test.py will call .venv-eval/bin/harbor automatically
# and spawn serve with .venv-muse/bin/python
```

## Run

```bash
cd /root/autodl-tmp/BIV/train   # example path on AutoDL
source .venv-muse/bin/activate

python scripts/test.py --dry-run

# Base Muse (Meta reference arm)
python scripts/test.py

# Real LoRA checkpoint
python scripts/test.py --ckpt outputs/.../checkpoint-e1-s50
python scripts/test.py --ckpt auto
```

Optional: `--base-url http://127.0.0.1:8000/v1` only if you already started `python -m eval.serve_openai ...` yourself (then `--ckpt` on `test.py` does **not** re-load weights — the running server does).

## Notes

- Not training JSONL. External Harbor datasets only.
- Meta TB used E2B; default here is `--env docker`.
- SWE uses mini-swe-agent ≈ Meta’s bash+file thin scaffold.
