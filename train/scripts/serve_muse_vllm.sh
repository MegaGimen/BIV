#!/usr/bin/env bash
# Start Muse Glimmer with vLLM on AutoDL (default port 6006 for custom service).
#
# AutoDL maps container 6006 → public https://<实例>.westd.seetacloud.com:8443
# Harbor / test.py should use:
#   export MUSE_BASE_URL=https://<实例>.westd.seetacloud.com:8443/v1
#
# Usage:
#   bash scripts/serve_muse_vllm.sh
#   bash scripts/serve_muse_vllm.sh --ckpt outputs/.../checkpoint-e1-s50
#   PORT=6006 CKPT=auto bash scripts/serve_muse_vllm.sh
#
# --ckpt loads a PEFT LoRA via vLLM --lora-modules (name: muse-lora).
# test.py --ckpt then calls model id "muse-lora"; without --ckpt uses Muse-Glimmer-30B.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-6006}"
HOST="${HOST:-0.0.0.0}"
SERVED_BASE="${SERVED_BASE:-Muse-Glimmer-30B}"
LORA_NAME="${LORA_NAME:-muse-lora}"
MODEL_PATH="${MODEL_PATH:-}"
CKPT="${CKPT:-}"
MAX_LORA_RANK="${MAX_LORA_RANK:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
DTYPE="${DTYPE:-bfloat16}"
TP="${TP:-1}"
# Blackwell (sm_120): FlashInfer JIT arch detect can spuriously fail warmup with
# "requires GPUs with sm75 or higher". Default off; override with =1 if desired.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      CKPT="$2"; shift 2 ;;
    --model-path|--model_path)
      MODEL_PATH="$2"; shift 2 ;;
    --port)
      PORT="$2"; shift 2 ;;
    --tp)
      TP="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,20p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  if [[ -d "$ROOT/outputs/models/Muse-Glimmer-30B" ]]; then
    MODEL_PATH="$ROOT/outputs/models/Muse-Glimmer-30B"
  else
    echo "ERROR: set MODEL_PATH or put weights at outputs/models/Muse-Glimmer-30B"
    exit 1
  fi
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm not found. In .venv-muse: pip install vllm"
  exit 1
fi

# Prefer the same interpreter as the vllm entrypoint (avoid system python3).
_VLLM_BIN="$(command -v vllm)"
_PY="$(dirname "$_VLLM_BIN")/python"
if [[ ! -x "$_PY" ]]; then
  _PY="python3"
fi

# Torch 2.13+cu130 needs nvidia-nccl-cu13. Installing vllm can pull
# nvidia-nccl-cu12 into the same nvidia/nccl/ path and break:
#   undefined symbol: ncclCommResume
if ! "$_PY" -c "import torch" 2>/dev/null; then
  echo "ERROR: torch import failed (often NCCL cu12 overwrote cu13)."
  echo "  Fix: pip uninstall -y nvidia-nccl-cu12"
  echo "       pip install --force-reinstall --no-deps 'nvidia-nccl-cu13==2.29.7'"
  "$_PY" -c "import torch" || true
  exit 1
fi

# Resolve CKPT=auto like test.py (epoch, step, kind).
if [[ -n "$CKPT" && "$(echo "$CKPT" | tr '[:upper:]' '[:lower:]')" == "auto" ]]; then
  SEARCH="${CKPT_SEARCH_DIR:-$ROOT/outputs/muse_glimmer_wm_mix}"
  CKPT="$(
    SEARCH="$SEARCH" "$_PY" - <<'PY'
import os, re
from pathlib import Path
out = Path(os.environ["SEARCH"])
pats = [
    (re.compile(r"^checkpoint-epoch(\d+)-end-s(\d+)$"), 2),
    (re.compile(r"^checkpoint-e(\d+)-s(\d+)$"), 1),
    (re.compile(r"^checkpoint-(\d+)$"), 0),
]
best = None
if out.is_dir():
    for p in out.iterdir():
        if not p.is_dir():
            continue
        epoch = step = kind = None
        for rx, k in pats:
            m = rx.match(p.name)
            if not m:
                continue
            if k == 0:
                epoch, step, kind = 0, int(m.group(1)), 0
            else:
                epoch, step, kind = int(m.group(1)), int(m.group(2)), k
            break
        if epoch is None:
            continue
        if not (p / "trainer_state.json").is_file():
            continue
        if not (
            (p / "adapter_model.safetensors").is_file()
            or (p / "adapter_config.json").is_file()
            or any(p.glob("*.safetensors"))
        ):
            continue
        key = (epoch, step, kind)
        if best is None or key > best[0]:
            best = (key, p)
print("" if best is None else best[1])
PY
  )"
  if [[ -z "$CKPT" ]]; then
    echo "ERROR: CKPT=auto but no complete checkpoint under $SEARCH"
    exit 1
  fi
  echo "[serve] CKPT=auto → $CKPT"
fi

echo "=== Muse vLLM serve ==="
echo "  model_path: $MODEL_PATH"
echo "  port:       $PORT  (AutoDL custom service → public :8443)"
echo "  served:     $SERVED_BASE"
if [[ -n "$CKPT" ]]; then
  echo "  lora:       $LORA_NAME ← $CKPT"
  echo "  Harbor/--model for this LoRA: $LORA_NAME"
else
  echo "  lora:       <none> (base only; Harbor --model $SERVED_BASE)"
fi
echo

CMD=(
  vllm serve "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_BASE"
  --dtype "$DTYPE"
  --tensor-parallel-size "$TP"
  --trust-remote-code
)
if [[ -n "$MAX_MODEL_LEN" ]]; then
  CMD+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "$CKPT" ]]; then
  if [[ ! -f "$CKPT/adapter_config.json" ]]; then
    echo "ERROR: $CKPT missing adapter_config.json (need PEFT LoRA ckpt)"
    exit 1
  fi
  CMD+=(
    --enable-lora
    --max-loras 1
    --max-lora-rank "$MAX_LORA_RANK"
    --lora-modules "${LORA_NAME}=${CKPT}"
  )
fi

echo "+ ${CMD[*]}"
exec "${CMD[@]}"
