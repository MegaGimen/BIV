#!/usr/bin/env bash
# Fresh venv for Chat Vector serve (Qwen3.5 + vLLM). Do NOT use train/.venv-muse.
#
# On AutoDL:
#   cd /root/autodl-tmp/BIV
#   bash merge/install_env.sh
#   source train/.venv/bin/activate
#   python merge/eval.py --max-model-len 65536
#
# Override location: VENV=/path/to/.venv bash merge/install_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$ROOT/train/.venv}"
PY="${PYTHON:-python3}"

if [[ "$VENV" == *".venv-muse"* ]]; then
  echo "ERROR: this env is for Qwen3.5 vLLM. Use train/.venv, not .venv-muse."
  exit 1
fi

echo "=== Chat Vector env ==="
echo "  python: $PY"
echo "  venv:   $VENV"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install -U pip setuptools wheel
# Qwen3.5 (qwen3_5_moe) is on vLLM nightly, not the PyPI stable cut.
python -m pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
python -m pip install -U modelscope huggingface_hub safetensors

python - <<'PY'
import sys
print("python", sys.version.split()[0], sys.executable)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        print("gpu", torch.cuda.get_device_name(0), "sm" + "".join(map(str, cap)))
except Exception as e:
    print("torch check:", e)
import vllm
print("vllm", getattr(vllm, "__version__", "?"))
PY

echo
echo "Activate: source $VENV/bin/activate"
echo "Serve:    python merge/eval.py --max-model-len 65536"
echo "If FlashInfer errors: VLLM_ATTENTION_BACKEND=FLASH_ATTN python merge/eval.py --max-model-len 65536"
