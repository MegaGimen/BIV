#!/usr/bin/env bash
# Install Muse-capable vLLM without Docker (AutoDL-friendly).
#
# PyPI vllm==0.27.1 has NO muse_glimmer model/parsers. Muse day-0 lives on
# vLLM main after PR #51655 (merged 2026-08-14). Until the next PyPI release,
# install a nightly wheel that includes Muse.
#
#   cd train
#   bash scripts/install_muse_vllm.sh
#   source .venv-vllm-muse/bin/activate
#   bash scripts/serve_muse_vllm.sh
#
# Override wheel:
#   VLLM_MUSE_WHEEL_URL=https://wheels.vllm.ai/<commit>/vllm-....whl bash scripts/install_muse_vllm.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-$ROOT/.venv-vllm-muse}"
# Nightly after Muse merge (has muse_glimmer.py + parsers). Update when needed.
DEFAULT_WHEEL='https://wheels.vllm.ai/cdc4824a21eaa986d4d1fee90a7e6465c9f706e6/vllm-0.27.2rc1.dev92%2Bgcdc4824a2-cp38-abi3-manylinux_2_28_x86_64.whl'
WHEEL_URL="${VLLM_MUSE_WHEEL_URL:-$DEFAULT_WHEEL}"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x /root/miniconda3/bin/python3 ]]; then
    PY=/root/miniconda3/bin/python3
  else
    PY="$(command -v python3)"
  fi
fi

echo "=== install Muse vLLM (no Docker) ==="
echo "  python: $PY"
echo "  venv:   $VENV"
echo "  wheel:  $WHEEL_URL"

"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip wheel setuptools

# Match AutoDL CUDA 13 / torch 2.13 when possible.
python -m pip install --upgrade 'torch==2.13.0' \
  --index-url https://download.pytorch.org/whl/cu130 \
  || python -m pip install --upgrade 'torch==2.13.0'

python -m pip install "$WHEEL_URL"

# vLLM deps may pull nvidia-nccl-cu12 over cu13 and break torch+cu130.
python -m pip uninstall -y nvidia-nccl-cu12 || true
python -m pip install --force-reinstall --no-deps 'nvidia-nccl-cu13==2.29.7'

python - <<'PY'
import pathlib
import torch
import vllm

print(f"torch={torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}")
print(f"vllm={vllm.__version__}")
root = pathlib.Path(vllm.__file__).parent
need = [
    root / "model_executor/models/muse_glimmer.py",
    root / "reasoning/muse_glimmer_reasoning_parser.py",
    root / "tool_parsers/muse_glimmer_tool_parser.py",
]
missing = [str(p) for p in need if not p.is_file()]
if missing:
    raise SystemExit("Muse files missing:\n  " + "\n  ".join(missing))
print("muse_glimmer model + parsers: OK")
PY

echo
echo "Activate with:  source $VENV/bin/activate"
echo "Then serve:     bash scripts/serve_muse_vllm.sh"
