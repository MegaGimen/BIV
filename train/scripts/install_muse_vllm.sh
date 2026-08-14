#!/usr/bin/env bash
# Upgrade .venv-muse to Muse day-0 vLLM without re-downloading torch/CUDA.
#
# Stock PyPI vllm==0.27.1 has NO muse_glimmer. Install a post-merge nightly
# wheel with --no-deps so existing .venv-muse packages are reused.
#
#   cd train && source .venv-muse/bin/activate
#   bash scripts/install_muse_vllm.sh
#   bash scripts/serve_muse_vllm.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-$ROOT/.venv-muse}"
DEFAULT_WHEEL='https://wheels.vllm.ai/cdc4824a21eaa986d4d1fee90a7e6465c9f706e6/vllm-0.27.2rc1.dev92%2Bgcdc4824a2-cp38-abi3-manylinux_2_28_x86_64.whl'
WHEEL_URL="${VLLM_MUSE_WHEEL_URL:-$DEFAULT_WHEEL}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: missing $VENV (expected existing Muse train env)"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "=== upgrade Muse vLLM in-place (no torch re-download) ==="
echo "  venv:  $VENV"
echo "  wheel: $WHEEL_URL"
python -c "import torch,vllm; print(f'before torch={torch.__version__} vllm={vllm.__version__}')"

python -m pip install --force-reinstall --no-deps "$WHEEL_URL"

# Nightly may have pulled cu12 NCCL earlier in other installs; keep cu13 for torch+cu130.
python -m pip uninstall -y nvidia-nccl-cu12 2>/dev/null || true
if ! python -c "import torch" 2>/dev/null; then
  python -m pip install --force-reinstall --no-deps 'nvidia-nccl-cu13==2.29.7'
fi

python - <<'PY'
import pathlib
import torch
import vllm

print(f"after torch={torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}")
print(f"after vllm={vllm.__version__}")
root = pathlib.Path(vllm.__file__).parent
need = [
    root / "model_executor/models/muse_glimmer.py",
    root / "reasoning/muse_glimmer_reasoning_parser.py",
    root / "tool_parsers/muse_glimmer_tool_parser.py",
]
missing = [str(p.relative_to(root)) for p in need if not p.is_file()]
if missing:
    raise SystemExit("Muse files missing: " + ", ".join(missing))
print("muse_glimmer model + parsers: OK")
PY

echo
echo "Serve with:  source $VENV/bin/activate && bash scripts/serve_muse_vllm.sh"
