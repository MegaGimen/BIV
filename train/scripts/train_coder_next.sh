#!/usr/bin/env bash
# Dual-GPU QLoRA SFT for Qwen3-Coder-Next on mix_v1.
# Expects step-2 token cache from: python scripts/tokenize.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MIX_DIR="${MIX_DIR:-data/processed/mix_v1}"
CONFIG="${CONFIG:-configs/axolotl/coder_next_qlora.yaml}"
# Set ALLOW_INLINE_PREPROCESS=1 to let axolotl tokenize during train (not recommended).
ALLOW_INLINE_PREPROCESS="${ALLOW_INLINE_PREPROCESS:-0}"

if [[ ! -f "$MIX_DIR/mix_manifest.json" ]]; then
  echo "Missing $MIX_DIR/mix_manifest.json — run prepare first:"
  echo "  python scripts/prepare_data.py --all --out-dir $MIX_DIR"
  exit 1
fi

echo "=== Mix line counts ==="
python - <<PY
import json
from pathlib import Path
m = json.loads(Path("$MIX_DIR/counts.json").read_text())
for k,v in sorted((m.get("jsonl_line_counts") or {}).items()):
    print(f"  {k}: {v:,}")
PY

if ! command -v axolotl >/dev/null 2>&1; then
  echo "axolotl not found. Install: pip install axolotl 'cut-cross-entropy'"
  exit 1
fi

if ! python scripts/tokenize.py --config "$CONFIG" --mix-dir "$MIX_DIR" --check; then
  if [[ "$ALLOW_INLINE_PREPROCESS" == "1" ]]; then
    echo "WARNING: token cache missing; continuing with inline preprocess (ALLOW_INLINE_PREPROCESS=1)"
  else
    echo ""
    echo "Token cache not ready. Run step 2 (CPU, no GPU needed):"
    echo "  python scripts/tokenize.py --config $CONFIG"
    echo "Or force inline preprocess during train:"
    echo "  ALLOW_INLINE_PREPROCESS=1 bash scripts/train_coder_next.sh"
    exit 1
  fi
else
  echo "Token cache OK — train will load prepared dataset."
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
echo "Launching axolotl on GPUs=$CUDA_VISIBLE_DEVICES config=$CONFIG"
exec axolotl train "$CONFIG"
