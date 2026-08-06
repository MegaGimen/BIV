#!/usr/bin/env bash
# Dual-GPU QLoRA SFT for Qwen3-Coder-Next on mix_v1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MIX_DIR="${MIX_DIR:-data/processed/mix_v1}"
CONFIG="${CONFIG:-configs/axolotl/coder_next_qlora.yaml}"

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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
echo "Launching axolotl on GPUs=$CUDA_VISIBLE_DEVICES config=$CONFIG"
exec axolotl train "$CONFIG"
