#!/usr/bin/env bash
# Dual-GPU QLoRA SFT for Qwen3-Coder-Next on ratio-sampled mix_v1.
# Expects step-2 from: python scripts/tokenize.py  (writes *.run.yaml + token cache)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MIX_DIR="${MIX_DIR:-data/processed/mix_v1}"
CONFIG="${CONFIG:-configs/axolotl/coder_next_qlora.yaml}"
RUN_CONFIG="${RUN_CONFIG:-configs/axolotl/coder_next_qlora.run.yaml}"
# Set ALLOW_INLINE_PREPROCESS=1 to let axolotl tokenize during train (not recommended).
ALLOW_INLINE_PREPROCESS="${ALLOW_INLINE_PREPROCESS:-0}"

if [[ ! -f "$MIX_DIR/mix_manifest.json" ]]; then
  echo "Missing $MIX_DIR/mix_manifest.json — run prepare first:"
  echo "  python scripts/prepare_data.py --all --out-dir $MIX_DIR"
  exit 1
fi

echo "=== Full corpus line counts (pre-sample) ==="
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

if [[ ! -f "$RUN_CONFIG" ]]; then
  echo "Missing $RUN_CONFIG — run ratio-sample + tokenize first:"
  echo "  python scripts/tokenize.py --config $CONFIG"
  exit 1
fi

if ! python scripts/tokenize.py --config "$CONFIG" --mix-dir "$MIX_DIR" --check; then
  if [[ "$ALLOW_INLINE_PREPROCESS" == "1" ]]; then
    echo "WARNING: token cache incomplete; continuing (ALLOW_INLINE_PREPROCESS=1)"
  else
    echo ""
    echo "Sampled token cache not ready. Run step 2 (CPU):"
    echo "  python scripts/tokenize.py --config $CONFIG"
    exit 1
  fi
else
  echo "Sampled token cache OK — train uses $RUN_CONFIG only."
fi

echo "=== Run config datasets (sampled) ==="
python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("$RUN_CONFIG").read_text())
for d in cfg.get("datasets") or []:
    p = Path(d["path"])
    n = sum(1 for _ in p.open("rb")) if p.is_file() else -1
    print(f"  {d['path']}: {n:,} rows")
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
echo "Launching axolotl on GPUs=$CUDA_VISIBLE_DEVICES config=$RUN_CONFIG"
exec axolotl train "$RUN_CONFIG"
