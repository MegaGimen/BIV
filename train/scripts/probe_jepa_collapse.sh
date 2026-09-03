#!/usr/bin/env bash
# Forward-only collapse probe on one GPU. No FSDP, no CP, no backward, no LoRA.
# Training stays 4-GPU: CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_jepa.sh
# Left [Enc(h);Enc(a)] (4096) vs right Enc(h,a,o) (2048), cross-dataset top-20.
#
#   cd train
#   CUDA_VISIBLE_DEVICES=0 bash scripts/probe_jepa_collapse.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/jepa/stage1.yaml}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
MAX_ROWS="${MAX_ROWS:-200}"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-rows|--top-k|--max-pairs|--out-dir|--config|--max-length|--model-dir|--mix-dir|--snippet|--print-pairs|--close-threshold)
      [[ $# -ge 2 ]] || { echo "missing value for $1"; exit 1; }
      if [[ "$1" == "--config" ]]; then CONFIG="$2"
      elif [[ "$1" == "--max-length" ]]; then MAX_LENGTH="$2"
      elif [[ "$1" == "--max-rows" ]]; then MAX_ROWS="$2"
      else EXTRA+=("$1" "$2"); fi
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Forward-only Stage 1 probe. One GPU, no FSDP.

  CUDA_VISIBLE_DEVICES=0 bash scripts/probe_jepa_collapse.sh
  bash scripts/probe_jepa_collapse.sh --max-rows 40
  bash scripts/probe_jepa_collapse.sh --print-pairs 0

Writes two files under outputs/jepa_collapse_probe/:
  left_zt_u-<stamp>.json   top-20 cross-dataset cosine on [z_t; u]
  right_hao-<stamp>.json   top-20 cross-dataset cosine on Enc(h,a,o)
Plus *-latest.json copies. Samples are file order, first N per mix source.
EOF
      exit 0
      ;;
    *)
      echo "unknown arg: $1"
      exit 1
      ;;
  esac
done

# Leftover from a 4-GPU train must not pull this into an accelerate mesh.
unset ACCELERATE_USE_PARALLELISM_CONFIG || true
unset PARALLELISM_CONFIG_DP_REPLICATE_SIZE || true
unset PARALLELISM_CONFIG_DP_SHARD_SIZE || true
unset PARALLELISM_CONFIG_TP_SIZE || true
unset PARALLELISM_CONFIG_CP_SIZE || true
unset PARALLELISM_CONFIG_CP_BACKEND || true
unset BIV_CP_SIZE || true
unset BIV_PARALLEL || true

echo "  single-GPU probe (no FSDP / no CP) max_rows=$MAX_ROWS max_length=$MAX_LENGTH"
python scripts/probe_jepa_collapse.py \
  --config "$CONFIG" \
  --max-length "$MAX_LENGTH" \
  --max-rows "$MAX_ROWS" \
  --cp-size 1 \
  "${EXTRA[@]}"
echo "Done."
