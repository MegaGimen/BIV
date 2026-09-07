#!/usr/bin/env bash
# Dual-GPU ACT probe, then merge that mask into Instruct.
# AgentWorld on GPU0, Instruct on GPU1, 1500 mix rows, models stay resident.
#
#   cd train
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/compare_act.sh
#   CUDA_VISIBLE_DEVICES=0,1 MAX_LENGTH=65536 bash scripts/compare_act.sh
#
# Merge reuses train/outputs/act/mask.json (no extra path). Skip it with SKIP_MERGE=1.
# One GPU still works (swaps models every --chunk-size rows):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/compare_act.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

JSONL="${JSONL:-data/processed/mix_v2}"
MAX_ROWS="${MAX_ROWS:-1500}"
MAX_LENGTH="${MAX_LENGTH:-32768}"

python scripts/compare_act.py \
  --jsonl "$JSONL" \
  --max-rows "$MAX_ROWS" \
  --max-length "$MAX_LENGTH" \
  "$@"

if [[ "${SKIP_MERGE:-0}" == "1" ]]; then
  echo "SKIP_MERGE=1: not running merge/act.py"
  exit 0
fi

python "$REPO/merge/act.py"
