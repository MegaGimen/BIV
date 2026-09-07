#!/usr/bin/env bash
# Dual-GPU ACT probe: AgentWorld on GPU0, Instruct on GPU1, 1500 mix rows.
# Models stay resident; both cards forward the same sample in parallel.
#
#   cd train
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/compare_act.sh
#   python ../merge/act.py --no-lm-head
#
# One GPU still works (swaps models every --chunk-size rows):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/compare_act.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

JSONL="${JSONL:-data/processed/mix_v2}"
MAX_ROWS="${MAX_ROWS:-1500}"
MAX_LENGTH="${MAX_LENGTH:-32768}"

exec python scripts/compare_act.py \
  --jsonl "$JSONL" \
  --max-rows "$MAX_ROWS" \
  --max-length "$MAX_LENGTH" \
  "$@"
