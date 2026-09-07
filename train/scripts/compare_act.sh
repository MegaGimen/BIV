#!/usr/bin/env bash
# Dual-GPU ACT compare, then merge that mask into Instruct.
# AgentWorld on GPU0, Instruct on GPU1. Spell every flag (defaults = live 1500).
#
#   cd train
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/compare_act.sh
#   CUDA_VISIBLE_DEVICES=0,1 MAX_ROWS=20 bash scripts/compare_act.sh   # smoke
#   CUDA_VISIBLE_DEVICES=0,1 MAX_LENGTH=65536 bash scripts/compare_act.sh
#   SKIP_MERGE=1 CUDA_VISIBLE_DEVICES=0,1 bash scripts/compare_act.sh
#
# Extra flags after the script name go to compare_act.py, e.g.
#   bash scripts/compare_act.sh --p 0.01
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORLD="${WORLD:-Qwen/Qwen-AgentWorld-35B-A3B}"
AGENT="${AGENT:-Qwen/Qwen3.5-35B-A3B}"
CACHE_DIR="${CACHE_DIR:-$REPO/merge/output/cache}"
OUT_DIR="${OUT_DIR:-outputs/act}"
SOURCE="${SOURCE:-modelscope}"
JSONL="${JSONL:-data/processed/mix_v2}"
MAX_ROWS="${MAX_ROWS:-1500}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-256}"
TOKENS="${TOKENS:-answer}"
P="${P:-0.01}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

set -- python scripts/compare_act.py \
  --world "$WORLD" \
  --agent "$AGENT" \
  --cache-dir "$CACHE_DIR" \
  --out-dir "$OUT_DIR" \
  --source "$SOURCE" \
  --jsonl "$JSONL" \
  --max-rows "$MAX_ROWS" \
  --max-length "$MAX_LENGTH" \
  --max-answer-tokens "$MAX_ANSWER_TOKENS" \
  --tokens "$TOKENS" \
  --p "$P" \
  --chunk-size "$CHUNK_SIZE" \
  --device-map "$DEVICE_MAP" \
  "$@"

echo "+ $*"
"$@"

if [[ "${SKIP_MERGE:-0}" == "1" ]]; then
  echo "SKIP_MERGE=1: not running merge/act.py"
  exit 0
fi

python "$REPO/merge/act.py"
