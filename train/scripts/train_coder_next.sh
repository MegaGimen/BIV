#!/usr/bin/env bash
# Dual-GPU QLoRA SFT (ms-swift) for Qwen3-Coder-Next.
#
# Requires explicit --max-length. Before training:
#   1) drop rows with length > max_length on each source cache
#   2) print retention counts
#   3) ask: 1=as-is / 2=rebalance 1:1:0.35 / 3=abort
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_coder_next.sh --max-length 16384
#   bash scripts/train_coder_next.sh --max-length 32768 --choice 2   # non-interactive
#
# Other overrides:
#   CONFIG=configs/swift/coder_next_qlora.yaml MIX_DIR=data/processed/mix_v1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CONFIG="${CONFIG:-configs/swift/coder_next_qlora.yaml}"
MIX_DIR="${MIX_DIR:-data/processed/mix_v1}"

MAX_LENGTH=""
CHOICE="${TRAIN_CHOICE:-}"
FORCE_PREP=0
EXTRA=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_coder_next.sh --max-length <N> [--choice 1|2|3] [--force-prep]

  --max-length N   required; clean = drop rows with token length > N
  --choice K       skip interactive prompt (1=as-is, 2=1:1:0.35, 3=abort)
  --force-prep     rebuild filtered run cache even if present

Env:
  CUDA_VISIBLE_DEVICES  default 0,1
  TRAIN_CHOICE          same as --choice
  CONFIG / MIX_DIR
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-length|--max_length|-m)
      [[ $# -ge 2 ]] || { echo "missing value for $1"; exit 1; }
      MAX_LENGTH="$2"
      shift 2
      ;;
    --choice|-c)
      [[ $# -ge 2 ]] || { echo "missing value for $1"; exit 1; }
      CHOICE="$2"
      shift 2
      ;;
    --force-prep)
      FORCE_PREP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

if [[ ${#EXTRA[@]} -gt 0 ]]; then
  echo "Unknown args: ${EXTRA[*]}"
  usage
  exit 1
fi

if [[ -z "$MAX_LENGTH" ]]; then
  echo "ERROR: --max-length is required (manual startup parameter)."
  echo ""
  usage
  exit 1
fi
if ! [[ "$MAX_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --max-length must be a positive integer, got: $MAX_LENGTH"
  exit 1
fi

if [[ ! -f "$MIX_DIR/mix_manifest.json" ]]; then
  echo "Missing $MIX_DIR/mix_manifest.json — run prepare first:"
  echo "  python scripts/prepare_data.py --all --out-dir $MIX_DIR"
  exit 1
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "ms-swift CLI not found. Install: pip install 'ms-swift>=3.11' deepspeed bitsandbytes"
  exit 1
fi

if ! python scripts/tokenize_data.py --config "$CONFIG" --mix-dir "$MIX_DIR" --check; then
  echo ""
  echo "Tokenize cache not ready. Run:"
  echo "  python scripts/tokenize_data.py --config $CONFIG"
  exit 1
fi

RUN_ENV="$(mktemp "${TMPDIR:-/tmp}/biv_train_env.XXXXXX")"
cleanup() { rm -f "$RUN_ENV"; }
trap cleanup EXIT

PREP_ARGS=(
  --config "$CONFIG"
  --max-length "$MAX_LENGTH"
  --write-env "$RUN_ENV"
)
if [[ -n "$CHOICE" ]]; then
  PREP_ARGS+=(--choice "$CHOICE")
fi
if [[ "$FORCE_PREP" -eq 1 ]]; then
  PREP_ARGS+=(--force)
fi

set +e
python scripts/train_prep_mix.py "${PREP_ARGS[@]}"
prep_rc=$?
set -e
if [[ "$prep_rc" -eq 3 ]]; then
  echo "Aborted by user (choice 3)."
  exit 3
fi
if [[ "$prep_rc" -ne 0 ]]; then
  echo "train_prep_mix.py failed (exit $prep_rc)."
  exit "$prep_rc"
fi

# shellcheck disable=SC1090
source "$RUN_ENV"

echo "=== ms-swift train ==="
echo "  tag=$TAG run_id=$RUN_ID choice=$TRAIN_CHOICE"
echo "  manifest=$MANIFEST"
echo "  model=$MODEL"
echo "  max_length=$MAX_LENGTH truncation=$TRUNC  (rows pre-cleaned)"
echo "  output_dir=$OUT_DIR"
echo "  cached_dataset:"
# shellcheck disable=SC2086
python - <<'PY'
import os
for p in os.environ["CACHED_DATASETS"].split():
    print(f"    {p}")
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(python - <<'PY'
import os
xs=[x for x in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if x.strip()]
print(len(xs) if xs else 1)
PY
)"
fi
export NPROC_PER_NODE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "  GPUs=$CUDA_VISIBLE_DEVICES NPROC_PER_NODE=$NPROC_PER_NODE"

# shellcheck disable=SC2086
exec swift sft \
  --model "$MODEL" \
  --tuner_type lora \
  --quant_method bnb \
  --quant_bits 4 \
  --cached_dataset $CACHED_DATASETS \
  --torch_dtype "$DTYPE" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$BS" \
  --gradient_accumulation_steps "$GAS" \
  --learning_rate "$LR" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --target_modules $TARGET_MODULES \
  --max_length "$MAX_LENGTH" \
  --truncation_strategy "$TRUNC" \
  --warmup_ratio "$WARMUP" \
  --logging_steps "$LOG_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_LIMIT" \
  --output_dir "$OUT_DIR" \
  --deepspeed "$DEEPSPEED" \
  --attn_impl flash_attn \
  --dataloader_num_workers 4
