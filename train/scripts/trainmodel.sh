#!/usr/bin/env bash
# Muse Glimmer-30B WM SFT via TRL + PEFT + Accelerate (branch: Muse).
# Auto parallel: 1 visible GPU → single; 2+ → DDP. FSDP2 via PARALLEL=fsdp2.
#
# Requires explicit --max-length. Before training:
#   1) structure-preserving right trunc (keep prefix ending on complete assistant)
#   2) print survivors (+ delete-ref for comparison)
#   3) ask: 1=as-is / 2=rebalance 1:1:0.35 / 3=abort
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0
#   bash scripts/trainmodel.sh --max-length 8192 --choice 1
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash scripts/trainmodel.sh --max-length 8192 --choice 1
#   QLORA=1 bash scripts/trainmodel.sh --max-length 8192 --choice 1
#   PARALLEL=fsdp2 bash scripts/trainmodel.sh --max-length 32768 --choice 1
#
# Env:
#   PARALLEL=single|ddp|fsdp2|auto   CONFIG=...  MIX_DIR=...  QLORA=0|1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CONFIG="${CONFIG:-configs/trl/muse_glimmer_30b_lora.yaml}"
MIX_DIR="${MIX_DIR:-data/processed/mix_v2}"

MAX_LENGTH=""
CHOICE="${TRAIN_CHOICE:-}"
FORCE_PREP=0
QLORA_FLAG=0
EXTRA=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/trainmodel.sh --max-length <N> [--choice 1|2|3] [--force-prep] [--qlora]

  --max-length N   required; struct-right trunc to complete assistant within N
  --choice K       skip interactive prompt (1=as-is, 2=1:1:0.35, 3=abort)
  --force-prep     rebuild filtered run cache even if present
  --qlora          4-bit QLoRA (also QLORA=1)

Env:
  CUDA_VISIBLE_DEVICES     default 0
  PARALLEL                 omit/auto | single | ddp | fsdp2
  TRAIN_CHOICE             same as --choice
  CONFIG / MIX_DIR / QLORA
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
    --qlora)
      QLORA_FLAG=1
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

_SAVE_PARALLEL="${PARALLEL-}"
# shellcheck disable=SC1090
source "$RUN_ENV"
if [[ -n "${_SAVE_PARALLEL}" ]]; then PARALLEL="${_SAVE_PARALLEL}"; fi

if [[ "$QLORA_FLAG" -eq 1 ]] || [[ "${QLORA:-0}" == "1" ]] || [[ "${QLORA:-}" == "true" ]]; then
  export QLORA=1
  QLORA_ARGS=(--qlora)
else
  QLORA_ARGS=()
fi

echo "=== Muse Glimmer TRL train ==="
echo "  tag=$TAG run_id=$RUN_ID choice=$TRAIN_CHOICE"
echo "  manifest=$MANIFEST"
echo "  model=$MODEL"
echo "  max_length=$MAX_LENGTH truncation=$TRUNC  (rows pre-cleaned)"
echo "  output_dir=$OUT_DIR"
echo "  qlora=${QLORA:-0}"
echo "  cached_dataset:"
# shellcheck disable=SC2086
python - <<'PY'
import os
for p in os.environ["CACHED_DATASETS"].split():
    print(f"    {p}")
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PARALLEL="${PARALLEL:-}"

NGPU="$(python - <<'PY'
import os
xs=[x for x in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if x.strip()]
print(len(xs) if xs else 1)
PY
)"

if [[ -z "$PARALLEL" || "$PARALLEL" == "auto" ]]; then
  if [[ "$NGPU" -le 1 ]]; then
    PARALLEL=single
  else
    PARALLEL=ddp
  fi
  echo "  auto PARALLEL=$PARALLEL (ngpu=$NGPU)"
fi

TRAIN_PY=(
  scripts/train_muse_trl.py
  --config "$CONFIG"
  --max-length "$MAX_LENGTH"
  --cached-datasets $CACHED_DATASETS
  --output-dir "$OUT_DIR"
  --model "$MODEL"
  --learning-rate "$LR"
  --num-epochs "$EPOCHS"
  --per-device-train-batch-size "$BS"
  --gradient-accumulation-steps "$GAS"
  --lora-rank "$LORA_RANK"
  --lora-alpha "$LORA_ALPHA"
  --logging-steps "$LOG_STEPS"
  --save-steps "$SAVE_STEPS"
  --save-total-limit "$SAVE_LIMIT"
  --warmup-ratio "$WARMUP"
  "${QLORA_ARGS[@]}"
)

LAUNCH=()
case "$PARALLEL" in
  single)
    echo "  single-GPU LoRA/QLoRA"
    LAUNCH=(python)
    ;;
  ddp|multi|multigpu)
    if [[ "$NGPU" -le 1 ]]; then
      echo "NOTE: ngpu=1 → PARALLEL=single"
      LAUNCH=(python)
    else
      ACCEL_CFG="${ACCELERATE_CONFIG:-configs/accelerate/muse_multi_ddp.yaml}"
      echo "  accelerate DDP num_processes=$NGPU config=$ACCEL_CFG"
      LAUNCH=(
        accelerate launch
        --config_file "$ACCEL_CFG"
        --num_processes "$NGPU"
        --mixed_precision bf16
      )
    fi
    ;;
  fsdp2|fsdp)
    if [[ "$NGPU" -le 1 ]]; then
      echo "WARNING: FSDP2 with 1 GPU is unnecessary; using single process."
      LAUNCH=(python)
    else
      ACCEL_CFG="${ACCELERATE_CONFIG:-configs/accelerate/muse_fsdp2.yaml}"
      echo "  accelerate FSDP2 num_processes=$NGPU config=$ACCEL_CFG"
      LAUNCH=(
        accelerate launch
        --config_file "$ACCEL_CFG"
        --num_processes "$NGPU"
        --mixed_precision bf16
      )
    fi
    ;;
  *)
    echo "ERROR: unknown PARALLEL=$PARALLEL (use single|ddp|fsdp2|auto)"
    exit 1
    ;;
esac

echo "  launch: ${LAUNCH[*]} ${TRAIN_PY[0]} …"
"${LAUNCH[@]}" "${TRAIN_PY[@]}"
echo "Done. Adapters → $OUT_DIR"
