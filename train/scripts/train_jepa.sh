#!/usr/bin/env bash
# Stage 1 JEPA: 4-GPU FSDP2 + Context Parallel, max_length=65536 (Muse recipe).
#
#   cd train
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash scripts/train_jepa.sh
#   bash scripts/train_jepa.sh --max-length 65536
#
# Auto: 1 GPU → python; 2+ → FSDP2+CP (cp_size=#GPUs). Override: PARALLEL=single|fsdp2|fsdp2_cp
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/jepa/stage1.yaml}"
MAX_LENGTH="${MAX_LENGTH:-65536}"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-length|--max_length|-m)
      [[ $# -ge 2 ]] || { echo "missing value for $1"; exit 1; }
      MAX_LENGTH="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || { echo "missing value for $1"; exit 1; }
      CONFIG="$2"
      shift 2
      ;;
    --model-dir|--mix-dir|--logging-dir|--max-steps)
      [[ $# -ge 2 ]] || { echo "missing value for $1"; exit 1; }
      EXTRA+=("$1" "$2")
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Stage 1 JEPA: 4-GPU FSDP2 + Context Parallel, max_length=65536.

  cd train
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  bash scripts/train_jepa.sh
  bash scripts/train_jepa.sh --max-length 65536

Env: PARALLEL=single|fsdp2|fsdp2_cp|auto  CONFIG=...  MAX_LENGTH=...
EOF
      exit 0
      ;;
    *)
      echo "unknown arg: $1"
      exit 1
      ;;
  esac
done

if ! [[ "$MAX_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --max-length must be a positive integer, got: $MAX_LENGTH"
  exit 1
fi

NGPU="$(python - <<'PY'
import os
xs=[x for x in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if x.strip()]
print(len(xs) if xs else 1)
PY
)"

PARALLEL="${PARALLEL:-auto}"
if [[ -z "$PARALLEL" || "$PARALLEL" == "auto" ]]; then
  if [[ "$NGPU" -le 1 ]]; then
    PARALLEL=single
  else
    PARALLEL=fsdp2_cp
  fi
  echo "  auto PARALLEL=$PARALLEL (ngpu=$NGPU, max_length=$MAX_LENGTH)"
fi

CP_SIZE=1
LAUNCH=()
case "$PARALLEL" in
  single)
    echo "  single-GPU JEPA"
    LAUNCH=(python)
    ;;
  fsdp2|fsdp)
    if [[ "$NGPU" -le 1 ]]; then
      PARALLEL=single
      LAUNCH=(python)
    else
      ACCEL_CFG="${ACCELERATE_CONFIG:-configs/accelerate/qwen35_moe_fsdp2.yaml}"
      echo "  accelerate FSDP2 (no CP) num_processes=$NGPU"
      LAUNCH=(
        accelerate launch
        --config_file "$ACCEL_CFG"
        --num_processes "$NGPU"
        --mixed_precision bf16
      )
    fi
    ;;
  fsdp2_cp|cp|fsdp2+cp)
    if [[ "$NGPU" -le 1 ]]; then
      echo "WARNING: CP needs ≥2 GPUs; falling back to single."
      PARALLEL=single
      LAUNCH=(python)
      CP_SIZE=1
    else
      CP_SIZE="$NGPU"
      ACCEL_CFG="${ACCELERATE_CONFIG:-configs/accelerate/qwen35_moe_fsdp2_cp.yaml}"
      echo "  accelerate FSDP2+CP"
      echo "    num_processes=$NGPU cp_size=$CP_SIZE (~max_length/$CP_SIZE tokens/GPU)"
      echo "    config=$ACCEL_CFG"
      export ACCELERATE_USE_PARALLELISM_CONFIG=true
      export PARALLELISM_CONFIG_DP_REPLICATE_SIZE=1
      export PARALLELISM_CONFIG_DP_SHARD_SIZE=1
      export PARALLELISM_CONFIG_TP_SIZE=1
      export PARALLELISM_CONFIG_CP_SIZE="$CP_SIZE"
      export PARALLELISM_CONFIG_CP_BACKEND=torch
      LAUNCH=(
        accelerate launch
        --config_file "$ACCEL_CFG"
        --num_processes "$NGPU"
        --mixed_precision bf16
        --use_fsdp
        --fsdp_version 2
        --use_parallelism_config
        --fsdp_transformer_layer_cls_to_wrap Qwen3_5MoeDecoderLayer
        --fsdp_activation_checkpointing false
        --parallelism_config_dp_replicate_size 1
        --parallelism_config_dp_shard_size 1
        --parallelism_config_tp_size 1
        --parallelism_config_cp_size "$CP_SIZE"
        --parallelism_config_cp_backend torch
      )
    fi
    ;;
  *)
    echo "ERROR: unknown PARALLEL=$PARALLEL (use single|fsdp2|fsdp2_cp|auto)"
    exit 1
    ;;
esac

export BIV_CP_SIZE="$CP_SIZE"
export BIV_PARALLEL="$PARALLEL"

TRAIN_PY=(
  scripts/train_jepa.py
  --config "$CONFIG"
  --max-length "$MAX_LENGTH"
  --cp-size "$CP_SIZE"
  "${EXTRA[@]}"
)

echo "  launch: ${LAUNCH[*]} ${TRAIN_PY[0]} --max-length $MAX_LENGTH --cp-size $CP_SIZE"
"${LAUNCH[@]}" "${TRAIN_PY[@]}"
echo "Done."
