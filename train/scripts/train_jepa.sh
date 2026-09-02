#!/usr/bin/env bash
# Stage 1 on AgentWorld: 32768 tokens, 4 GPUs as 2 groups of 2.
# Each group does Context Parallel (cp_size=2); the two groups train on
# different rows (dp_replicate_size=2). Same 2x2 layout as JEPALLM's 32k path.
#
#   cd train
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash scripts/train_jepa.sh
#   bash scripts/train_jepa.sh --save-steps 1 --max-steps 2
#   bash scripts/train_jepa.sh --resume
#
# 4 GPUs → 2x2 (needs qwen35_moe_fsdp2_cp2x2.yaml). Other GPU counts fall
# back to a single CP group (cp_size=NGPU), no dp_replicate.
# Optional long-context: PARALLEL=fsdp2_cp MAX_LENGTH=65536 bash scripts/train_jepa.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Variable-length prefix turns fragment the caching allocator; keep this on
# unless the user already set a policy.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/jepa/stage1.yaml}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
RUN_TAG="${RUN_TAG:-jepa}"
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
    --model-dir|--mix-dir|--logging-dir|--max-steps|--save-steps|--log-steps|--collapse-steps|--resume-from)
      [[ $# -ge 2 ]] || { echo "missing value for $1"; exit 1; }
      if [[ "$1" == "--resume-from" ]]; then
        EXTRA+=(--resume "$2")
      elif [[ "$1" == "--collapse-steps" ]]; then
        EXTRA+=(--log-steps "$2")
      else
        EXTRA+=("$1" "$2")
      fi
      shift 2
      ;;
    --resume)
      if [[ $# -ge 2 && "$2" != -* ]]; then
        EXTRA+=(--resume "$2")
        shift 2
      else
        EXTRA+=(--resume)
        shift
      fi
      ;;
    -h|--help)
      cat <<'EOF'
Stage 1: 32768 tokens, 4 GPUs as 2x2 (dp_replicate=2, cp=2 each).

  cd train
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  bash scripts/train_jepa.sh
  bash scripts/train_jepa.sh --save-steps 1 --max-steps 2
  bash scripts/train_jepa.sh --resume
  bash scripts/train_jepa.sh --resume outputs/jepa_stage1/checkpoint-e0-s25

4 GPUs: two groups of 2, each group CP-shards the sequence, groups eat
different data. Losses all-reduce into one TensorBoard curve (prefix jepa-).
Other GPU counts: single CP group, cp_size=NGPU.

     --save-steps N       (default yaml 25; 1 smokes FSDP save)
     --log-steps N        (default yaml 5; loss + collapse, not save)
     --resume             newest complete ckpt under output_dir
     --resume PATH / --resume-from PATH
     --max-steps N

Optional 65536 / 4-way CP: PARALLEL=fsdp2_cp MAX_LENGTH=65536 bash scripts/train_jepa.sh
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

case "$CONFIG" in
  /*) ;;
  *) CONFIG="$ROOT/$CONFIG" ;;
esac
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG"
  exit 1
fi

NGPU="$(python - <<'PY'
import os, shutil, subprocess
xs = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
if xs:
    print(len(xs))
elif shutil.which("nvidia-smi"):
    out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    n = sum(1 for line in out.splitlines() if line.strip().startswith("GPU"))
    print(n if n else 1)
else:
    print(1)
PY
)"

PARALLEL="${PARALLEL:-auto}"
USE_2X2=0
if [[ "$PARALLEL" == "fsdp2_cp2x2" ]]; then
  USE_2X2=1
elif [[ "$PARALLEL" == "auto" || -z "$PARALLEL" ]]; then
  if [[ "$NGPU" -eq 4 ]]; then
    USE_2X2=1
  fi
fi

if [[ "$USE_2X2" -eq 1 ]]; then
  if [[ "$NGPU" -ne 4 ]]; then
    echo "WARNING: 2x2 needs exactly 4 GPUs (got $NGPU). Falling back to a single"
    echo "         ${NGPU}-way CP group (no dp_replicate)."
    USE_2X2=0
    PARALLEL=fsdp2_cp
  fi
fi

CP_SIZE=1
LAUNCH=()
if [[ "$USE_2X2" -eq 1 ]]; then
  CP_SIZE=2
  ACCEL_CFG="${ACCELERATE_CONFIG:-configs/accelerate/qwen35_moe_fsdp2_cp2x2.yaml}"
  echo "  accelerate FSDP2+CP, 2 groups of 2 (dp_replicate=2, cp_size=$CP_SIZE)"
  echo "    config=$ACCEL_CFG  max_length=$MAX_LENGTH (~$((MAX_LENGTH / CP_SIZE)) tokens/GPU)"
  export ACCELERATE_USE_PARALLELISM_CONFIG=true
  export PARALLELISM_CONFIG_DP_REPLICATE_SIZE=2
  export PARALLELISM_CONFIG_DP_SHARD_SIZE=1
  export PARALLELISM_CONFIG_TP_SIZE=1
  export PARALLELISM_CONFIG_CP_SIZE="$CP_SIZE"
  export PARALLELISM_CONFIG_CP_BACKEND=torch
  export BIV_CP_SIZE="$CP_SIZE"
  export BIV_PARALLEL="fsdp2_cp2x2"
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
    --parallelism_config_dp_replicate_size 2
    --parallelism_config_dp_shard_size 1
    --parallelism_config_tp_size 1
    --parallelism_config_cp_size "$CP_SIZE"
    --parallelism_config_cp_backend torch
  )
else
  if [[ -z "$PARALLEL" || "$PARALLEL" == "auto" ]]; then
    if [[ "$NGPU" -le 1 ]]; then
      PARALLEL=single
    else
      PARALLEL=fsdp2_cp
    fi
    echo "  auto PARALLEL=$PARALLEL (ngpu=$NGPU, max_length=$MAX_LENGTH)"
  fi
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
        echo "  accelerate FSDP2+CP (single group)"
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
      echo "ERROR: unknown PARALLEL=$PARALLEL (use auto|single|fsdp2|fsdp2_cp|fsdp2_cp2x2)"
      exit 1
      ;;
  esac
  export BIV_CP_SIZE="$CP_SIZE"
  export BIV_PARALLEL="$PARALLEL"
fi

TRAIN_PY=(
  scripts/train_jepa.py
  --config "$CONFIG"
  --max-length "$MAX_LENGTH"
  --cp-size "$CP_SIZE"
  --run-tag "$RUN_TAG"
  "${EXTRA[@]}"
)

echo "  launch: ${LAUNCH[*]} ${TRAIN_PY[0]} --max-length $MAX_LENGTH --cp-size $CP_SIZE --run-tag $RUN_TAG"
"${LAUNCH[@]}" "${TRAIN_PY[@]}"
echo "Done."
