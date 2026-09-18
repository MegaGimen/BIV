#!/usr/bin/env bash
# Stage 2 Step 1: frozen JEPA world + frozen Instruct, train draft/scorer/W.
# Same GPU mesh as Stage 1. Point --jepa-ckpt at a complete Stage 1 dir
# (adapter + jepa.pt) or use auto (newest under outputs/jepa_stage1).
#
#   cd train
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_jepa_s2.sh --jepa-ckpt auto
#   bash scripts/train_jepa_s2.sh --jepa-ckpt outputs/jepa_stage1/checkpoint-epoch2-end-s13630
#   bash scripts/train_jepa_s2.sh --save-steps 1 --max-steps 2
#
# N GPUs → single CP group (cp_size=N), seq-split on. GDN heads must divide N
# (Qwen3.5-35B-A3B: 2 or 4, not 3). Opt-in 4-GPU 2x2 (weight-only CP, no seq
# split): PARALLEL=fsdp2_cp2x2. Weight-only CP: --no-seq-split.
# 2-GPU 65536:
#   CUDA_VISIBLE_DEVICES=0,1 MAX_LENGTH=65536 bash scripts/train_jepa.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Variable-length prefix turns fragment the caching allocator; keep this on
# unless the user already set a policy.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/jepa/stage2.yaml}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
RUN_TAG="${RUN_TAG:-jepa-s2}"
EXTRA=()
SEQ_SPLIT_CLI=""

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
    --model-dir|--mix-dir|--logging-dir|--max-steps|--save-steps|--log-steps|--collapse-steps|--resume-from|--grad-accum|--jepa-ckpt|--jepa-dir|--world-dir|--instruct-dir)
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
    --ghost|--seq-split|--no-seq-split)
      EXTRA+=("$1")
      if [[ "$1" == "--seq-split" || "$1" == "--no-seq-split" ]]; then
        SEQ_SPLIT_CLI="$1"
      fi
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Stage 2 Step 1: same mesh as Stage 1. Frozen World+JEPA + frozen Instruct.

  cd train
  CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_jepa_s2.sh --jepa-ckpt auto
  bash scripts/train_jepa_s2.sh --jepa-ckpt outputs/jepa_stage1/checkpoint-epoch2-end-s13630
  bash scripts/train_jepa_s2.sh --save-steps 1 --max-steps 2
  bash scripts/train_jepa_s2.sh --resume

Default: single CP group, cp_size=NGPU, seq-split on. TensorBoard prefix jepa-.
GDN heads must divide N (this 35B: 2 or 4 GPUs). Opt-in 2x2:
PARALLEL=fsdp2_cp2x2 (weight-only CP, two data groups).

     --save-steps N       (default yaml 25; 1 smokes FSDP save)
     --log-steps N        (default yaml 5; loss + collapse, not save)
     --resume             newest complete ckpt under output_dir
     --resume PATH / --resume-from PATH
     --max-steps N
     --grad-accum N      (override yaml; smoke one optimizer step with 1)
     --seq-split         default on for FSDP+CP; shard tokens, GDN all-to-all
     --no-seq-split      weight-only CP (full sequence on each rank)
     --ghost             2-step feasibility run: no ckpt, no TensorBoard,
                         output_dir=outputs/jepa_ghost_seqcp (never jepa_stage2)
     --jepa-ckpt PATH    Stage 1 dir, or auto (newest complete under jepa_dir)

  CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_jepa_s2.sh --jepa-ckpt auto
  CUDA_VISIBLE_DEVICES=0,1 MAX_LENGTH=32768 bash scripts/train_jepa_s2.sh --jepa-ckpt auto
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
  echo "    config=$ACCEL_CFG  max_length=$MAX_LENGTH (weight-only CP; pass --seq-split to override)"
  export ACCELERATE_USE_PARALLELISM_CONFIG=true
  export PARALLELISM_CONFIG_DP_REPLICATE_SIZE=2
  export PARALLELISM_CONFIG_DP_SHARD_SIZE=1
  export PARALLELISM_CONFIG_TP_SIZE=1
  export PARALLELISM_CONFIG_CP_SIZE="$CP_SIZE"
  export PARALLELISM_CONFIG_CP_BACKEND=torch
  export BIV_CP_SIZE="$CP_SIZE"
  export BIV_PARALLEL="fsdp2_cp2x2"
  if [[ -z "$SEQ_SPLIT_CLI" ]]; then
    EXTRA+=(--no-seq-split)
  fi
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
        echo "    num_processes=$NGPU cp_size=$CP_SIZE (seq-split $MAX_LENGTH/$CP_SIZE tokens/GPU; CP-folded FSDP shards weights)"
        echo "    config=$ACCEL_CFG"
        if [[ "$NGPU" -eq 2 && "$MAX_LENGTH" -ge 65536 ]]; then
          echo "    2-GPU ${MAX_LENGTH}: seq-split + checkpoint_wrapper after prepare"
        fi
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
  scripts/train_jepa_s2.py
  --config "$CONFIG"
  --max-length "$MAX_LENGTH"
  --cp-size "$CP_SIZE"
  --run-tag "$RUN_TAG"
  "${EXTRA[@]}"
)

echo "  launch: ${LAUNCH[*]} ${TRAIN_PY[0]} --max-length $MAX_LENGTH --cp-size $CP_SIZE --run-tag $RUN_TAG"
"${LAUNCH[@]}" "${TRAIN_PY[@]}"
echo "Done."
