#!/usr/bin/env bash
# Stage 1 LLM-JEPA, 32768-length smoke: 4 GPUs as 2 groups of 2, each group
# doing Context Parallel (cp_size=2) on different data (dp_replicate_size=2).
# Roughly 2x throughput vs train_jepallm.sh's 65536/4-way-CP run, at the cost
# of truncating longer rows. Separate config/output_dir/TensorBoard tag from
# train_jepallm.sh so neither run overwrites the other.
#
#   cd train
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash scripts/train_jepallm_32k.sh
#   bash scripts/train_jepallm_32k.sh --save-steps 1 --max-steps 2
#   bash scripts/train_jepallm_32k.sh --resume
#
# Needs exactly 4 GPUs (2x2). Fewer/more: fall back to train_jepallm.sh with
# --max-length 32768 (single 4-way CP group, no dp_replicate speedup).
#
# accelerate's ParallelismConfig refuses dp_replicate>1 + cp>1 with
# dp_shard_size==1 out of the box (raises "pure data parallelism... cannot be
# used with... context parallelism"). train_jepallm.py's build_parallelism_config()
# routes around that — see its docstring for why it's safe here. If you hit
# that ValueError again, it means that workaround isn't being reached (e.g.
# PARALLELISM_CONFIG_DP_REPLICATE_SIZE not set before Accelerator() runs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/jepa/jepallm_32k.yaml}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
RUN_TAG="${RUN_TAG:-jepallm32k}"
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
Stage 1 LLM-JEPA, 32768 smoke: 4 GPUs as 2x2 (dp_replicate=2, cp=2 each).

  cd train
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  bash scripts/train_jepallm_32k.sh
  bash scripts/train_jepallm_32k.sh --save-steps 1 --max-steps 2
  bash scripts/train_jepallm_32k.sh --resume
  bash scripts/train_jepallm_32k.sh --resume outputs/jepallm32k_stage1/checkpoint-e0-s25

Separate from train_jepallm.sh: config=configs/jepa/jepallm_32k.yaml,
output_dir=outputs/jepallm32k_stage1, TensorBoard tags=jepallm32k-g0-<stamp> and
jepallm32k-g1-<stamp> (one run per dp_replicate group, same shared stamp —
`tensorboard --logdir /root/tf-logs` shows both curves together).
Needs exactly 4 GPUs; falls back to a single 4-way CP group (no dp_replicate
speedup) otherwise.

     --save-steps N       (default yaml 25; 1 smokes FSDP save)
     --log-steps N        (default yaml 5; loss + collapse, not save)
     --resume             newest complete ckpt under output_dir
     --resume PATH / --resume-from PATH
     --max-steps N
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

if [[ "$NGPU" -ne 4 ]]; then
  echo "WARNING: 2x2 needs exactly 4 GPUs (got $NGPU). Falling back to a single"
  echo "         ${NGPU}-way CP group (no dp_replicate speedup)."
  export PARALLEL=fsdp2_cp
  exec bash "$ROOT/scripts/train_jepallm.sh" --config "$CONFIG" --max-length "$MAX_LENGTH" "${EXTRA[@]}"
fi

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

TRAIN_PY=(
  scripts/train_jepallm.py
  --config "$CONFIG"
  --max-length "$MAX_LENGTH"
  --cp-size "$CP_SIZE"
  --run-tag "$RUN_TAG"
  "${EXTRA[@]}"
)

echo "  launch: ${LAUNCH[*]} ${TRAIN_PY[0]} --max-length $MAX_LENGTH --cp-size $CP_SIZE --run-tag $RUN_TAG"
"${LAUNCH[@]}" "${TRAIN_PY[@]}"
echo "Done."
