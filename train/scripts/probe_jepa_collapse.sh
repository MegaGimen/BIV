#!/usr/bin/env bash
# Forward-only collapse probe. No backward, no LoRA.
# Default 200 rows = 25 optimizer steps × grad_accum 8 on one replica group.
#
#   cd train
#   CUDA_VISIBLE_DEVICES=0 bash scripts/probe_jepa_collapse.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/probe_jepa_collapse.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/jepa/stage1.yaml}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
MAX_ROWS="${MAX_ROWS:-200}"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-rows|--close-threshold|--max-pairs|--out-dir|--config|--max-length|--model-dir|--mix-dir)
      [[ $# -ge 2 ]] || { echo "missing value for $1"; exit 1; }
      if [[ "$1" == "--config" ]]; then CONFIG="$2"
      elif [[ "$1" == "--max-length" ]]; then MAX_LENGTH="$2"
      elif [[ "$1" == "--max-rows" ]]; then MAX_ROWS="$2"
      else EXTRA+=("$1" "$2"); fi
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Forward-only JEPA collapse probe (no training).

  CUDA_VISIBLE_DEVICES=0 bash scripts/probe_jepa_collapse.sh
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/probe_jepa_collapse.sh
  bash scripts/probe_jepa_collapse.sh --max-rows 40
  bash scripts/probe_jepa_collapse.sh --close-threshold 0.85

1 GPU: plain python (no accelerate mesh). 2–3 GPUs: FSDP2+CP.
4 GPUs: same 2x2 as train_jepa.sh.
Writes outputs/jepa_collapse_probe/collapse_probe-<stamp>.json
EOF
      exit 0
      ;;
    *)
      echo "unknown arg: $1"
      exit 1
      ;;
  esac
done

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

# Leftover from a 4-GPU train: yaml/env still ask for a 4-rank CP mesh.
unset ACCELERATE_USE_PARALLELISM_CONFIG || true
unset PARALLELISM_CONFIG_DP_REPLICATE_SIZE || true
unset PARALLELISM_CONFIG_DP_SHARD_SIZE || true
unset PARALLELISM_CONFIG_TP_SIZE || true
unset PARALLELISM_CONFIG_CP_SIZE || true
unset PARALLELISM_CONFIG_CP_BACKEND || true
unset BIV_CP_SIZE || true
unset BIV_PARALLEL || true

TRAIN_PY=(
  scripts/probe_jepa_collapse.py
  --config "$CONFIG"
  --max-length "$MAX_LENGTH"
  --max-rows "$MAX_ROWS"
)

if [[ "$NGPU" -le 1 ]]; then
  echo "  single-GPU probe (no accelerate / no CP) max_rows=$MAX_ROWS max_length=$MAX_LENGTH"
  python "${TRAIN_PY[@]}" --cp-size 1 "${EXTRA[@]}"
elif [[ "$NGPU" -eq 4 ]]; then
  CP_SIZE=2
  ACCEL_CFG="${ACCELERATE_CONFIG:-configs/accelerate/qwen35_moe_fsdp2_cp2x2.yaml}"
  export ACCELERATE_USE_PARALLELISM_CONFIG=true
  export PARALLELISM_CONFIG_DP_REPLICATE_SIZE=2
  export PARALLELISM_CONFIG_DP_SHARD_SIZE=1
  export PARALLELISM_CONFIG_TP_SIZE=1
  export PARALLELISM_CONFIG_CP_SIZE="$CP_SIZE"
  export PARALLELISM_CONFIG_CP_BACKEND=torch
  export BIV_CP_SIZE="$CP_SIZE"
  echo "  probe FSDP2+CP 2x2 cp_size=$CP_SIZE max_rows=$MAX_ROWS max_length=$MAX_LENGTH"
  accelerate launch \
    --config_file "$ACCEL_CFG" \
    --num_processes "$NGPU" \
    --mixed_precision bf16 \
    --use_fsdp \
    --fsdp_version 2 \
    --use_parallelism_config \
    --fsdp_transformer_layer_cls_to_wrap Qwen3_5MoeDecoderLayer \
    --fsdp_activation_checkpointing false \
    --parallelism_config_dp_replicate_size 2 \
    --parallelism_config_dp_shard_size 1 \
    --parallelism_config_tp_size 1 \
    --parallelism_config_cp_size "$CP_SIZE" \
    --parallelism_config_cp_backend torch \
    "${TRAIN_PY[@]}" --cp-size "$CP_SIZE" "${EXTRA[@]}"
else
  CP_SIZE="$NGPU"
  ACCEL_CFG="${ACCELERATE_CONFIG:-configs/accelerate/qwen35_moe_fsdp2_cp.yaml}"
  export ACCELERATE_USE_PARALLELISM_CONFIG=true
  export PARALLELISM_CONFIG_DP_REPLICATE_SIZE=1
  export PARALLELISM_CONFIG_DP_SHARD_SIZE=1
  export PARALLELISM_CONFIG_TP_SIZE=1
  export PARALLELISM_CONFIG_CP_SIZE="$CP_SIZE"
  export PARALLELISM_CONFIG_CP_BACKEND=torch
  export BIV_CP_SIZE="$CP_SIZE"
  echo "  probe FSDP2+CP cp_size=$CP_SIZE (yaml defaults to 4; CLI overrides) max_rows=$MAX_ROWS"
  accelerate launch \
    --config_file "$ACCEL_CFG" \
    --num_processes "$NGPU" \
    --mixed_precision bf16 \
    --use_fsdp \
    --fsdp_version 2 \
    --use_parallelism_config \
    --fsdp_transformer_layer_cls_to_wrap Qwen3_5MoeDecoderLayer \
    --fsdp_activation_checkpointing false \
    --parallelism_config_dp_replicate_size 1 \
    --parallelism_config_dp_shard_size 1 \
    --parallelism_config_tp_size 1 \
    --parallelism_config_cp_size "$CP_SIZE" \
    --parallelism_config_cp_backend torch \
    "${TRAIN_PY[@]}" --cp-size "$CP_SIZE" "${EXTRA[@]}"
fi
echo "Done."
