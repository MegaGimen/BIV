#!/usr/bin/env bash
# Forward-only collapse probe on the 32k/2x2 layout. No backward, no LoRA.
# Default 200 rows = 25 optimizer steps × grad_accum 8 on one replica group.
#
#   cd train
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash scripts/probe_jepallm_collapse.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/jepa/jepallm_32k.yaml}"
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

  bash scripts/probe_jepallm_collapse.sh
  bash scripts/probe_jepallm_collapse.sh --max-rows 40
  bash scripts/probe_jepallm_collapse.sh --close-threshold 0.85

Writes outputs/jepallm32k_collapse_probe/collapse_probe-<stamp>.json
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

if [[ "$NGPU" -ne 4 ]]; then
  echo "WARNING: 2x2 probe expects 4 GPUs (got $NGPU). Launching with cp_size=$NGPU, no dp_replicate."
  CP_SIZE="$NGPU"
  ACCEL_CFG="${ACCELERATE_CONFIG:-configs/accelerate/qwen35_moe_fsdp2_cp.yaml}"
  export PARALLELISM_CONFIG_DP_REPLICATE_SIZE=1
  export PARALLELISM_CONFIG_CP_SIZE="$CP_SIZE"
else
  CP_SIZE=2
  ACCEL_CFG="${ACCELERATE_CONFIG:-configs/accelerate/qwen35_moe_fsdp2_cp2x2.yaml}"
  export ACCELERATE_USE_PARALLELISM_CONFIG=true
  export PARALLELISM_CONFIG_DP_REPLICATE_SIZE=2
  export PARALLELISM_CONFIG_DP_SHARD_SIZE=1
  export PARALLELISM_CONFIG_TP_SIZE=1
  export PARALLELISM_CONFIG_CP_SIZE="$CP_SIZE"
  export PARALLELISM_CONFIG_CP_BACKEND=torch
  export BIV_CP_SIZE="$CP_SIZE"
fi

echo "  probe FSDP2+CP cp_size=$CP_SIZE max_rows=$MAX_ROWS max_length=$MAX_LENGTH"
accelerate launch \
  --config_file "$ACCEL_CFG" \
  --num_processes "$NGPU" \
  --mixed_precision bf16 \
  --use_fsdp \
  --fsdp_version 2 \
  --use_parallelism_config \
  --fsdp_transformer_layer_cls_to_wrap Qwen3_5MoeDecoderLayer \
  --fsdp_activation_checkpointing false \
  scripts/probe_jepallm_collapse.py \
  --config "$CONFIG" \
  --max-length "$MAX_LENGTH" \
  --cp-size "$CP_SIZE" \
  --max-rows "$MAX_ROWS" \
  "${EXTRA[@]}"
echo "Done."
