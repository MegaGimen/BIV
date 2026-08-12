#!/usr/bin/env bash
# QLoRA SFT (ms-swift) for Kimi-Dev-72B (dense Qwen2.5-72B). Branch: */msswift
# Auto parallel: 1 visible GPU → single; 2+ → sequence parallel (Ulysses+Ring).
# FSDP remains available via PARALLEL=fsdp (weight shard; not default).
#
# Requires explicit --max-length. Before training:
#   1) structure-preserving right trunc (keep prefix ending on complete assistant)
#   2) print survivors (+ delete-ref for comparison)
#   3) ask: 1=as-is / 2=rebalance 1:1:0.35 / 3=abort
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0          # single ~96GB
#   bash scripts/trainmodel.sh --max-length 8192 --choice 2
#   export CUDA_VISIBLE_DEVICES=0,1,2,3    # multi → SP (long-context)
#   bash scripts/trainmodel.sh --max-length 32768 --choice 1
#   PARALLEL=fsdp bash scripts/trainmodel.sh --max-length 32768 --choice 1
#   # experimental FSDP2 + SP (≈ Axolotl CP); prefer 4 GPUs, SP=2:
#   PARALLEL=sp_fsdp SEQUENCE_PARALLEL_SIZE=2 bash scripts/trainmodel.sh --max-length 32768 --choice 1
#
# Other overrides:
#   PARALLEL=single|sp|fsdp|sp_fsdp|deepspeed|device_map  CONFIG=...  MIX_DIR=...
#   SEQUENCE_PARALLEL_SIZE=N  (sp: default=NGPU; sp_fsdp: default=2 when NGPU>=4)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CONFIG="${CONFIG:-configs/swift/kimi_dev_72b_qlora.yaml}"
MIX_DIR="${MIX_DIR:-data/processed/mix_v2}"

MAX_LENGTH=""
CHOICE="${TRAIN_CHOICE:-}"
FORCE_PREP=0
EXTRA=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/trainmodel.sh --max-length <N> [--choice 1|2|3] [--force-prep]

  --max-length N   required; struct-right trunc to complete assistant within N
  --choice K       skip interactive prompt (1=as-is, 2=1:1:0.35, 3=abort)
  --force-prep     rebuild filtered run cache even if present

Env:
  CUDA_VISIBLE_DEVICES     default 0 (use 0,1,2,3 for 4-GPU)
  PARALLEL                 omit/auto | single | sp | fsdp | sp_fsdp | deepspeed | device_map
  SEQUENCE_PARALLEL_SIZE   sp: default=NGPU; sp_fsdp: default=2 if NGPU>=4
  TRAIN_CHOICE             same as --choice
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

# prep env may blank PARALLEL / FSDP_CONFIG / SEQUENCE_PARALLEL_SIZE — keep caller exports.
_SAVE_PARALLEL="${PARALLEL-}"
_SAVE_FSDP_CONFIG="${FSDP_CONFIG-}"
_SAVE_SEQUENCE_PARALLEL_SIZE="${SEQUENCE_PARALLEL_SIZE-}"
# shellcheck disable=SC1090
source "$RUN_ENV"
if [[ -n "${_SAVE_PARALLEL}" ]]; then PARALLEL="${_SAVE_PARALLEL}"; fi
if [[ -n "${_SAVE_FSDP_CONFIG}" ]]; then FSDP_CONFIG="${_SAVE_FSDP_CONFIG}"; fi
if [[ -n "${_SAVE_SEQUENCE_PARALLEL_SIZE}" ]]; then SEQUENCE_PARALLEL_SIZE="${_SAVE_SEQUENCE_PARALLEL_SIZE}"; fi

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

# Default visible devices: single GPU. Multi-GPU users must export CUDA_VISIBLE_DEVICES=0,1,...
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# PARALLEL from env/config; empty → auto: 1 GPU → single, 2+ → sp
PARALLEL="${PARALLEL:-}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_MEMORY="${MAX_MEMORY:-}"
DEEPSPEED="${DEEPSPEED:-zero3}"
FSDP_CONFIG="${FSDP_CONFIG:-configs/swift/fsdp_qlora_kimi_dev_72b.json}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NGPU="$(python - <<'PY'
import os
xs=[x for x in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if x.strip()]
print(len(xs) if xs else 1)
PY
)"

if [[ -z "$PARALLEL" ]]; then
  if [[ "$NGPU" -le 1 ]]; then
    PARALLEL=single
  else
    PARALLEL=sp
  fi
  echo "  auto PARALLEL=$PARALLEL (ngpu=$NGPU)"
elif [[ "$NGPU" -le 1 && ( "$PARALLEL" == "fsdp" || "$PARALLEL" == "sp" || "$PARALLEL" == "sp_fsdp" ) && "${PARALLEL_FORCE:-}" != "1" ]]; then
  echo "NOTE: ngpu=1 → using PARALLEL=single (set PARALLEL_FORCE=1 to keep $PARALLEL)"
  PARALLEL=single
fi

# single: plain 1-process QLoRA on one GPU (best for ~80–96GB cards).
# sp: DDP + ms-swift sequence_parallel_size (Ulysses+Ring) — default multi-GPU for long context.
# fsdp: accelerate FSDP2 + QLoRA (weight shard; no SP).
# sp_fsdp: EXPERIMENTAL FSDP2 + sequence_parallel (msswift SP ≈ Axolotl CP). Prefer 4 GPUs
#          with SEQUENCE_PARALLEL_SIZE=2. Not an official ms-swift cookbook combo.
# device_map: legacy model-parallel (bnb CPU offload often breaks at step 0).
# deepspeed: DDP+ZeRO (QLoRA often OOMs on rank0 during load on 48GB).
EXTRA_PARALLEL_ARGS=()
LAUNCH=()
unset BIV_BNB_CPU_OFFLOAD

_fsdp_common() {
  # Shared FSDP2 launch + QLoRA args (GPU load, no device_map).
  export ACCELERATE_USE_FSDP=true
  export FSDP_VERSION=2
  if [[ ! -f "$FSDP_CONFIG" ]]; then
    echo "ERROR: FSDP config not found: $FSDP_CONFIG"
    exit 1
  fi
  FSDP_RUN_CONFIG="$(mktemp "${TMPDIR:-/tmp}/biv_fsdp_XXXXXX.json")"
  cleanup_fsdp() { rm -f "$FSDP_RUN_CONFIG"; }
  trap cleanup_fsdp EXIT
  python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("$FSDP_CONFIG").read_text())
cfg["num_processes"] = int("$NGPU")
cfg.pop("_comment", None)
fc = cfg.get("fsdp_config") or {}
ver = fc.get("fsdp_version", cfg.get("fsdp_version", 2))
print(f"  wrote FSDP config num_processes={cfg['num_processes']} fsdp_version={ver} → $FSDP_RUN_CONFIG")
Path("$FSDP_RUN_CONFIG").write_text(json.dumps(cfg, indent=2) + "\n")
PY
  EXTRA_PARALLEL_ARGS+=(
    --bnb_4bit_quant_storage bfloat16
    --bnb_4bit_compute_dtype bfloat16
    --optim adamw_torch_8bit
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  )
  LAUNCH=(accelerate launch --config_file "$FSDP_RUN_CONFIG")
}

if [[ "$PARALLEL" == "single" ]]; then
  unset NPROC_PER_NODE
  unset NNODES
  unset ACCELERATE_USE_FSDP
  # Let ms-swift place the model on the only visible CUDA device; no FSDP/DeepSpeed.
  EXTRA_PARALLEL_ARGS+=(
    --bnb_4bit_compute_dtype bfloat16
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  )
  NPROC_PER_NODE="(single process)"
  echo "  single-GPU QLoRA (no FSDP/DeepSpeed/SP)"
elif [[ "$PARALLEL" == "sp" ]]; then
  if [[ "$NGPU" -le 1 ]]; then
    echo "WARNING: PARALLEL=sp with 1 GPU is a no-op; prefer PARALLEL=single."
  fi
  unset ACCELERATE_USE_FSDP
  unset NNODES
  SP_SIZE="${SEQUENCE_PARALLEL_SIZE:-$NGPU}"
  if ! [[ "$SP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SEQUENCE_PARALLEL_SIZE must be a positive integer, got: $SP_SIZE"
    exit 1
  fi
  if [[ "$SP_SIZE" -gt "$NGPU" ]]; then
    echo "ERROR: SEQUENCE_PARALLEL_SIZE=$SP_SIZE > visible GPUs ($NGPU)"
    exit 1
  fi
  # Launch via torchrun so ranks actually start; keep bnb patches in run_swift_sft.py.
  unset NPROC_PER_NODE
  MASTER_PORT="${MASTER_PORT:-29501}"
  LAUNCH=(
    torchrun
    --nproc_per_node="$NGPU"
    --master_port="$MASTER_PORT"
  )
  EXTRA_PARALLEL_ARGS+=(
    --sequence_parallel_size "$SP_SIZE"
    --padding_free true
    --bnb_4bit_compute_dtype bfloat16
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  )
  NPROC_PER_NODE="(torchrun nproc=$NGPU)"
  echo "  sequence parallel: torchrun nproc=$NGPU sequence_parallel_size=$SP_SIZE port=$MASTER_PORT"
elif [[ "$PARALLEL" == "fsdp" ]]; then
  if [[ "$NGPU" -le 1 ]]; then
    echo "WARNING: PARALLEL=fsdp with 1 GPU is unnecessary; prefer PARALLEL=single on 96GB."
  fi
  unset NPROC_PER_NODE
  unset NNODES
  _fsdp_common
  NPROC_PER_NODE="(accelerate num_processes=$NGPU)"
  echo "  ACCELERATE_USE_FSDP=$ACCELERATE_USE_FSDP FSDP_VERSION=$FSDP_VERSION (FSDP2 only, GPU load, offload=false)"
elif [[ "$PARALLEL" == "sp_fsdp" ]]; then
  if [[ "$NGPU" -lt 2 ]]; then
    echo "ERROR: PARALLEL=sp_fsdp needs >=2 GPUs (prefer 4 with SEQUENCE_PARALLEL_SIZE=2)"
    exit 1
  fi
  unset NPROC_PER_NODE
  unset NNODES
  # Prefer SP=2 on 4+ even GPUs (Axolotl-like CP=2); override with SEQUENCE_PARALLEL_SIZE.
  if [[ -n "${SEQUENCE_PARALLEL_SIZE:-}" ]]; then
    SP_SIZE="$SEQUENCE_PARALLEL_SIZE"
  elif [[ "$NGPU" -ge 4 && $((NGPU % 2)) -eq 0 ]]; then
    SP_SIZE=2
  else
    SP_SIZE="$NGPU"
  fi
  if ! [[ "$SP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SEQUENCE_PARALLEL_SIZE must be a positive integer, got: $SP_SIZE"
    exit 1
  fi
  if [[ "$SP_SIZE" -gt "$NGPU" ]]; then
    echo "ERROR: SEQUENCE_PARALLEL_SIZE=$SP_SIZE > visible GPUs ($NGPU)"
    exit 1
  fi
  if [[ $((NGPU % SP_SIZE)) -ne 0 ]]; then
    echo "ERROR: NGPU=$NGPU not divisible by SEQUENCE_PARALLEL_SIZE=$SP_SIZE"
    exit 1
  fi
  _fsdp_common
  EXTRA_PARALLEL_ARGS+=(
    --sequence_parallel_size "$SP_SIZE"
    --padding_free true
  )
  NPROC_PER_NODE="(accelerate+sp nproc=$NGPU sp=$SP_SIZE)"
  echo "  EXPERIMENTAL sp_fsdp: FSDP2 num_processes=$NGPU + sequence_parallel_size=$SP_SIZE (not cookbook-backed)"
elif [[ "$PARALLEL" == "device_map" ]]; then
  echo "WARNING: parallel=device_map is legacy; bnb+CPU offload often fails at first train step."
  if [[ -n "${NPROC_PER_NODE:-}" || -n "${NNODES:-}" ]]; then
    echo "NOTE: unsetting NPROC_PER_NODE/NNODES for device_map."
  fi
  unset NPROC_PER_NODE
  unset NNODES
  export BIV_BNB_CPU_OFFLOAD=1
  if [[ -z "$MAX_MEMORY" ]]; then
    _per="${BIV_MAX_MEMORY_PER_GPU:-44GiB}"
    _cpu="${BIV_CPU_MEMORY:-200GiB}"
    MAX_MEMORY="$(
      BIV_MAX_MEMORY_PER_GPU="$_per" BIV_CPU_MEMORY="$_cpu" python - <<'PY'
import os
xs = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
n = len(xs) if xs else 1
per = os.environ["BIV_MAX_MEMORY_PER_GPU"]
cpu = os.environ["BIV_CPU_MEMORY"]
parts = [f'{i}: "{per}"' for i in range(n)]
parts.append(f'"cpu": "{cpu}"')
print("{" + ", ".join(parts) + "}")
PY
    )"
  fi
  EXTRA_PARALLEL_ARGS+=(--device_map "$DEVICE_MAP")
  if [[ -n "$MAX_MEMORY" ]]; then
    EXTRA_PARALLEL_ARGS+=(--max_memory "$MAX_MEMORY")
  fi
  NPROC_PER_NODE="(unset — no torchrun)"
elif [[ "$PARALLEL" == "deepspeed" ]]; then
  NPROC_PER_NODE="${NPROC_PER_NODE:-$NGPU}"
  export NPROC_PER_NODE
  EXTRA_PARALLEL_ARGS+=(--deepspeed "$DEEPSPEED")
else
  echo "ERROR: unknown PARALLEL=$PARALLEL (use single, sp, fsdp, sp_fsdp, device_map, or deepspeed)"
  exit 1
fi

# REQUIRED: FlashAttention only (no sdpa/eager fallback). Long-context memory depends on it.
# Hub download of kernels-community/flash-attn2 is NOT enough — need Python package flash_attn.
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
if [[ "$ATTN_IMPL" != "flash_attn" && "$ATTN_IMPL" != "flash_attention_2" && "$ATTN_IMPL" != "kernels-community/flash-attn2" ]]; then
  echo "ERROR: Only FlashAttention is allowed (got ATTN_IMPL=$ATTN_IMPL)."
  echo "  Unset ATTN_IMPL or set ATTN_IMPL=flash_attn"
  exit 1
fi

echo "Checking FlashAttention (hard requirement)…"
if ! python - <<'PY'
import sys

def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)

try:
    import torch
except Exception as e:
    fail(f"torch import failed: {e}")

print(f"  torch={torch.__version__} cuda={torch.version.cuda}")

fa_ok = False
try:
    import flash_attn
    from flash_attn import flash_attn_func  # noqa: F401
    fa_ok = True
    print(f"  flash_attn package OK: {getattr(flash_attn, '__version__', '?')}")
except Exception as e:
    print(f"  flash_attn package MISSING/broken: {e}")

if not fa_ok:
    fail(
        "ERROR: FlashAttention is REQUIRED and the Python package is not installed. "
        "Install: pip install flash-attn --no-build-isolation. "
        "Refusing to start without FA (no sdpa fallback)."
    )
print("  FlashAttention check passed.", flush=True)
PY
then
  exit 1
fi



echo "  GPUs=$CUDA_VISIBLE_DEVICES NPROC_PER_NODE=$NPROC_PER_NODE"
echo "  parallel=$PARALLEL attn_impl=$ATTN_IMPL"
case "$PARALLEL" in
  single) echo "  mode=single-GPU QLoRA" ;;
  sp) echo "  mode=sequence_parallel size=${SEQUENCE_PARALLEL_SIZE:-$NGPU}" ;;
  fsdp) echo "  fsdp_config=$FSDP_CONFIG (runtime=$FSDP_RUN_CONFIG)" ;;
  device_map) echo "  device_map=$DEVICE_MAP max_memory=${MAX_MEMORY:-"(unset)"}" ;;
  deepspeed) echo "  deepspeed=$DEEPSPEED" ;;
esac

# Visible CUDA devices (catch "CUDA_VISIBLE_DEVICES=0,1" without export)
python - <<'PY'
import os, torch
print(
    f"  torch.cuda.device_count={torch.cuda.device_count()} "
    f"visible={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}",
    flush=True,
)
if torch.cuda.device_count() < 1:
    raise SystemExit("ERROR: no CUDA devices visible to torch")
PY

# Patched entry (bnb fixes). FSDP → accelerate; SP → torchrun; else plain python.
SFT_ENTRY=(python "$ROOT/scripts/run_swift_sft.py")
if [[ "$PARALLEL" == "deepspeed" ]]; then
  SFT_ENTRY=(swift sft)
elif [[ "$PARALLEL" == "fsdp" || "$PARALLEL" == "sp" ]]; then
  SFT_ENTRY=("${LAUNCH[@]}" "$ROOT/scripts/run_swift_sft.py")
fi
# single / device_map: plain python run_swift_sft.py (already set)

EXTRA_MODEL_ARGS=()
if [[ -n "${TEMPLATE:-}" ]]; then
  EXTRA_MODEL_ARGS+=(--template "$TEMPLATE")
fi
if [[ -n "${MODEL_TYPE:-}" ]]; then
  EXTRA_MODEL_ARGS+=(--model_type "$MODEL_TYPE")
fi

# shellcheck disable=SC2086
exec "${SFT_ENTRY[@]}" \
  --model "$MODEL" \
  --tuner_type lora \
  --quant_method bnb \
  --quant_bits 4 \
  --bnb_4bit_use_double_quant false \
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
  "${EXTRA_PARALLEL_ARGS[@]}" \
  "${EXTRA_MODEL_ARGS[@]}" \
  --attn_impl "$ATTN_IMPL" \
  --dataloader_num_workers 4
