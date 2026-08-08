#!/usr/bin/env bash
# Multi-GPU QLoRA SFT (ms-swift) for Qwen3-Coder-Next.
# Default: single-process device_map (split 4bit weights across visible GPUs at load).
# DeepSpeed DDP+QLoRA materializes ~40GB+ on rank0 during load → OOM on 48GB cards.
#
# Requires explicit --max-length. Before training:
#   1) structure-preserving right trunc (keep prefix ending on complete assistant)
#   2) print survivors (+ delete-ref for comparison)
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
MIX_DIR="${MIX_DIR:-data/processed/mix_v2}"

MAX_LENGTH=""
CHOICE="${TRAIN_CHOICE:-}"
FORCE_PREP=0
EXTRA=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_coder_next.sh --max-length <N> [--choice 1|2|3] [--force-prep]

  --max-length N   required; struct-right trunc to complete assistant within N
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
PARALLEL="${PARALLEL:-device_map}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_MEMORY="${MAX_MEMORY:-}"
DEEPSPEED="${DEEPSPEED:-zero3}"

# device_map: one process owns all visible GPUs (model-parallel load).
# deepspeed: one process per GPU (only viable if each card can hold full QLoRA weights).
if [[ "$PARALLEL" == "device_map" ]]; then
  # Must force: a stale NPROC_PER_NODE=2 from the env reintroduces rank0 full-load OOM.
  if [[ -n "${NPROC_PER_NODE:-}" && "$NPROC_PER_NODE" != "1" ]]; then
    echo "NOTE: parallel=device_map forces NPROC_PER_NODE=1 (was $NPROC_PER_NODE)."
  fi
  NPROC_PER_NODE=1
else
  if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    NPROC_PER_NODE="$(python - <<'PY'
import os
xs=[x for x in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if x.strip()]
print(len(xs) if xs else 1)
PY
)"
  fi
fi
export NPROC_PER_NODE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA_PARALLEL_ARGS=()
if [[ "$PARALLEL" == "device_map" ]]; then
  # L20 ~46GB can hold the whole ~40–45GB 4bit checkpoint alone, so device_map=auto
  # without a cap parks everything on GPU0. Cap each GPU below full-model size to force
  # a split, but keep the sum >> model size so nothing spills to CPU/disk (bnb forbids that
  # unless llm_int8_enable_fp32_cpu_offload). 28GiB×2 was too tight → CPU spill ValueError.
  # Override: BIV_MAX_MEMORY_PER_GPU=40GiB | none (skip --max_memory).
  if [[ -z "$MAX_MEMORY" ]]; then
    _per="${BIV_MAX_MEMORY_PER_GPU:-40GiB}"
    if [[ "$_per" != "none" && "$_per" != "skip" && "$_per" != "0" ]]; then
      MAX_MEMORY="$(
        BIV_MAX_MEMORY_PER_GPU="$_per" python - <<'PY'
import os
xs = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
n = len(xs) if xs else 1
per = os.environ["BIV_MAX_MEMORY_PER_GPU"]
# Accelerate wants int keys (0,1,…), not JSON "0"/"1".
print("{" + ", ".join(f'{i}: "{per}"' for i in range(n)) + "}")
PY
      )"
    fi
  fi
  EXTRA_PARALLEL_ARGS+=(--device_map "$DEVICE_MAP")
  if [[ -n "$MAX_MEMORY" ]]; then
    EXTRA_PARALLEL_ARGS+=(--max_memory "$MAX_MEMORY")
  fi
elif [[ "$PARALLEL" == "deepspeed" ]]; then
  EXTRA_PARALLEL_ARGS+=(--deepspeed "$DEEPSPEED")
else
  echo "ERROR: unknown PARALLEL=$PARALLEL (use device_map or deepspeed)"
  exit 1
fi

# REQUIRED: FlashAttention only (no sdpa/eager fallback). Long-context memory depends on it.
# Hub download of kernels-community/flash-attn2 is NOT enough — need Python package flash_attn
# (transformers checks import flash_attn before / besides hub kernels).
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

# Path 1 (what transformers flash_attn path needs): pip package
fa_ok = False
try:
    import flash_attn
    from flash_attn import flash_attn_func  # noqa: F401
    fa_ok = True
    print(f"  flash_attn package OK: {getattr(flash_attn, '__version__', '?')}")
except Exception as e:
    print(f"  flash_attn package MISSING/broken: {e}")

# Path 2: hub kernels (optional complement; alone often NOT enough for transformers)
ker_ok = False
try:
    from kernels import get_kernel
    import kernels
    print(f"  kernels={getattr(kernels, '__version__', '?')}")
    k = get_kernel("kernels-community/flash-attn2")
    if getattr(k, "flash_attn_func", None) is None and getattr(k, "flash_attn_varlen_func", None) is None:
        print("  hub kernel loaded but missing flash_attn_func")
    else:
        ker_ok = True
        print("  hub kernel kernels-community/flash-attn2 OK")
except Exception as e:
    print(f"  hub kernel load failed: {e}")

if not fa_ok:
    fail(
        "\nERROR: FlashAttention is REQUIRED and the Python package is not installed.\n"
        "  `hf download kernels-community/flash-attn2` only fills the HF cache;\n"
        "  it does NOT register the `flash_attn` module that transformers checks.\n\n"
        "  Install on this GPU venv:\n"
        "    pip install -U packaging ninja\n"
        "    pip install flash-attn --no-build-isolation\n\n"
        "  Then verify:\n"
        "    python -c \"import flash_attn; print(flash_attn.__version__)\"\n"
        "  Refusing to start without FA (no sdpa fallback).\n"
    )
if not ker_ok:
    print(
        "  note: hub kernel not loadable yet; proceeding with flash_attn package only.",
        flush=True,
    )
print("  FlashAttention check passed.", flush=True)
PY
then
  exit 1
fi

echo "  GPUs=$CUDA_VISIBLE_DEVICES NPROC_PER_NODE=$NPROC_PER_NODE"
echo "  parallel=$PARALLEL attn_impl=$ATTN_IMPL"
if [[ "$PARALLEL" == "device_map" ]]; then
  echo "  device_map=$DEVICE_MAP max_memory=${MAX_MEMORY:-"(unset)"}"
else
  echo "  deepspeed=$DEEPSPEED"
fi

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
  "${EXTRA_PARALLEL_ARGS[@]}" \
  --attn_impl "$ATTN_IMPL" \
  --dataloader_num_workers 4
