#!/usr/bin/env bash
# Dual-GPU QLoRA SFT (ms-swift) for Qwen3-Coder-Next on ratio-sampled mix.
# Expects: python scripts/tokenize_data.py  (+ optional: python scripts/stat.py)
#
# Env overrides:
#   MAX_LENGTH=16384 TRUNCATION_STRATEGY=right CUDA_VISIBLE_DEVICES=0,1
#   CONFIG=configs/swift/coder_next_qlora.yaml
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CONFIG="${CONFIG:-configs/swift/coder_next_qlora.yaml}"
MIX_DIR="${MIX_DIR:-data/processed/mix_v1}"

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

eval "$(python - <<'PY'
import json, os, shlex
from pathlib import Path
import yaml

ROOT = Path(".").resolve()
config_path = Path(os.environ.get("CONFIG", "configs/swift/coder_next_qlora.yaml"))
cfg = yaml.safe_load(config_path.read_text())
cache_root = Path(cfg.get("cache_root", "outputs/swift_cache/coder_next_mix_v1"))
if not cache_root.is_absolute():
    cache_root = ROOT / cache_root
latest_path = cache_root / "LATEST"
if not latest_path.is_file():
    raise SystemExit(f"Missing {latest_path} — run tokenize_data.py")
latest = latest_path.read_text(encoding="utf-8").strip()
manifest = cache_root / latest / "tokenize_manifest.json"
m = json.loads(manifest.read_text(encoding="utf-8"))
train = cfg.get("train") or {}
tg = train.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]
cached = []
for _name, rel in m["cached_train"].items():
    p = Path(rel)
    cached.append(str(p if p.is_absolute() else ROOT / p))

def exp(k, v):
    print(f"export {k}=" + shlex.quote(str(v)))

exp("TAG", latest)
exp("MANIFEST", manifest)
exp("MODEL", m.get("model") or cfg.get("model"))
exp("MAX_LENGTH", os.environ.get("MAX_LENGTH", train.get("max_length", 16384)))
exp(
    "TRUNC",
    os.environ.get("TRUNCATION_STRATEGY", train.get("truncation_strategy", "right")),
)
exp("LR", train.get("learning_rate", 1e-4))
exp("EPOCHS", train.get("num_epochs", 2))
exp("OUT_DIR", train.get("output_dir", "outputs/swift_coder_next_wm_mix"))
exp("LORA_RANK", train.get("lora_rank", 16))
exp("LORA_ALPHA", train.get("lora_alpha", 16))
exp("BS", train.get("per_device_train_batch_size", 1))
exp("GAS", train.get("gradient_accumulation_steps", 8))
exp("DEEPSPEED", train.get("deepspeed", "zero2"))
exp("TARGET_MODULES", " ".join(tg))
exp("CACHED_DATASETS", " ".join(cached))
exp("DTYPE", train.get("torch_dtype", "bfloat16"))
exp("WARMUP", train.get("warmup_ratio", 0.03))
exp("LOG_STEPS", train.get("logging_steps", 10))
exp("SAVE_STEPS", train.get("save_steps", 200))
exp("SAVE_LIMIT", train.get("save_total_limit", 3))
PY
)"

echo "=== ms-swift train ==="
echo "  tag=$TAG"
echo "  manifest=$MANIFEST"
echo "  model=$MODEL"
echo "  max_length=$MAX_LENGTH truncation=$TRUNC"
echo "  cached_dataset:"
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
