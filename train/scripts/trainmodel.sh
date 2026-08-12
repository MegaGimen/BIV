#!/usr/bin/env bash
# QLoRA SFT (Axolotl) with FSDP2 + context parallel.
# Branch: Kimi-Dev-72B/Axolotl
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash scripts/trainmodel.sh --max-length 32768 --choice 1 --force-prep
#
# Install:
#   pip install 'axolotl[ring-flash-attn]'
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CONFIG="${CONFIG:-configs/axolotl/kimi_dev_72b_qlora.yaml}"
MIX_DIR="${MIX_DIR:-data/processed/mix_v2}"

MAX_LENGTH=""
CHOICE="${TRAIN_CHOICE:-}"
FORCE_PREP=0
EXTRA=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/trainmodel.sh --max-length <N> [--choice 1|2|3] [--force-prep]

  --max-length N   required; sets Axolotl sequence_len
  --choice K       1=use mix jsonl as-is, 2=rebalance 1:1:0.35 into run dir, 3=abort
  --force-prep     rebuild run yaml / sampled jsonl

Env:
  CUDA_VISIBLE_DEVICES  default 0 (use 0,1,2,3 for 4-GPU CP×FSDP)
  CONFIG / MIX_DIR
  CONTEXT_PARALLEL_SIZE override (default: 2 when NGPU even and >=2, else 1)
                            dp_shard_size is always NGPU / CONTEXT_PARALLEL_SIZE
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
  echo "ERROR: --max-length is required."
  usage
  exit 1
fi
if ! [[ "$MAX_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --max-length must be a positive integer, got: $MAX_LENGTH"
  exit 1
fi

if [[ ! -f "$MIX_DIR/mix_manifest.json" ]]; then
  echo "Missing $MIX_DIR/mix_manifest.json — run:"
  echo "  python scripts/prepare_data.py --all --out-dir $MIX_DIR"
  exit 1
fi

if ! command -v axolotl >/dev/null 2>&1 && ! python -c "import axolotl" >/dev/null 2>&1; then
  echo "Axolotl not found. Install: pip install 'axolotl[ring-flash-attn]'"
  exit 1
fi

if [[ -z "$CHOICE" ]]; then
  echo "Prep choice for mix under max_length=$MAX_LENGTH:"
  echo "  1) as-is (point Axolotl at mix_v2 jsonl)"
  echo "  2) rebalance sample wm_code:wm_os:anti_forget = 1:1:0.35 into a run dir"
  echo "  3) abort"
  read -r -p "Choice [1/2/3]: " CHOICE
fi
if [[ "$CHOICE" == "3" ]]; then
  echo "Aborted by user (choice 3)."
  exit 3
fi
if [[ "$CHOICE" != "1" && "$CHOICE" != "2" ]]; then
  echo "ERROR: choice must be 1, 2, or 3 (got $CHOICE)"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NGPU="$(python - <<'PY'
import os
xs=[x for x in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if x.strip()]
print(len(xs) if xs else 1)
PY
)"
# CP × dp_shard must equal NGPU. Default CP=2 on even multi-GPU (4→2×2 FSDP+CP).
if [[ -n "${CONTEXT_PARALLEL_SIZE:-}" ]]; then
  CP_SIZE="$CONTEXT_PARALLEL_SIZE"
elif [[ "$NGPU" -ge 2 && $((NGPU % 2)) -eq 0 ]]; then
  CP_SIZE=2
else
  CP_SIZE=1
fi
if ! [[ "$CP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CONTEXT_PARALLEL_SIZE must be positive integer, got: $CP_SIZE"
  exit 1
fi
if [[ "$CP_SIZE" -gt "$NGPU" ]]; then
  echo "ERROR: CONTEXT_PARALLEL_SIZE=$CP_SIZE > visible GPUs ($NGPU)"
  exit 1
fi
if [[ $((NGPU % CP_SIZE)) -ne 0 ]]; then
  echo "ERROR: NGPU=$NGPU not divisible by CONTEXT_PARALLEL_SIZE=$CP_SIZE"
  exit 1
fi
DP_SHARD_SIZE=$((NGPU / CP_SIZE))

RUN_ROOT="outputs/axolotl_runs/ml${MAX_LENGTH}_c${CHOICE}"
RUN_YAML="$RUN_ROOT/train.run.yaml"
mkdir -p "$RUN_ROOT"

if [[ "$FORCE_PREP" -eq 1 || ! -f "$RUN_YAML" ]]; then
  CONFIG="$CONFIG" MIX_DIR="$MIX_DIR" MAX_LENGTH="$MAX_LENGTH" CHOICE="$CHOICE" \
  RUN_ROOT="$RUN_ROOT" RUN_YAML="$RUN_YAML" NGPU="$NGPU" CP_SIZE="$CP_SIZE" \
  DP_SHARD_SIZE="$DP_SHARD_SIZE" \
  python - <<'PY'
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import yaml

root = Path(".").resolve()
config = Path(os.environ["CONFIG"])
if not config.is_absolute():
    config = root / config
mix_dir = Path(os.environ["MIX_DIR"])
if not mix_dir.is_absolute():
    mix_dir = root / mix_dir
run_root = Path(os.environ["RUN_ROOT"])
if not run_root.is_absolute():
    run_root = root / run_root
run_yaml = Path(os.environ["RUN_YAML"])
if not run_yaml.is_absolute():
    run_yaml = root / run_yaml
max_length = int(os.environ["MAX_LENGTH"])
choice = os.environ["CHOICE"]
ngpu = int(os.environ["NGPU"])
cp_size = int(os.environ["CP_SIZE"])
dp_shard_size = int(os.environ["DP_SHARD_SIZE"])

cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
# Strip BIV-only keys Axolotl does not understand.
cfg.pop("biv_mix", None)

sources = {
    "wm_code": mix_dir / "wm_code" / "train.jsonl",
    "wm_os": mix_dir / "wm_os" / "train.jsonl",
    "anti_forget": mix_dir / "anti_forget" / "train.jsonl",
}
for k, p in sources.items():
    if not p.is_file():
        raise SystemExit(f"missing dataset: {p}")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


seed = 42
rng = random.Random(seed)

if choice == "2":
    code = load_jsonl(sources["wm_code"])
    os_rows = load_jsonl(sources["wm_os"])
    anti = load_jsonl(sources["anti_forget"])
    n_os = len(os_rows)
    n_code = min(len(code), n_os)
    n_anti = min(len(anti), max(1, int(round(n_os * 0.35))))
    code_s = rng.sample(code, n_code) if n_code < len(code) else list(code)
    os_s = rng.sample(os_rows, n_code) if n_code < len(os_rows) else list(os_rows)
    anti_s = rng.sample(anti, n_anti) if n_anti < len(anti) else list(anti)
    out = {
        "wm_code": run_root / "wm_code.jsonl",
        "wm_os": run_root / "wm_os.jsonl",
        "anti_forget": run_root / "anti_forget.jsonl",
    }
    write_jsonl(out["wm_code"], code_s)
    write_jsonl(out["wm_os"], os_s)
    write_jsonl(out["anti_forget"], anti_s)
    dataset_paths = out
    print(
        f"rebalanced: code={n_code} os={n_code} anti={n_anti} → {run_root}",
        flush=True,
    )
else:
    dataset_paths = sources
    print("using mix jsonl as-is", flush=True)

cfg["sequence_len"] = max_length
cfg["output_dir"] = str(run_root / "checkpoints")
cfg["dataset_prepared_path"] = str(run_root / "prepared")
cfg["context_parallel_size"] = cp_size
cfg["dp_shard_size"] = dp_shard_size
cfg["datasets"] = [
    {
        "path": str(dataset_paths["wm_code"]),
        "type": "chat_template",
        "field_messages": "messages",
    },
    {
        "path": str(dataset_paths["wm_os"]),
        "type": "chat_template",
        "field_messages": "messages",
    },
    {
        "path": str(dataset_paths["anti_forget"]),
        "type": "chat_template",
        "field_messages": "messages",
    },
]

# Prefer local model_dir when present.
base = cfg.get("base_model")
if isinstance(base, str) and not Path(base).exists():
    # keep hub id / path as configured
    pass

run_yaml.parent.mkdir(parents=True, exist_ok=True)
run_yaml.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
meta = {
    "max_length": max_length,
    "choice": choice,
    "ngpu": ngpu,
    "context_parallel_size": cp_size,
    "dp_shard_size": dp_shard_size,
    "config_template": str(config),
    "run_yaml": str(run_yaml),
}
(run_root / "run_manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(f"wrote {run_yaml}", flush=True)
print(json.dumps(meta, indent=2), flush=True)
PY
else
  echo "Reusing existing run yaml: $RUN_YAML (pass --force-prep to rebuild)"
fi

echo "=== Axolotl train ==="
echo "  config=$RUN_YAML"
echo "  max_length=$MAX_LENGTH choice=$CHOICE"
echo "  GPUs=$CUDA_VISIBLE_DEVICES ngpu=$NGPU context_parallel_size=$CP_SIZE dp_shard_size=$DP_SHARD_SIZE (CP×FSDP mesh=$CP_SIZE×$DP_SHARD_SIZE)"

# FlashAttention hard check (Axolotl CP requires FA + ring-flash-attn)
python - <<'PY'
import sys

def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)

try:
    import torch
    import flash_attn
    from flash_attn import flash_attn_func  # noqa: F401
    print(f"  torch={torch.__version__} flash_attn={getattr(flash_attn, '__version__', '?')}")
except Exception as e:
    fail(f"FlashAttention required for Axolotl CP: {e}")

try:
    import ring_flash_attn  # noqa: F401
    print("  ring_flash_attn OK")
except Exception as e:
    print(f"  WARNING: ring_flash_attn import failed ({e}); CP may fail at runtime")
    print("  Install: pip install 'axolotl[ring-flash-attn]'")

if torch.cuda.device_count() < 1:
    fail("ERROR: no CUDA devices visible")
print(f"  torch.cuda.device_count={torch.cuda.device_count()}", flush=True)
PY

# Prefer axolotl CLI; fall back to python -m
if command -v axolotl >/dev/null 2>&1; then
  exec axolotl train "$RUN_YAML"
fi
exec python -m axolotl.cli.train "$RUN_YAML"
