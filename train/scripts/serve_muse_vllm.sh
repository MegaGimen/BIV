#!/usr/bin/env bash
# Start Muse Glimmer with Muse-capable vLLM on AutoDL (port 6006 → public :8443).
#
# Requires Muse day-0 vLLM (NOT stock pip 0.27.1). Prefer:
#   bash scripts/install_muse_vllm.sh
#   source .venv-muse/bin/activate
#
# Official Muse flags (Meta / vLLM recipe):
#   --enable-auto-tool-choice
#   --tool-call-parser muse_glimmer
#   --reasoning-parser muse_glimmer
#   --generation-config auto
#
# Usage:
#   bash scripts/serve_muse_vllm.sh              # default: latest LoRA ckpt under train out_dir
#   bash scripts/serve_muse_vllm.sh --ckpt auto   # same (explicit)
#   bash scripts/serve_muse_vllm.sh --ckpt outputs/.../checkpoint-e0-s2150
#   bash scripts/serve_muse_vllm.sh --base        # no LoRA (Harbor model Muse-Glimmer-30B)
#   MAX_LENGTH=65536 CHOICE=1 bash scripts/serve_muse_vllm.sh
#   bash scripts/serve_muse_vllm.sh --tp 2
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Prefer Muse train/serve env (.venv-muse). Do not create a second venv.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -x "$ROOT/.venv-muse/bin/vllm" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT/.venv-muse/bin/activate"
  fi
fi

PORT="${PORT:-6006}"
HOST="${HOST:-0.0.0.0}"
SERVED_BASE="${SERVED_BASE:-Muse-Glimmer-30B}"
LORA_NAME="${LORA_NAME:-muse-lora}"
MODEL_PATH="${MODEL_PATH:-}"
# Default: auto-pick latest complete LoRA checkpoint (override with --ckpt PATH or --base).
CKPT="${CKPT:-auto}"
BASE_ONLY=0
MAX_LORA_RANK="${MAX_LORA_RANK:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
DTYPE="${DTYPE:-bfloat16}"
TP="${TP:-1}"
CONFIG="${CONFIG:-configs/trl/muse_glimmer_30b_lora.yaml}"
MAX_LENGTH="${MAX_LENGTH:-}"
CHOICE="${CHOICE:-1}"
# Blackwell (sm_120): FlashInfer JIT can false-fail; default off.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Meta recipe flags (override with =0 to disable).
ENABLE_MUSE_PARSERS="${ENABLE_MUSE_PARSERS:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      CKPT="$2"; shift 2 ;;
    --base|--no-lora)
      BASE_ONLY=1
      CKPT=""
      shift
      ;;
    --model-path|--model_path)
      MODEL_PATH="$2"; shift 2 ;;
    --port)
      PORT="$2"; shift 2 ;;
    --tp)
      TP="$2"; shift 2 ;;
    --max-length|--max_length)
      MAX_LENGTH="$2"; shift 2 ;;
    --choice)
      CHOICE="$2"; shift 2 ;;
    --config)
      CONFIG="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,30p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$BASE_ONLY" -eq 1 ]]; then
  CKPT=""
fi

if [[ -z "$MODEL_PATH" ]]; then
  if [[ -d "$ROOT/outputs/models/Muse-Glimmer-30B" ]]; then
    MODEL_PATH="$ROOT/outputs/models/Muse-Glimmer-30B"
  else
    echo "ERROR: set MODEL_PATH or put weights at outputs/models/Muse-Glimmer-30B"
    exit 1
  fi
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm not found. Run: bash scripts/install_muse_vllm.sh"
  exit 1
fi

_VLLM_BIN="$(command -v vllm)"
_PY="$(dirname "$_VLLM_BIN")/python"
if [[ ! -x "$_PY" ]]; then
  _PY="python3"
fi

if ! "$_PY" -c "import torch" 2>/dev/null; then
  echo "ERROR: torch import failed (often NCCL cu12 overwrote cu13)."
  echo "  Fix: pip uninstall -y nvidia-nccl-cu12"
  echo "       pip install --force-reinstall --no-deps 'nvidia-nccl-cu13==2.29.7'"
  "$_PY" -c "import torch" || true
  exit 1
fi

if ! "$_PY" -c "import pathlib,vllm; p=pathlib.Path(vllm.__file__).parent/'model_executor/models/muse_glimmer.py'; raise SystemExit(0 if p.is_file() else 1)"; then
  echo "ERROR: this vLLM build has no muse_glimmer (stock PyPI wheel)."
  echo "  Install Muse nightly: bash scripts/install_muse_vllm.sh"
  echo "  Then: source .venv-muse/bin/activate && bash scripts/serve_muse_vllm.sh"
  exit 1
fi

_resolve_search_dir() {
  # 1) explicit CKPT_SEARCH_DIR / OUT_DIR
  # 2) CONFIG + MAX_LENGTH + CHOICE → train output_dir_mlN_cK
  # 3) newest outputs/muse_glimmer_wm_mix* that has a complete ckpt
  if [[ -n "${CKPT_SEARCH_DIR:-}" ]]; then
    echo "$CKPT_SEARCH_DIR"
    return
  fi
  if [[ -n "${OUT_DIR:-}" ]]; then
    echo "$OUT_DIR"
    return
  fi
  ROOT="$ROOT" CONFIG="$CONFIG" MAX_LENGTH="${MAX_LENGTH:-}" CHOICE="$CHOICE" "$_PY" - <<'PY'
import os, re
from pathlib import Path

root = Path(os.environ["ROOT"])
cfg_path = Path(os.environ["CONFIG"])
if not cfg_path.is_absolute():
    cfg_path = root / cfg_path
base = "outputs/muse_glimmer_wm_mix"
text = cfg_path.read_text(encoding="utf-8") if cfg_path.is_file() else ""
try:
    import yaml
    cfg = yaml.safe_load(text) or {}
    base = str((cfg.get("train") or {}).get("output_dir") or base)
except Exception:
    m = re.search(r"(?m)^\s*output_dir:\s*(\S+)", text)
    if m:
        base = m.group(1).strip().strip("'\"")
ml = (os.environ.get("MAX_LENGTH") or "").strip()
choice = os.environ.get("CHOICE") or "1"
if ml.isdigit():
    out = root / f"{base}_ml{ml}_c{choice}"
    print(out)
    raise SystemExit(0)

# No max-length: pick newest mix_* dir that contains a complete ckpt.
pats = [
    re.compile(r"^checkpoint-epoch(\d+)-end-s(\d+)$"),
    re.compile(r"^checkpoint-e(\d+)-s(\d+)$"),
    re.compile(r"^checkpoint-(\d+)$"),
]
candidates = []
parent = root / Path(base).parent if "/" in base or base.startswith("outputs") else root / "outputs"
# Also scan root/outputs for muse_glimmer_wm_mix*
scan_roots = {root / "outputs", (root / base).parent if not Path(base).is_absolute() else Path(base).parent}
for sr in scan_roots:
    if not sr.is_dir():
        continue
    for d in sr.iterdir():
        if not d.is_dir():
            continue
        if "muse_glimmer_wm_mix" not in d.name:
            continue
        has = False
        for p in d.iterdir():
            if not p.is_dir():
                continue
            if not any(rx.match(p.name) for rx in pats):
                continue
            if (p / "trainer_state.json").is_file() and (
                (p / "adapter_config.json").is_file()
                or (p / "adapter_model.safetensors").is_file()
                or any(p.glob("*.safetensors"))
            ):
                has = True
                break
        if has:
            candidates.append(d)
if candidates:
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(candidates[0])
else:
    # Fall back to bare base path (may be empty → clear error later)
    print(root / base if not Path(base).is_absolute() else Path(base))
PY
}

# Resolve CKPT=auto → latest complete LoRA under train output_dir.
if [[ -n "$CKPT" && "$(echo "$CKPT" | tr '[:upper:]' '[:lower:]')" == "auto" ]]; then
  SEARCH="$(_resolve_search_dir)"
  echo "[serve] ckpt search_dir=$SEARCH"
  CKPT="$(
    SEARCH="$SEARCH" "$_PY" - <<'PY'
import os, re
from pathlib import Path
out = Path(os.environ["SEARCH"])
pats = [
    (re.compile(r"^checkpoint-epoch(\d+)-end-s(\d+)$"), 2),
    (re.compile(r"^checkpoint-e(\d+)-s(\d+)$"), 1),
    (re.compile(r"^checkpoint-(\d+)$"), 0),
]
best = None
if out.is_dir():
    for p in out.iterdir():
        if not p.is_dir():
            continue
        epoch = step = kind = None
        for rx, k in pats:
            m = rx.match(p.name)
            if not m:
                continue
            if k == 0:
                epoch, step, kind = 0, int(m.group(1)), 0
            else:
                epoch, step, kind = int(m.group(1)), int(m.group(2)), k
            break
        if epoch is None:
            continue
        if not (p / "trainer_state.json").is_file():
            continue
        if not (
            (p / "adapter_model.safetensors").is_file()
            or (p / "adapter_config.json").is_file()
            or any(p.glob("*.safetensors"))
        ):
            continue
        key = (epoch, step, kind)
        if best is None or key > best[0]:
            best = (key, p)
print("" if best is None else best[1])
PY
  )"
  if [[ -z "$CKPT" ]]; then
    echo "ERROR: CKPT=auto but no complete checkpoint under $SEARCH"
    echo "  Hint: set MAX_LENGTH / CHOICE / CKPT_SEARCH_DIR, or pass --ckpt PATH, or --base"
    exit 1
  fi
  echo "[serve] CKPT=auto → $CKPT"
fi

# Sidecar for Harbor/eval hosts (copy step into MUSE_EVAL_STEP if desired).
META_PATH="$ROOT/outputs/.muse_vllm_serve.json"
mkdir -p "$ROOT/outputs"
if [[ -n "$CKPT" ]]; then
  CKPT_NAME="$(basename "$CKPT")"
  CKPT_STEP="$(
    CKPT_NAME="$CKPT_NAME" "$_PY" - <<'PY'
import os, re
name = os.environ["CKPT_NAME"]
m = re.search(r"(?:^|[-_])s(\d+)(?:$|[-_])", name, re.I)
if m:
    print(m.group(1))
else:
    m = re.match(r"^checkpoint-(\d+)$", name)
    print(m.group(1) if m else "0")
PY
  )"
  "$_PY" - <<PY
import json
from pathlib import Path
meta = {
    "mode": "lora",
    "lora_name": "${LORA_NAME}",
    "served_base": "${SERVED_BASE}",
    "ckpt": str(Path("${CKPT}").resolve()),
    "ckpt_name": "${CKPT_NAME}",
    "step": int("${CKPT_STEP}"),
    "harbor_model_id": "${LORA_NAME}",
}
Path("${META_PATH}").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(f"[serve] wrote meta → ${META_PATH}")
print(f"[serve] Harbor: model id = ${LORA_NAME}")
print(f"[serve] Optional on eval host: export MUSE_EVAL_ARM=${CKPT_NAME} MUSE_EVAL_STEP=${CKPT_STEP}")
PY
else
  "$_PY" - <<PY
import json
from pathlib import Path
meta = {
    "mode": "base",
    "lora_name": None,
    "served_base": "${SERVED_BASE}",
    "ckpt": None,
    "ckpt_name": None,
    "step": 0,
    "harbor_model_id": "${SERVED_BASE}",
}
Path("${META_PATH}").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(f"[serve] wrote meta → ${META_PATH}")
print(f"[serve] Harbor: model id = ${SERVED_BASE}")
print("[serve] Optional on eval host: export MUSE_EVAL_ARM=base MUSE_EVAL_STEP=0")
PY
fi

echo "=== Muse vLLM serve ==="
echo "  venv:       ${VIRTUAL_ENV:-"(PATH vllm)"}"
echo "  model_path: $MODEL_PATH"
echo "  port:       $PORT  (AutoDL custom service → public :8443)"
echo "  served:     $SERVED_BASE"
echo "  muse_parsers: ENABLE_MUSE_PARSERS=$ENABLE_MUSE_PARSERS"
echo "  flashinfer_sampler: VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
if [[ -n "$CKPT" ]]; then
  echo "  lora:       $LORA_NAME ← $CKPT"
else
  echo "  lora:       <none> (base only; Harbor --model $SERVED_BASE / test.py --base)"
fi
echo

CMD=(
  vllm serve "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_BASE"
  --dtype "$DTYPE"
  --tensor-parallel-size "$TP"
  --generation-config auto
)
# Muse native path does not need trust_remote_code; keep optional for odd exports.
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  CMD+=(--trust-remote-code)
fi
if [[ "$ENABLE_MUSE_PARSERS" == "1" ]]; then
  CMD+=(
    --enable-auto-tool-choice
    --tool-call-parser muse_glimmer
    --reasoning-parser muse_glimmer
  )
fi
if [[ -n "$MAX_MODEL_LEN" ]]; then
  CMD+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "$CKPT" ]]; then
  if [[ ! -f "$CKPT/adapter_config.json" ]]; then
    echo "ERROR: $CKPT missing adapter_config.json (need PEFT LoRA ckpt)"
    exit 1
  fi
  # Text-only Harbor/agent path: disable mm so LoRA does not wrap the
  # vision encoder (otherwise lora_shrink asserts during mm profile).
  CMD+=(
    --limit-mm-per-prompt '{"image":0,"video":0}'
    --enable-lora
    --max-loras 1
    --max-lora-rank "$MAX_LORA_RANK"
    --lora-modules "${LORA_NAME}=${CKPT}"
  )
fi

echo "+ ${CMD[*]}"
exec "${CMD[@]}"
