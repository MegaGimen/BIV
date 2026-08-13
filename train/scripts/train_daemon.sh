#!/usr/bin/env bash
# Muse training watchdog: relaunch trainmodel.sh with the *same* args on any
# abnormal exit (OOM / SIGKILL / CTRL+C / crash). Never shrinks hyperparams.
#
# Usage (same flags as trainmodel.sh):
#   bash scripts/train_daemon.sh --max-length 65536 --choice 1 \
#     --resume-from outputs/muse_glimmer_wm_mix_ml65536_c1/checkpoint-200
#
# Behavior:
#   - Passes all CLI args through to scripts/trainmodel.sh unchanged (no fallback
#     batch/seq/parallel knobs).
#   - On restart after a failure, replaces --resume-from with the newest complete
#     checkpoint under the run output_dir (so progress is not lost).
#   - CTRL+C (SIGINT) → kill training tree → daemon restarts (for crash tests).
#   - SIGTERM / SIGQUIT → stop daemon (no restart). Or: touch "$STOP_FILE".
#
# Env:
#   DAEMON_RESTART_DELAY   seconds between restarts (default 30)
#   DAEMON_MAX_RESTARTS    0 = unlimited (default 0)
#   DAEMON_STOP_FILE       default: outputs/.train_daemon_stop
#   CONFIG / MIX_DIR / …   same as trainmodel.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TRAIN_SH="$ROOT/scripts/trainmodel.sh"
DELAY="${DAEMON_RESTART_DELAY:-30}"
MAX_RESTARTS="${DAEMON_MAX_RESTARTS:-0}"
STOP_FILE="${DAEMON_STOP_FILE:-$ROOT/outputs/.train_daemon_stop}"
CONFIG="${CONFIG:-configs/trl/muse_glimmer_30b_lora.yaml}"

if [[ ! -f "$TRAIN_SH" ]]; then
  echo "ERROR: missing $TRAIN_SH"
  exit 1
fi

USER_RESUME="${RESUME_FROM:-}"
MAX_LENGTH=""
CHOICE="${TRAIN_CHOICE:-}"
PASS_ARGS=()

args=("$@")
i=0
n=${#args[@]}
while [[ $i -lt $n ]]; do
  a="${args[$i]}"
  case "$a" in
    --resume-from|--resume_from)
      i=$((i + 1))
      if [[ $i -ge $n ]]; then
        echo "ERROR: $a needs a path"
        exit 1
      fi
      USER_RESUME="${args[$i]}"
      ;;
    --max-length|--max_length|-m)
      PASS_ARGS+=("$a")
      i=$((i + 1))
      if [[ $i -ge $n ]]; then
        echo "ERROR: $a needs a value"
        exit 1
      fi
      MAX_LENGTH="${args[$i]}"
      PASS_ARGS+=("$MAX_LENGTH")
      ;;
    --choice|-c)
      PASS_ARGS+=("$a")
      i=$((i + 1))
      if [[ $i -ge $n ]]; then
        echo "ERROR: $a needs a value"
        exit 1
      fi
      CHOICE="${args[$i]}"
      PASS_ARGS+=("$CHOICE")
      ;;
    *)
      PASS_ARGS+=("$a")
      ;;
  esac
  i=$((i + 1))
done

if [[ -z "$MAX_LENGTH" ]]; then
  echo "ERROR: --max-length is required (same as trainmodel.sh)"
  exit 1
fi
if [[ -z "$CHOICE" ]]; then
  CHOICE=1
fi

resolve_out_dir() {
  CONFIG="$CONFIG" MAX_LENGTH="$MAX_LENGTH" CHOICE="$CHOICE" ROOT="$ROOT" python3 - <<'PY'
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
out = f"{base}_ml{os.environ['MAX_LENGTH']}_c{os.environ['CHOICE']}"
p = Path(out)
print(p if p.is_absolute() else (root / p))
PY
}

OUT_DIR="$(resolve_out_dir)"
mkdir -p "$(dirname "$STOP_FILE")" "$OUT_DIR"
rm -f "$STOP_FILE"

find_latest_ckpt() {
  OUT_DIR="$OUT_DIR" python3 - <<'PY'
import os, re
from pathlib import Path

out = Path(os.environ["OUT_DIR"])
if not out.is_dir():
    raise SystemExit(0)

pat_roll = re.compile(r"^checkpoint-e(\d+)-s(\d+)$")
pat_epoch = re.compile(r"^checkpoint-epoch(\d+)-end-s(\d+)$")
pat_digit = re.compile(r"^checkpoint-(\d+)$")

best = None  # (step, kind_rank, path)
for p in out.iterdir():
    if not p.is_dir():
        continue
    name = p.name
    step = None
    rank = 0
    m = pat_epoch.match(name)
    if m:
        step, rank = int(m.group(2)), 2
    else:
        m = pat_roll.match(name)
        if m:
            step, rank = int(m.group(2)), 1
        else:
            m = pat_digit.match(name)
            if m:
                step, rank = int(m.group(1)), 0
    if step is None:
        continue
    if not (p / "trainer_state.json").is_file():
        continue
    if not (
        (p / "adapter_model.safetensors").is_file()
        or (p / "pytorch_model_fsdp.bin").is_file()
        or any(p.glob("*.safetensors"))
    ):
        continue
    key = (step, rank)
    if best is None or key > (best[0], best[1]):
        best = (step, rank, p)

if best is None:
    raise SystemExit(0)
print(best[2])
PY
}

CHILD_PID=""
CHILD_PGID=""
STOP_DAEMON=0

kill_train_tree() {
  local sig="${1:-TERM}"
  if [[ -n "$CHILD_PGID" ]]; then
    kill "-$sig" -- "-$CHILD_PGID" 2>/dev/null || true
  fi
  if [[ -n "$CHILD_PID" ]]; then
    kill "-$sig" "$CHILD_PID" 2>/dev/null || true
  fi
}

on_term() {
  echo "[daemon] got stop signal → shutting down (no restart)"
  STOP_DAEMON=1
  kill_train_tree TERM
}
on_int() {
  echo "[daemon] SIGINT → kill training tree, will restart after delay"
  kill_train_tree INT
}

trap on_term TERM QUIT
trap on_int INT

echo "=== Muse train daemon ==="
echo "  train:        $TRAIN_SH"
echo "  out_dir:      $OUT_DIR"
echo "  delay:        ${DELAY}s"
echo "  max_restarts: ${MAX_RESTARTS} (0=unlimited)"
echo "  stop file:    $STOP_FILE  (or SIGTERM)"
echo "  user_resume:  ${USER_RESUME:-<none>}"
echo "  passthrough:  ${PASS_ARGS[*]}"
echo

attempt=0
while true; do
  if [[ -f "$STOP_FILE" ]]; then
    echo "[daemon] stop file present → exit"
    exit 0
  fi
  if [[ "$STOP_DAEMON" -eq 1 ]]; then
    exit 0
  fi

  attempt=$((attempt + 1))
  if [[ "$MAX_RESTARTS" -gt 0 && "$attempt" -gt "$MAX_RESTARTS" ]]; then
    echo "[daemon] reached DAEMON_MAX_RESTARTS=$MAX_RESTARTS → exit"
    exit 1
  fi

  launch_args=("${PASS_ARGS[@]}")
  resume_path=""
  if [[ "$attempt" -eq 1 ]]; then
    if [[ -n "$USER_RESUME" ]]; then
      resume_path="$USER_RESUME"
    else
      resume_path="$(find_latest_ckpt || true)"
    fi
  else
    resume_path="$(find_latest_ckpt || true)"
    if [[ -z "$resume_path" && -n "$USER_RESUME" ]]; then
      resume_path="$USER_RESUME"
    fi
  fi

  unset RESUME_FROM || true
  if [[ -n "$resume_path" ]]; then
    if [[ ! -d "$resume_path" ]]; then
      echo "[daemon] WARNING: resume path missing: $resume_path (starting without resume)"
    else
      export RESUME_FROM="$resume_path"
      launch_args+=(--resume-from "$resume_path")
      echo "[daemon] attempt=$attempt resume_from=$resume_path"
    fi
  else
    echo "[daemon] attempt=$attempt (fresh start, no checkpoint yet)"
  fi

  echo "[daemon] launch: bash scripts/trainmodel.sh ${launch_args[*]}"
  # New session: tty CTRL+C hits our trap; we forward to the accelerate PGID.
  set +e
  setsid bash "$TRAIN_SH" "${launch_args[@]}" &
  CHILD_PID=$!
  CHILD_PGID="$CHILD_PID"
  wait "$CHILD_PID"
  code=$?
  set -e
  CHILD_PID=""
  CHILD_PGID=""

  if [[ "$STOP_DAEMON" -eq 1 || -f "$STOP_FILE" ]]; then
    echo "[daemon] stop requested after train exit_code=$code"
    exit "$code"
  fi

  if [[ "$code" -eq 0 ]]; then
    echo "[daemon] training finished successfully (exit 0) → daemon exits"
    exit 0
  fi
  if [[ "$code" -eq 3 ]]; then
    echo "[daemon] train aborted (exit 3) → daemon exits (no restart)"
    exit 3
  fi

  echo "[daemon] train exited code=$code → restart in ${DELAY}s (same args, no param fallback)"
  # Give NCCL/CUDA a moment to release after OOM kills.
  sleep "$DELAY"
done
