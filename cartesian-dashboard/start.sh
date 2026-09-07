#!/bin/bash
# Start Cartesian FastAPI (Agent A/B) + Vite frontend.
set -euo pipefail

ROOT=/home/BIV
DASH="$ROOT/cartesian-dashboard"

# shellcheck disable=SC1091
set -a
source "$ROOT/.env"
set +a

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/.nvm/versions/node/v24.12.0/bin:$ROOT/.venv/bin:$PATH"

cd "$ROOT"
"$ROOT/.venv/bin/python" -m cartesian.server &
API_PID=$!

cd "$DASH/client"
npm run dev &
UI_PID=$!

cleanup() {
  kill "$API_PID" "$UI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait
