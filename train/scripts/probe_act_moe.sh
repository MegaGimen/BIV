#!/usr/bin/env bash
# CPU MoE coverage probe for compare_act. Does not steal GPUs, does not forward.
#
#   cd train
#   bash scripts/probe_act_moe.sh
#   bash scripts/probe_act_moe.sh --skip-modules    # headers only
#   bash scripts/probe_act_moe.sh --world-only
#
# Needs merge/output/cache already populated (the GPU host). Writes
# train/outputs/act_moe_probe/report.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Structure probe: never attach to a card that compare/train is using.
unset CUDA_VISIBLE_DEVICES

python scripts/probe_act_moe.py "$@"
