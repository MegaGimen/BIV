#!/usr/bin/env bash
# Compat alias. Live entry is train_jepa.sh (N-way FSDP2+CP, seq-split).
exec bash "$(cd "$(dirname "$0")" && pwd)/train_jepa.sh" "$@"
