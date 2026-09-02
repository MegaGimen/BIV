#!/usr/bin/env bash
# Compat alias. Live 32k / 2x2 entry is train_jepa.sh.
exec bash "$(cd "$(dirname "$0")" && pwd)/train_jepa.sh" "$@"
