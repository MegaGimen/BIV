#!/bin/bash
# BIV entrypoint: Cartesian Agent A/B dashboard on nanobot runtime.
set -euo pipefail

cd /home/BIV
# shellcheck disable=SC1091
set -a
source /home/BIV/.env
set +a

exec /bin/bash /home/BIV/cartesian-dashboard/start.sh
