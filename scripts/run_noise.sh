#!/usr/bin/env bash
# Run one config several times on the same tasks and report how much identical runs differ (the noise floor).
# Usage: scripts/run_noise.sh [config-name] [replicates]      defaults: flash 3.   Cost with flash: about $0.25 per replicate.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-flash}"; N="${2:-3}"
: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY first (configs/flash.json reads it)}"
STAMP=$(date +%Y%m%d-%H%M%S)
DIRS=()
for i in $(seq 1 "$N"); do
  miniagent-eval run --config "configs/$CFG.json" --split dev --tasks b --repeat 3 --workers 4 --run-id "$STAMP-noise$i-$CFG" >/dev/null
  DIRS+=("results/$STAMP-noise$i-$CFG")
  echo "replicate $i done: results/$STAMP-noise$i-$CFG"
done
miniagent-eval noise "${DIRS[@]}" --out "results/$STAMP-noise-$CFG.md"
