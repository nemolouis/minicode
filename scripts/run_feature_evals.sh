#!/usr/bin/env bash
# Run the baseline and every feat-* config on the same tasks and print paired comparisons against the baseline.
# Needs DEEPSEEK_API_KEY. Rough cost: 13 tasks x 3 repeats x ~$0.006 = about $0.25 per config.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY first (configs/flash.json reads it)}"
STAMP=$(date +%Y%m%d-%H%M%S)
COMMON=(--split dev --tasks b --repeat 3 --workers 4)
run() { minicode-eval run --config "configs/$1.json" "${COMMON[@]}" --run-id "$STAMP-$1" >/dev/null; echo "results/$STAMP-$1"; }
BASE=$(run flash)
echo "baseline: $BASE"; minicode-eval report "$BASE" | head -20
for cfg in feat-todo-flash feat-explore-flash feat-readfirst-flash feat-diag-flash feat-perms-flash; do
  OUT=$(run "$cfg")
  echo; echo "=== $cfg vs flash ==="; minicode-eval compare "$BASE" "$OUT"
done
echo; echo "Write the outcome next to each config's hypothesis in docs/ablations.md, including predictions that were wrong."
