#!/usr/bin/env bash
#
# Convenience wrapper: skip Stage 1 by pointing at an existing
# baseline_results.json. Useful when re-running Stage 2/3 with different
# memory or reference sources while keeping the baseline fixed.
#
# Usage:
#   RUN_NAME=reanalysis2 \
#   MEMORY_SRC=/path/playbook.txt \
#   BASELINE_PATH=runs/exp1/stage1/baseline_results.json \
#     bash memory_utility/scripts/run_with_existing_baseline.sh

set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
: "${BASELINE_PATH:?BASELINE_PATH is required}"
export BASELINE_PATH
exec bash "$SCRIPT_DIR/run_stage0_to_3_only.sh"
