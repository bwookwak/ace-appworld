#!/usr/bin/env bash
#
# Convenience wrapper: skip Stage 0 by pointing at an existing memory.json.
# Delegates to run_stage0_to_3_only.sh with MEMORY_PATH set.
#
# Usage:
#   RUN_NAME=reanalysis1 MEMORY_PATH=runs/exp1/memory.json \
#     bash memory_utility/scripts/run_with_existing_memory.sh

set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
: "${MEMORY_PATH:?MEMORY_PATH is required}"
export MEMORY_PATH
exec bash "$SCRIPT_DIR/run_stage0_to_3_only.sh"
