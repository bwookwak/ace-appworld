#!/usr/bin/env bash
#
# Full end-to-end pipeline. Stages 4–7 are skeletons in v1; this script
# will still invoke them (they'll exit with their skeleton message),
# preserving the CLI shape and output file conventions for when they're
# filled in.
#
# See scripts/run_stage0_to_3_only.sh for the v1 recommended entry point
# and full env-var documentation.
#
# All env vars from run_stage0_to_3_only.sh apply here. Extras:
#   BUCKETS_PATH    — skip Stage 4 with this buckets.json
#   VERIFIED_PATH   — skip Stage 5 with this verified.json
#   STAGE6_DIR      — skip Stage 6 with this directory
#   REPORT_PATH     — skip Stage 7 with this report.md

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Run Stages 0-3 first via the v1 script
bash "$SCRIPT_DIR/run_stage0_to_3_only.sh"

: "${RUN_NAME:?RUN_NAME is required}"
PY="${PY:-/home/bwoo/.conda/envs/ace/bin/python}"
export APPWORLD_ROOT="$REPO_ROOT"
export APPWORLD_PY="$PY"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

FORCE_FLAG=""
if [[ "${FORCE:-0}" == "1" ]]; then FORCE_FLAG="--force"; fi

RUN_DIR="$REPO_ROOT/memory_utility/runs/$RUN_NAME"
STAGE0_OUT="$RUN_DIR/memory.json"
STAGE3_OUT="$RUN_DIR/stage3/insight_stats.json"

# --- Stage 4 ---
STAGE4_OUT="$RUN_DIR/stage4/buckets.json"
if [[ -n "${BUCKETS_PATH:-}" ]]; then
    mkdir -p "$(dirname "$STAGE4_OUT")"
    cp -f "$BUCKETS_PATH" "$STAGE4_OUT"
else
    "$PY" -m memory_utility.stage4_bucket \
        --stats "$STAGE3_OUT" \
        --out "$STAGE4_OUT" \
        $FORCE_FLAG || echo "[warn] stage4 is a skeleton"
fi

# --- Stage 5 ---
STAGE5_OUT="$RUN_DIR/stage5/verified.json"
if [[ -n "${VERIFIED_PATH:-}" ]]; then
    mkdir -p "$(dirname "$STAGE5_OUT")"
    cp -f "$VERIFIED_PATH" "$STAGE5_OUT"
else
    "$PY" -m memory_utility.stage5_verify \
        --memory "$STAGE0_OUT" \
        --buckets "$STAGE4_OUT" \
        --out "$STAGE5_OUT" \
        $FORCE_FLAG || echo "[warn] stage5 is a skeleton"
fi

# --- Stage 6 ---
STAGE6_DIR_PATH="$RUN_DIR/stage6"
if [[ -n "${STAGE6_DIR:-}" ]]; then
    mkdir -p "$STAGE6_DIR_PATH"
    cp -rf "$STAGE6_DIR"/* "$STAGE6_DIR_PATH/" 2>/dev/null || true
else
    "$PY" -m memory_utility.stage6_final_eval \
        --memory "$STAGE0_OUT" \
        --verified "$STAGE5_OUT" \
        --out-dir "$STAGE6_DIR_PATH" \
        $FORCE_FLAG || echo "[warn] stage6 is a skeleton"
fi

# --- Stage 7 ---
REPORT_OUT="$RUN_DIR/report.md"
if [[ -n "${REPORT_PATH:-}" ]]; then
    cp -f "$REPORT_PATH" "$REPORT_OUT"
else
    "$PY" -m memory_utility.stage7_report \
        --run-dir "$RUN_DIR" \
        --out "$REPORT_OUT" \
        $FORCE_FLAG || echo "[warn] stage7 is a skeleton"
fi

echo "==================================================="
echo "  Full pipeline complete."
echo "  report.md : $REPORT_OUT"
echo "==================================================="
