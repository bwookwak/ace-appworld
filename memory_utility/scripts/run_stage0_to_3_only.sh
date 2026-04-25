#!/usr/bin/env bash
#
# Recommended entry point for the v1 (Stages 0–3) pipeline.
#
# Usage:
#   RUN_NAME=my_exp1 \
#   MEMORY_SRC=/path/to/your/playbook.txt \
#   TASKS=dev SAMPLE_SIZE=10 DUMMIES_N=5 \
#   bash memory_utility/scripts/run_stage0_to_3_only.sh
#
# Env vars (required):
#   RUN_NAME        — run identifier, creates runs/<RUN_NAME>/
#   MEMORY_SRC      — path to source playbook/memory file (.txt/.json/.jsonl)
# Env vars (optional):
#   TASKS           — dataset name (default: dev)
#   SAMPLE_SIZE     — integer, first-N tasks from dataset (unset = full dataset)
#   TASK_IDS        — comma-separated override (supersedes SAMPLE_SIZE)
#   DUMMIES_N       — number of dummies to append (default: 0)
#   DUMMY_DOMAINS   — csv (default: cooking,sports_coaching,personal_finance,general_health,home_gardening,music_practice)
#   REFERENCE_DETECTION — citation|judge|both (default: both)
#   REFERENCE_SOURCE    — citation|judge|union|intersection (Stage 3; default: union)
#   JUDGE_MODEL     — (default: gpt-4o-mini)
#   JUDGE_PROVIDER  — (default: openai)
#   NUM_PROCESSES   — parallelism for appworld run (default: 1)
#   FORCE           — 1 to bust all caches
#   MEMORY_PATH     — skip Stage 0: use this memory.json directly
#   BASELINE_PATH   — skip Stage 1: use this baseline_results.json directly
#   STAGE2_DIR      — skip Stage 2: use this directory directly
#   PY              — python binary (default: /home/bwoo/.conda/envs/ace/bin/python)
#
# Outputs all go under runs/<RUN_NAME>/.
RUN_NAME="test_run"
MEMORY_SRC="/home/bwoo/workspace/ace-appworld/experiments/playbooks/appworld_offline_trained_no_gt_playbook.txt"
TASKS="dev"
SAMPLE_SIZE="5"
DUMMIES_N="3"


set -euo pipefail

: "${RUN_NAME:?RUN_NAME is required}"

if [[ -z "${MEMORY_PATH:-}" ]]; then
    : "${MEMORY_SRC:?Either MEMORY_SRC (Stage 0 input) or MEMORY_PATH (skip Stage 0) is required}"
fi

# Resolve repo root as the parent of this script's dir.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

PY="${PY:-/home/bwoo/.conda/envs/ace/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "Python binary not found or not executable: $PY" >&2
    echo "Set PY to your ace env's python, e.g. PY=/home/bwoo/.conda/envs/ace/bin/python" >&2
    exit 1
fi

export APPWORLD_ROOT="$REPO_ROOT"
export APPWORLD_PY="$PY"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

TASKS="${TASKS:-dev}"
DUMMIES_N="${DUMMIES_N:-0}"
DUMMY_DOMAINS="${DUMMY_DOMAINS:-cooking,sports_coaching,personal_finance,general_health,home_gardening,music_practice}"
REFERENCE_DETECTION="${REFERENCE_DETECTION:-both}"
REFERENCE_SOURCE="${REFERENCE_SOURCE:-union}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
FORCE_FLAG=""
if [[ "${FORCE:-0}" == "1" ]]; then FORCE_FLAG="--force"; fi

RUN_DIR="$REPO_ROOT/memory_utility/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"

echo "==================================================="
echo "  memory_utility pipeline — Stages 0–3"
echo "  RUN_NAME=$RUN_NAME"
echo "  RUN_DIR=$RUN_DIR"
echo "  TASKS=$TASKS  SAMPLE_SIZE=${SAMPLE_SIZE:-unset}  TASK_IDS=${TASK_IDS:-unset}"
echo "  DUMMIES_N=$DUMMIES_N  REFERENCE_DETECTION=$REFERENCE_DETECTION  REFERENCE_SOURCE=$REFERENCE_SOURCE"
echo "==================================================="

SAMPLE_ARGS=()
if [[ -n "${SAMPLE_SIZE:-}" ]]; then
    SAMPLE_ARGS+=(--sample-size "$SAMPLE_SIZE")
fi
if [[ -n "${TASK_IDS:-}" ]]; then
    SAMPLE_ARGS+=(--task-ids "$TASK_IDS")
fi

# --- Stage 0 ---
STAGE0_OUT="$RUN_DIR/memory.json"
if [[ -n "${MEMORY_PATH:-}" ]]; then
    echo "[stage0] --from existing MEMORY_PATH=$MEMORY_PATH"
    mkdir -p "$RUN_DIR"
    cp -f "$MEMORY_PATH" "$STAGE0_OUT"
else
    "$PY" -m memory_utility.stage0_memory \
        --from-file "$MEMORY_SRC" \
        --append-dummies "$DUMMIES_N" \
        --dummy-domains "$DUMMY_DOMAINS" \
        --judge-model "$JUDGE_MODEL" \
        --judge-provider "$JUDGE_PROVIDER" \
        --out "$STAGE0_OUT" \
        $FORCE_FLAG
fi

# --- Stage 1 ---
STAGE1_OUT="$RUN_DIR/stage1/baseline_results.json"
if [[ -n "${BASELINE_PATH:-}" ]]; then
    "$PY" -m memory_utility.stage1_baseline \
        --from-file "$BASELINE_PATH" \
        --out "$STAGE1_OUT"
else
    "$PY" -m memory_utility.stage1_baseline \
        --tasks "$TASKS" \
        "${SAMPLE_ARGS[@]}" \
        --num-processes "$NUM_PROCESSES" \
        --out "$STAGE1_OUT" \
        $FORCE_FLAG
fi

# --- Stage 2 ---
STAGE2_DIR_PATH="$RUN_DIR/stage2"
if [[ -n "${STAGE2_DIR:-}" ]]; then
    "$PY" -m memory_utility.stage2_instrumented \
        --memory "$STAGE0_OUT" \
        --from-file "$STAGE2_DIR" \
        --out-dir "$STAGE2_DIR_PATH"
else
    "$PY" -m memory_utility.stage2_instrumented \
        --memory "$STAGE0_OUT" \
        --tasks "$TASKS" \
        "${SAMPLE_ARGS[@]}" \
        --reference-detection "$REFERENCE_DETECTION" \
        --judge-model "$JUDGE_MODEL" \
        --judge-provider "$JUDGE_PROVIDER" \
        --num-processes "$NUM_PROCESSES" \
        --out-dir "$STAGE2_DIR_PATH" \
        $FORCE_FLAG
fi

# --- Stage 3 ---
STAGE3_OUT="$RUN_DIR/stage3/insight_stats.json"
"$PY" -m memory_utility.stage3_stats \
    --baseline "$STAGE1_OUT" \
    --instrumented "$STAGE2_DIR_PATH" \
    --memory "$STAGE0_OUT" \
    --reference-source "$REFERENCE_SOURCE" \
    --out "$STAGE3_OUT" \
    $FORCE_FLAG

echo "==================================================="
echo "  Pipeline (Stages 0–3) complete."
echo "  memory.json               : $STAGE0_OUT"
echo "  baseline_results.json     : $STAGE1_OUT"
echo "  stage2 outputs            : $STAGE2_DIR_PATH"
echo "  insight_stats.json        : $STAGE3_OUT"
echo "  logs                      : $RUN_DIR/logs/"
echo "==================================================="
