#!/usr/bin/env bash
#
# Phase B: snapshot_200, snapshot_300 × dev 57.
# Agent는 gpt-5-mini (사용자 결정). Stage 1 baseline은 dev_phaseA_gpt5mini 의
# baseline_results.json 을 그대로 재사용 (gpt-5-mini × initial 8-bullet × dev 57).
#
# 사용법:
#   bash memory_utility/scripts/run_phase_b.sh
#   ONLY=snap200 bash ...

set -u
set -o pipefail

REPO_ROOT="/home/bwoo/workspace/ace-appworld"
PY="/home/bwoo/.conda/envs/ace/bin/python"

export APPWORLD_ROOT="$REPO_ROOT"
export APPWORLD_PROJECT_PATH="$REPO_ROOT"
export APPWORLD_PY="$PY"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

SNAP_DIR="$REPO_ROOT/experiments/runs/ACE_offline_5epoch/playbooks"
SRC_BASELINE="$REPO_ROOT/memory_utility/runs/dev_phaseA_gpt5mini/stage1/baseline_results.json"

# Phase B agent — gpt-5-mini via OpenAI direct
AGENT_MODEL="${AGENT_MODEL:-gpt-5-mini}"
AGENT_PROVIDER="${AGENT_PROVIDER:-openai}"

if [[ ! -f "$SRC_BASELINE" ]]; then
    echo "source baseline not found: $SRC_BASELINE" >&2
    echo "Phase A의 gpt5mini sub-run이 끝나야 Phase B Stage 1을 재사용할 수 있습니다." >&2
    exit 1
fi

SNAPS=("snap200|200" "snap300|300")
NUM_PROCESSES="${NUM_PROCESSES:-1}"
TASKS="${TASKS:-dev}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-4o-mini}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openrouter}"

SAMPLE_ARGS=()
[[ -n "$SAMPLE_SIZE" ]] && SAMPLE_ARGS+=(--sample-size "$SAMPLE_SIZE")

run_one_snap() {
    local slug="$1" n="$2"
    local RUN_NAME="dev_phaseB_${slug}"
    local RUN_DIR="$REPO_ROOT/memory_utility/runs/$RUN_NAME"
    local SNAP_FILE="$SNAP_DIR/trained_playbook_snapshot_${n}.txt"

    if [[ ! -f "$SNAP_FILE" ]]; then
        echo "snapshot not found: $SNAP_FILE" >&2
        return 1
    fi

    mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stage1"

    echo "==================================================="
    echo "  Phase B sub-run: $RUN_NAME"
    echo "  memory src: $SNAP_FILE"
    echo "  agent: $AGENT_MODEL ($AGENT_PROVIDER)"
    echo "  tasks=$TASKS sample=${SAMPLE_SIZE:-full} num_processes=$NUM_PROCESSES"
    echo "  started: $(date -Iseconds)"
    echo "==================================================="

    # 1. Stage 0 — 새 snapshot + 3 dummies
    "$PY" -m memory_utility.stage0_memory \
        --from-file "$SNAP_FILE" \
        --append-dummies 3 \
        --judge-model "$JUDGE_MODEL" \
        --judge-provider "$JUDGE_PROVIDER" \
        --out "$RUN_DIR/memory.json" \
        --force \
        > "$RUN_DIR/logs/stage0.log" 2>&1
    echo "  Stage 0 rc=$?"

    # 2. Stage 1 재사용 — dev_phaseA_gpt5mini 의 gpt-5-mini baseline
    cp -f "$SRC_BASELINE" "$RUN_DIR/stage1/baseline_results.json"
    echo "  Stage 1 reused from dev_phaseA_gpt5mini"

    # 3. Stage 2a — gpt-5-mini agent on new snapshot
    "$PY" -m memory_utility.stage2a_run \
        --memory "$RUN_DIR/memory.json" \
        --tasks "$TASKS" "${SAMPLE_ARGS[@]}" \
        --num-processes "$NUM_PROCESSES" \
        --agent-model "$AGENT_MODEL" \
        --agent-provider "$AGENT_PROVIDER" \
        --out-dir "$RUN_DIR/stage2" \
        --force \
        > "$RUN_DIR/logs/stage2a.log" 2>&1
    echo "  Stage 2a rc=$?"

    # 4. Stage 2b (judge)
    "$PY" -m memory_utility.stage2b_analyze \
        --memory "$RUN_DIR/memory.json" \
        --stage2a-dir "$RUN_DIR/stage2" \
        --out-dir "$RUN_DIR/stage2" \
        --reference-detection both \
        --judge-model "$JUDGE_MODEL" \
        --judge-provider "$JUDGE_PROVIDER" \
        --force \
        > "$RUN_DIR/logs/stage2b.log" 2>&1
    echo "  Stage 2b rc=$?"

    # 5. Stage 3 × 4
    for SRC in citation judge union intersection; do
        "$PY" -m memory_utility.stage3_stats \
            --baseline "$RUN_DIR/stage1/baseline_results.json" \
            --instrumented "$RUN_DIR/stage2/" \
            --memory "$RUN_DIR/memory.json" \
            --reference-source "$SRC" \
            --out "$RUN_DIR/stage3/insight_stats_${SRC}.json" \
            --force \
            > "$RUN_DIR/logs/stage3_${SRC}.log" 2>&1
    done

    echo "  done: $(date -Iseconds)"
    echo ""
}

for entry in "${SNAPS[@]}"; do
    IFS='|' read -r slug n <<< "$entry"
    if [[ -n "${ONLY:-}" && "${ONLY}" != "$slug" ]]; then
        continue
    fi
    run_one_snap "$slug" "$n" || echo "[phase_b] $slug failed, continuing"
done

echo "==================================================="
echo "Phase B complete: $(date -Iseconds)"
echo "==================================================="
