#!/usr/bin/env bash
#
# Phase A: 5개 약한 agent 모델 × dev 57 × snapshot_100 memory.
# 각 모델별로 Stage 1 + Stage 2a 병렬, 그다음 Stage 2b (openrouter/gpt-4o-mini),
# 마지막으로 Stage 3를 4개 reference_source로 모두 계산.
#
# 사용법:
#   bash memory_utility/scripts/run_phase_a.sh                 # 전체 5 모델
#   ONLY=gpt5mini bash memory_utility/scripts/run_phase_a.sh   # 특정 모델 하나만
#
# 필요 env:
#   OPENAI_API_KEY  (진짜 OpenAI 키, sk-proj-...)
#   GEMINI_API_KEY
#   OPENROUTER_API_KEY

set -u
set -o pipefail

REPO_ROOT="/home/bwoo/workspace/ace-appworld"
PY="/home/bwoo/.conda/envs/ace/bin/python"

export APPWORLD_ROOT="$REPO_ROOT"
export APPWORLD_PROJECT_PATH="$REPO_ROOT"
export APPWORLD_PY="$PY"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

SRC_MEMORY="$REPO_ROOT/memory_utility/runs/dev_full_snap100/memory.json"
if [[ ! -f "$SRC_MEMORY" ]]; then
    echo "source memory.json not found: $SRC_MEMORY" >&2
    exit 1
fi

# (slug, name, provider)  — 모델 순서
MODELS=(
    "gpt5mini|gpt-5-mini|openai"
    "gpt5nano|gpt-5-nano|openai"
    "gemma4|gemma-4-26b-a4b-it|gemini"
    "qwen35|qwen/qwen3.5-35b-a3b|openrouter"
    "gptoss120|openai/gpt-oss-120b|openrouter"
)

NUM_PROCESSES="${NUM_PROCESSES:-1}"
TASKS="${TASKS:-dev}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-4o-mini}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openrouter}"

SAMPLE_ARGS=()
[[ -n "$SAMPLE_SIZE" ]] && SAMPLE_ARGS+=(--sample-size "$SAMPLE_SIZE")

run_one() {
    local slug="$1" name="$2" provider="$3"
    local RUN_NAME="dev_phaseA_${slug}"
    local RUN_DIR="$REPO_ROOT/memory_utility/runs/$RUN_NAME"
    mkdir -p "$RUN_DIR/logs"

    echo "==================================================="
    echo "  Phase A sub-run: $RUN_NAME"
    echo "  model=$name  provider=$provider"
    echo "  tasks=$TASKS  sample_size=${SAMPLE_SIZE:-full}  num_processes=$NUM_PROCESSES"
    echo "  started: $(date -Iseconds)"
    echo "==================================================="

    # 1. memory.json 재사용
    cp -f "$SRC_MEMORY" "$RUN_DIR/memory.json"

    # 2. Stage 1 + Stage 2a 병렬
    (
        "$PY" -m memory_utility.stage1_baseline \
            --tasks "$TASKS" "${SAMPLE_ARGS[@]}" \
            --num-processes "$NUM_PROCESSES" \
            --agent-model "$name" \
            --agent-provider "$provider" \
            --out "$RUN_DIR/stage1/baseline_results.json" \
            --force \
            > "$RUN_DIR/logs/stage1.log" 2>&1
    ) &
    local S1=$!
    (
        "$PY" -m memory_utility.stage2a_run \
            --memory "$RUN_DIR/memory.json" \
            --tasks "$TASKS" "${SAMPLE_ARGS[@]}" \
            --num-processes "$NUM_PROCESSES" \
            --agent-model "$name" \
            --agent-provider "$provider" \
            --out-dir "$RUN_DIR/stage2" \
            --force \
            > "$RUN_DIR/logs/stage2a.log" 2>&1
    ) &
    local S2A=$!

    local RC1=0 RC2=0
    wait "$S1" || RC1=$?
    wait "$S2A" || RC2=$?
    echo "  Stage 1 rc=$RC1  Stage 2a rc=$RC2"
    if [[ $RC1 -ne 0 || $RC2 -ne 0 ]]; then
        echo "  FAILED. logs: $RUN_DIR/logs/"
        return 1
    fi

    # 3. Stage 2b (judge) — openrouter gpt-4o-mini
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

    # 4. Stage 3 × 4 reference sources
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

for entry in "${MODELS[@]}"; do
    IFS='|' read -r slug name provider <<< "$entry"
    if [[ -n "${ONLY:-}" && "${ONLY}" != "$slug" ]]; then
        continue
    fi
    # SKIP=slug1,slug2 — 콤마 구분 슬러그 목록은 건너뜀
    if [[ -n "${SKIP:-}" && ",${SKIP}," == *",${slug},"* ]]; then
        echo "[phase_a] skipping $slug (SKIP=${SKIP})"
        continue
    fi
    run_one "$slug" "$name" "$provider" || echo "[phase_a] $slug failed, continuing to next model"
done

echo "==================================================="
echo "Phase A complete: $(date -Iseconds)"
echo "==================================================="
