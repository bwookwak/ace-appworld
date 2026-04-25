#!/usr/bin/env bash
#
# Baseline (no memory) 평가 — test_normal split.
#
# 무엇을 하는가:
#   memory_utility의 Stage 1만 단독으로 돌립니다. Stage 1은 ACE evaluation
#   에이전트(`ace_evaluation_react`)를 `appworld_initial_playbook.txt`(시드
#   플레이북, 5개의 최소 bullet)로 실행합니다. 즉, 학습된 ACE 플레이북이
#   없는 상태(no memory)와 사실상 동일합니다.
#   (memory_utility/README.md 의 Caveats 섹션 참고: "very close to no-memory
#   but carries a tiny amount of scaffolding".)
#
# 출력 위치:
#   memory_utility/runs/<RUN_NAME>/
#     ├── stage1/
#     │   ├── baseline_results.json     ← 핵심 결과 (per_task + aggregate)
#     │   ├── baseline_playbook.txt     ← 사용한 플레이북 동결본
#     │   ├── llm_failures_eval.jsonl   ← LLM 실패 로그 (있다면)
#     │   └── config_canonical/         ← 재현용 jsonnet
#     └── logs/
#         └── stage1_baseline.log
#
#   per-task trajectory:
#     experiments/outputs/memutil_<RUN_NAME>_stage1/tasks/<task_id>/logs/lm_calls.jsonl
#
# 추가 집계가 필요하면 (TGC/SGC, difficulty 분해표):
#   appworld evaluate memutil_<RUN_NAME>_stage1 test_normal
#
# 실시간 로그:
#   tail -f memory_utility/runs/<RUN_NAME>/logs/stage1_baseline.log
#
# Ctrl+C 하면 trap이 자식 프로세스를 같이 종료합니다.

# ============================================================
# TODO(사용자): 아래 변수들을 본인 실행에 맞게 수정하세요
# ============================================================

# TODO: 이 run의 이름. memory_utility/runs/<RUN_NAME>/ 아래로 결과가 쌓입니다.
RUN_NAME="baseline_test_normal"

# TODO: 평가 split. 보통 test_normal 고정.
#       (dev=56 | train=89 | test_normal=167 | test_challenge=416)
TASKS="test_normal"

# TODO: 앞에서부터 N개만 돌리려면 숫자, 전체 167개를 돌리려면 빈 문자열.
SAMPLE_SIZE=""

# TODO: 베이스라인 플레이북. 기본은 시드(initial) 플레이북.
#       다른 플레이북으로 baseline을 갈음하고 싶다면 여기를 바꾸세요.
BASELINE_PLAYBOOK="/home/bwoo/workspace/ace-appworld/experiments/playbooks/appworld_initial_playbook.txt"

# TODO: 동시에 돌릴 태스크 수. OpenRouter rate limit를 보고 정하세요.
NUM_PROCESSES="1"

# TODO: 1이면 config_hash 캐시를 무시하고 강제 재실행.
FORCE="0"

# (선택) 평가까지 자동으로 이어서 돌리려면 1.
#   appworld evaluate memutil_<RUN_NAME>_stage1 test_normal 를 끝에 호출합니다.
RUN_APPWORLD_EVALUATE="1"

# ============================================================
# 아래부터는 보통 수정 불필요
# ============================================================

set -euo pipefail

: "${RUN_NAME:?RUN_NAME is required}"
: "${TASKS:?TASKS is required}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

PY="${PY:-/home/bwoo/.conda/envs/ace/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "Python binary not found or not executable: $PY" >&2
    exit 1
fi

if [[ ! -f "$BASELINE_PLAYBOOK" ]]; then
    echo "BASELINE_PLAYBOOK not found: $BASELINE_PLAYBOOK" >&2
    exit 1
fi

export APPWORLD_ROOT="$REPO_ROOT"
export APPWORLD_PROJECT_PATH="$REPO_ROOT"
export APPWORLD_PY="$PY"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

FORCE_FLAG=""
if [[ "$FORCE" == "1" ]]; then FORCE_FLAG="--force"; fi

RUN_DIR="$REPO_ROOT/memory_utility/runs/$RUN_NAME"
STAGE1_OUT="$RUN_DIR/stage1/baseline_results.json"
STAGE1_LOG="$RUN_DIR/logs/stage1_baseline.log"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stage1"

SAMPLE_ARGS=()
if [[ -n "$SAMPLE_SIZE" ]]; then
    SAMPLE_ARGS+=(--sample-size "$SAMPLE_SIZE")
fi

EXPERIMENT_NAME="memutil_${RUN_NAME}_stage1"

echo "==================================================="
echo "  Baseline (no memory) — Stage 1 only"
echo "  RUN_NAME=$RUN_NAME"
echo "  TASKS=$TASKS  SAMPLE_SIZE=${SAMPLE_SIZE:-all}"
echo "  NUM_PROCESSES=$NUM_PROCESSES  FORCE=$FORCE"
echo "  BASELINE_PLAYBOOK=$BASELINE_PLAYBOOK"
echo "  RUN_DIR=$RUN_DIR"
echo "  EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "==================================================="

# Ctrl+C / SIGTERM 받으면 백그라운드 자식도 같이 죽임
STAGE1_PID=""
cleanup() {
    echo ""
    echo "[baseline] cleanup: terminating background job..."
    [[ -n "$STAGE1_PID" ]] && kill "$STAGE1_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "[baseline] launching Stage 1..."
echo "[baseline]   tail -f $STAGE1_LOG"
echo ""

(
    "$PY" -m memory_utility.stage1_baseline \
        --tasks "$TASKS" \
        "${SAMPLE_ARGS[@]}" \
        --baseline-playbook "$BASELINE_PLAYBOOK" \
        --num-processes "$NUM_PROCESSES" \
        --out "$STAGE1_OUT" \
        $FORCE_FLAG
) > "$STAGE1_LOG" 2>&1 &
STAGE1_PID=$!
echo "[baseline] Stage 1 PID=$STAGE1_PID"

RC1=0
wait "$STAGE1_PID" || RC1=$?
STAGE1_PID=""
trap - INT TERM EXIT

echo ""
echo "[baseline] Stage 1 done rc=$RC1"
if [[ $RC1 -ne 0 ]]; then
    echo "[baseline] Stage 1 failed. Check log: $STAGE1_LOG" >&2
    exit 1
fi

echo "==================================================="
echo "  Stage 1 outputs:"
echo "    baseline_results.json : $STAGE1_OUT"
echo "    log                   : $STAGE1_LOG"
echo "    trajectories          : experiments/outputs/$EXPERIMENT_NAME/tasks/"
echo "==================================================="

if [[ "$RUN_APPWORLD_EVALUATE" == "1" ]]; then
    echo ""
    echo "[baseline] running 'appworld evaluate $EXPERIMENT_NAME $TASKS' for aggregate metrics..."
    cd "$REPO_ROOT"
    appworld evaluate "$EXPERIMENT_NAME" "$TASKS" || {
        echo "[baseline] WARNING: 'appworld evaluate' failed; you can rerun manually:" >&2
        echo "           appworld evaluate $EXPERIMENT_NAME $TASKS" >&2
    }
fi
