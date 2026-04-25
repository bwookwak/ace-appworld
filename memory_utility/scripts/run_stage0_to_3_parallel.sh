#!/usr/bin/env bash
#
# Parallel version of run_stage0_to_3_only.sh.
#
# 실행 흐름:
#   Stage 0                     (순차; Stage 2가 memory.json을 필요로 함)
#     |
#     +--> Stage 1  ]  동시 실행 (서로 다른 플레이북/실험명, 의존성 없음)
#     +--> Stage 2  ]
#     |
#   Stage 3                     (Stage 1 & 2 모두 끝난 뒤 순수 계산)
#
# 병렬 구간의 로그는 각각 runs/<RUN_NAME>/logs/stage{1,2}.parallel.log 에 적힙니다.
# 실시간으로 보고 싶으면:
#   tail -f memory_utility/runs/<RUN_NAME>/logs/stage1.parallel.log &
#   tail -f memory_utility/runs/<RUN_NAME>/logs/stage2.parallel.log &
#
# Ctrl+C 하면 trap이 두 자식 프로세스를 같이 종료합니다.

# ============================================================
# TODO(사용자): 아래 변수들을 본인 실행에 맞게 수정하세요
# ============================================================

# TODO: 이 run의 이름. memory_utility/runs/<RUN_NAME>/ 아래로 출력물이 쌓입니다.
RUN_NAME="test_run_parallel"

# TODO: Stage 0 입력으로 쓸 플레이북 파일 경로 (.txt / .json / .jsonl)
#        예시는 snapshot_100 (130 insights, 약 1.12 epoch). 다른 체크포인트로 바꾸려면 여기만 수정.
MEMORY_SRC="/home/bwoo/workspace/ace-appworld/experiments/runs/ACE_offline_5epoch/playbooks/trained_playbook_snapshot_100.txt"

# TODO: 태스크셋 (dev=56 | train=89 | test_normal=167 | test_challenge=416)
TASKS="dev"

# TODO: dataset의 앞에서부터 몇 개만 쓸지. 전체 돌리려면 빈 문자열("")로 두세요.
SAMPLE_SIZE="20"

# TODO: 비-AppWorld 도메인에서 생성할 더미 insight 개수. 0이면 더미 생성 안 함.
DUMMIES_N="3"

# TODO: 각 stage 내부의 태스크 병렬도. Stage 1 ∥ Stage 2 까지 곱해지므로 OpenRouter rate limit 주의.
#       예: NUM_PROCESSES=3 이면 동시에 최대 3(stage1) + 3(stage2) = 6개 agent 호출이 나감.
NUM_PROCESSES="1"

# TODO: Stage 2 reference 수집 방법.
#       both    — citation + judge 둘 다 (기본, calibration 가능)
#       citation — 에이전트 자체 인용만 쓰고 judge는 끔 (가장 싸고 빠름, 20~30% 비용 절감)
#       judge   — judge만 씀 (citation 라인 무시, 더 비쌈)
REFERENCE_DETECTION="both"

# TODO: judge / dummy-generation 모델. gpt-4o-mini가 싸고 빠름.
JUDGE_MODEL="gpt-5.4-mini"

# TODO: Stage 3의 insight→task 매핑 소스.
#       union — citation OR judge (기본, 재현율 우선)
#       intersection — citation AND judge (정밀도 우선)
#       citation / judge — 단일 소스만
REFERENCE_SOURCE="union"

# TODO: 1이면 모든 stage의 config_hash 캐시를 무시하고 재실행.
FORCE="0"

# (선택) 외부 파일로 특정 stage를 건너뛰기. 필요 없으면 빈 문자열로 두세요.
MEMORY_PATH=""      # Stage 0 skip: 이미 있는 memory.json 경로
BASELINE_PATH=""    # Stage 1 skip: 이미 있는 baseline_results.json 경로
STAGE2_DIR=""       # Stage 2 skip: 이미 있는 stage2/ 디렉토리 경로

# ============================================================
# 아래부터는 보통 수정 불필요
# ============================================================

set -euo pipefail

: "${RUN_NAME:?RUN_NAME is required}"

if [[ -z "$MEMORY_PATH" ]]; then
    : "${MEMORY_SRC:?MEMORY_SRC (Stage 0 input) 또는 MEMORY_PATH (skip Stage 0) 중 하나는 필요합니다}"
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

PY="${PY:-/home/bwoo/.conda/envs/ace/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "Python binary not found or not executable: $PY" >&2
    exit 1
fi

export APPWORLD_ROOT="$REPO_ROOT"
export APPWORLD_PY="$PY"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DUMMY_DOMAINS="cooking,sports_coaching,personal_finance,general_health,home_gardening,music_practice"
JUDGE_PROVIDER="openai"

FORCE_FLAG=""
if [[ "$FORCE" == "1" ]]; then FORCE_FLAG="--force"; fi

RUN_DIR="$REPO_ROOT/memory_utility/runs/$RUN_NAME"
mkdir -p "$RUN_DIR/logs"

SAMPLE_ARGS=()
if [[ -n "$SAMPLE_SIZE" ]]; then
    SAMPLE_ARGS+=(--sample-size "$SAMPLE_SIZE")
fi

echo "==================================================="
echo "  memory_utility pipeline — Stages 0–3 (PARALLEL)"
echo "  RUN_NAME=$RUN_NAME"
echo "  RUN_DIR=$RUN_DIR"
echo "  TASKS=$TASKS  SAMPLE_SIZE=${SAMPLE_SIZE:-unset}"
echo "  DUMMIES_N=$DUMMIES_N  REFERENCE_DETECTION=$REFERENCE_DETECTION  REFERENCE_SOURCE=$REFERENCE_SOURCE"
echo "  NUM_PROCESSES=$NUM_PROCESSES   (※ Stage 1 ∥ Stage 2 이므로 동시 agent 호출은 2×$NUM_PROCESSES)"
echo "==================================================="

# Ctrl+C / SIGTERM 받으면 백그라운드 자식도 같이 죽임
STAGE1_PID=""
STAGE2_PID=""
cleanup() {
    echo ""
    echo "[parallel] cleanup: terminating background jobs..."
    [[ -n "$STAGE1_PID" ]] && kill "$STAGE1_PID" 2>/dev/null || true
    [[ -n "$STAGE2_PID" ]] && kill "$STAGE2_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# --- Stage 0 (sequential) ---
STAGE0_OUT="$RUN_DIR/memory.json"
if [[ -n "$MEMORY_PATH" ]]; then
    echo "[stage0] skip — copying existing MEMORY_PATH=$MEMORY_PATH"
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

# --- Stage 1 ∥ Stage 2 ---
STAGE1_OUT="$RUN_DIR/stage1/baseline_results.json"
STAGE2_DIR_PATH="$RUN_DIR/stage2"
STAGE1_LOG="$RUN_DIR/logs/stage1.parallel.log"
STAGE2_LOG="$RUN_DIR/logs/stage2.parallel.log"

echo ""
echo "[parallel] launching Stage 1 & Stage 2 concurrently..."
echo "[parallel]   tail -f $STAGE1_LOG"
echo "[parallel]   tail -f $STAGE2_LOG"
echo ""

# Stage 1 in background
(
    if [[ -n "$BASELINE_PATH" ]]; then
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
) > "$STAGE1_LOG" 2>&1 &
STAGE1_PID=$!
echo "[parallel] Stage 1 PID=$STAGE1_PID"

# Stage 2 in background
(
    if [[ -n "$STAGE2_DIR" ]]; then
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
) > "$STAGE2_LOG" 2>&1 &
STAGE2_PID=$!
echo "[parallel] Stage 2 PID=$STAGE2_PID"

# Wait for both; capture individual exit codes (do NOT trip set -e here).
RC1=0
wait "$STAGE1_PID" || RC1=$?
RC2=0
wait "$STAGE2_PID" || RC2=$?
STAGE1_PID=""
STAGE2_PID=""

echo ""
echo "[parallel] Stage 1 done rc=$RC1"
echo "[parallel] Stage 2 done rc=$RC2"
# Disable the cleanup-on-exit trap now that both are reaped — we want the
# script to return normally instead of re-invoking kill on stale PIDs.
trap - INT TERM EXIT

if [[ $RC1 -ne 0 || $RC2 -ne 0 ]]; then
    echo ""
    echo "[parallel] one or both parallel stages failed. Check logs:"
    echo "           $STAGE1_LOG"
    echo "           $STAGE2_LOG"
    exit 1
fi

# --- Stage 3 (sequential) ---
STAGE3_OUT="$RUN_DIR/stage3/insight_stats.json"
"$PY" -m memory_utility.stage3_stats \
    --baseline "$STAGE1_OUT" \
    --instrumented "$STAGE2_DIR_PATH" \
    --memory "$STAGE0_OUT" \
    --reference-source "$REFERENCE_SOURCE" \
    --out "$STAGE3_OUT" \
    $FORCE_FLAG

echo "==================================================="
echo "  Pipeline (Stages 0–3, parallel) complete."
echo "  memory.json               : $STAGE0_OUT"
echo "  baseline_results.json     : $STAGE1_OUT"
echo "  stage2 outputs            : $STAGE2_DIR_PATH"
echo "  insight_stats.json        : $STAGE3_OUT"
echo "  parallel logs             : $STAGE1_LOG"
echo "                              $STAGE2_LOG"
echo "==================================================="
