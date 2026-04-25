#!/usr/bin/env bash
#
# Chained orchestrator: Phase A (gpt5mini는 SKIP) → Phase B (gpt-5-mini × snap200/300).
#
# Phase A 의 4개 모델 (gpt5nano, gemma4, qwen35, gptoss120) 끝나면
# 자동으로 Phase B 를 시작.
#
# 만약 Phase A 가 도중에 실패해도 Phase B 는 시도 (Phase A 의 gpt5mini Stage 1은 이미 있음).

set -u
REPO_ROOT="/home/bwoo/workspace/ace-appworld"
SCRIPTS="$REPO_ROOT/memory_utility/scripts"

echo "==================================================="
echo "[chained] START $(date -Iseconds)"
echo "==================================================="

# Phase A — gpt5mini는 이미 완료됐으므로 skip
SKIP="${SKIP:-gpt5mini}" bash "$SCRIPTS/run_phase_a.sh"
RC_A=$?
echo "[chained] Phase A done rc=$RC_A"

# Phase A의 sub-run 일부가 실패해도 Phase B는 돌릴 수 있음
# (Phase B는 Phase A의 gpt5mini Stage 1 baseline에만 의존)
echo ""
echo "[chained] Starting Phase B..."
bash "$SCRIPTS/run_phase_b.sh"
RC_B=$?
echo "[chained] Phase B done rc=$RC_B"

echo "==================================================="
echo "[chained] DONE $(date -Iseconds)  rc_a=$RC_A rc_b=$RC_B"
echo "==================================================="
