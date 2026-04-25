# dry_run snapshot

Reference artifacts showing the exact shape of each Stage 0–3 output file
so you can inspect schemas without running the full pipeline.

## What's real vs synthetic

| File | Status | Notes |
|---|---|---|
| `memory.json` | **mostly real** | Real Stage 0 output from `appworld_initial_playbook.txt` (8 insights). The 3 appended dummy insights are hand-crafted (not LLM-generated) to avoid unauthorized API spend. A real Stage 0 run would LLM-generate these via the judge model. `_note` fields mark hand-crafted entries. |
| `dummy_generation_log.jsonl` | synthetic | Schema-valid demo of what a real run's audit log looks like. `_note` marks each entry. |
| `stage1/baseline_results.json` | synthetic | Real 5 dev task IDs (`50e1ac9_1`, `50e1ac9_2`, `50e1ac9_3`, `fac291d_1`, `fac291d_2`), synthetic success/step numbers. `_note` marks the config. |
| `stage2/instrumented_summary.json` | synthetic | Same 5 tasks; synthetic citation/judge signals carefully chosen so Stage 3 produces interpretable output. |
| `stage2/instrumented_logs.jsonl` | synthetic | Per-step records derived from the summary for completeness. |
| `stage2/judge_calibration.json` | synthetic | Computed from the synthetic per-step records; real Jaccard arithmetic. |
| `stage3/insight_stats.json` | **real** | Produced by actually running `python -m memory_utility.stage3_stats` against the files above. Demonstrates that Stage 3 math works end-to-end on real-shape data. |

## What Stage 3 shows here

With 5 synthetic tasks (baseline: 2/5 pass, full-memory: 3/5 pass) and
11 insights (8 real + 3 dummies):

- `shr-00005` cited on 3 tasks, `lift_mean = 0.33`, `has_positive_spike = True`
- `api-00004` cited on 1 task (the one that flips), `lift_mean = 1.00`
- `shr-00006`, `misc-00008` cited but with `lift_mean = 0.00`
- **All 3 dummies** have `reference_count_total = 0` — exactly what we
  want: dummies should be un-cited by both agent and judge.

## Running a real pipeline yourself

```bash
# The ace conda env must already have appworld + deps installed.
export OPENAI_API_KEY=...        # for Stage 0 dummies + Stage 2 judge
# (OPENROUTER_API_KEY is used by the agent's deepseek calls — set if your
# current agent config points at OpenRouter.)

RUN_NAME=my_first_run \
MEMORY_SRC=/home/bwoo/workspace/ace-appworld/experiments/playbooks/appworld_offline_trained_no_gt_playbook.txt \
TASKS=dev SAMPLE_SIZE=5 DUMMIES_N=3 \
bash /home/bwoo/workspace/ace-appworld/memory_utility/scripts/run_stage0_to_3_only.sh
```

Approximate cost for that invocation (5 dev tasks, ~40 steps each):
- Agent runs (Stage 1 + Stage 2): ~$1–3 (deepseek/deepseek-v3.2 via OpenRouter)
- Judge replays (Stage 2): ~$0.10–0.50 (gpt-4o-mini)
- Dummy generation (Stage 0): ~$0.01

Wall time: ~15–20 minutes (single-process). Parallelize with
`NUM_PROCESSES=4` if you have quota.

## Inspecting outputs

```bash
# Peek at the canonical Stage 3 output
jq '.global, .per_insight | to_entries | map(select(.value.reference_count_total > 0)) | map({id: .key, refs: .value.reference_count_total, lift: .value.lift_mean})' \
    stage3/insight_stats.json

# Peek at what the real Stage 0 wrote for one insight
jq '.insights[0]' memory.json
```
