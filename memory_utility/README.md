# memory_utility

Modular pipeline for estimating the utility of each insight in an AppWorld
playbook without running full per-insight ablation on every insight. See
`REPO_NOTES.md` for how this wraps the existing ace-appworld framework.

## TL;DR — run v1 (Stages 0–3) on 10 dev tasks

```bash
# Ensure ace env is active or export the python path
export OPENAI_API_KEY=...          # required for judge + dummy generation

RUN_NAME=dry_run1 \
MEMORY_SRC=/home/bwoo/workspace/ace-appworld/experiments/playbooks/appworld_offline_trained_no_gt_playbook.txt \
TASKS=dev SAMPLE_SIZE=10 DUMMIES_N=3 \
bash memory_utility/scripts/run_stage0_to_3_only.sh
```

Outputs land under `memory_utility/runs/$RUN_NAME/`.

## Pipeline shape

| Stage | Module | Output | Role |
|-------|--------|--------|------|
| 0     | `stage0_memory` | `memory.json` | Canonicalize source playbook; append dummies (LLM-generated, length-matched) |
| 1     | `stage1_baseline` | `stage1/baseline_results.json` | Run agent with MINIMAL playbook; collect per-task success |
| 2     | `stage2_instrumented` | `stage2/instrumented_logs.jsonl` + `instrumented_summary.json` + `judge_calibration.json` | Run agent with FULL memory; log which insights were cited/judged per step |
| 3     | `stage3_stats` | `stage3/insight_stats.json` | Pure computation: per-insight reference counts + lift (single-seed) |
| 4–7   | `stage4_bucket`, `stage5_verify`, `stage6_final_eval`, `stage7_report` | (skeletons) | CLI surface + TODO markers; v1 stops at Stage 3 |

Each stage is an **independently runnable CLI** (`python -m memory_utility.stageN_xxx ...`)
that reads JSON/JSONL input files and writes JSON/JSONL outputs. Stages
communicate only via files. There's a `--from-file ...` escape hatch on
every stage so you can substitute an existing artifact and skip the
stage.

## Run directory layout

```
memory_utility/runs/<RUN_NAME>/
├── memory.json                           # Stage 0
├── dummy_generation_log.jsonl            # Stage 0 (audit log)
├── logs/
│   ├── stage0_memory.log
│   ├── stage1_baseline.log
│   ├── stage2_instrumented.log
│   └── stage3_stats.log
├── stage1/
│   ├── baseline_results.json             # Stage 1 canonical
│   ├── baseline_playbook.txt             # frozen copy of the baseline playbook
│   └── config_canonical/                 # generated jsonnet for reproducibility
├── stage2/
│   ├── instrumented_summary.json         # Stage 2 canonical
│   ├── instrumented_logs.jsonl           # one JSON line per step
│   ├── judge_calibration.json            # citation vs judge agreement
│   ├── full_memory_playbook.txt          # playbook rendered from memory.json
│   ├── generator_prompt.txt              # amended prompt (with [cited_insights:] instruction)
│   ├── trajectories/<task_id>.jsonl      # copy of each task's lm_calls.jsonl
│   └── config_canonical/
├── stage3/
│   └── insight_stats.json                # Stage 3 canonical
└── stage{4,5,6,7}/ ...                   # populated once those stages are filled in
```

## Raw data you can re-analyze

- **Per-task trajectories** (Stage 1 and Stage 2):
  `experiments/outputs/memutil_<RUN_NAME>_stage{1,2}/tasks/<task_id>/logs/lm_calls.jsonl`
- **Judge raw responses** (Stage 2): embedded in each record of
  `stage2/instrumented_logs.jsonl` under `judge_raw_response`.
- **Dummy generation trace** (Stage 0):
  `dummy_generation_log.jsonl` (prompt, completion, parsed result, cost).
- **Per-task cost**: `experiments/outputs/memutil_<RUN_NAME>_stage{1,2}/tasks/<task_id>/misc/cost.txt`

## Re-run Stage 3 with a different reference source (no re-running Stage 2)

Stage 3 is pure computation. To swap `--reference-source` without re-running the agent:

```bash
/home/bwoo/.conda/envs/ace/bin/python -m memory_utility.stage3_stats \
  --baseline runs/<RUN_NAME>/stage1/baseline_results.json \
  --instrumented runs/<RUN_NAME>/stage2/ \
  --memory runs/<RUN_NAME>/memory.json \
  --reference-source intersection \
  --min-references-per-task 2 \
  --out runs/<RUN_NAME>/stage3/insight_stats_intersection.json \
  --force
```

## Skipping stages with external inputs

All stages accept `--from-file` for substitution. Shell envs mirror this:

| Skip | Env var |
|---|---|
| Stage 0 | `MEMORY_PATH=/path/to/memory.json` |
| Stage 1 | `BASELINE_PATH=/path/to/baseline_results.json` |
| Stage 2 | `STAGE2_DIR=/path/to/stage2/` |
| Stage 4 | `BUCKETS_PATH=...` (full pipeline only) |
| Stage 5 | `VERIFIED_PATH=...` (full pipeline only) |

`FORCE=1` busts all caches for a full re-run.

## Key design decisions (already fixed; see main design doc)

- **Single seed.** Lift per task ∈ {−1, 0, +1}. No multi-seed averaging.
- **Citation style:** per-step `[cited_insights: id1, id2]` line, appended as
  instruction (9) to the generator prompt (Stage 2 only). Stage 1 uses the
  ORIGINAL, unmodified prompt with a minimal playbook.
- **Judge backend:** reuses the repo's `LiteLLMGenerator` (OpenAI-compatible
  under the hood). Default model `gpt-4o-mini`. Swap via `--judge-model`.
- **Dummies:** LLM-generated, length-matched to real insights' token distribution,
  drawn from non-AppWorld domains (cooking, sports_coaching, personal_finance,
  general_health, home_gardening, music_practice by default).

## Environment

- Python: `/home/bwoo/.conda/envs/ace/bin/python` (the `ace` conda env).
  Override with `PY=<path>` env var.
- `APPWORLD_ROOT` must be set (the shell scripts handle this automatically).
- For judge + dummy generation: `OPENAI_API_KEY` (or `OPENROUTER_API_KEY`) in env.

## Caveats

- **Deletion-based ablation confound** (relevant from Stage 5 onward, when
  implemented): removing an insight shortens the prompt, which itself can
  move agent behavior. Report must disclose this.
- **OpenRouter provider routing:** even at temperature 0 + fixed seed, the
  agent's model (`deepseek/deepseek-v3.2`) may be routed to a different
  provider between the baseline and instrumented runs. Determinism is
  best-effort, not guaranteed. Single-seed choice in v1 is aware of this.
- **Stage 1 baseline** uses `appworld_initial_playbook.txt` (5 minimal
  bullets) rather than a truly empty playbook. It's very close to no-memory
  but carries a tiny amount of scaffolding (e.g., `complete_task()` reminder).

## v1 completion checklist

- [x] Stages 0, 1, 2, 3 fully implemented
- [x] Stages 4, 5, 6, 7 skeletons with CLI + CUSTOMIZE markers
- [x] `common/` utilities: io, config_hash, progress, litellm_client, playbook, paths, appworld_runner
- [x] Shell scripts: run_stage0_to_3_only, run_full_pipeline, run_with_existing_memory, run_with_existing_baseline
- [x] JSON schemas for Stage 0–3 outputs
- [x] Dry-run snapshot under `memory_utility/examples/dry_run/`
