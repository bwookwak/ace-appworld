# REPO_NOTES — ace-appworld findings for memory-utility pipeline

Written before coding the pipeline. All paths are relative to the repo root
`/home/bwoo/workspace/ace-appworld`.

## A. Agent loop

- CLI entry: `appworld run <config_name>` (installed script from
  `src/appworld/cli.py:458` — `run()` function). It loads
  `experiments/configs/<config_name>.jsonnet`, applies
  `--override '<json>'`, and dispatches into
  `appworld_experiments.code.<runner_type>.run:run_experiment`.
- For ACE this is `experiments/code/ace/run.py:11` (`run_experiment`).
  That file decides the agent class from `run_type`:
  - `ace-adaptation` → `StarAgent` (trains playbook via curator/reflector)
  - `ace-evaluation` → `Agent` (read-only playbook, **this is what we want**)
  - `non-ace-evaluation` → `BaseAgent`
- The evaluation agent class is registered as `ace_evaluation_react` in
  `experiments/code/ace/evaluation_react.py:13`. Its playbook is loaded
  **read-only** from `trained_playbook_file_path` (line 33–35).
- Single-task entry is exposed on the CLI as
  `appworld run <config> --task-id <task_id>` (see `cli.py:468`).
- Per-task step loop is in
  `experiments/code/ace/evaluation_agent.py:67` (`solve_task`). It
  opens an `AppWorld` context, iterates up to `max_steps` (default 40
  from configs), calls `world.execute()` each step, and checks
  `world.task_completed()`. Success is verified by running
  `appworld evaluate <experiment_name> <dataset>` which walks every
  per-task output dir.
- **Success signal for per-task:**
  `experiments/outputs/<exp>/tasks/<task_id>/evaluation/report.md`
  contains `Num Failed Tests : 0` on success (produced by
  `appworld evaluate`). There is also a consolidated JSON at
  `experiments/outputs/<exp>/evaluations/<dataset>.json` after running
  `appworld evaluate`. Step count = count of agent turns in
  `logs/lm_calls.jsonl` (one LLM call per step).

## B. Memory system

- Memory = a **playbook**: a plain text file where each insight is one
  line (sometimes wrapped) in the format `[<id>] <content>`.
  IDs look like `shr-00001`, `api-00004`, `psw-00007`, `misc-00008`.
  IDs are sectioned under markdown headers like
  `## STRATEGIES AND HARD RULES`, `## COMMON MISTAKES AND CORRECT STRATEGIES`,
  `## TROUBLESHOOTING AND PITFALLS:`, `## OTHERS`.
- Canonical parser: `experiments/code/ace/playbook.py:13`
  (`parse_playbook_line`). Pattern: `\[([^\]]+)\]\s*(.*)`.
- Injection point: the generator prompt template
  `experiments/prompts/appworld_react_generator_prompt.txt:30-33` contains
  `### PLAYBOOK BEGIN\n{{ playbook }}\n### PLAYBOOK END`. The Jinja
  `playbook` variable is set in
  `experiments/code/ace/adaptation_react.py:62` and
  `evaluation_react.py:51`.
- Existing playbook files:
  - `experiments/playbooks/appworld_initial_playbook.txt` — **23 lines,
    minimal starter playbook** (shr-00001, shr-00005, shr-00006, shr-00021, misc-00008)
  - `experiments/playbooks/appworld_offline_trained_no_gt_playbook.txt`
    — **333 lines, the fully-trained no-GT playbook** (this is the
    "full memory" for the pipeline)
  - `experiments/playbooks/appworld_offline_trained_with_gt_playbook.txt`
    — 0 lines (empty)
  - `experiments/playbooks/appworld_online_trained_playbook.txt`
    — 373 lines
  - `experiments/runs/ACE_offline_5epoch/playbooks/trained_playbook.txt`
    (+ periodic `*_snapshot_{50,100,...,400}.txt`)
- The generator prompt **already contains in-context examples where
  the agent cites insight IDs inline** (`[shr-00005]`, `[api-00004]`,
  `[psw-00007]`, `[misc-00008]`). Citation instrumentation builds on a
  pattern the model is already imitating.

## C. Logging

Per-task outputs under `experiments/outputs/<exp_name>/tasks/<task_id>/`:

- `logs/loggger.log` — Rich-panel human-readable agent trace
- `logs/lm_calls.jsonl` — **one JSON line per LLM call**, each line has
  `{"id", "input": {"model", "messages": [...]}, "output": {...}, ...}`.
  This is the authoritative trajectory source for post-hoc judge.
- `logs/api_calls.jsonl` — AppWorld API calls the agent made
- `logs/environment_io.md` — human-readable environment I/O
- `evaluation/report.md` — unit-test results (success/failure text)
- `misc/cost.txt` — single float (total cost in dollars)
- `checkpoints/`, `dbs/` — AppWorld environment snapshots

Run-level outputs under `experiments/runs/<config_name>/`:

- `playbooks/` — `trained_playbook.txt`, `_snapshot_<N>.txt`,
  `reflections.jsonl`, `llm_failures.jsonl`, `llm_failures_eval.jsonl`
- `results/learning_curve.jsonl` — per-checkpoint score records
- `results/failed_tasks_ckpt_<N>.json` — list of failed task IDs
- `configs/eval_ckpt_<N>.jsonnet` — generated wrapper configs

## D. Config & runtime

- Config format: **Jsonnet** with `extVar("APPWORLD_PROJECT_PATH")`.
- Main working configs:
  - `experiments/configs/ACE_offline_5epoch.jsonnet` — adaptation (trains)
  - `experiments/configs/ACE_eval_checkpoint_template.jsonnet` — evaluation
    template (read-only playbook). Used as a base via `import` by
    per-checkpoint wrappers like `eval_ckpt_50.jsonnet`.
- Model: `deepseek/deepseek-v3.2` via `openrouter` provider,
  temperature 0, seed 100. Configured in both configs above.
- Seed: `config.agent.appworld_config.random_seed = 123` (env seed) +
  `model_config.seed = 100` (LLM seed). Temperature 0, so determinism is
  effectively set unless OpenRouter routes to a different provider.
- CLI entrypoints:
  - `appworld run <config>` (run agent; supports `--task-id`,
    `--override`, `--num-processes`)
  - `appworld evaluate <exp_name> <dataset>` (score outputs —
    success/failure, writes consolidated JSON)
  - `python experiments/code/ace/orchestrator.py ...` — user's wrapper
    that runs adaptation + concurrent eval of each checkpoint
- `back_fill.sh` (root): re-runs eval for checkpoints 50/100/150/200/250
  via `appworld run eval_ckpt_<N> && appworld evaluate eval_ckpt_<N> test_normal`.
  This is **not** related to our pipeline — it's the user's retry loop
  for orchestrator evals.

## E. Eval pipeline (to reuse)

- `appworld run <eval_config>` solves every task and writes outputs to
  `experiments/outputs/<eval_config>/tasks/<task_id>/`.
- `appworld evaluate <eval_config> <dataset>` walks those outputs and
  produces per-task reports plus `experiments/outputs/<eval_config>/evaluations/<dataset>.json`.
- The pipeline can **reuse this exact flow** for Stage 1 (baseline) and
  Stage 2 (full-memory instrumented run). Each stage gets its own generated
  wrapper jsonnet config pointing `trained_playbook_file_path` at the
  stage-specific playbook file.

## F. Task sets

Task IDs are loaded from `data/datasets/<name>.txt` (one task ID per line):

- `train.txt` — 89 tasks
- `dev.txt` — 56 tasks
- `test_normal.txt` — 167 tasks (the canonical eval split the user uses)
- `test_challenge.txt` — 416 tasks

Subset selection:

- `--override '{"config": {"dataset": "dev"}}'` to swap dataset
- `--override '{"config": {"sample_size": 5}}'` to cap count
- `--override '{"config": {"task_ids": ["task1", "task2"]}}'` for an
  explicit list (supported by `run.py:22`)
- `--task-id <id>` for a single task on the CLI

No existing formal "small dry-run" subset — recommend using
`--override '{"config": {"sample_size": 5}}'` against `dev` for dry runs.

## G. Environment & deps

- Python ≥3.11, conda env `ace` (as instructed).
- Declared deps (pyproject.toml): `openai>=1.74.0`, `together`,
  `sambanova`, `jsonnet`, `typer`, `sqlmodel`, `fastapi`, `pydantic`,
  `libcst`, `httpx`, `requests`, `tqdm`.
- **litellm is NOT in pyproject.toml** but the project uses an
  OpenAI-compatible client in `experiments/code/ace/lite_llm_generator.py`
  via OpenRouter. The pipeline spec calls for `litellm` as the judge
  backend — will need to add `litellm` to the env (or reuse
  `LiteLLMGenerator` which currently wraps `openai.OpenAI` under the hood).
  See "Questions" §2.
- Cost tracking via `experiments/code/ace/cost_tracker.py`.

## H. Output conventions

- Run artefacts live under `experiments/runs/<config_name>/` (managed by
  orchestrator).
- Raw per-task outputs from `appworld run` live under
  `experiments/outputs/<config_name>/tasks/<task_id>/` — this is fixed
  by `path_store`, so the pipeline cannot redirect it. We'll **mirror
  relevant files into `runs/<run_name>/stageN/...`** so the pipeline
  directory is self-contained.
- Recommendation: place the new pipeline at
  `/home/bwoo/workspace/ace-appworld/memory_utility/` (root-level, next
  to `src/` and `experiments/`). All pipeline run artefacts go under
  `/home/bwoo/workspace/ace-appworld/memory_utility/runs/<RUN_NAME>/`.

## I. Mapping the spec's "insight" onto the repo

- Each playbook bullet (`[id] content`) is one "insight".
- `insight.id` = the bracketed ID (e.g., `shr-00005`, `api-00004`).
- `insight.text` = bullet content (without the `[id] ` prefix).
- `insight.source` = `"provided"` for existing bullets from a source
  playbook, `"dummy"` for generated ones.
- Dummy insertion = appending `[dummy-00001] <off-domain text>` lines
  into a `## OTHERS` (or new `## DUMMY`) section, optionally shuffled
  among real bullets.

## J. Key files for the pipeline to touch (reference, no edits yet)

| Concern | Path | Line(s) |
|---|---|---|
| Run CLI | `src/appworld/cli.py` | 458 |
| Runner dispatch | `experiments/code/ace/run.py` | 11 |
| Eval agent (playbook-load point) | `experiments/code/ace/evaluation_react.py` | 33, 51 |
| Eval agent solve loop | `experiments/code/ace/evaluation_agent.py` | 67 |
| Generator prompt (injection) | `experiments/prompts/appworld_react_generator_prompt.txt` | 30–33 |
| Playbook parser | `experiments/code/ace/playbook.py` | 13 |
| Per-task trajectory | `experiments/outputs/<exp>/tasks/<tid>/logs/lm_calls.jsonl` | — |
| Per-task success | `experiments/outputs/<exp>/tasks/<tid>/evaluation/report.md` | — |
| Eval template config | `experiments/configs/ACE_eval_checkpoint_template.jsonnet` | — |

## K. Conflicts / deviations from the spec (raised with user)

1. **Playbook format is a flat `.txt` with `[id] content` bullets, not
   JSON.** The pipeline's `memory.json` will be a **derived** canonical
   form; Stage 0 will render it back into a `.txt` playbook that the
   evaluation agent can consume. Each stage's "memory in the run" is
   the `.txt` it writes; the `memory.json` is the source of truth the
   pipeline passes around.
2. **No native way to "run with no memory".** The evaluation agent
   requires `trained_playbook_file_path` to exist. Baseline will write
   an empty `.txt` (or one containing only section headers) — see
   "Questions" §3.
3. **Reference detection via citation:** the prompt already models the
   cite-by-id pattern. We'll amend the **key instructions** block
   (lines 947–963) to *mandate* the structured line
   `[cited_insights: id1, id2]` at each step. Diff will be shown before
   finalizing — see "Questions" §1.
4. **Reference detection via judge (post-hoc):** we have
   `logs/lm_calls.jsonl` with full step-level messages + assistant
   output. We can replay these for the judge without re-running the
   agent. Good news for cost.
5. **Seed:** model is temperature 0 + seed 100, but routed through
   OpenRouter which may fall back across providers. Determinism is
   **best-effort**, not guaranteed. Spec acknowledges single-seed v1
   anyway; noting as caveat.
6. **litellm is not a current dep** — will be added (spec §2.3 calls
   for it). Alternatively we could reuse the existing
   `LiteLLMGenerator` (it is still just `openai.OpenAI` under the
   hood, despite the name) for judge calls to avoid a new dep. See
   "Questions" §2.
