"""Stage 1 — Baseline Evaluation.

Run the ACE evaluation agent with a MINIMAL playbook (the repo's
`appworld_initial_playbook.txt` by default, per the user's design
decision) on the task subset, then read per-task success + step count
out of the appworld output tree.

CLI forms:
  --from-file PATH    : skip running, load an existing baseline_results.json
  (default)           : generate stage config, run agent, evaluate, write results

Output shape (runs/<RUN>/stage1/baseline_results.json):
  {
    "per_task": { "tid": {"success": 0/1, "step_count": N, "trajectory_path": "..."} },
    "aggregate": {...},
    "config": {...},
    "config_hash": "..."
  }
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Any

from .common import paths
from .common.appworld_runner import (
    evaluate_one_task,
    read_task_cost,
    read_trajectory,
    run_appworld,
    task_has_output,
    write_stage_config,
)
from .common.config_hash import hash_config, stage_cache_ok
from .common.io import read_json, read_text, write_json, write_text
from .common.progress import stage_context

STAGE_NAME = "stage1_baseline"


def _build_jsonnet(
    playbook_path: str,
    failure_log_path: str,
    dataset: str | None,
    agent_model: str | None = None,
    agent_provider: str | None = None,
    openrouter_min_throughput_p90: int = 100,
    openrouter_max_latency_p90: int = 1,
) -> str:
    """Generate a wrapper jsonnet that extends the eval template with stage-local overrides.

    If `agent_model` is given, override the agent's `generator_model_config.name`
    (and `provider` when `agent_provider` is set). When provider == "openrouter",
    a tightened `openrouter_provider` block is injected (throughput/latency
    thresholds). For non-openrouter providers, `openrouter_provider` is cleared
    so the base template's routing hints don't leak through.
    """
    lines: list[str] = []
    lines.append(f'local base = import "{paths.EVAL_TEMPLATE_CONFIG}";')
    lines.append("base + {")
    lines.append("  config+: {")
    if dataset:
        lines.append(f'    dataset: {json.dumps(dataset)},')
    lines.append("    agent+: {")
    lines.append(f'      trained_playbook_file_path: {json.dumps(playbook_path)},')
    lines.append("      generator_model_config+: {")
    lines.append(f'        failure_log_path: {json.dumps(failure_log_path)},')
    lines.append('        on_failure: "warn",')
    lines.append("        max_retries: 10,")
    if agent_model:
        lines.append(f'        name: {json.dumps(agent_model)},')
        # OpenAI GPT-5 family: temperature must be 1 (default), and a bunch of
        # legacy params are rejected outright (stop, seed, logprobs, penalties,
        # response_format). Null them out so the stage jsonnet overrides the
        # ACE template's deepseek-optimized defaults.
        if "gpt-5" in agent_model.lower():
            lines.append("        temperature: 1,")
            lines.append("        stop: null,")
            lines.append("        seed: null,")
            lines.append("        logprobs: null,")
            lines.append("        top_logprobs: null,")
            lines.append("        frequency_penalty: null,")
            lines.append("        presence_penalty: null,")
            lines.append("        response_format: null,")
    if agent_provider:
        lines.append(f'        provider: {json.dumps(agent_provider)},')
        if agent_provider == "openrouter":
            lines.append("        openrouter_provider: {")
            lines.append('          sort: "throughput",')
            lines.append(f'          preferred_min_throughput: {{ p90: {openrouter_min_throughput_p90} }},')
            lines.append(f'          preferred_max_latency:    {{ p90: {openrouter_max_latency_p90} }},')
            lines.append("          allow_fallbacks: true,")
            lines.append("        },")
        else:
            # Override away the base template's openrouter_provider so LiteLLM
            # doesn't send provider routing hints to a non-openrouter backend.
            lines.append("        openrouter_provider: null,")
    lines.append("      },")
    lines.append("    },")
    lines.append("  },")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _resolve_task_ids(
    dataset: str,
    sample_size: int | None,
    task_ids_csv: str | None,
) -> list[str]:
    if task_ids_csv:
        return [t.strip() for t in task_ids_csv.split(",") if t.strip()]
    ids = paths.load_task_ids(dataset)
    if sample_size is not None:
        ids = ids[:sample_size]
    return ids


def _validate_and_copy(from_file: str, out_path: str) -> dict[str, Any]:
    blob = read_json(from_file)
    for key in ("per_task", "aggregate", "config"):
        if key not in blob:
            raise ValueError(f"--from-file {from_file} missing required key '{key}'")
    write_json(out_path, blob)
    return blob


def run(
    out_path: str,
    tasks: str = "dev",
    sample_size: int | None = None,
    task_ids_csv: str | None = None,
    baseline_playbook: str = paths.INITIAL_PLAYBOOK_PATH,
    from_file: str | None = None,
    force: bool = False,
    run_name_hint: str | None = None,
    num_processes: int = 1,
    agent_model: str | None = None,
    agent_provider: str | None = None,
    openrouter_min_throughput_p90: int = 100,
    openrouter_max_latency_p90: int = 1,
) -> str:
    stage_dir = os.path.dirname(os.path.abspath(out_path))
    run_dir = os.path.dirname(stage_dir)
    run_name = os.path.basename(run_dir.rstrip("/")) or (run_name_hint or "default")
    experiment_name = f"memutil_{run_name}_stage1"

    if from_file:
        with stage_context(STAGE_NAME, run_dir):
            print(f"[{STAGE_NAME}] --from-file mode: validating and copying {from_file}")
            _validate_and_copy(from_file, out_path)
        return out_path

    task_ids = _resolve_task_ids(tasks, sample_size, task_ids_csv)
    cfg = {
        "tasks_dataset": tasks,
        "sample_size": sample_size,
        "task_ids_csv": task_ids_csv,
        "resolved_task_ids": task_ids,
        "baseline_playbook": os.path.abspath(baseline_playbook),
        "experiment_name": experiment_name,
        "agent_model": agent_model,
        "agent_provider": agent_provider,
        "openrouter_min_throughput_p90": openrouter_min_throughput_p90,
        "openrouter_max_latency_p90": openrouter_max_latency_p90,
    }
    cfg_hash = hash_config(cfg, input_file_paths=[baseline_playbook])
    if not force and stage_cache_ok(out_path, cfg_hash):
        print(f"[{STAGE_NAME}] cache hit: {out_path}")
        return out_path

    with stage_context(STAGE_NAME, run_dir) as ctx:
        # Freeze the baseline playbook into the stage dir for reproducibility.
        os.makedirs(stage_dir, exist_ok=True)
        frozen_playbook = os.path.join(stage_dir, "baseline_playbook.txt")
        shutil.copyfile(baseline_playbook, frozen_playbook)
        print(f"[{STAGE_NAME}] frozen baseline playbook → {frozen_playbook}")
        # Stage 1 uses the ORIGINAL generator prompt (per user design decision).
        failure_log = os.path.join(stage_dir, "llm_failures_eval.jsonl")
        jsonnet_text = _build_jsonnet(
            playbook_path=frozen_playbook,
            failure_log_path=failure_log,
            dataset=tasks,
            agent_model=agent_model,
            agent_provider=agent_provider,
            openrouter_min_throughput_p90=openrouter_min_throughput_p90,
            openrouter_max_latency_p90=openrouter_max_latency_p90,
        )
        canonical, mirrored = write_stage_config(
            jsonnet_text, experiment_name, mirror_dir=os.path.join(stage_dir, "config_canonical"),
        )
        print(f"[{STAGE_NAME}] config: canonical={canonical} mirror={mirrored}")
        # Build override for sample_size / task_ids (dataset is already in jsonnet)
        override: dict[str, Any] = {}
        if task_ids_csv:
            override = {"config": {"task_ids": task_ids}}
        elif sample_size is not None:
            override = {"config": {"sample_size": sample_size}}

        print(f"[{STAGE_NAME}] invoking appworld run on {len(task_ids)} tasks ({tasks})...")
        ctx.progress(done=0, total=len(task_ids), msg="appworld_run")
        rc = run_appworld(
            experiment_name=experiment_name,
            override=override or None,
            num_processes=num_processes,
        )
        if rc != 0:
            print(f"[{STAGE_NAME}] WARNING: appworld run returned rc={rc}; proceeding to evaluate what's present")

        # Evaluate each task individually; robust to partial runs.
        print(f"[{STAGE_NAME}] scoring tasks...")
        per_task: dict[str, Any] = {}
        total_steps = 0
        total_cost = 0.0
        n_success = 0
        n_ran = 0
        for tid in ctx.iter(task_ids, total=len(task_ids), desc="evaluate"):
            if not task_has_output(experiment_name, tid):
                per_task[tid] = {
                    "success": 0, "step_count": 0, "cost_usd": 0.0,
                    "trajectory_path": None, "note": "no_output",
                }
                continue
            traj = read_trajectory(experiment_name, tid)
            step_count = len(traj)
            total_steps += step_count
            try:
                ev = evaluate_one_task(tid, experiment_name)
                success = 1 if ev["success"] else 0
            except Exception as exc:
                print(f"[{STAGE_NAME}] evaluate_task({tid}) failed: {exc}")
                success = 0
                ev = {"success": False, "num_failed": -1, "num_passed": -1}
            cost = read_task_cost(experiment_name, tid)
            total_cost += cost
            n_ran += 1
            n_success += success
            traj_path = os.path.join(
                paths.task_output_dir(experiment_name, tid),
                "logs", "lm_calls.jsonl",
            )
            per_task[tid] = {
                "success": success,
                "step_count": step_count,
                "cost_usd": cost,
                "num_passed": ev["num_passed"],
                "num_failed": ev["num_failed"],
                "trajectory_path": traj_path if os.path.exists(traj_path) else None,
            }

        aggregate = {
            "n_tasks": len(task_ids),
            "n_ran": n_ran,
            "n_success": n_success,
            "success_rate": (n_success / n_ran) if n_ran else 0.0,
            "mean_steps": (total_steps / n_ran) if n_ran else 0.0,
            "total_cost_usd": round(total_cost, 4),
        }
        blob = {
            "per_task": per_task,
            "aggregate": aggregate,
            "config": cfg,
            "config_hash": cfg_hash,
            "experiment_name": experiment_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        write_json(out_path, blob)
        print(f"[{STAGE_NAME}] wrote {out_path}  (success={n_success}/{n_ran}, avg_steps={aggregate['mean_steps']:.1f})")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 1: Baseline evaluation")
    p.add_argument("--out", required=True, help="Output baseline_results.json path")
    p.add_argument("--tasks", default="dev", help="Dataset name (dev/train/test_normal/test_challenge)")
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--task-ids", default=None, help="Comma-separated list of task IDs")
    p.add_argument("--baseline-playbook", default=paths.INITIAL_PLAYBOOK_PATH)
    p.add_argument("--from-file", default=None, help="Skip running; copy existing baseline_results.json")
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--agent-model", default=None,
                   help="Override agent generator_model_config.name (e.g. gpt-5-mini, qwen/qwen3.5-35b-a3b)")
    p.add_argument("--agent-provider", default=None,
                   help="Override agent provider (openai|openrouter|gemini|...)")
    p.add_argument("--openrouter-throughput-p90", type=int, default=100,
                   help="OpenRouter preferred_min_throughput.p90 (only used when agent-provider=openrouter)")
    p.add_argument("--openrouter-latency-p90", type=int, default=1,
                   help="OpenRouter preferred_max_latency.p90 seconds (only when agent-provider=openrouter)")
    args = p.parse_args(argv)
    run(
        out_path=args.out,
        tasks=args.tasks,
        sample_size=args.sample_size,
        task_ids_csv=args.task_ids,
        baseline_playbook=args.baseline_playbook,
        from_file=args.from_file,
        force=args.force,
        num_processes=args.num_processes,
        agent_model=args.agent_model,
        agent_provider=args.agent_provider,
        openrouter_min_throughput_p90=args.openrouter_throughput_p90,
        openrouter_max_latency_p90=args.openrouter_latency_p90,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
