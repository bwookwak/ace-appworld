"""Stage 2a — Instrumented Full-Memory Run (AGENT ONLY).

Runs the ACE evaluation agent with the FULL memory + citation-instrumented
prompt. Does NOT call the judge or emit instrumented_logs — that's Stage 2b.

Splitting agent-run from analysis lets us:
  - retry/extend only the agent phase without re-paying judge cost
  - survive agent crashes (Stage 2b still works on whatever output exists)
  - rerun Stage 2b with different judge models / sampling rates cheaply

Outputs under `<out_dir>/`:
  - full_memory_playbook.txt     (rendered from memory.json)
  - generator_prompt.txt         (amended with citation instruction)
  - config_canonical/*.jsonnet   (the stage-local wrapper config)
  - stage2a_manifest.json        (records which tasks have agent output)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

from .common import paths
from .common.appworld_runner import run_appworld, task_has_output, write_stage_config
from .common.config_hash import hash_config, stage_cache_ok
from .common.io import read_json, read_text, write_json, write_text
from .common.playbook import render_playbook


STAGE_NAME = "stage2a_run"


CITATION_INSTRUCTION = (
    "(9) After your code block on each step, emit exactly one additional "
    "line of the form:\n"
    "[cited_insights: id1, id2]\n"
    "listing the playbook bullet IDs you actually relied on for this step. "
    "Use IDs exactly as they appear in brackets in the playbook (e.g. shr-00005, api-00004). "
    "If you did not use any playbook bullet for this step, write:\n"
    "[cited_insights: none]\n"
    "This line must appear AFTER the closing ``` of your code block. "
    "Do not mix it into the code. Do not skip it."
)


def _amend_generator_prompt(src_text: str) -> str:
    anchor = "Using these APIs and playbook, generate code to solve the actual task"
    idx = src_text.rfind(anchor)
    if idx == -1:
        return src_text.rstrip() + "\n\n" + CITATION_INSTRUCTION + "\n\n" + anchor + ":\n"
    head = src_text[:idx].rstrip()
    tail = src_text[idx:]
    return head + "\n\n" + CITATION_INSTRUCTION + "\n\n\n\n\nUSER:\n" + tail


def _build_jsonnet(
    playbook_path: str,
    generator_prompt_path: str,
    failure_log_path: str,
    dataset: str | None,
    agent_model: str | None = None,
    agent_provider: str | None = None,
    openrouter_min_throughput_p90: int = 100,
    openrouter_max_latency_p90: int = 1,
) -> str:
    lines: list[str] = []
    lines.append(f'local base = import "{paths.EVAL_TEMPLATE_CONFIG}";')
    lines.append("base + {")
    lines.append("  config+: {")
    if dataset:
        lines.append(f'    dataset: {json.dumps(dataset)},')
    lines.append("    agent+: {")
    lines.append(f'      trained_playbook_file_path: {json.dumps(playbook_path)},')
    lines.append(f'      generator_prompt_file_path: {json.dumps(generator_prompt_path)},')
    lines.append("      generator_model_config+: {")
    lines.append(f'        failure_log_path: {json.dumps(failure_log_path)},')
    lines.append('        on_failure: "warn",')
    lines.append("        max_retries: 10,")
    if agent_model:
        lines.append(f'        name: {json.dumps(agent_model)},')
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


def run(
    memory_path: str,
    out_dir: str,
    tasks: str = "dev",
    sample_size: int | None = None,
    task_ids_csv: str | None = None,
    force: bool = False,
    num_processes: int = 1,
    agent_model: str | None = None,
    agent_provider: str | None = None,
    openrouter_min_throughput_p90: int = 100,
    openrouter_max_latency_p90: int = 1,
) -> str:
    from .common.progress import stage_context
    run_dir = os.path.dirname(os.path.abspath(out_dir).rstrip("/"))
    run_name = os.path.basename(run_dir.rstrip("/")) or "default"
    experiment_name = f"memutil_{run_name}_stage2"
    manifest_path = os.path.join(out_dir, "stage2a_manifest.json")

    memory_blob = read_json(memory_path)
    memory_insights = list(memory_blob["insights"])
    task_ids = _resolve_task_ids(tasks, sample_size, task_ids_csv)

    cfg = {
        "memory_path": os.path.abspath(memory_path),
        "tasks_dataset": tasks,
        "sample_size": sample_size,
        "task_ids_csv": task_ids_csv,
        "resolved_task_ids": task_ids,
        "experiment_name": experiment_name,
        "agent_model": agent_model,
        "agent_provider": agent_provider,
        "openrouter_min_throughput_p90": openrouter_min_throughput_p90,
        "openrouter_max_latency_p90": openrouter_max_latency_p90,
    }
    cfg_hash = hash_config(cfg, input_file_paths=[memory_path, paths.GENERATOR_PROMPT_PATH])
    if not force and stage_cache_ok(manifest_path, cfg_hash):
        print(f"[{STAGE_NAME}] cache hit: {manifest_path}")
        return out_dir

    os.makedirs(out_dir, exist_ok=True)
    with stage_context(STAGE_NAME, run_dir) as ctx:
        playbook_text = render_playbook(memory_insights)
        frozen_playbook = os.path.join(out_dir, "full_memory_playbook.txt")
        write_text(frozen_playbook, playbook_text)
        print(f"[{STAGE_NAME}] rendered {len(memory_insights)} insights → {frozen_playbook}")

        src_prompt = read_text(paths.GENERATOR_PROMPT_PATH)
        amended_prompt = _amend_generator_prompt(src_prompt)
        frozen_prompt = os.path.join(out_dir, "generator_prompt.txt")
        write_text(frozen_prompt, amended_prompt)
        print(f"[{STAGE_NAME}] amended prompt → {frozen_prompt}")

        failure_log = os.path.join(out_dir, "llm_failures_eval.jsonl")
        jsonnet_text = _build_jsonnet(
            playbook_path=frozen_playbook,
            generator_prompt_path=frozen_prompt,
            failure_log_path=failure_log,
            dataset=tasks,
            agent_model=agent_model,
            agent_provider=agent_provider,
            openrouter_min_throughput_p90=openrouter_min_throughput_p90,
            openrouter_max_latency_p90=openrouter_max_latency_p90,
        )
        canonical, mirrored = write_stage_config(
            jsonnet_text, experiment_name, mirror_dir=os.path.join(out_dir, "config_canonical"),
        )
        print(f"[{STAGE_NAME}] config: canonical={canonical} mirror={mirrored}")

        # Tasks already done? (appworld may skip but we also let the user see)
        pre_done = [tid for tid in task_ids if task_has_output(experiment_name, tid)]
        pre_missing = [tid for tid in task_ids if tid not in pre_done]
        print(f"[{STAGE_NAME}] pre-existing agent outputs: {len(pre_done)}/{len(task_ids)}")
        if pre_done:
            print(f"[{STAGE_NAME}]   (appworld may skip or re-run these — test with a single task if unsure)")

        override: dict[str, Any] = {}
        if task_ids_csv:
            override = {"config": {"task_ids": task_ids}}
        elif sample_size is not None:
            override = {"config": {"sample_size": sample_size}}

        print(f"[{STAGE_NAME}] invoking appworld run on {len(task_ids)} tasks ({tasks})...")
        rc = run_appworld(
            experiment_name=experiment_name,
            override=override or None,
            num_processes=num_processes,
        )
        if rc != 0:
            print(f"[{STAGE_NAME}] WARNING: appworld run returned rc={rc}; capturing partial state in manifest")

        post_done = [tid for tid in task_ids if task_has_output(experiment_name, tid)]
        post_missing = [tid for tid in task_ids if tid not in post_done]
        print(f"[{STAGE_NAME}] post-run agent outputs: {len(post_done)}/{len(task_ids)}")

        manifest = {
            "experiment_name": experiment_name,
            "playbook_path": frozen_playbook,
            "generator_prompt_path": frozen_prompt,
            "config_mirrored": mirrored,
            "task_ids_requested": task_ids,
            "tasks_with_output": post_done,
            "tasks_missing_output": post_missing,
            "appworld_rc": rc,
            "config": cfg,
            "config_hash": cfg_hash,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        write_json(manifest_path, manifest)
        print(f"[{STAGE_NAME}] wrote {manifest_path}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 2a: Agent-only run with citation instrumentation")
    p.add_argument("--memory", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tasks", default="dev")
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--task-ids", default=None)
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--agent-model", default=None)
    p.add_argument("--agent-provider", default=None)
    p.add_argument("--openrouter-throughput-p90", type=int, default=100)
    p.add_argument("--openrouter-latency-p90", type=int, default=1)
    args = p.parse_args(argv)
    run(
        memory_path=args.memory,
        out_dir=args.out_dir,
        tasks=args.tasks,
        sample_size=args.sample_size,
        task_ids_csv=args.task_ids,
        agent_model=args.agent_model,
        agent_provider=args.agent_provider,
        openrouter_min_throughput_p90=args.openrouter_throughput_p90,
        openrouter_max_latency_p90=args.openrouter_latency_p90,
        force=args.force,
        num_processes=args.num_processes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
