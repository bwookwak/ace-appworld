import json
import os
from typing import Any

from appworld.task import Task, load_task_ids
from appworld_experiments.code.ace.base_agent import BaseAgent
from appworld_experiments.code.ace.evaluation_agent import Agent
from appworld_experiments.code.ace.adaptation_agent import StarAgent


def run_experiment(
    experiment_name: str,
    runner_config: dict[str, Any],
    task_id: str | None = None,
    num_processes: int = 1,
    process_index: int = 0,
) -> None:
    run_type = runner_config.pop("run_type")
    agent_config = runner_config.pop("agent")
    dataset_name = runner_config.pop("dataset", None)
    sample_size = runner_config.pop("sample_size", None)
    custom_task_ids = runner_config.pop("task_ids", None)
    num_epochs = runner_config.pop("num_epochs", 1)
    resume_from_checkpoint = runner_config.pop("resume_from_checkpoint", False)

    if runner_config:
        raise Exception(f"Unexpected keys in the runner config: {runner_config}")

    if task_id:
        task_ids = [task_id]
    elif custom_task_ids:
        task_ids = list(custom_task_ids)
        print(f"Using custom task list: {len(task_ids)} tasks")
    else:
        if dataset_name is None:
            raise Exception("Either 'dataset' or 'task_ids' must be specified in the config")
        task_ids = load_task_ids(dataset_name)
        if sample_size is not None:
            task_ids = task_ids[:sample_size]

    for task_id_ in task_ids:
        Task.load(task_id=task_id_)

    task_ids = task_ids * num_epochs

    # Resume support: only meaningful for adaptation runs.
    if resume_from_checkpoint:
        if run_type != "ace-adaptation":
            print(
                f"[WARN] resume_from_checkpoint=True is only supported for ace-adaptation; "
                f"got run_type={run_type}. Ignoring."
            )
        else:
            trained_path = agent_config.get("trained_playbook_file_path")
            if not trained_path:
                raise ValueError(
                    "resume_from_checkpoint=True but agent.trained_playbook_file_path is not set."
                )
            base_no_ext = trained_path[:-4] if trained_path.endswith(".txt") else trained_path
            state_path = f"{base_no_ext}_checkpoint_state.json"
            if not os.path.exists(state_path):
                print(
                    f"[INFO] resume_from_checkpoint=True but no checkpoint_state.json at "
                    f"{state_path}. Starting fresh."
                )
            else:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                last_idx = int(state["global_task_index"])  # number of completed tasks so far
                if last_idx >= len(task_ids):
                    print(
                        f"[INFO] All {len(task_ids)} tasks already completed in previous run "
                        f"(last_idx={last_idx}). Nothing to do."
                    )
                    return
                snapshot_path = state.get("playbook_snapshot_path")
                if snapshot_path and os.path.exists(snapshot_path):
                    agent_config["initial_playbook_file_path"] = snapshot_path
                    print(f"[RESUME] loading playbook from {snapshot_path}")
                else:
                    print(
                        f"[WARN] checkpoint snapshot not found at {snapshot_path}; "
                        f"falling back to current trained_playbook_file_path."
                    )
                    if os.path.exists(trained_path):
                        agent_config["initial_playbook_file_path"] = trained_path
                agent_config["global_task_offset"] = last_idx
                last_completed = state.get("last_completed_task_id")
                remaining = len(task_ids) - last_idx
                print(
                    f"[RESUME] Resuming adaptation from global task index {last_idx} "
                    f"(last completed: {last_completed}). Remaining: {remaining} tasks."
                )
                task_ids = task_ids[last_idx:]

    if run_type == "ace-adaptation":
        agent = StarAgent.from_dict(agent_config)
    elif run_type == "ace-evaluation":
        agent = Agent.from_dict(agent_config)
    elif run_type == "non-ace-evaluation":
        agent = BaseAgent.from_dict(agent_config)
    else:
        raise ValueError(f"Unknown run_type: {run_type}")

    agent.solve_tasks(
        task_ids=task_ids,
        experiment_name=experiment_name,
        num_processes=num_processes,
        process_index=process_index,
    )
