"""Shared path constants for the pipeline.

The pipeline must run against a specific ace-appworld checkout; everything
is anchored on `REPO_ROOT`. If `APPWORLD_ROOT` is set in the environment
(as required to import `appworld`), that's our anchor.
"""
from __future__ import annotations

import os

# Repo root resolution: prefer APPWORLD_ROOT, fall back to this file's parent.
_env_root = os.environ.get("APPWORLD_ROOT")
if _env_root:
    REPO_ROOT = os.path.abspath(_env_root)
else:
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
EXPERIMENT_CONFIGS_DIR = os.path.join(EXPERIMENTS_DIR, "configs")
EXPERIMENT_PROMPTS_DIR = os.path.join(EXPERIMENTS_DIR, "prompts")
EXPERIMENT_PLAYBOOKS_DIR = os.path.join(EXPERIMENTS_DIR, "playbooks")
EXPERIMENT_OUTPUTS_DIR = os.path.join(EXPERIMENTS_DIR, "outputs")
DATASETS_DIR = os.path.join(REPO_ROOT, "data", "datasets")

INITIAL_PLAYBOOK_PATH = os.path.join(EXPERIMENT_PLAYBOOKS_DIR, "appworld_initial_playbook.txt")
GENERATOR_PROMPT_PATH = os.path.join(EXPERIMENT_PROMPTS_DIR, "appworld_react_generator_prompt.txt")
EVAL_TEMPLATE_CONFIG = os.path.join(EXPERIMENT_CONFIGS_DIR, "ACE_eval_checkpoint_template.jsonnet")

PIPELINE_ROOT = os.path.join(REPO_ROOT, "memory_utility")
PIPELINE_RUNS_DIR = os.path.join(PIPELINE_ROOT, "runs")


def run_dir_for(run_name: str) -> str:
    return os.path.join(PIPELINE_RUNS_DIR, run_name)


def task_output_dir(experiment_name: str, task_id: str) -> str:
    return os.path.join(EXPERIMENT_OUTPUTS_DIR, experiment_name, "tasks", task_id)


def load_task_ids(dataset_name: str) -> list[str]:
    path = os.path.join(DATASETS_DIR, f"{dataset_name}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
